"""Caption pool → text prompt phrases (negative / alias). LLM refinement (hygiene filter).

Phrases the captioner attaches to mined negative crops cannot be used as-is —
many **describe the suppression target itself** (containing the class name, a
part of the target, the target's object family). Using such phrases as
suppression prompts **suppresses the true positives**. The judgment needs world
knowledge (e.g. 'red cap' is part of a coca-cola bottle but a distinct object
next to a yellow ball) — string matching or text embeddings cannot separate
these, hence an LLM.

**This stage is a filter, not a gate** — it only judges semantic conflict;
whether a surviving phrase actually reduces FPs is decided by the adoption gate.

Uses vLLM's `/v1/chat/completions` + `json_schema` (same server as the planner).
A reply that does not parse as JSON is asked once more (one-off degeneration);
if it still fails, or on server error, the fallback is **to drop the phrase** —
no suppression is the status quo, while wrong suppression costs recall. The
failure warning keeps the finish reason and the head and tail of the reply.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import requests

from ..clients.llm import JSON_ATTEMPTS, REQUEST_TIMEOUT, LLMReplyError, parse_reply
from ..clients.llm import post_with_retry

log = logging.getLogger(__name__)

_ARTICLE = re.compile(r"^(?:a|an|the)\s+")


def _ask_json(sess: requests.Session, base_url: str, payload: dict) -> dict:
    """A JSON reply — asked once more if it does not parse (the same rule as
    clients.llm.LLMClient.chat_json). The last failure is raised as
    LLMReplyError (finish reason and the head and tail of the reply included)."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    for attempt in range(1, JSON_ATTEMPTS + 1):
        resp = post_with_retry(sess, url, payload, timeout=REQUEST_TIMEOUT)
        try:
            return parse_reply(resp.json())
        except LLMReplyError as exc:
            if attempt == JSON_ATTEMPTS:
                raise
            log.warning("refinement reply did not parse as JSON (%s) — asking again", exc)


def normalize_caption(text: str) -> str:
    """Normalize a caption — lowercase, collapse whitespace, strip punctuation and
    a leading article.

    The frequency table counts exact matches, so surface variants such as
    "A damaged cable." / "damaged cable" split the count and understate the true
    mode strength. Fold them here before counting. The article is stripped only
    if something remains (protects article-only captions).

    **Idempotent** — articles and punctuation are stripped **alternately** until
    a fixed point. Repeating only one leaves punctuation exposed by removing an
    article (`the "red cap"` → `"red cap`), which then hides the next article, so
    f(f(x)) ≠ f(x). There are two call paths (pool construction in mining,
    re-counting in refine_*); if non-idempotent, one phrase gets two forms and
    the frequency table, cross-pool filter and verbatim check disagree.
    """
    t = " ".join(str(text).split()).lower()
    while True:
        prev = t
        t = t.strip(".,;:!?\"'").strip()
        stripped = _ARTICLE.sub("", t)
        if stripped:                      # keep article-only captions
            t = stripped
        if t == prev:
            return t

SYSTEM = ("You filter candidate *negative* phrases for an open-vocabulary object "
          "detector. Answer with JSON only.")

# This prompt is **a safety filter only**. The refiner sees only phrase text, not
# the crops, so it does not judge "visually confusable?" — the phrase came from
# the caption of a source crop (a false detection, another class's GT, or a
# background false detection), and its effect is measured by the gate. The
# refiner checks only rules 1-4, answerable from text alone: using a synonym,
# part, or parent category of the target as a negative suppresses the target
# itself. Other category names are blocked too: the client emulates suppression
# target-agnostically (a negative removes any overlapping positive, not only the
# target's — clients.yoloe), so a negative naming another category suppresses
# that category.
# Where the phrase pool came from — stated truthfully per source (the
# text_negatives value). gt_crop is not a false detection but another class's GT crop
SOURCE_INTRO = {
    "fp_crop": 'crops where the detector wrongly fired for "{target}"',
    "gt_crop": 'ground-truth crops of the other categories, which may look like "{target}"',
    "background": 'wrong detections for "{target}" that overlap no ground-truth object',
}

RULES = """The detector is being prompted to find these categories:
{categories}

For the category "{target}", an automatic captioner looked at {source} and
produced these phrases (with how often each phrase occurred):
{pool}

These phrases are candidate NEGATIVE prompts for "{target}". A negative
suppresses every detection it overlaps, of any category, so a phrase naming
another category in the list would suppress that category too. Do NOT judge whether the confusion looks plausible to you: you cannot see
the crops, and a separate measurement afterwards decides whether each phrase
actually helps.

Your only job is a SAFETY filter. KEEP every phrase unless one of the following
holds -- each is answerable from the phrase text alone:
1. It names, contains, or is a synonym of "{target}" or of any other category
   in the list above.
2. It names a PART or an ATTRIBUTE of "{target}" (e.g. for a cola bottle,
   "red cap" is its cap -- suppressing it would suppress the bottle itself).
3. It names the object family or material that "{target}" belongs to (e.g. for
   a damaged wire, "black cable" and "steel wire" describe the target's own
   family).
4. It is so generic that it would match the target as well (e.g. "object",
   "thing", "red shape").

Return the phrases you keep, unchanged, in descending order of count, and each
rejected phrase with the number of the rule (1, 2, 3 or 4) that applies. Do not
invent other grounds for rejection."""

SCHEMA = {
    "name": "negative_phrases",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "keep": {"type": "array", "items": {"type": "string"}},
            "rejected": {
                "type": "array",
                "items": {
                    "type": "object",
                    # the reason is the rule number only — free text lengthens the
                    # reply, and a truncated reply leaves the whole class with keep=[]
                    "properties": {"phrase": {"type": "string"},
                                   "rule": {"type": "integer", "enum": [1, 2, 3, 4]}},
                    "required": ["phrase", "rule"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["keep", "rejected"],
        "additionalProperties": False,
    },
}


def _names(phrase: str, name: str) -> bool:
    """Does phrase contain name as whole words, or name contain phrase — 'category'
    does not contain 'cat'. Plurals (-s, -es) count as the same name."""
    p, n = phrase.lower(), name.lower()

    def within(a, b):
        return re.search(rf"(?<![a-z0-9]){re.escape(a)}(?:e?s)?(?![a-z0-9])", b) is not None
    return within(n, p) or within(p, n)


def refine_negatives(base_url: str, model: str, target: str, categories: list[str],
                     captions: list[str], *, max_keep: int = 5,
                     source: str = "fp_crop",
                     session: requests.Session | None = None) -> dict:
    """Captions → {"keep": [phrase], "rejected": [{phrase, rule}]}.

    `source` is where the pool came from (fp_crop, gt_crop, background) — the
    prompt's opening says so. The max_keep cap is applied by code (it is not in the
    prompt). Other category names are blocked too — emulated suppression removes any
    overlapping positive, so a negative naming another category suppresses it.
    Returns keep=[] on failure — dropping is the safe direction.
    """
    pool = Counter(p for c in captions if c and (p := normalize_caption(c)))
    if not pool:
        return {"keep": [], "rejected": [], "error": "empty caption pool"}
    pool_txt = "\n".join(f"- {p!r} ×{n}" for p, n in pool.most_common())
    prompt = RULES.format(categories="\n".join(f"- {c!r}" for c in categories),
                          target=target, pool=pool_txt,
                          source=SOURCE_INTRO[source].format(target=target))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        # A large pool makes a long phrase list — too small a budget truncates
        # the response and breaks parsing (json_schema enforcement does not
        # prevent truncation).
        "max_tokens": 1536,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
    }
    sess = session or requests.Session()
    try:
        parsed = _ask_json(sess, base_url, payload)
        keep = [p for p in parsed.get("keep", []) if isinstance(p, str) and p.strip()]
        # Mechanical double-check for rule 1 in case the LLM misses it — cheap and
        # harmless. Every category name is checked (whole words)
        safe, dropped = [], list(parsed.get("rejected", []))
        for p in keep:
            hit = next((c for c in [target, *categories] if _names(p, c)), None)
            if hit is not None:
                dropped.append({"phrase": p, "rule": 1,
                                "note": f"mechanical check: contains category name {hit!r}"})
                continue
            safe.append(p)
        # the cap is cut by code in count order — not relying on the LLM's order
        safe.sort(key=lambda p: -pool.get(normalize_caption(p), 0))
        return {"keep": safe[:max_keep], "rejected": dropped}
    except Exception as exc:                                  # noqa: BLE001
        log.warning("negative phrase refinement failed (%s: %s) — dropping phrases",
                    type(exc).__name__, exc)
        return {"keep": [], "rejected": [], "error": f"{type(exc).__name__}: {exc}"}


ALIAS_SYSTEM = ("You improve the *positive* text prompt of an open-vocabulary "
                "object detector. Answer with JSON only.")

ALIAS_RULES = """The detector is prompted to find these categories:
{categories}

For the category "{target}", an automatic captioner looked at crops of GROUND
TRUTH instances of "{target}" and produced these phrases (with counts):
{pool}

For CONTRAST, the same captioner described ground-truth crops of the OTHER
categories like this:
{others}

Pick AT MOST ONE phrase from the "{target}" list to ADD alongside "{target}"
as an extra text prompt for the same category. A good addition is a plain
visual noun phrase describing what "{target}" instances look like -- useful
when the category name is a slug, jargon, or under-descriptive.

Rules:
1. The phrase must describe "{target}" itself -- not a part of it, not the
   background. A bare family noun ("bottle", "animal") is too broad; a family
   noun plus a distinguishing attribute ("red soda can") is good.
2. It must NOT describe (or be easily confused with) any OTHER category -- in
   particular, avoid anything that closely resembles another category's phrase
   list above.
3. Prefer specific over generic; never pick phrases like "object" or "thing".
4. Return the phrase EXACTLY as it appears in the "{target}" list -- do not
   invent, merge, or reword phrases.
5. If no phrase adds anything beyond the name "{target}" itself, return null.

Give the reason in one short sentence."""

ALIAS_SCHEMA = {
    "name": "alias_phrase",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "alias": {"type": ["string", "null"]},
            # a truncated reply breaks parsing and falls back to alias=None (max_tokens 256)
            "reason": {"type": "string", "maxLength": 200},
        },
        "required": ["alias", "reason"],
        "additionalProperties": False,
    },
}


def refine_alias(base_url: str, model: str, target: str, categories: list[str],
                 captions: list[str], *,
                 other_captions: dict[str, list[str]] | None = None,
                 session: requests.Session | None = None) -> dict:
    """Captions → {"alias": phrase|None, "reason": str}.

    - `other_captions`: caption pools of the other categories — empirical
      contrast evidence for the distinctiveness check (rule 2). Names alone cannot
      tell whether two classes' caption distributions overlap.
    - The returned phrase must be **verbatim from the pool** (normalized form) —
      captioner output is a vocabulary sample grounded in crops, and synthesized
      phrases have no grounding. The mechanical rule-4 check drops out-of-pool
      phrases.
    - alias=None on failure — using the class name as-is is the safe direction
      (symmetric with negatives' "drop": both fall back to the status quo).
    """
    pool = Counter(p for c in captions if c and (p := normalize_caption(c)))
    if not pool:
        return {"alias": None, "reason": "empty caption pool"}
    pool_txt = "\n".join(f"- {p!r} ×{n}" for p, n in pool.most_common())
    others_txt = "(none)"
    if other_captions:
        blocks = []
        for other, caps in sorted(other_captions.items()):
            table = Counter(p for c in caps if c and (p := normalize_caption(c)))
            if table:
                blocks.append(f"{other!r}:\n" + "\n".join(
                    f"  - {p!r} ×{n}" for p, n in table.most_common()))
        if blocks:
            others_txt = "\n".join(blocks)
    prompt = ALIAS_RULES.format(
        categories="\n".join(f"- {c!r}" for c in categories),
        target=target, pool=pool_txt, others=others_txt)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": ALIAS_SYSTEM},
                     {"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": ALIAS_SCHEMA},
    }
    sess = session or requests.Session()
    try:
        parsed = _ask_json(sess, base_url, payload)
        alias = parsed.get("alias")
        if not (isinstance(alias, str) and alias.strip()):
            return {"alias": None, "reason": str(parsed.get("reason", ""))}
        alias = normalize_caption(alias)
        # Mechanical rule-4 check — out-of-pool phrases (synthesized/reworded) are ungrounded
        if alias not in pool:
            return {"alias": None,
                    "reason": f"mechanical check: {alias!r} is not in the caption pool (not verbatim)"}
        if alias == target.lower():
            return {"alias": None, "reason": "same as the class name — adds nothing"}
        # Mechanical rule-2 check — drop if it contains/is contained in another
        # category name as whole words (containing its own name is harmless — just
        # more specific)
        for c in categories:
            if c != target and _names(alias, c):
                return {"alias": None,
                        "reason": f"mechanical check: contains/contained in another category {c!r}"}
        return {"alias": alias, "reason": str(parsed.get("reason", ""))}
    except Exception as exc:                                  # noqa: BLE001
        log.warning("alias refinement failed (%s: %s) — keeping the class name",
                    type(exc).__name__, exc)
        return {"alias": None, "reason": f"{type(exc).__name__}: {exc}"}

# ----------------------------------------------------- VLM captioner (alias source)

VLM_CAPTION_SYSTEM = ("You write short search phrases for objects in image crops. "
                      "These phrases become text prompts for an open-vocabulary "
                      "object detector. Answer with JSON only.")

VLM_CAPTION_RULES = """Below are {n} crops. Each is a tight crop of ONE object, and all of
them show instances of the same object category. Crops can be small and
upscaled, so they may look soft.

For every crop, give 1-3 short phrases someone would type to search for that
object:
- Each phrase must contain a noun naming the object type, optionally with 1-2
  visual attributes (e.g. "rusty bolted plate", "orange clownfish"); never an
  attribute alone.
- Do NOT describe the background, the crop border, or the image quality.
- Plain lowercase, 1-4 words, no sentences, no punctuation.
- Use consistent wording for crops that look alike, so the phrase counts are
  meaningful.
- If the object is hard to make out, give your single best guess of its type as
  a plain noun (e.g. "metal plate"). Never return an empty list.

Return one entry per crop, with "crop" set to the number shown before it."""


def vlm_caption_schema(n: int) -> dict:
    """Response schema for one batch — exactly one entry per crop, 1-3 phrases each."""
    return {
        "name": "crop_captions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "captions": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "items": {
                        "type": "object",
                        "properties": {
                            "crop": {"type": "integer", "enum": list(range(n))},
                            "phrases": {"type": "array", "minItems": 1, "maxItems": 3,
                                        "items": {"type": "string", "maxLength": 40}},
                        },
                        "required": ["crop", "phrases"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["captions"],
            "additionalProperties": False,
        },
    }


VLM_BATCH = 4            # crops shown per call — more blurs the descriptions, and a
                         # server capping images per prompt (--limit-mm-per-prompt)
                         # rejects an over-limit batch whole (400). Same as
                         # diagnose.describe.CROPS_PER_CALL.


def vlm_caption(rows: list[dict], chat, *, open_image=None,
                batch: int = VLM_BATCH) -> dict[str, list[str]]:
    """Crops → phrase lists. Captioner contract: rows → {id: [phrase, ...]}.

    The VLM looks at each crop directly, yielding free-form visual descriptions
    rather than "class name + context" phrases that the alias safety rules (own
    name / other-category confusion) would reject. Calls are costly, so crops are
    batched. A batch never mixes classes (the part of the id before '#' is the
    class) — mixed, the "same wording for look-alikes" rule carries one class's
    phrases over to another. Small crops are upscaled by the same rule as the
    diagnosis descriptions. Call failures and invalid responses degrade to **an
    empty result**
    (mining yields no material, and the loop rejects the candidate as
    no_material)."""
    from PIL import Image

    from ..core.coco import crop_image, upscale_for_vlm
    from ..core.io import data_uri

    def _open(path):
        return Image.open(path)

    opener = open_image or _open
    out: dict[str, list[str]] = {}
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(str(r["id"]).rsplit("#", 1)[0], []).append(r)
    chunks = [group[start:start + batch] for group in by_class.values()
              for start in range(0, len(group), batch)]
    for chunk in chunks:
        parts: list[dict] = []
        ids: list[str] = []
        for r in chunk:
            try:
                im = opener(r["image"])
                crop = crop_image(im.convert("RGB"), r["bbox_xyxy"])
            except Exception:            # noqa: BLE001 — a failed crop drops only that row
                log.warning("crop failed — skipping: %s", r.get("id"))
                continue
            parts.append({"type": "text", "text": f"crop {len(ids)}:"})
            parts.append({"type": "image_url",
                          "image_url": {"url": data_uri(upscale_for_vlm(crop))}})
            ids.append(r["id"])
        if not ids:
            continue
        parts.insert(0, {"type": "text", "text": VLM_CAPTION_RULES.format(n=len(ids))})
        try:
            raw = chat([{"role": "system", "content": VLM_CAPTION_SYSTEM},
                        {"role": "user", "content": parts}],
                       schema=vlm_caption_schema(len(ids)))
        except Exception as exc:         # noqa: BLE001 — fallback is an empty result
            log.warning("VLM caption call failed (%s) — skipping this batch", exc)
            continue
        for item in (raw or {}).get("captions") or []:
            if not isinstance(item, dict):
                continue
            i = item.get("crop")
            phrases = [str(x).strip() for x in (item.get("phrases") or [])
                       if str(x).strip()]
            if isinstance(i, int) and 0 <= i < len(ids) and phrases:
                out[ids[i]] = phrases
    if len(out) < len(rows):
        # a failed call, an unopenable crop or an invalid reply leaves that crop without a caption
        log.warning("VLM caption: %d of %d crops have no caption",
                    len(rows) - len(out), len(rows))
    return out
