"""VLM descriptions of FP/FN crops — **diagnostic signal only** (never on the
decision path).

Every sample that will appear in the prompt is described
(diagnose.sample.pick_detail). Describing only some would let the call budget,
not severity, decide which get a note, and the planner would read "no note"
as "less important" when it was merely cut. Crops per call are bounded by
batching (CROPS_PER_CALL).

Crops come **only from train detail** — holdout diagnosis is anonymized
aggregates, so it structurally cannot supply crops. Results are merged into
diagnosis["train"]["crop_notes"] and appear only in the planner prompt
(proposal material); adoption is still decided solely by the holdout CI gate.

The artifact (crop_notes.json) stores crop metadata, per-batch responses, and
notes; resume replays it without re-calling — failures are recorded too and
not retried (same as proposal.json replay).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from ..core.coco import Split, crop_image
from ..core.io import data_uri
from .sample import pick_detail

log = logging.getLogger(__name__)

ARTIFACT = "crop_notes.json"
CROPS_PER_CALL = 4       # crops per call — too many at once blurs the descriptions,
                         # and servers cap images per prompt: under vLLM
                         # --limit-mm-per-prompt=4 a batch of 6 is rejected **whole**
                         # (400, no notes at all). The default must stay under the
                         # cap — over it the result is not fewer notes but none.
MIN_CROP_PX = 96         # upscale if the short side is smaller — error crops are
                         # often a few dozen px, too small for the VLM to read shape

SCHEMA = {
    "name": "crop_notes",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"crop": {"type": "integer"},
                                   "note": {"type": "string"}},
                    "required": ["crop", "note"],
                },
            },
        },
        "required": ["notes"],
    },
}

PROMPT_HEAD = (
    "You will see cropped regions from an object detector's error analysis on "
    "the train split. For each crop, state in one short sentence what the "
    "region actually contains and the most likely visual cause of the error "
    "(e.g. reflection, look-alike object, occlusion, unusual viewpoint, tiny "
    "or blurry object). The kind tells you which error it is: "
    "background = a false positive on this region; "
    "class_confusion = detected as the wrong class; "
    "missed = this ground-truth object was not detected at all; "
    "missed_small = a small ground-truth object that was not detected; "
    "below_threshold = detected, but scored below the class threshold; "
    "localization = detected, but the box does not overlap the object well; "
    "duplicate = an extra near-duplicate box on an already-detected object. "
    "Crops may be upscaled from a few dozen pixels, so they can look soft. "
    "Answer with JSON only.")


def select_crops(diag_train: dict) -> list[dict]:
    """Crops to describe — every prompt sample (no kind filter, no count cap).

    Samples and descriptions must go through the same selection so each note
    lands on its sample as `note` (planner.llm._merge_notes). Only rows without
    a box or image are dropped — no crop can be made from them.
    """
    return [r for side in ("fp", "fn")
            for r in pick_detail(diag_train.get(side) or [])
            if r.get("image") and r.get("box")]


def _prepared(crop: Image.Image) -> Image.Image:
    """Upscale, keeping aspect ratio, if the short side is below MIN_CROP_PX."""
    w, h = crop.size
    short = min(w, h)
    if short <= 0 or short >= MIN_CROP_PX:
        return crop
    scale = MIN_CROP_PX / short
    return crop.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                       Image.LANCZOS)


def _crop_payloads(split: Split, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Rows → (crop metadata, data URIs). Rows whose image is missing are silently skipped."""
    by_name = {e.file_name: e for e in split.images}
    metas: list[dict] = []
    uris: list[str] = []
    for r in rows:
        entry = by_name.get(r["image"])
        if entry is None:
            continue
        with Image.open(split.image_path(entry)) as im:
            crop = crop_image(im.convert("RGB"), r["box"])
        meta = {"crop": len(metas), "kind": r["kind"],
                "category": r.get("category"),
                "category_id": r.get("category_id"), "image": r["image"],
                "box": list(r["box"])}
        if "score" in r:
            meta["score"] = r["score"]
        metas.append(meta)
        uris.append(data_uri(_prepared(crop)))
    return metas, uris


def _describe_batch(chat, batch: list[dict], uris: list[str]) -> tuple[dict, dict | None, str | None]:
    """Describe one batch — (crop index → note, raw response, error).

    Crop numbers in the prompt are **0-based within the batch**. Using global
    numbers would silently misalign when the model restarts counting at 0."""
    parts: list[dict] = [{"type": "text", "text": PROMPT_HEAD}]
    for i, m in enumerate(batch):
        # Give the class by name — a numeric id is no clue for the description
        label = m.get("category") or f"id {m.get('category_id')}"
        parts.append({"type": "text",
                      "text": f"crop {i} (kind={m['kind']}, class={label}):"})
        parts.append({"type": "image_url",
                      "image_url": {"url": uris[m["crop"]]}})
    raw, error = None, None
    try:
        raw = chat([{"role": "user", "content": parts}], schema=SCHEMA)
    except Exception as exc:  # noqa: BLE001 — a failed description just means no description
        error = f"{type(exc).__name__}: {exc}"
        log.warning("crop description call failed (%s) — continuing this batch without notes", error)
    out: dict[int, str] = {}
    if isinstance(raw, dict) and isinstance(raw.get("notes"), list):
        for n in raw["notes"]:
            if not isinstance(n, dict):
                continue
            note = str(n.get("note", "")).strip()
            idx = n.get("crop")
            if note and isinstance(idx, int) and 0 <= idx < len(batch):
                out[batch[idx]["crop"]] = note
    return out, raw, error


def crop_notes(split: Split, diag_train: dict, chat,
               out_path: Path) -> list[dict] | None:
    """Describe sample crops — [{"crop", "kind", "category_id", "image", "box",
    ("score",) "note"}] or None (no notes / failure). Replay: if out_path exists,
    return the stored notes without re-calling.

    A failed batch does not discard the other batches' notes."""
    out_path = Path(out_path)
    if out_path.exists():
        return json.loads(out_path.read_text()).get("notes") or None
    rows = select_crops(diag_train)
    if not rows:
        return None
    metas, uris = _crop_payloads(split, rows)
    if not metas:
        return None

    by_idx: dict[int, str] = {}
    calls: list[dict] = []
    for start in range(0, len(metas), CROPS_PER_CALL):
        batch = metas[start:start + CROPS_PER_CALL]
        got, raw, error = _describe_batch(chat, batch, uris)
        by_idx.update(got)
        calls.append({"crops": [m["crop"] for m in batch],
                      "response": raw, "error": error})
    notes = [{**m, "note": by_idx[m["crop"]]}
             for m in metas if by_idx.get(m["crop"])] or None
    log.info("crop notes: %d/%d (%d calls)",
             len(notes or []), len(metas), len(calls))
    # Prompt images (base64) are not stored — the crop metadata is enough to replay
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"crops": metas, "notes": notes, "calls": calls},
        indent=2, ensure_ascii=False, default=str))
    return notes
