"""phrases — normalization idempotence, mechanical double-check in refine, safety-filter prompt contract."""

import json
from collections import Counter

import pytest

from agentic_configuration_search.infer import phrases
from agentic_configuration_search.infer.phrases import normalize_caption

ALIAS = "shiny widget"


def test_normalize_caption():
    assert normalize_caption("  A Damaged   Cable. ") == "damaged cable"
    assert normalize_caption("THE broken wire") == "broken wire"
    assert normalize_caption("an apple") == "apple"
    assert normalize_caption("a") == "a"                     # a lone article is preserved


def test_normalize_caption_is_idempotent():
    """There are two call paths (pool construction, refine_* re-aggregation), so it must be idempotent —
    otherwise one phrase gets two surface forms: (1) the frequency table splits, (2) the cross-pool
    filter catches only one form and is bypassed, (3) phrases the LLM picked from the pool fail the verbatim check."""
    f = normalize_caption
    for raw in ("A The Cap.", "a a bottle", "the the wire", "  An A Ring ",
                'the "red cap"', "an 'empty bottle'", 'The "Soda Bottle".',
                "the ...cap", "the ; wire", 'a \'the "a cap"\''):
        once = f(raw)
        assert f(once) == once, raw
    assert f('the "red cap"') == "red cap"
    assert f("A The Cap.") == "cap"
    assert f("a") == "a"


def test_normalize_caption_merges_surface_variants():
    """The frequency table counts exact matches — surface variants must fold into one key for mode strength to be right."""
    caps = ['the "red cap"', '"red cap"', 'a red cap', 'red cap.',
            "The Red Cap", "the 'red cap'"]
    assert dict(Counter(normalize_caption(c) for c in caps)) == {"red cap": 6}


def test_refine_prompt_is_safety_filter_only():
    """The refine prompt must be a **safety filter** — it does not ask the LLM about confusability.
    Confusion is judged by the gate through measurement."""
    low = " ".join(phrases.RULES.lower().split())
    assert "do not judge whether the confusion" in low
    assert "cannot see the crops" in low
    assert "keep every phrase unless" in low
    assert "keep a phrase only if" not in low   # the old keep criterion must not come back
    for token in ("synonym", "part or an attribute", "family or material", "generic"):
        assert token in low, token
    assert "do not invent other" in low
    # the cap (max_keep) is cut by code — in the prompt, safe phrases over the cap have nowhere to go
    assert "at most" not in low and "max_keep" not in phrases.RULES


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


def test_refine_alias_rejects_synthesized_phrase(monkeypatch):
    """Out-of-pool phrases (synthesized or paraphrased) have no grounding — the mechanical check drops them."""
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"alias": "totally new phrase",
                                                   "reason": "x"}))
    out = phrases.refine_alias("http://x", "m", "cat a", ["cat a"], [ALIAS])
    assert out["alias"] is None and "verbatim" in out["reason"]


def test_refine_alias_returns_normalized_pool_phrase(monkeypatch):
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"alias": "Shiny Widget",
                                                   "reason": "x"}))
    out = phrases.refine_alias("http://x", "m", "cat a", ["cat a"],
                               ["A Shiny Widget."])
    assert out["alias"] == ALIAS                              # returned in normalized form


def test_refine_alias_rejects_other_category_overlap(monkeypatch):
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"alias": "red cat b", "reason": "x"}))
    out = phrases.refine_alias("http://x", "m", "cat a", ["cat a", "cat b"],
                               ["a red cat b"])
    assert out["alias"] is None and "cat b" in out["reason"]


def test_refine_negatives_rule1_double_check(monkeypatch):
    """Even if the LLM misses rule 1 (the target's own name), the mechanical check filters it out.
    Another category's name is fine as a negative — a negative is attached only to the
    target's own prompts."""
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"keep": ["red cat a", "cat b", "blue ring"],
                                                   "rejected": []}))
    out = phrases.refine_negatives("http://x", "m", "cat a", ["cat a", "cat b"],
                                   ["red cat a", "cat b", "blue ring"])
    assert out["keep"] == ["cat b", "blue ring"]
    assert [r["phrase"] for r in out["rejected"]] == ["red cat a"]
    assert out["rejected"][0]["rule"] == 1


def test_refine_negatives_name_check_is_whole_word(monkeypatch):
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"keep": ["category label", "black cat"],
                                                   "rejected": []}))
    out = phrases.refine_negatives("http://x", "m", "cat", ["cat"],
                                   ["category label", "black cat"])
    assert out["keep"] == ["category label"]


def test_refine_negatives_prompt_follows_the_source(monkeypatch):
    """A gt_crop source is not a false detection — the prompt must state the source truthfully."""
    sent = []

    def capture(session, url, payload, timeout):
        sent.append(payload["messages"][-1]["content"])
        return _FakeResp({"keep": [], "rejected": []})
    monkeypatch.setattr(phrases, "post_with_retry", capture)
    for src in ("fp_crop", "gt_crop", "background"):
        phrases.refine_negatives("http://x", "m", "cat a", ["cat a", "cat b"], ["p"],
                                 source=src)
    fp, gt, bg = (" ".join(t.lower().split()) for t in sent)
    assert "wrongly fired" in fp
    assert "ground-truth crops of the other categories" in gt and "wrongly fired" not in gt
    assert "overlap no ground-truth object" in bg


def test_refine_negatives_rejections_are_rule_numbers():
    rej = phrases.SCHEMA["schema"]["properties"]["rejected"]["items"]
    assert rej["properties"]["rule"]["enum"] == [1, 2, 3, 4]
    assert "reason" not in rej["properties"]


def test_refine_alias_other_category_check_is_whole_word(monkeypatch):
    """With a 'fish' category, 'translucent jellyfish' does not contain another category's name."""
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"alias": "translucent jellyfish",
                                                   "reason": "x"}))
    out = phrases.refine_alias("http://x", "m", "jellyfish", ["jellyfish", "fish"],
                               ["translucent jellyfish"])
    assert out["alias"] == "translucent jellyfish"


def test_refine_alias_reason_is_capped():
    assert phrases.ALIAS_SCHEMA["schema"]["properties"]["reason"]["maxLength"] <= 200


def test_refine_failure_drops_phrases(monkeypatch):
    """The fallback for LLM server errors or parse failures is "drop" — no suppression
    means current behavior. Server errors are not asked again (retrying is
    post_with_retry's job)."""
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("down")
    monkeypatch.setattr(phrases, "post_with_retry", boom)
    assert phrases.refine_negatives("http://x", "m", "t", ["t"], ["p"])["keep"] == []
    assert phrases.refine_alias("http://x", "m", "t", ["t"], ["p"])["alias"] is None
    assert len(calls) == 2


class _RawResp:
    """A reply that does not parse as JSON — cut off at the token limit."""
    def __init__(self, text):
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text},
                             "finish_reason": "length"}]}


def test_refine_asks_again_after_unparseable_reply(monkeypatch, caplog):
    import logging
    replies = iter([_RawResp('{"keep": ["blue'),
                    _FakeResp({"keep": ["blue ring"], "rejected": []})])
    monkeypatch.setattr(phrases, "post_with_retry", lambda *a, **k: next(replies))
    with caplog.at_level(logging.WARNING):
        out = phrases.refine_negatives("http://x", "m", "cat a", ["cat a"], ["blue ring"])
    assert out["keep"] == ["blue ring"]
    assert any('{"keep": ["blue' in r.getMessage() and "length" in r.getMessage()
               for r in caplog.records)


def test_refine_failure_after_retry_logs_the_reply(monkeypatch, caplog):
    """If both replies fail to parse, the fallback applies (drop / keep the class
    name) and the warning keeps the head and tail of the reply."""
    import logging
    calls = []

    def broken(*a, **k):
        calls.append(1)
        return _RawResp('{"alias": "half')
    monkeypatch.setattr(phrases, "post_with_retry", broken)
    with caplog.at_level(logging.WARNING):
        assert phrases.refine_negatives("http://x", "m", "t", ["t"], ["p"])["keep"] == []
        assert phrases.refine_alias("http://x", "m", "t", ["t"], ["p"])["alias"] is None
    assert len(calls) == 4
    fails = [r.getMessage() for r in caplog.records if "refinement failed" in r.getMessage()]
    assert len(fails) == 2 and all('{"alias": "half' in m for m in fails)


def test_vlm_caption_rows_contract():
    """VLM captioner contract: rows -> {id: [phrase, ...]}.
    It actually shows the crop (multimodal), so captions are not trapped in the class-name context."""
    from PIL import Image
    from agentic_configuration_search.infer.phrases import vlm_caption

    img = Image.new("RGB", (40, 30), "red")
    seen = {}

    def chat(messages, **kw):
        seen["parts"] = messages[-1]["content"]          # the user message after the system one
        return {"captions": [{"crop": 0, "phrases": ["shiny red disc", "red dot"]},
                             {"crop": 1, "phrases": ["red dot"]}]}

    rows = [{"id": "cat#0", "image": "a.png", "bbox_xyxy": [0, 0, 20, 20]},
            {"id": "cat#1", "image": "a.png", "bbox_xyxy": [5, 5, 25, 25]}]
    out = vlm_caption(rows, chat, open_image=lambda p: img, batch=8)
    assert out["cat#0"] == ["shiny red disc", "red dot"]
    assert out["cat#1"] == ["red dot"]
    uris = [p for p in seen["parts"] if p.get("type") == "image_url"]
    assert len(uris) == 2 and uris[0]["image_url"]["url"].startswith("data:image")
    # a broken response yields an empty result instead of an exception — mining degrades to no material
    assert vlm_caption(rows, lambda m, **k: {"bad": 1}, open_image=lambda p: img) == {}


def test_vlm_batches_stay_under_server_image_limit():
    """Both paths that send crops to the VLM in batches (the alias=vlm captioner
    and diagnosis crop notes) keep the same limit — a server capping images per
    prompt (--limit-mm-per-prompt) rejects an over-limit batch whole (400)."""
    from agentic_configuration_search.diagnose import describe
    from agentic_configuration_search.infer import phrases

    assert phrases.VLM_BATCH == describe.CROPS_PER_CALL <= 4


def _vlm_rows(names_counts, size=(20, 20)):
    rows = []
    for name, n in names_counts:
        for i in range(n):
            rows.append({"id": f"{name}#{i}", "image": "a.png",
                         "bbox_xyxy": [0, 0, size[0], size[1]]})
    return rows


def _echo_chat(calls):
    """One phrase per crop — keeps a record of the calls."""
    def chat(messages, **kw):
        user = messages[-1]["content"]
        n = sum(1 for p in user if p.get("type") == "image_url")
        calls.append({"messages": messages, "schema": kw.get("schema"), "n": n})
        return {"captions": [{"crop": i, "phrases": ["red disc"]} for i in range(n)]}
    return chat


def test_vlm_caption_batches_never_mix_classes():
    """One call holds crops of one class only — mixed, the "same wording for
    look-alikes" rule carries one class's phrases over to another."""
    from PIL import Image
    calls = []
    rows = _vlm_rows([("cat a", 3), ("cat b", 3)])
    out = phrases.vlm_caption(rows, _echo_chat(calls),
                              open_image=lambda p: Image.new("RGB", (40, 40)), batch=4)
    assert [c["n"] for c in calls] == [3, 3]
    assert set(out) == {r["id"] for r in rows}


def test_vlm_caption_schema_is_exact_per_batch():
    """One entry per crop, 1-3 phrases per entry — allowing an empty list lets the
    model give nothing on small crops."""
    from PIL import Image
    calls = []
    phrases.vlm_caption(_vlm_rows([("cat a", 3)]), _echo_chat(calls),
                        open_image=lambda p: Image.new("RGB", (40, 40)), batch=4)
    caps = calls[0]["schema"]["schema"]["properties"]["captions"]
    assert caps["minItems"] == caps["maxItems"] == 3
    item = caps["items"]["properties"]
    assert item["crop"]["enum"] == [0, 1, 2]
    assert item["phrases"]["minItems"] == 1 and item["phrases"]["maxItems"] == 3
    assert item["phrases"]["items"]["maxLength"] <= 40


def test_vlm_caption_upscales_small_crops():
    """Small crops are upscaled by the same rule as the diagnosis descriptions."""
    import base64
    import io

    from PIL import Image

    from agentic_configuration_search.core.coco import MIN_VLM_CROP_PX
    calls = []
    phrases.vlm_caption(_vlm_rows([("cat a", 1)], size=(12, 24)), _echo_chat(calls),
                        open_image=lambda p: Image.new("RGB", (40, 40)))
    url = next(p for p in calls[0]["messages"][-1]["content"]
               if p.get("type") == "image_url")["image_url"]["url"]
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert min(img.size) == MIN_VLM_CROP_PX


def test_vlm_caption_prompt_intent():
    """One batch is one category, phrases carry a type noun, a best guess when unclear — never empty."""
    low = " ".join(phrases.VLM_CAPTION_RULES.lower().split())
    assert "same object category" in low
    assert "noun" in low and "never an attribute alone" in low
    assert "never return an empty list" in low
    assert "background" in low                       # no background, border or image quality
    assert "detector" in phrases.VLM_CAPTION_SYSTEM.lower()


def test_refine_negatives_cap_keeps_the_most_frequent(monkeypatch):
    """The cap is cut by code in count order — not relying on the order the LLM returned."""
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"keep": ["rare ring", "common disc"],
                                                   "rejected": []}))
    out = phrases.refine_negatives("http://x", "m", "cat a", ["cat a"],
                                   ["rare ring", "common disc", "common disc"], max_keep=1)
    assert out["keep"] == ["common disc"]


def test_name_check_covers_plurals(monkeypatch):
    """Plurals are the same name — 'soccer balls' is a phrase for the target 'soccer ball'."""
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"keep": ["soccer balls", "glass boxes", "ballroom"],
                                                   "rejected": []}))
    out = phrases.refine_negatives("http://x", "m", "soccer ball", ["soccer ball"],
                                   ["soccer balls", "glass boxes", "ballroom"])
    assert out["keep"] == ["glass boxes", "ballroom"]
