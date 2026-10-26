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
    """Even if the LLM misses rule 1 (contains the category name), the mechanical check filters it out."""
    monkeypatch.setattr(phrases, "post_with_retry",
                        lambda *a, **k: _FakeResp({"keep": ["red cat a", "blue ring"],
                                                   "rejected": []}))
    out = phrases.refine_negatives("http://x", "m", "cat a", ["cat a"],
                                   ["red cat a", "blue ring"])
    assert out["keep"] == ["blue ring"]
    assert any("rule 1 mechanical check" in r["reason"] for r in out["rejected"])


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
        seen["parts"] = messages[0]["content"]
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
