"""Candidates per round (planner.max_candidates) — a config knob.

The default 3 equals the earlier hard-coded LLMPlanner(max_candidates=3). The
prompt's "Propose up to N", the response schema's maxItems, the k of the re-ask
and of the re-ask after carried duplicates, and the deterministic planner (and
the LLM's deterministic fallback) all follow it.
"""

import pytest

from agentic_configuration_search.config import ConfigError, PlannerCfg, config_from_dict
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from tests.test_empty_round import _cands, _cfg, _planner_chat
from tests.test_loop import _fake_infer, _fakes, _gt_lookup, _hist
from tests.test_loop import big_root  # noqa: F401 — fixture reuse


def test_default_is_three_and_must_be_positive():
    assert PlannerCfg().max_candidates == 3
    assert config_from_dict({"dataset": "x"}).planner.max_candidates == 3
    assert config_from_dict({"dataset": "x", "planner": {
        "max_candidates": 2}}).planner.max_candidates == 2
    with pytest.raises(ConfigError, match="max_candidates"):
        config_from_dict({"dataset": "x", "planner": {"max_candidates": 0}})


def test_upper_bound_keeps_the_response_within_output_tokens():
    """Above 3 the longest response the schema allows can exceed
    OUTPUT_MAX_TOKENS (4096) — a cut response breaks the JSON and silently falls
    back to deterministic. config enforces the cap (tests/test_planner.py pins
    the truncation guard to it)."""
    from agentic_configuration_search.config import PLANNER_MAX_CANDIDATES

    assert PLANNER_MAX_CANDIDATES == 3
    assert config_from_dict({"dataset": "x", "planner": {
        "max_candidates": 3}}).planner.max_candidates == 3
    with pytest.raises(ConfigError, match=r"max_candidates.*1\.\.3"):
        config_from_dict({"dataset": "x", "planner": {"max_candidates": 4}})


def _spy(*responses):
    chat, calls, _ = _planner_chat(*responses)
    schemas = []

    def wrapped(messages, **kw):
        if (kw.get("schema") or {}).get("name") == "proposal":
            schemas.append(kw["schema"]["schema"]["properties"]["candidates"]
                           ["maxItems"])
        return chat(messages, **kw)
    return wrapped, calls, schemas


@pytest.mark.parametrize("k", [1, 2])
def test_config_value_reaches_prompt_schema_and_reask(big_root, tmp_path,  # noqa: F811
                                                      monkeypatch, k):
    """If the first response exceeds k, the overflow is dropped as max_candidates,
    and if the rest are all rejected before holdout the re-ask also asks for at
    most k."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    first = _cands(*[(f"t{i}", {"tiling": 2 + i}) for i in range(3)])
    chat, calls, schemas = _spy(first, _cands(("ex", {"exemplar": "base"})))
    run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=chat, rounds=1,
               cfg=_cfg(big_root, rounds=1, planner={"max_candidates": k}),
               **_fakes())
    assert f"Propose up to {k} candidates" in calls[0][-1]["content"]
    rows = _hist(tmp_path / "run")[1]["candidates"]
    over = [c for c in rows if c.get("skipped") == "max_candidates"]
    assert len(over) == 3 - k
    assert len(calls) == 2, "the first candidates were all rejected, so it must re-ask"
    assert f"Propose up to {k} new candidates" in calls[1][-1]["content"]
    assert schemas == [k, k]


def test_deterministic_planner_follows_the_knob(big_root, tmp_path,  # noqa: F811
                                                monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    cfg = config_from_dict({"dataset": str(big_root), "rounds": 1,
                            "planner": {"max_candidates": 1}})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
               rounds=1, cfg=cfg, **_fakes())
    import json
    prop = json.loads((tmp_path / "run" / "rounds" / "round_001"
                       / "proposal.json").read_text())
    assert len(prop["candidates"]) == 1
