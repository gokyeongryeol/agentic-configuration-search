"""Candidate ids are unique within a run — the loop adds a round prefix
(r<round>_).

Why: in validation runs the LLM restarted ids from cand_0 each round (cand_0 in
both R1 and R2). Lookups are per round, so it was not a functional bug, but
reports and logs became ambiguous. Carried candidates keep carry_<original id> —
the original id already has a round prefix.
"""

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import _round_ids, run_rounds
from tests.test_empty_round import _cands, _cfg, _planner_chat
from tests.test_loop import _fake_infer, _fakes, _gt_lookup, _hist
from tests.test_loop import big_root  # noqa: F401 — fixture reuse


def test_round_ids_prefix_strip_and_dedupe():
    items = [{"id": "cand_0"}, {"id": "r1_cand_0"},
             {"id": "carry_r1_cand_1", "source": "carryover"},
             {"id": "human_a"}, {"id": "cand_0"}]
    out = _round_ids(items, 3, taken={"r3_human_a"})
    assert [c["id"] for c in out] == ["r3_cand_0", "r3_cand_0_2",
                                      "carry_r1_cand_1", "r3_human_a_2",
                                      "r3_cand_0_3"]
    assert items[0]["id"] == "cand_0"           # the input (replay artifact) is unchanged


def test_planner_made_carry_id_is_normalized():
    """A planner imitating a carry id (carry_r1_cand_1) does not make a carried
    candidate — it gets no prefix exemption (it could clash across rounds) and
    the report does not treat it as carried. Only real carried candidates made by
    the loop (source=carryover) are exempt."""
    items = [{"id": "carry_r1_cand_1"}, {"id": "carry_r1_cand_1", "source": "carryover"},
             {"id": "carry_carry_x"}]
    out = _round_ids(items, 3, taken=set())
    assert [c["id"] for c in out] == ["r3_cand_1", "carry_r1_cand_1", "r3_x"]
    assert all("source" not in c for i, c in enumerate(out) if i != 1)


def test_ids_are_unique_across_rounds_and_reasks(big_root, tmp_path,  # noqa: F811
                                                 monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    # round 1: tiling (rejected as unused) → exemplar on re-ask (adopted)
    # round 2: from cand_0 again
    chat, _, _ = _planner_chat(_cands(("cand_0", {"tiling": 2})),
                               _cands(("cand_0", {"exemplar": "base"})),
                               _cands(("cand_0", {"alias": "vlm"})))
    run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=chat, rounds=2,
               cfg=_cfg(big_root, rounds=2), **_fakes())
    h = _hist(tmp_path / "run")
    ids = [(e["round"], c["id"]) for e in h[1:] for c in e["candidates"]]
    assert all(cid.startswith(f"r{r}_") or cid.startswith("carry_")
               for r, cid in ids), ids
    plain = [cid for _, cid in ids]
    assert len(plain) == len(set(plain)), ids
    assert "r1_cand_0" in plain and "r1_cand_0_2" in plain     # re-ask clash
    assert h[1]["decision"]["id"] == "r1_cand_0_2"
    assert h[1]["champion"]["id"] == "r1_cand_0_2"
    assert any(cid.startswith("r2_") for cid in plain)
