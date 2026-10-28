"""A proposal that is the champion itself does not use up the round's re-ask.

Why: a validation run's only candidate in a round was the
request that had adopted the current champion ({lora, rescorer}; the rescorer
refit came out identity, so the champion label was {lora}). Combo dedup dropped
it as combo_tried, and the round's one re-ask was spent on it.

- A combination equal to the champion (its current axes or the request axes
  that adopted it) is recorded as skipped="champion_combo" (not combo_tried)
- If every candidate of a proposal is such a no-op drop (champion_combo,
  no_change), the planner is re-asked once per round (FREE_REASKS) without
  counting toward planner.max_reasks
- Replay: k of the re-ask artifact (proposal_reask<k>.json) runs in order
  whether or not the re-ask was free, and resume replays the same artifacts
"""

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import (FREE_REASKS, NOOP_SKIPS, Loop,
                                                      run_rounds)
from agentic_configuration_search.planner.axes import validate_axes
from tests.test_empty_round import _cands, _cfg, _planner_chat
from tests.test_loop import _fake_infer, _fakes, _gt_lookup, _hist
from tests.test_loop import big_root  # noqa: F401 — fixture reuse


def _hist_with_relabeled_champion():
    asked = validate_axes({"lora": True, "rescorer": True})
    champ = validate_axes({"lora": True})
    return [
        {"round": 0, "champion": {"axes": validate_axes({}), "round": 0,
                                  "id": "baseline"}, "candidates": [],
         "decision": {"kind": "baseline", "adopted": True}},
        {"round": 1, "champion": {"axes": champ, "round": 1, "id": "r1_a"},
         "candidates": [{"id": "r1_a", "axes": asked, "adopted": True}],
         "decision": {"adopted": True, "id": "r1_a"}},
    ]


def test_filter_names_the_champion_combo(big_root, tmp_path):  # noqa: F811
    loop = Loop(_cfg(big_root), big_root, tmp_path / "run")
    hist = _hist_with_relabeled_champion()
    same = {"id": "c", "axes": validate_axes({"lora": True, "rescorer": True})}
    label = {"id": "e", "axes": validate_axes({"lora": True})}
    other = {"id": "d", "axes": validate_axes({"lora": True, "tiling": 2})}
    hist[1]["candidates"].append({"id": "r1_b", "axes": other["axes"]})
    fresh, dropped = loop._filter([same, label, other], hist)
    assert fresh == []
    reasons = {d["id"]: d["skipped"] for d in dropped}
    assert reasons == {"c": "champion_combo", "e": "champion_combo",
                       "d": "combo_tried"}
    assert "champion" in next(d for d in dropped if d["id"] == "c")["note"]
    assert {"champion_combo", "no_change"} == set(NOOP_SKIPS) and FREE_REASKS == 1


def _relabel_run(big_root, tmp_path, monkeypatch, *responses, **cfg):  # noqa: F811
    """R1 adopts exemplar=base + tiling 2 — tiles go unused in the stub, so the
    champion label is exemplar=base while the request stays on the history row.
    R2's first proposal is that request verbatim."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    first = _cands(("ex", {"exemplar": "base", "tiling": 2}))
    chat, calls, _ = _planner_chat(first, first, *responses)
    out = run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=chat, rounds=2,
                     cfg=_cfg(big_root, rounds=2, **cfg), **_fakes())
    return out, calls


def test_champion_combo_gets_a_free_reask(big_root, tmp_path, monkeypatch):  # noqa: F811
    out, calls = _relabel_run(big_root, tmp_path, monkeypatch,
                              _cands(("t2", {"tiling": 2})),        # free re-ask
                              _cands(("al", {"alias": "vlm"})))     # counted re-ask
    assert out.decisions[1]["adopted"] is True
    assert out.champion["axes"]["exemplar"] == "base"
    h = _hist(tmp_path / "run")
    rows = {c["id"]: c for c in h[2]["candidates"]}
    assert rows["r2_ex"]["skipped"] == "champion_combo"
    assert rows["r2_t2"]["reask"] == 1 and rows["r2_al"]["reask"] == 2
    assert len(calls) == 4                     # R1 once + R2 first, free, counted
    d2 = out.decisions[2]
    assert d2["reasks"] == 2 and d2["free_reasks"] == 1
    rd = tmp_path / "run" / "rounds" / "round_002"
    assert (rd / "proposal_reask1.json").exists() and (rd / "proposal_reask2.json").exists()
    # the re-ask message carries why: it was the champion itself
    assert "champion_combo" in calls[2][-1]["content"]


def test_free_reask_is_bounded_per_round(big_root, tmp_path, monkeypatch):  # noqa: F811
    """Even the free re-ask happens once per round if it only returns the champion
    combo again — after that planner.max_reasks counts (no endless round)."""
    same = _cands(("again", {"exemplar": "base", "tiling": 2}))
    out, calls = _relabel_run(big_root, tmp_path, monkeypatch, same, same, same,
                              planner={"max_reasks": 1})
    assert len(calls) == 4                     # R1 + R2 first, free 1, counted 1
    assert out.decisions[2]["reasks"] == 2 and out.decisions[2]["free_reasks"] == 1


def test_max_reasks_zero_turns_every_reask_off(big_root, tmp_path, monkeypatch):  # noqa: F811
    out, calls = _relabel_run(big_root, tmp_path, monkeypatch,
                              _cands(("t2", {"tiling": 2})), planner={"max_reasks": 0})
    assert len(calls) == 2 and "reasks" not in out.decisions[2]


def test_ordinary_drops_do_not_get_a_free_reask(big_root, tmp_path, monkeypatch):  # noqa: F811
    """Drops of non-champion combinations (combo_tried and the like) are not free."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    chat, calls, _ = _planner_chat(_cands(("t2", {"tiling": 2})),
                                   _cands(("t2b", {"tiling": 2})),   # combo_tried
                                   _cands(("al", {"alias": "vlm"})))
    out = run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=chat, rounds=1,
                     cfg=_cfg(big_root, rounds=1, planner={"max_reasks": 1}),
                     **_fakes())
    assert len(calls) == 2 and "free_reasks" not in out.decisions[1]


def test_free_reask_replays_on_resume(big_root, tmp_path, monkeypatch):  # noqa: F811
    out, calls = _relabel_run(big_root, tmp_path, monkeypatch,
                              _cands(("t2", {"tiling": 2})),
                              _cands(("al", {"alias": "vlm"})))
    run = tmp_path / "run"
    rows1 = _hist(run)[2]["candidates"]
    lines = (run / "history.jsonl").read_text().splitlines()
    (run / "history.jsonl").write_text("\n".join(lines[:2]) + "\n")

    def boom(messages, **kw):
        if (kw.get("schema") or {}).get("name") == "proposal":
            raise AssertionError("resume called the planner")
        return {}

    again = run_rounds(big_root, out_dir=run, llm_chat=boom, rounds=2,
                       cfg=_cfg(big_root, rounds=2), resume=True, **_fakes())
    assert again.decisions[2]["reasks"] == 2 and again.decisions[2]["free_reasks"] == 1
    rows2 = _hist(run)[2]["candidates"]
    assert [(c["id"], c.get("skipped"), c.get("reask")) for c in rows2] == \
        [(c["id"], c.get("skipped"), c.get("reask")) for c in rows1]


def test_prompt_lists_the_request_that_adopted_the_champion():
    """The tried-combination list carries the request that adopted the champion
    as the champion itself — the planner had read it as a new combination."""
    from agentic_configuration_search.planner.llm import _combos_line

    block = _combos_line(_hist_with_relabeled_champion())
    assert "{rescorer=true} = the request that adopted the current champion" in block
    assert "champion_combo" in block
