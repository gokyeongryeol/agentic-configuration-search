"""Empty rounds — a round that put nothing on holdout should not use up a whole round.

Why: when every candidate is rejected around screen (axis_unused, same_setting,
no_material), a round measures nothing on holdout, yet each such round used to
redo the diagnosis (full train inference), the VLM crop notes and the planner
call, and add one to the saturation count.

- re-ask within the same round (planner.max_reasks)
- stop after consecutive empty rounds (saturation.empty_rounds)
- reuse the diagnosis while the champion is unchanged (diagnosis_key.json)
"""

import json

import pytest

from agentic_configuration_search.config import ConfigError, config_from_dict
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop import runner
from agentic_configuration_search.loop.runner import run_rounds
from tests.test_loop import _fake_infer, _fakes, _gt_lookup, _hist
from tests.test_loop import big_root  # noqa: F401 — fixture reuse


def _planner_chat(*responses):
    """The next response on each planner call (proposal schema) — an empty
    proposal once they run out. Crop-note calls are counted separately. calls
    collects the messages sent to the planner."""
    calls, notes = [], []

    def chat(messages, **kw):
        if (kw.get("schema") or {}).get("name") != "proposal":
            notes.append(messages)
            return {}
        calls.append(messages)
        i = len(calls) - 1
        return responses[i] if i < len(responses) else {"candidates": [],
                                                         "stop": False}
    return chat, calls, notes


def _cands(*pairs):
    return {"candidates": [{"id": cid, "axes": axes, "rationale": "r"}
                           for cid, axes in pairs], "stop": False}


def _cfg(root, **over):
    d = {"dataset": str(root), "planner": {"kind": "llm"}}
    for k, v in over.items():
        if isinstance(v, dict):
            d.setdefault(k, {}).update(v)
        else:
            d[k] = v
    return config_from_dict(d)


# --- config -----------------------------------------------------------------

def test_empty_round_config_defaults_and_validation():
    cfg = config_from_dict({"dataset": "x"})
    assert cfg.planner.max_reasks == 1
    assert cfg.saturation.empty_rounds == 2
    # 0 is off
    assert config_from_dict({"dataset": "x", "planner": {"max_reasks": 0},
                             "saturation": {"empty_rounds": 0}}).planner.max_reasks == 0
    with pytest.raises(ConfigError, match="max_reasks"):
        config_from_dict({"dataset": "x", "planner": {"max_reasks": -1}})
    with pytest.raises(ConfigError, match="empty_rounds"):
        config_from_dict({"dataset": "x", "saturation": {"empty_rounds": -1}})


# --- 1. re-ask within the same round ---------------------------------------------

def test_empty_round_reasks_planner_within_the_round(big_root, tmp_path,  # noqa: F811
                                                     monkeypatch):
    """If the first proposal is all rejected before holdout, the round is not
    closed — the planner is asked again in the same conversation. The rejection
    reasons go into the follow-up, and re-asked candidates take the same path
    (screen, holdout) and stay in the same round's history marked reask. The
    diagnosis runs once."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    chat, calls, _ = _planner_chat(_cands(("t2", {"tiling": 2})),
                                   _cands(("ex", {"exemplar": "base"})))
    n_diag = []
    orig = runner.diagnose
    monkeypatch.setattr(runner, "diagnose",
                        lambda *a, **k: (n_diag.append(1), orig(*a, **k))[1])
    out = run_rounds(big_root, out_dir=tmp_path / "run", cfg=_cfg(big_root, rounds=1),
                     llm_chat=chat, rounds=1, **_fakes())

    assert len(calls) == 2
    follow = calls[1]
    assert follow[-2]["role"] == "assistant"            # continues the first response as a conversation
    msg = follow[-1]["content"]
    assert "t2" in msg and "axis_unused" in msg and "tiling=2" in msg
    d1 = out.decisions[1]
    assert d1["adopted"] is True and d1["id"] == "r1_ex"
    assert d1["reasks"] == 1
    rows = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}
    assert rows["r1_t2"]["skipped"] == "axis_unused" and "reask" not in rows["r1_t2"]
    assert rows["r1_ex"]["reask"] == 1 and rows["r1_ex"]["adopted"] is True
    rd = tmp_path / "run" / "rounds" / "round_001"
    stored = json.loads((rd / "proposal_reask1.json").read_text())
    assert stored["message"] == msg and stored["response"]["candidates"][0]["id"] == "ex"
    assert (rd / "settings" / "r1_ex").is_dir()
    # the round diagnosis (train + holdout) runs once. The other 2 calls are the
    # end-of-run final champion diagnosis (rounds/final) — round 1 adopted, so
    # the champion changed
    assert len(n_diag) == 4
    assert json.loads((tmp_path / "run" / "rounds" / "final" / "diagnosis_key.json")
                      .read_text())["champion"]["id"] == "r1_ex"


def test_reask_is_capped_and_can_be_disabled(big_root, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    unused = [_cands((f"a{i}", {"alias": "vlm", "tiling": 2 + i})) for i in range(5)]

    chat, calls, _ = _planner_chat(_cands(("t2", {"tiling": 2})), *unused)
    run_rounds(big_root, out_dir=tmp_path / "off", llm_chat=chat, rounds=1,
               cfg=_cfg(big_root, rounds=1, planner={"max_reasks": 0}), **_fakes())
    assert len(calls) == 1
    assert not (tmp_path / "off" / "rounds" / "round_001" / "proposal_reask1.json").exists()

    chat, calls, _ = _planner_chat(_cands(("t2", {"tiling": 2})),
                                   _cands(("t3", {"tiling": 3})),
                                   _cands(("t4", {"tiling": 4})),
                                   _cands(("t5", {"tiling": 5})))
    out = run_rounds(big_root, out_dir=tmp_path / "two", llm_chat=chat, rounds=1,
                     cfg=_cfg(big_root, rounds=1, planner={"max_reasks": 2}), **_fakes())
    assert len(calls) == 3                               # first proposal + 2 re-asks
    assert out.decisions[1]["reasks"] == 2
    assert not out.decisions[1].get("evaluated")


def test_reask_candidates_go_through_the_same_filters(big_root, tmp_path,  # noqa: F811
                                                      monkeypatch):
    """Re-asked candidates go through the shared filter too — a combination
    already proposed this round is combo_tried, and a grid at or above one left
    unused this round is tiling_blocked."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    seen = []
    orig = runner.materialize
    monkeypatch.setattr(runner, "materialize",
                        lambda cfg, ds, axes, *a, **k: (seen.append(dict(axes)),
                                                        orig(cfg, ds, axes, *a, **k))[1])
    # the stub has no false positives, so fp_crop mining comes out empty (no_material)
    chat, calls, _ = _planner_chat(_cands(("t2", {"tiling": 2}),
                                          ("tn", {"text_negatives": "fp_crop"})),
                                   _cands(("again", {"text_negatives": "fp_crop"}),
                                          ("t3", {"tiling": 3})))
    run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=chat, rounds=1,
               cfg=_cfg(big_root, rounds=1), **_fakes())
    assert "no_material" in calls[1][-1]["content"]
    rows = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}
    assert rows["r1_tn"]["skipped"] == "no_material"
    assert rows["r1_again"]["skipped"] == "combo_tried" and rows["r1_again"]["reask"] == 1
    assert rows["r1_t3"]["skipped"] == "tiling_blocked" and rows["r1_t3"]["reask"] == 1
    assert not any(a.get("tiling") == 3 for a in seen)


def test_reask_ids_never_collide_with_the_first_answer(big_root, tmp_path,  # noqa: F811
                                                       monkeypatch):
    """A re-asked candidate reusing an id from the first response is renamed —
    the same id would collide in the materials directory (settings/<id>) and the
    history rows."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    chat, _, _ = _planner_chat(_cands(("c", {"tiling": 2})),
                               _cands(("c", {"exemplar": "base"})))
    run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=chat, rounds=1,
               cfg=_cfg(big_root, rounds=1), **_fakes())
    ids = [c["id"] for c in _hist(tmp_path / "run")[1]["candidates"]]
    assert len(ids) == len(set(ids)) and "r1_c_2" in ids, ids


def test_reask_replays_on_resume_without_llm(big_root, tmp_path, monkeypatch):  # noqa: F811
    """Resuming an interrupted round replays proposal.json and
    proposal_reask<k>.json — zero LLM calls, the same candidates and decisions."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    # the re-asked alias is evaluated but at Δ=0 — champion.json must stay at the
    # baseline to mimic a round that died just before writing history. The alias
    # equals the champion on fit_dev, so by default it would drop out before
    # holdout (fit_dev_identical) — that verdict is turned off; this test only
    # checks the replay contract
    chat, calls, _ = _planner_chat(_cands(("t2", {"tiling": 2})),
                                   _cands(("al", {"alias": "vlm"})))
    out_dir = tmp_path / "run"
    cfg = _cfg(big_root, rounds=1, screen={"skip_fit_dev_losers": False})
    first = run_rounds(big_root, out_dir=out_dir, llm_chat=chat, rounds=1, cfg=cfg,
                       **_fakes())
    assert first.decisions[1]["evaluated"] and not first.decisions[1]["adopted"]
    rows1 = _hist(out_dir)[1]["candidates"]
    # roll back as if round 1 died just before being written to history
    lines = (out_dir / "history.jsonl").read_text().splitlines()
    (out_dir / "history.jsonl").write_text(lines[0] + "\n")

    def boom(messages, **kw):
        raise AssertionError("resume called the LLM")

    n_diag = _diag_spy(monkeypatch)          # the interrupted round reuses its diagnosis too
    again = run_rounds(big_root, out_dir=out_dir, llm_chat=boom, rounds=1, cfg=cfg,
                       resume=True, **_fakes())
    assert again.decisions[1]["evaluated"].keys() == first.decisions[1]["evaluated"].keys()
    assert again.decisions[1]["reasks"] == 1
    assert n_diag == []
    rows2 = _hist(out_dir)[1]["candidates"]
    assert [(c["id"], c.get("skipped"), c.get("reask")) for c in rows2] == \
        [(c["id"], c.get("skipped"), c.get("reask")) for c in rows1]


def test_planner_reask_replays_its_artifact(tmp_path):
    """LLMPlanner.reask replay — the same artifact_dir gives the same proposal without calls."""
    from agentic_configuration_search.planner.llm import LLMPlanner

    chat, calls, _ = _planner_chat(_cands(("a", {"alias": "vlm"})),
                                   _cands(("b", {"tiling": 2})))
    p = LLMPlanner(chat=chat, artifact_dir=tmp_path)
    p.propose([], {})
    outcomes = [{"id": "a", "axes": {"alias": "vlm"}, "skipped": "no_material",
                 "note": "alias mining came out empty"}]
    r1 = p.reask([], {}, outcomes, k=1, taken={"a"})
    assert [c["id"] for c in r1["candidates"]] == ["b"] and len(calls) == 2
    assert "no_material" in calls[1][-1]["content"]

    def boom(messages, **kw):
        raise AssertionError("replay called the LLM")
    r2 = LLMPlanner(chat=boom, artifact_dir=tmp_path).reask([], {}, outcomes, k=1,
                                                           taken={"a"})
    assert r2 == r1


def test_reask_without_a_first_response_does_not_call(tmp_path):
    """Without a conversation to continue (the first call failed, deterministic fallback), no re-ask."""
    from agentic_configuration_search.planner.llm import LLMPlanner

    n = []

    def flaky(messages, **kw):
        n.append(1)
        raise RuntimeError("server down")
    p = LLMPlanner(chat=flaky, artifact_dir=tmp_path)
    p.propose([], {})
    assert p.reask([], {}, [], k=1, taken=set()) is None and len(n) == 1
    assert not p.reask_path(1).exists()


def test_deterministic_planner_does_not_reask(big_root, tmp_path, monkeypatch):  # noqa: F811
    """The deterministic planner is not re-asked — it is the fixed path and the A/B control."""
    from agentic_configuration_search.planner import deterministic

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    n = []
    monkeypatch.setattr(deterministic, "propose",
                        lambda *a, **k: (n.append(1),
                                         {"planner": "deterministic",
                                          "candidates": [], "stop": False})[1])
    human = tmp_path / "c.yaml"
    human.write_text("- {id: t2, axes: {tiling: 2}}\n")
    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
                     rounds=1, candidates_file=human, **_fakes())
    assert len(n) == 1 and not out.decisions[1].get("evaluated")
    assert "reasks" not in out.decisions[1]


# --- 2. stop after consecutive empty rounds ---------------------------------------

def _always_blocked_tiling():
    """Only tiling, a different grid each call — after the first grid goes unused, all are folded."""
    grids = iter(range(2, 100))

    def chat(messages, **kw):
        if (kw.get("schema") or {}).get("name") != "proposal":
            return {}
        g = next(grids)
        return _cands((f"t{g}", {"tiling": min(g, 6), "alias": "vlm"} if g > 6
                       else {"tiling": g}))
    return chat


def test_consecutive_empty_rounds_stop_the_run(big_root, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    cfg = _cfg(big_root, rounds=8, planner={"max_reasks": 0})
    out = run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=_always_blocked_tiling(),
                     rounds=8, cfg=cfg, **_fakes())
    assert out.stopped == "empty_rounds"
    assert [d["round"] for d in out.decisions] == [0, 1, 2]   # before saturation.rounds (5)
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["stopped"] == "empty_rounds"

    off = run_rounds(big_root, out_dir=tmp_path / "off", llm_chat=_always_blocked_tiling(),
                     rounds=8, cfg=_cfg(big_root, rounds=8, planner={"max_reasks": 0},
                                        saturation={"empty_rounds": 0}), **_fakes())
    assert off.stopped == "saturation" and len(off.decisions) == 6


def test_resume_restores_the_empty_round_counter(big_root, tmp_path,  # noqa: F811
                                                 monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    cfg = _cfg(big_root, rounds=8, planner={"max_reasks": 0})
    out = run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=_always_blocked_tiling(),
                     rounds=8, cfg=cfg, **_fakes())
    assert out.stopped == "empty_rounds"

    def boom(messages, **kw):
        raise AssertionError("a stopped run called the LLM")
    again = run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=boom, rounds=8,
                       cfg=cfg, resume=True, **_fakes())
    assert again.stopped == "empty_rounds"
    assert len(again.decisions) == len(out.decisions)


def test_trailing_empty_counts_rounds_that_evaluated_nothing():
    from agentic_configuration_search.loop.runner import _empty_round, _trailing_empty

    base = {"round": 0, "decision": {"kind": "baseline", "adopted": True}}
    full = {"round": 1, "decision": {"adopted": False, "evaluated": {"a": {}}}}
    empty = {"round": 2, "decision": {"adopted": False, "evaluated": {}}}
    none = {"round": 3, "decision": {"adopted": False, "kind": "no_candidates"}}
    assert _empty_round(empty["decision"]) and _empty_round(none["decision"])
    assert not _empty_round(full["decision"]) and not _empty_round(base["decision"])
    assert _trailing_empty([base, full, empty, none]) == 2
    assert _trailing_empty([base, empty, full]) == 0
    assert _trailing_empty([base]) == 0


# --- 3. reuse the diagnosis while the champion is unchanged ------------------------

def _diag_spy(monkeypatch):
    n = []
    orig = runner.diagnose
    monkeypatch.setattr(runner, "diagnose",
                        lambda *a, **k: (n.append(1), orig(*a, **k))[1])
    return n


def test_diagnosis_is_reused_while_the_champion_is_unchanged(big_root, tmp_path,  # noqa: F811
                                                             monkeypatch):
    """The next round under an unchanged champion does not redo the diagnosis
    (train re-inference) or the crop notes — it copies the previous round's
    artifacts into the new round directory."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    n_diag = _diag_spy(monkeypatch)
    n_infer = []
    orig_infer = runner.infer_for_groups
    monkeypatch.setattr(runner, "infer_for_groups",
                        lambda *a, **k: (n_infer.append(1), orig_infer(*a, **k))[1])
    # the alias gives the same setting (Δ=0) — the champion stays at the baseline
    chat, calls, notes = _planner_chat(_cands(("al", {"alias": "vlm"})),
                                       _cands(("al2", {"text_negatives": "fp_crop"})))
    out = run_rounds(big_root, out_dir=tmp_path / "run", llm_chat=chat, rounds=2,
                     cfg=_cfg(big_root, rounds=2, planner={"max_reasks": 0}),
                     **_fakes())
    assert not any(d.get("adopted") for d in out.decisions[1:])
    assert len(out.decisions) == 3
    assert len(n_diag) == 2 and len(n_infer) == 1         # in round 1 only
    assert len(notes) >= 1
    n_notes = len(notes)
    r1, r2 = (tmp_path / "run" / "rounds" / f"round_00{i}" for i in (1, 2))
    for name in ("diagnosis_train.json", "diagnosis_test.json", "diagnosis_key.json"):
        assert (r2 / name).read_text() == (r1 / name).read_text(), name
    assert (r2 / "crop_notes.json").exists() or not (r1 / "crop_notes.json").exists()
    assert len(notes) == n_notes                          # the notes are not called again either
    # the planner gets the same diagnosis
    assert "Per class" in calls[1][-1]["content"]


def test_diagnosis_is_recomputed_after_adoption_or_config_change(big_root, tmp_path,  # noqa: F811
                                                                 monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    n_diag = _diag_spy(monkeypatch)
    # exemplar is adopted in round 1 — round 2 diagnoses again with the new champion
    chat, _, _ = _planner_chat(_cands(("ex", {"exemplar": "base"})),
                               _cands(("al", {"alias": "vlm"})))
    out = run_rounds(big_root, out_dir=tmp_path / "a", llm_chat=chat, rounds=2,
                     cfg=_cfg(big_root, rounds=2), **_fakes())
    assert out.decisions[1]["adopted"] and len(out.decisions) == 3
    assert len(n_diag) == 4

    # a changed diagnosis input (config) recomputes even with the same champion.
    # The alias equals the champion on fit_dev, so by default it drops out before
    # holdout (fit_dev_identical) and the re-ask spends the second answer in
    # round 1 — that verdict is turned off
    n_diag.clear()
    keep = {"skip_fit_dev_losers": False}
    chat, _, _ = _planner_chat(_cands(("al", {"alias": "vlm"})),
                               _cands(("tn", {"text_negatives": "fp_crop"})))
    run_rounds(big_root, out_dir=tmp_path / "b", llm_chat=chat, rounds=1,
               cfg=_cfg(big_root, rounds=1, screen=keep), **_fakes())
    assert len(n_diag) == 2
    run_rounds(big_root, out_dir=tmp_path / "b", llm_chat=chat, rounds=2,
               cfg=_cfg(big_root, rounds=2, diagnose={"small_object_ratio": 0.3},
                        screen=keep),
               resume=True, **_fakes())
    assert len(n_diag) == 4


def test_failed_reask_closes_the_round(tmp_path):
    """A failed re-ask call is None — the round closes even with max_reasks left. The
    failure record stays as an artifact, and resume replays the same verdict without a call."""
    from agentic_configuration_search.planner.llm import LLMPlanner

    n = []

    def chat(messages, **kw):
        n.append(1)
        if len(n) > 1:
            raise RuntimeError("server down")
        return _cands(("a", {"alias": "vlm"}))
    p = LLMPlanner(chat=chat, artifact_dir=tmp_path)
    p.propose([], {})
    assert p.reask([], {}, [], k=1, taken={"a"}) is None and len(n) == 2
    assert "server down" in json.loads(p.reask_path(1).read_text())["error"]

    def boom(messages, **kw):
        raise AssertionError("the replay called the LLM")
    assert LLMPlanner(chat=boom, artifact_dir=tmp_path).reask(
        [], {}, [], k=1, taken={"a"}) is None
