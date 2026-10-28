"""planner.axis_status — axes turned off automatically during a run (OFF after retire_after failures in a row)."""

import pytest

from agentic_configuration_search.config import ConfigError, config_from_dict
from agentic_configuration_search.planner.axes import axis_values, validate_axes
from agentic_configuration_search.planner.axis_status import axis_status

BASE = validate_axes({})
RES = validate_axes({"rescorer": True})


def _row(r, champ, cands, adopted=None, evaluated=()):
    ev = {cid: {"micro_f_beta": 0.5, "gate": {"adopted": cid == adopted}}
          for cid in evaluated}
    return {"round": r, "champion": {"axes": champ},
            "candidates": cands,
            "decision": {"adopted": adopted is not None, "id": adopted, "evaluated": ev}}


def _cand(cid, axes, **kw):
    return {"id": cid, "axes": validate_axes(axes), **kw}


def _held_out_fail(r, champ, cid, axes):
    return _row(r, champ, [_cand(cid, axes)], evaluated=[cid])


H0 = [_row(0, BASE, [])]


def test_two_held_out_failures_in_a_row_turn_the_axis_off():
    h = H0 + [_held_out_fail(1, BASE, "a", {"exemplar": "base"}),
              _held_out_fail(2, BASE, "b", {"exemplar": "fn_arm", "exemplar_fn_k": 2})]
    st = axis_status(h, 2)
    assert st.axes["exemplar"].status == "off" and st.axes["exemplar"].off_since == 2
    assert st.block(validate_axes({"exemplar": "base", "exemplar_base_k": 8}), BASE)
    # changed together with an open axis it is not blocked — whose fault a combination's result is cannot be told
    assert st.block(validate_axes({"exemplar": "base", "tiling": 2}), BASE) is None


def test_a_win_resets_the_streak_and_none_never_turns_off():
    h = H0 + [_held_out_fail(1, BASE, "a", {"tiling": 2}),
              _row(2, BASE, [_cand("b", {"tiling": 3}, gate_passed_but_lost=True)],
                   evaluated=["b"]),
              _held_out_fail(3, BASE, "c", {"tiling": 4})]
    assert axis_status(h, 2).axes["tiling"].status == "open"
    h2 = H0 + [_held_out_fail(1, BASE, "a", {"alias": "vlm"}),
               _held_out_fail(2, BASE, "b", {"alias": "vlm"})]
    assert axis_status(h2, None).axes["alias"].status == "open"     # record only
    assert axis_status(h2, None).block(validate_axes({"alias": "vlm"}), BASE) is None


def test_empty_material_does_not_count():
    """No material is a problem of the value — another source value (e.g. gt_crop) may work."""
    h = H0 + [_row(1, BASE, [_cand("a", {"text_negatives": "background"},
                                   skipped="no_material", empty_axis="text_negatives")]),
              _row(2, BASE, [_cand("b", {"text_negatives": "fp_crop"},
                                   skipped="no_material", empty_axis="text_negatives")])]
    s = axis_status(h, 2).axes["text_negatives"]
    assert s.status == "open" and s.no_material == ["background", "fp_crop"]


def test_unused_counts_only_for_the_unused_axis():
    h = H0 + [_row(1, BASE, [_cand("a", {"exemplar": "base", "tiling": 2},
                                   skipped="axis_unused", unused_axes=["tiling"])]),
              _row(2, BASE, [_cand("b", {"tiling": 3}, skipped="axis_unused",
                                   unused_axes=["tiling"])])]
    st = axis_status(h, 2)
    assert st.axes["tiling"].status == "off" and st.axes["exemplar"].measured == 0


def test_champion_change_reopens_once_and_champion_axes_are_on():
    h = H0 + [_held_out_fail(1, BASE, "a", {"exemplar": "base"}),
              _held_out_fail(2, BASE, "b", {"exemplar": "fn_arm"}),
              _row(3, RES, [_cand("c", {"rescorer": True}, adopted=True)],
                   adopted="c", evaluated=["c"])]
    st = axis_status(h, 2)
    assert st.axes["exemplar"].status == "probe"
    assert st.block(validate_axes({"exemplar": "base"}), RES) is None
    assert st.axes["rescorer"].status == "on"
    h2 = h + [_held_out_fail(4, RES, "d", {"rescorer": True, "exemplar": "base"})]
    assert axis_status(h2, 2).axes["exemplar"].status == "off"     # one more failure: OFF again


def test_board_shows_status_and_record():
    from agentic_configuration_search.planner.board import board_lines
    h = H0 + [_held_out_fail(1, BASE, "a", {"exemplar": "base"}),
              _held_out_fail(2, BASE, "b", {"exemplar": "fn_arm"})]
    lines = board_lines(h, axis_values(), axis_status(h, 2))
    ex = next(l for l in lines if l.startswith("exemplar | ")).split(" | ")
    assert ex[1] == "OFF since R2" and ex[3] == "held-out fail x2 (last failed in R1, R2)"
    assert "only together with an axis that is not OFF" in ex[6]
    assert "axis_retired" in lines[0]
    tiling = next(l for l in lines if l.startswith("tiling | ")).split(" | ")
    assert tiling[3] == "never tried in this run"


def test_one_round_counts_once_per_axis_and_a_win_beats_its_siblings():
    """Once per round × axis — three variants losing in one round are one failure,
    and any win in the round is a win, whatever the row order."""
    lost3 = _row(1, BASE, [_cand(c, {"tiling": g}, skipped="fit_dev_lost")
                           for c, g in (("a", 2), ("b", 3), ("c", 4))])
    s = axis_status(H0 + [lost3], 2).axes["tiling"]
    assert s.status == "open" and s.streak == 1 and s.fails == {"fit_dev lost": 1}
    mixed = _row(1, BASE, [_cand("w", {"text_negatives": "fp_crop"},
                                 gate_passed_but_lost=True),
                           _cand("l1", {"text_negatives": "gt_crop"}, skipped="fit_dev_lost"),
                           _cand("l2", {"text_negatives": "gt_crop", "tiling": 2},
                                 skipped="fit_dev_lost")], evaluated=["w"])
    s = axis_status(H0 + [_held_out_fail(0.5, BASE, "p", {"text_negatives": "fp_crop"}),
                          mixed], 2).axes["text_negatives"]
    assert s.streak == 0 and s.wins == 1


def test_same_setting_counts_only_when_it_equals_the_champion():
    h = H0 + [_row(r, BASE, [_cand(f"s{r}", {"text_negatives": "fp_crop"},
                                   skipped="same_setting", same_as=same)])
              for r, same in ((1, "sibling"), (2, "tried"))]
    assert axis_status(h, 2).axes["text_negatives"].streak == 0
    h = H0 + [_row(r, BASE, [_cand(f"s{r}", {"text_negatives": "fp_crop"},
                                   skipped="same_setting", same_as="champion")])
              for r in (1, 2)]
    assert axis_status(h, 2).axes["text_negatives"].status == "off"


def test_turning_an_axis_off_is_not_a_try_of_it():
    from agentic_configuration_search.planner.axis_status import _changed_axes
    champ = validate_axes({"visual_negatives": "fp_crop", "visual_negative_k": 4,
                           "exemplar": "fn_arm", "exemplar_fn_k": 4})
    off = dict(champ, visual_negatives=False, visual_negative_k=None)
    assert _changed_axes(off, champ) == set()
    assert _changed_axes(dict(champ, visual_negative_k=8), champ) == {"visual_negatives"}


def test_arm_gates_and_unused_axes_charge_only_their_axis():
    h = H0 + [_row(r, BASE, [_cand(f"a{r}", {"lora": True, "alias": "vlm"},
                                   skipped="lora_gate")]) for r in (1, 2)]
    st = axis_status(h, 2)
    assert st.axes["lora"].status == "off" and st.axes["alias"].measured == 0
    # a relabelled candidate — the unused tiling is unused, the holdout result is exemplar's
    h = H0 + [_row(r, BASE, [_cand(f"t{r}", {"tiling": 2, "exemplar": "base"},
                                   unused_axes=["tiling"])], evaluated=[f"t{r}"])
              for r in (1, 2)]
    st = axis_status(h, 2)
    assert st.axes["tiling"].fails == {"unused": 2}
    assert st.axes["exemplar"].fails == {"held-out fail": 2}


def test_classifier_tripwire_is_empty_material_not_a_failure():
    h = H0 + [_row(r, BASE, [_cand(f"c{r}", {"crop_classifier": True},
                                   skipped="classifier_gate",
                                   empty_axis="crop_classifier")]) for r in (1, 2)]
    s = axis_status(h, 2).axes["crop_classifier"]
    assert s.status == "open" and s.measured == 0 and s.no_material


def test_retired_rows_are_not_tried_combos():
    from agentic_configuration_search.planner.deterministic import tried_combos
    c = _cand("x", {"exemplar": "base"}, skipped="axis_retired")
    assert tried_combos(H0 + [_row(1, BASE, [c])]) == set()


def test_tied_failure_kinds_are_deterministic():
    """A tie takes the fixed order — the same record whatever the hash seed (set order)."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = ("from agentic_configuration_search.planner.axes import validate_axes as v;"
            "from agentic_configuration_search.planner.axis_status import axis_status as a;"
            "B=v({});R=lambda r,c,e=():{'round':r,'champion':{'axes':B},'candidates':c,"
            "'decision':{'adopted':False,'evaluated':{i:{'gate':{}} for i in e}}};"
            "h=[R(0,[]),R(1,[{'id':'x','axes':v({'tiling':2})},"
            "{'id':'y','axes':v({'tiling':3}),'skipped':'fit_dev_lost'}],['x'])];"
            "print(a(h,2).axes['tiling'].fails)")
    src = str(Path(__file__).resolve().parents[1] / "src")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed),
                                "PYTHONPATH": src}, check=True).stdout.strip()
            for seed in range(1, 7)}
    assert outs == {"{'held-out fail': 1}"}


def test_tried_but_unmeasured_axis_says_so():
    """An axis that was only outside screen's top K — proposed, but without a measured outcome."""
    h = H0 + [_row(1, BASE, [_cand("a", {"alias": "vlm"})])]
    s = axis_status(h, 2).axes["alias"]
    assert s.tried == 1 and s.measured == 0
    assert "tried, no measured outcome yet" in s.record()


def test_retired_rows_are_not_listed_as_tried_combos_in_the_prompt():
    from agentic_configuration_search.planner.ledger import tried_under_champion
    c = _cand("x", {"exemplar": "base"}, skipped="axis_retired")
    assert tried_under_champion(H0 + [_row(1, BASE, [c])]) == []


def test_constraint_lines_follow_disabled_axes():
    from agentic_configuration_search.planner.axis_status import AxisStatus
    from agentic_configuration_search.planner.llm import _constraint_lines
    full = " ".join(_constraint_lines((), AxisStatus(2)))
    assert "compound_server_state" in full and "tiling_blocked" in full
    part = " ".join(_constraint_lines(("lora", "tiling"), AxisStatus(2)))
    assert "compound_server_state" not in part and "tiling_blocked" not in part
    assert "Once crop_classifier is on" in part and "lora" not in part


def test_llm_fallback_candidates_are_tagged():
    import json
    import tempfile
    from pathlib import Path
    from agentic_configuration_search.planner.llm import LLMPlanner
    diag = {"train": {"summary": {"fn": {"missed": 5}}}}
    with tempfile.TemporaryDirectory() as td:
        out = LLMPlanner(chat=lambda m, **k: None, artifact_dir=Path(td)).propose(H0, diag)
        assert json.loads((Path(td) / "proposal.json").read_text())["fallback"] \
            == "deterministic"
    assert out["candidates"]
    assert all(c["source"] == "fallback" for c in out["candidates"])


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_config_rejects_bad_retire_after(bad):
    with pytest.raises(ConfigError, match="axis_retire_after"):
        config_from_dict({"dataset": "x", "search": {"axis_retire_after": bad}})
    assert config_from_dict({"dataset": "x",
                             "search": {"axis_retire_after": None}}).search.axis_retire_after is None


# ---- runner shared filter ----------------------------------------------------------

from agentic_configuration_search.loop.runner import Loop  # noqa: E402
from tests.test_empty_round import _cfg  # noqa: E402
from tests.test_loop import big_root  # noqa: E402,F401 — fixture reuse


def _retired_history():
    return H0 + [_held_out_fail(1, BASE, "r1_a", {"exemplar": "base"}),
                 _held_out_fail(2, BASE, "r2_b", {"exemplar": "fn_arm"})]


def test_filter_drops_planner_candidates_on_off_axes(big_root, tmp_path):  # noqa: F811
    loop = Loop(_cfg(big_root), big_root, tmp_path / "run")
    assert loop.cfg.planner.kind == "llm"
    hist = _retired_history()
    off = {"id": "x", "axes": validate_axes({"exemplar": "base", "exemplar_base_k": 2})}
    human = {**off, "id": "h", "source": "human"}
    combo = {"id": "y", "axes": validate_axes({"exemplar": "base", "tiling": 2})}
    fresh, dropped = loop._filter([off, human, combo], hist)
    assert {d["id"]: d["skipped"] for d in dropped} == {"x": "axis_retired"}
    assert [c["id"] for c in fresh] == ["h", "y"]


def test_filter_leaves_the_deterministic_control_alone(big_root, tmp_path):  # noqa: F811
    import dataclasses
    cfg = _cfg(big_root)
    cfg = dataclasses.replace(cfg, planner=dataclasses.replace(cfg.planner,
                                                               kind="deterministic"))
    loop = Loop(cfg, big_root, tmp_path / "run")
    off = {"id": "x", "axes": validate_axes({"exemplar": "base", "exemplar_base_k": 2})}
    fresh, dropped = loop._filter([off], _retired_history())
    assert [c["id"] for c in fresh] == ["x"] and not dropped


def test_filter_spares_the_llm_fallback_and_uses_the_given_status_history(
        big_root, tmp_path):  # noqa: F811
    """The deterministic fallback inside the LLM planner knows nothing of OFF — blocking
    it would make an empty round. A re-ask uses the status counted from the round-start
    history (the same verdict as its prompt)."""
    loop = Loop(_cfg(big_root), big_root, tmp_path / "run")
    off = {"id": "x", "axes": validate_axes({"exemplar": "base", "exemplar_base_k": 2})}
    fresh, _ = loop._filter([{**off, "source": "fallback"}], _retired_history())
    assert [c["id"] for c in fresh] == ["x"]
    start = H0 + [_held_out_fail(1, BASE, "r1_a", {"exemplar": "base"})]
    so_far = start + [_row(2, BASE, [_cand("r2_b", {"exemplar": "fn_arm"},
                                           skipped="fit_dev_lost")])]
    assert loop._filter([off], so_far)[1]                # counting this round in: OFF
    fresh, dropped = loop._filter([off], so_far, status_history=start)
    assert [c["id"] for c in fresh] == ["x"] and not dropped


def test_lora_tripwire_is_empty_material_not_a_failure():
    h = H0 + [_row(r, BASE, [_cand(f"l{r}", {"lora": True}, skipped="lora_gate",
                                   empty_axis="lora", arm_gate={"tripwire": True})])
              for r in (1, 2)]
    s = axis_status(h, 2).axes["lora"]
    assert s.status == "open" and s.measured == 0 and s.no_material


def test_digest_skips_champion_values_and_off_axes():
    """The digest lists only what can be proposed now — not values the champion already uses or OFF axes."""
    from agentic_configuration_search.planner.llm import _digest_lines
    diag = {"train": {"summary": {"fn": {"below_threshold": 9}}}}
    champ = validate_axes({"rescorer": True, "exemplar": "base"})
    line = _digest_lines(diag, champion=champ, off={"lora"})[0]
    axes = line.split("->")[1]
    assert "rescorer" not in axes and "lora" not in axes
    assert "exemplar=fn_arm" in axes and "crop_classifier" in axes


def test_tripwire_and_empty_sources_are_reported_apart():
    """A tripwire that skipped training (value True) and an empty source value are
    written apart, and a value is written once."""
    from agentic_configuration_search.diagnose.report import _outcome as report_outcome
    from agentic_configuration_search.planner.ledger import _outcome as ledger_outcome
    h = H0 + [_row(r, BASE, [_cand(f"c{r}", {"crop_classifier": True},
                                   skipped="classifier_gate", empty_axis="crop_classifier"),
                             _cand(f"a{r}", {"alias": "vlm"}, skipped="no_material",
                                   empty_axis="alias")]) for r in (1, 2)]
    st = axis_status(h, 2)
    assert st.axes["crop_classifier"].record() == "not trained, too little data x2"
    alias = st.axes["alias"].record()
    assert "empty material for vlm" in alias and "vlm, vlm" not in alias
    row = h[1]["candidates"][0]
    assert ledger_outcome(row, {}) == "no_material" and report_outcome(row, {}) == "no_material"


def test_digest_says_when_nothing_can_be_proposed_now():
    """When a cause has remedies but nothing can be proposed now it is "no axis to
    propose now", not "not addressable by a setting". Values with empty material
    under the current champion are not recommended either."""
    from agentic_configuration_search.planner.llm import NO_REMEDY, NONE_NOW, _digest_lines
    diag = {"train": {"summary": {"fn": {"localization": 4}, "fp": {"duplicate": 2,
                                                                   "class_confusion": 3}}}}
    lines = {l.split(":")[0][2:]: l for l in _digest_lines(
        diag, champion=validate_axes({"lora": True}),
        empty={("alias", "vlm"), ("text_negatives", "gt_crop")})}
    assert lines["localization"].endswith(NONE_NOW)     # lora is already on
    assert lines["duplicate"].endswith(NO_REMEDY)        # no remedy at all
    assert "alias" not in lines["class_confusion"] and "text_negatives" not in lines["class_confusion"]


def test_digest_skips_folded_and_lower_tiling_grids():
    """Folded grids and grids at or below the champion's are not recommended — the same verdict as the board and constraints."""
    from agentic_configuration_search.planner.llm import NONE_NOW, _digest_lines
    diag = {"train": {"summary": {"fn": {"missed_small": 5}}}}

    class Folded:                       # grid >= 2 folded (the fold verdict itself is tested with gates)
        def block(self, axes):
            return "folded" if (axes.get("tiling") or 0) >= 2 else None
    assert _digest_lines(diag, champion=BASE, gate=Folded())[0].endswith(NONE_NOW)
    champ = validate_axes({"tiling": 3})
    line = _digest_lines(diag, champion=champ)[0]
    assert "tiling" in line                                # 4 and up remain
    assert _digest_lines(diag, champion=validate_axes({"tiling": 6}))[0].endswith(NONE_NOW)


def test_empty_inherited_axis_is_not_charged_to_the_champion_value():
    """When changing exemplar re-mines an inherited negative axis that comes out empty,
    the champion's value is not recorded as having no material."""
    champ = validate_axes({"exemplar": "base", "text_negatives": "fp_crop"})
    h = [_row(0, champ, []),
         _row(1, champ, [_cand("e", {**champ, "exemplar": "fn_arm"}, skipped="no_material",
                               empty_axis="text_negatives")])]
    assert axis_status(h, 2).axes["text_negatives"].no_material == []


def test_failure_rounds_follow_the_failure_kinds():
    h = H0 + [_held_out_fail(1, BASE, "a", {"text_negatives": "fp_crop"}),
              _row(2, BASE, [_cand("b", {"text_negatives": "gt_crop"}, skipped="no_material",
                                   empty_axis="text_negatives")])]
    rec = axis_status(h, 2).axes["text_negatives"].record()
    assert rec == "held-out fail x1 (last failed in R1); empty material for gt_crop"
