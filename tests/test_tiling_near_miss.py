"""One-step near-miss exception of the tiling fold gate.

Even if grid g was screened and not used under the current champion, g+1 is
opened **once** when two conditions hold together:

(a) noise — the one-sided upper bound of (tiled − untiled) per-class mean F
    over a paired bootstrap of fit_dev images (the prescreen subset, if
    prescreened), `tiling_f.gap_ub`, is at or above 0.
(b) magnification shortfall — for s*, a lower quantile of the TP sizes (input
    px) of the class with the most missed_small in the train sizes counted at
    the champion's fit_dev screen thresholds (not tau_test),
    m·mag(g) < s* ≤ m·mag(g+1)
    (m = that class's missed_small median px). With fewer TPs than
    `search.tiling_tp_min`, (b) cannot be judged → no exception.

The verdict is a pure function reading only history rows (`tiling_f.gap_ub`,
`tiling_shortfall`).
"""

import json

import numpy as np
import pytest

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from agentic_configuration_search.planner.axes import validate_axes
from tests.test_loop import _fakes, _gt_lookup, _hist
from tests.test_loop import big_root  # noqa: F401 — fixture reuse
from tests.test_time_efficiency import _entry, _llm_by_round, _row


def _unused(cid, g, *, near=True, short=True, prescreen=False, **extra):
    row = _row(cid, g, used=False, skipped="axis_unused", **extra)
    row["tiling_f"] = {"tiled": 0.70, "untiled": 0.72,
                       "gap_ub": 0.01 if near else -0.02}
    row["tiling_shortfall"] = {"grid": g, "class": "cat b", "n_tp": 40,
                               "tp_px": 30.0, "missed_px": 18.0,
                               "mag_g": 1.5, "mag_next": 2.0, "shortfall": short}
    if prescreen:
        row["prescreen"] = {"n_images": 20, "of": 80, "passed": False}
        row["note"] = "prescreen: ..."
    return row


def _gate(hist):
    from agentic_configuration_search.planner.gates import tiling_gate
    return tiling_gate(hist)


def _blocked(gate, g, **extra):
    return gate.block(validate_axes({"tiling": g, **extra})) is not None


# --- gate truth table -------------------------------------------------------------

@pytest.mark.parametrize("near,short,opens", [
    (True, True, True), (True, False, False), (False, True, False),
    (False, False, False)])
def test_exception_truth_table(near, short, opens):
    hist = [_entry(0, 0, []), _entry(1, 0, [_unused("t2", 2, near=near, short=short)])]
    gate = _gate(hist)
    assert gate.folded_at == 2
    assert _blocked(gate, 2)                     # g itself stays folded
    assert _blocked(gate, 3) is (not opens)      # only g+1 opens, by exception
    assert _blocked(gate, 4) and _blocked(gate, 6)   # g+2 and up stay blocked
    assert (gate.open_grid == 3) is opens


def test_exception_is_consumed_after_g_plus_1_is_screened_unused():
    hist = [_entry(0, 0, []),
            _entry(1, 0, [_unused("t2", 2)]),
            _entry(2, 0, [_unused("t3", 3)])]     # g+1, screened by exception, also unused
    gate = _gate(hist)
    assert gate.folded_at == 2 and gate.open_grid is None
    assert _blocked(gate, 3) and _blocked(gate, 4)   # no new exception at g+1
    assert not any("ONCE" in line for line in gate.describe())


def test_exception_is_consumed_after_g_plus_1_is_screened_used():
    """If g+1 was used, the normal rule applies — the exception is spent and the
    fold at g holds."""
    hist = [_entry(0, 0, []),
            _entry(1, 0, [_unused("t2", 2)]),
            _entry(2, 0, [_row("t3", 3, used=True, fit_dev=0.6)])]
    gate = _gate(hist)
    assert gate.open_grid is None
    assert _blocked(gate, 3, alias="vlm") and _blocked(gate, 4)


def test_blocked_g_plus_1_attempt_does_not_consume():
    """A g+1 that never reached screen (rejected for another reason) does not
    spend the exception."""
    hist = [_entry(0, 0, []),
            _entry(1, 0, [_unused("t2", 2),
                          _row("t3x", 3, skipped="axis_disabled")])]
    assert _gate(hist).open_grid == 3


def test_exception_resets_when_champion_changes():
    hist = [_entry(0, 0, []),
            _entry(1, 0, [_unused("t2", 2)]),
            _entry(2, 0, [_unused("t3", 3)]),
            _entry(3, 3, [], champ_axes={"exemplar": "base"})]
    gate = _gate(hist)
    assert gate.folded_at is None and gate.open_grid is None
    for g in (2, 3, 4):
        assert not _blocked(gate, g, exemplar="base")


def test_exception_reads_prescreen_rejections():
    """Prescreen rejections follow the same rule — with the gap_ub from the subset."""
    hist = [_entry(0, 0, []), _entry(1, 0, [_unused("t3", 3, prescreen=True)])]
    gate = _gate(hist)
    assert gate.folded_at == 3 and gate.open_grid == 4
    assert not _blocked(gate, 4) and _blocked(gate, 5)


def test_exception_needs_every_unused_row_at_the_fold_to_qualify():
    hist = [_entry(0, 0, []),
            _entry(1, 0, [_unused("t2", 2), _unused("t2a", 2, near=False,
                                                     alias="vlm")])]
    assert _gate(hist).open_grid is None


def test_exception_ignores_rows_without_evidence_and_the_top_grid():
    bare = _row("t2", 2, used=False, skipped="axis_unused")   # no gap_ub or shortfall
    assert _gate([_entry(0, 0, []), _entry(1, 0, [bare])]).open_grid is None
    top = _unused("t6", 6)                                       # g+1 is outside the axis
    assert _gate([_entry(0, 0, []), _entry(1, 0, [top])]).open_grid is None


def test_exception_is_a_pure_function_of_history():
    """resume — serializing and re-reading history gives the same verdict."""
    hist = [_entry(0, 0, []), _entry(1, 0, [_unused("t2", 2)])]
    again = json.loads(json.dumps(hist))
    a, b = _gate(hist), _gate(again)
    assert (a.folded_at, a.open_grid, a.describe()) == \
        (b.folded_at, b.open_grid, b.describe())


def test_describe_states_the_open_grid_and_why():
    gate = _gate([_entry(0, 0, []), _entry(1, 0, [_unused("t2", 2)])])
    lines = gate.describe()
    fold = [ln for ln in lines if "folded" in ln]
    assert fold and "grid >= 2" in fold[0]
    once = [ln for ln in lines if "ONCE" in ln]
    assert len(once) == 1
    text = once[0]
    assert "grid 3" in text and "near-miss" in text
    assert "+0.010" in text                       # the gap upper bound
    assert "18 px" in text and "30 px" in text    # missed median and TP s*
    assert "\n" not in text


# --- (a) bootstrap near-miss -------------------------------------------------------

def test_bootstrap_ci_upper_tail():
    from agentic_configuration_search.core.metrics import bootstrap_ci

    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 2.5, 3.5, 4.5]
    out = bootstrap_ci(a, b, lambda xs: float(np.mean(xs)), n_boot=200,
                       seed=0, alpha=0.05, tail="upper")
    assert out["lo"] is None and out["hi"] == pytest.approx(0.5)


def _img(tp, fp, fn):
    """Per-class (tp, fp, fn) of one image — a single class."""
    return np.array([[tp, fp, fn]], dtype=float)


def test_gap_ub_identical_configs_is_a_near_miss():
    from agentic_configuration_search.eval.screen import tiling_gap_ub

    rows = [_img(1, 0, 0), _img(1, 1, 0), _img(0, 0, 1), _img(2, 0, 0)]
    ub = tiling_gap_ub(rows, rows, beta=1.0, n_boot=300, seed=0, alpha=0.05)
    assert ub == pytest.approx(0.0)              # upper bound ≥ 0 → near-miss


def test_gap_ub_clear_loss_is_not_a_near_miss():
    from agentic_configuration_search.eval.screen import tiling_gap_ub

    untiled = [_img(1, 0, 0)] * 30
    tiled = [_img(0, 1, 1)] * 30                   # loses on every image
    ub = tiling_gap_ub(untiled, tiled, beta=1.0, n_boot=300, seed=0, alpha=0.05)
    assert ub < 0


def test_gap_ub_small_loss_on_few_images_is_within_noise():
    from agentic_configuration_search.eval.screen import tiling_gap_ub

    untiled = [_img(1, 0, 0)] * 10
    tiled = [_img(1, 0, 0)] * 9 + [_img(0, 1, 1)]  # loses on one image only
    ub = tiling_gap_ub(untiled, tiled, beta=1.0, n_boot=500, seed=0, alpha=0.05)
    assert ub >= 0
    # with more samples, the same rate of loss is outside noise — why a prescreen
    # subset gives wider bounds
    ub_big = tiling_gap_ub(untiled * 20, tiled * 20, beta=1.0, n_boot=500,
                           seed=0, alpha=0.05)
    assert ub_big < 0


def test_gap_ub_point_matches_tiling_f_statistic():
    """The bootstrap statistic is tiling_f's yardstick — per-class mean F(β)."""
    from agentic_configuration_search.eval.screen import class_mean_f

    rows = [np.array([[1, 0, 0], [0, 1, 1]], dtype=float),
            np.array([[1, 0, 1], [1, 0, 0]], dtype=float)]
    # class 1: tp2 fp0 fn1 → F 0.8, class 2: tp1 fp1 fn1 → F 0.5
    assert class_mean_f(rows, beta=1.0) == pytest.approx(0.65)


def test_screen_records_gap_ub_with_gate_settings(big_root, tmp_path,  # noqa: F811
                                                  monkeypatch):
    """Screen puts gap_ub into tiling_f — with tiled equal to untiled the bound is 0."""
    from tests.test_loop import _fake_infer

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    chat = _llm_by_round({1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}]})
    from agentic_configuration_search.config import config_from_dict
    cfg = config_from_dict({"dataset": str(big_root), "rounds": 1,
                            "planner": {"kind": "llm", "max_reasks": 0},
                            "gate": {"bootstrap": 200}})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=1, cfg=cfg, **_fakes())
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["t2"]
    assert rec["tiling_used"] is False
    assert rec["tiling_f"]["gap_ub"] == pytest.approx(0.0)
    # big_root's GT are all above the cut (0.04) — without missed_small there is
    # no magnification verdict either
    assert "tiling_shortfall" not in rec


# --- (b) s* and the magnification shortfall ----------------------------------------

def _diag(tp_ratios: dict, missed_small: dict) -> dict:
    """A train diagnosis — fn (classified at tau_test, what the planner sees) and
    tiling_sizes (classified at the screen thresholds, the material for s*)
    filled with the same values. s* reads only tiling_sizes."""
    fn = [{"kind": "missed_small", "category": name, "size_ratio": r}
          for name, rs in missed_small.items() for r in rs]
    fn.append({"kind": "missed", "category": "cat a", "size_ratio": 0.5})
    return {"fn": fn, "tiling_sizes": {
        "thresholds": "fit_dev",
        "tp": {k: sorted(v) for k, v in tp_ratios.items()},
        "missed_small": {k: sorted(v) for k, v in missed_small.items()}}}


def test_tp_floor_reads_only_screen_threshold_sizes():
    """s* and m come only from tiling_sizes counted at the fit_dev thresholds —
    the fn list classified at tau_test (holdout-estimated) is only the diagnosis
    the planner sees, not material for s*."""
    from agentic_configuration_search.planner.gates import tp_floor

    d = _diag({"cat b": [0.04] * 30}, {"cat b": [0.02] * 3})
    d["fn"] = []                                      # even with an empty tau_test classification
    assert tp_floor(d, quantile=0.1, tp_min=20)["class"] == "cat b"
    d = _diag({"cat b": [0.04] * 30}, {"cat b": [0.02] * 3})
    del d["tiling_sizes"]                             # no material, no verdict
    assert tp_floor(d, quantile=0.1, tp_min=20) is None


def test_tp_floor_uses_the_class_with_most_small_misses():
    from agentic_configuration_search.planner.gates import tp_floor

    tps = {"cat a": [0.01] * 50, "cat b": [0.03 + 0.001 * i for i in range(40)]}
    d = _diag(tps, {"cat a": [0.02], "cat b": [0.02, 0.022, 0.024]})
    out = tp_floor(d, quantile=0.1, tp_min=20)
    assert out["class"] == "cat b" and out["n_tp"] == 40
    assert out["tp_px"] == pytest.approx((0.03 + 0.004) * 640, abs=0.1)
    assert out["missed_px"] == pytest.approx(0.022 * 640, abs=0.1)
    assert out["available"] is True


def test_tp_floor_too_few_tps_is_unavailable():
    from agentic_configuration_search.planner.gates import tiling_shortfall, tp_floor

    d = _diag({"cat b": [0.05] * 5}, {"cat b": [0.02, 0.02]})
    out = tp_floor(d, quantile=0.1, tp_min=20)
    assert out["available"] is False and out["n_tp"] == 5
    v = tiling_shortfall(d, 2, overlap=0.5, quantile=0.1, tp_min=20)
    assert v["shortfall"] is False
    assert tp_floor(_diag({"cat b": [0.05] * 30}, {}), quantile=0.1,
                    tp_min=20) is None                          # no missed_small


@pytest.mark.parametrize("m,g,expect", [
    (20.0, 2, True),    # 20·1.5=30 < 32 ≤ 20·2=40
    (22.0, 2, False),   # 22·1.5=33 ≥ 32 — g already reached it (not a magnification issue)
    (12.0, 2, False),   # 12·2=24 < 32 — g+1 does not reach it either
    (12.0, 4, True),    # 12·2.5=30 < 32 ≤ 12·3=36
])
def test_shortfall_window(m, g, expect):
    from agentic_configuration_search.planner.gates import tiling_shortfall

    tps = {"cat b": [32 / 640] * 30}
    d = _diag(tps, {"cat b": [m / 640]})
    v = tiling_shortfall(d, g, overlap=0.5, quantile=0.1, tp_min=20)
    assert v["shortfall"] is expect and v["grid"] == g


def test_size_profile_uses_the_given_thresholds():
    """tiling_sizes are counted by applying the **given** thresholds (the fit_dev
    screen thresholds) to the floor records — not records filtered at tau_test."""
    from tests.test_diagnose import IOU, ONTO, _records
    from agentic_configuration_search.diagnose.errors import size_profile

    recs, _floor = _records()                       # a.png in recs: a 0.9 TP + a 0.7 FP
    low = size_profile(recs, {1: 0.5, 2: 0.5}, ONTO, iou_thr=IOU)
    assert low["thresholds"] == "fit_dev"
    assert low["tp"] == {"cat a": [0.2]}             # the 20×20 TP on a.png
    high = size_profile(recs, {1: 0.99, 2: 0.99}, ONTO, iou_thr=IOU)
    assert high["tp"] == {}                           # other thresholds, other profile


def test_diagnose_no_longer_carries_tp_sizes():
    from tests.test_diagnose import IOU, ONTO, _records
    from agentic_configuration_search.diagnose.errors import diagnose

    recs, floor = _records()
    train = diagnose(recs, floor, {1: 0.5, 2: 0.5}, ONTO, iou_thr=IOU,
                     anonymize=False)
    assert "tp_size_ratios" not in train and "tiling_sizes" not in train


def test_config_knobs_defaults_and_validation():
    from agentic_configuration_search.config import ConfigError, config_from_dict

    s = config_from_dict({"dataset": "x"}).search
    assert (s.tiling_tp_quantile, s.tiling_tp_min) == (0.1, 20)
    for bad in ({"tiling_tp_quantile": 0.0}, {"tiling_tp_quantile": 1.0},
                {"tiling_tp_min": 0}):
        with pytest.raises(ConfigError):
            config_from_dict({"dataset": "x", "search": bad})


# --- prompt -----------------------------------------------------------------------

def test_prompt_rule_paragraph_states_the_exception():
    from agentic_configuration_search.planner.llm import PROMPT

    flat = " ".join(PROMPT.split())
    assert "gap_ub" in flat and "exactly once" in flat
    assert "tiling_blocked" in flat


def test_small_block_reports_champion_tp_floor():
    from agentic_configuration_search.planner.llm import _small_block

    tps = {"cat b": [32 / 640] * 30}
    d = {"train": _diag(tps, {"cat b": [20 / 640] * 4})}
    text = _small_block(d, overlap=0.5, tp_quantile=0.1, tp_min=20)
    assert "champion TP p10" in text and "32 px" in text and "cat b" in text
    # with too few TPs that line is absent (the block itself stays)
    thin = {"train": _diag({"cat b": [0.04] * 3}, {"cat b": [20 / 640] * 4})}
    text = _small_block(thin, overlap=0.5, tp_quantile=0.1, tp_min=20)
    assert "median" in text and "champion TP" not in text


def test_llm_folded_lines_carry_the_exception():
    from agentic_configuration_search.planner.llm import _folded_axis_lines

    hist = [_entry(0, 0, []), _entry(1, 0, [_unused("t2", 2)])]
    assert any("ONCE" in ln for ln in _folded_axis_lines(hist))


# --- tiling_f of an adopted tiling candidate ----------------------------------------

def test_rescored_screen_keeps_tiling_f(monkeypatch):
    """Even when the rescorer arm replaces the fit_dev scoring, tiling_f (the
    evidence for the tile decision) stays — the reason it went missing from
    adopted tiling+rescorer rows."""
    from agentic_configuration_search.loop import rescore

    monkeypatch.setattr(rescore, "rescored_split_score",
                        lambda *a, **k: {"per_class": {1: {"f_beta": 0.9}},
                                         "micro_f_beta": 0.9})
    setting = {"configs": [], "per_class": {1: {"f_beta": 0.8}},
               "fit_dev_micro_f_beta": 0.8, "modality_diagnostic": {},
               "tiling_f": {"tiled": 0.8, "untiled": 0.7, "gap_ub": 0.2}}

    class _Ds:
        fit_dev = None
    out = rescore.rescored_screen(None, _Ds(), setting, None, {})
    assert out["tiling_f"] == setting["tiling_f"]
    assert out["fit_dev_micro_f_beta"] == 0.9


def test_adopted_tiling_row_carries_tiling_f(big_root, tmp_path,  # noqa: F811
                                             monkeypatch):
    from tests.test_loop import _tile_helps_infer

    monkeypatch.setattr(gateway, "infer_config", _tile_helps_infer(_gt_lookup(big_root)))
    chat = _llm_by_round({1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}]})
    from agentic_configuration_search.config import config_from_dict
    cfg = config_from_dict({"dataset": str(big_root), "rounds": 1,
                            "planner": {"kind": "llm", "max_reasks": 0},
                            "gate": {"bootstrap": 200}})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=1, cfg=cfg, **_fakes())
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["t2"]
    assert rec["tiling_used"] is True
    assert {"tiled", "untiled", "gap_ub"} <= set(rec["tiling_f"])


# --- end to end: the exception opens g+1 once ----------------------------------------

def _shortfall_infer(by_path):
    """The text pass finds all of class 1 and only class-2 objects of 30px or
    more. Tiles change nothing (tiled == untiled → gap_ub 0, a near-miss)."""
    def fake(ctx, client, config, queries, ontology, image_paths,
             alias_to_name=None, **kw):
        out = []
        for p in image_paths:
            e = by_path[str(p)]
            out.append([{"box": list(a.bbox_xyxy), "score": 0.9,
                         "category_id": a.category_id, "sources": ["full"]}
                        for a in e.annotations
                        if a.category_id == 1
                        or a.bbox_xyxy[2] - a.bbox_xyxy[0] >= 30])
        return out
    return fake


def test_loop_opens_g_plus_1_once(big_root, tmp_path, monkeypatch):  # noqa: F811
    """big_root class 2: 20px (r≈0.224, missed_small), 30 and 40px TPs. At overlap
    0.6 (640 px input side), m≈143px·1.4=201 < s*≈215 ≤ 143·1.8=258 — grid 2
    falls short on magnification, grid 3 reaches it."""
    from agentic_configuration_search.config import config_from_dict

    monkeypatch.setattr(gateway, "infer_config",
                        _shortfall_infer(_gt_lookup(big_root)))
    chat = _llm_by_round({
        1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}],
        2: [{"id": "t3", "axes": {"tiling": 3}, "rationale": "y"},
            {"id": "t4", "axes": {"tiling": 4}, "rationale": "z"}],
        3: [{"id": "t3b", "axes": {"tiling": 3, "alias": "vlm"}, "rationale": "w"}]})
    cfg = config_from_dict({
        "dataset": str(big_root), "rounds": 3,
        "planner": {"kind": "llm", "max_reasks": 0},
        "gate": {"bootstrap": 200},
        "saturation": {"rounds": 5, "empty_rounds": 0},
        "diagnose": {"small_object_ratio": 0.3},
        "search": {"tiling_overlap": 0.6, "tiling_tp_min": 1}})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=3, cfg=cfg, **_fakes())
    h = _hist(tmp_path / "run")
    r1 = {c["id"]: c for c in h[1]["candidates"]}
    t2 = r1["t2"]
    assert t2["tiling_used"] is False and t2["tiling_f"]["gap_ub"] >= 0
    assert t2["tiling_shortfall"]["shortfall"] is True
    assert t2["tiling_shortfall"]["class"] == "cat b"
    r2 = {c["id"]: c for c in h[2]["candidates"]}
    assert r2["t3"].get("skipped") != "tiling_blocked"      # opened by the exception
    assert "tiling_used" in r2["t3"]
    assert r2["t4"]["skipped"] == "tiling_blocked"            # g+2 is blocked
    r3 = {c["id"]: c for c in h[3]["candidates"]}
    assert r3["t3b"]["skipped"] == "tiling_blocked"           # spent after one use


def test_shortfall_does_not_depend_on_holdout_thresholds(big_root, tmp_path,  # noqa: F811
                                                         monkeypatch):
    """Perturbing tau_test (estimated on holdout) leaves s*, m, the shortfall
    verdict and the fold gate's exception (whether g+1 opens) unchanged — s*
    comes from train TPs counted at the champion's fit_dev screen thresholds.
    Raising tau_test above every score (0.9) leaves neither TPs nor missed_small
    in the tau_test-based train diagnosis."""
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop import runner as runner_mod

    def run(out, perturb):
        monkeypatch.setattr(gateway, "infer_config",
                            _shortfall_infer(_gt_lookup(big_root)))
        real = runner_mod.evaluate_on_holdout

        def fake_eval(*a, **k):
            out_ = real(*a, **k)
            if perturb:
                out_["tau_test"] = {cid: 0.95 for cid in out_["tau_test"]}
            return out_
        monkeypatch.setattr(runner_mod, "evaluate_on_holdout", fake_eval)
        chat = _llm_by_round({
            1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}],
            2: [{"id": "t3", "axes": {"tiling": 3}, "rationale": "y"}]})
        cfg = config_from_dict({
            "dataset": str(big_root), "rounds": 2,
            "planner": {"kind": "llm", "max_reasks": 0},
            "gate": {"bootstrap": 200},
            "saturation": {"rounds": 5, "empty_rounds": 0},
            "diagnose": {"small_object_ratio": 0.3},
            "search": {"tiling_overlap": 0.6, "tiling_tp_min": 1}})
        run_rounds(big_root, out_dir=out, planner="llm", llm_chat=chat,
                   rounds=2, cfg=cfg, **_fakes())
        return _hist(out)

    a = run(tmp_path / "a", False)
    b = run(tmp_path / "b", True)
    assert set(b[0]["decision"]["tau_test"].values()) == {0.95}   # the perturbation took
    assert a[0]["decision"]["tau_test"] != b[0]["decision"]["tau_test"]

    def view(h):
        return [(c["id"], c.get("skipped"), c.get("tiling_used"),
                 c.get("tiling_shortfall"))
                for e in h[1:] for c in e["candidates"]]
    va, vb = view(a), view(b)
    assert va[0][3]["shortfall"] is True                    # the exception actually triggers
    assert va[1][1] != "tiling_blocked"                     # g+1 opened
    assert va == vb
