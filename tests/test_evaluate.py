"""eval/evaluate — holdout τ estimation (legacy field name tau_test) and the adoption gate."""

import pytest

from agentic_detector_config.config import config_from_dict
from agentic_detector_config.core.data import load_dataset
from agentic_detector_config.eval.evaluate import evaluate_on_holdout, gate
from agentic_detector_config.fitting.materials import Materials
from agentic_detector_config.infer import gateway
from agentic_detector_config.infer.queries import config_grid

CFG = config_from_dict({"dataset": "x"})


def _setting(per_class):
    return {"configs": config_grid(), "per_class": per_class}


def test_tau_is_estimated_on_test_not_carried_from_screen(dataset_root, monkeypatch):
    """The fit_dev threshold carried in per_class is discarded; τ is re-estimated on the holdout."""
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    test = ds.holdout

    def fake_infer(ctx, client, config, queries, ontology, image_paths,
                   alias_to_name=None, **kw):
        # GT hit (0.6) + low-score FP (0.2) — the F1-max cut must be 0.6
        out = []
        for e in test.images:
            dets = [{"box": list(a.bbox_xyxy), "score": 0.6,
                     "category_id": a.category_id} for a in e.annotations]
            dets.append({"box": [70.0, 60.0, 90.0, 76.0], "score": 0.2,
                         "category_id": e.annotations[0].category_id})
            out.append(dets)
        return out

    monkeypatch.setattr(gateway, "infer_config", fake_infer)
    # the 0.9 threshold from screen (which would cut everything → F1 0) must be ignored
    setting = _setting({1: {"config_id": 0, "modality": "text", "threshold": 0.9,
                            "f_beta": 0.0},
                        2: {"config_id": 0, "modality": "text", "threshold": 0.9,
                            "f_beta": 0.0}})
    out = evaluate_on_holdout(CFG, object(), test, ds.ontology, setting, Materials())
    # Re-estimated on this split, not the 0.9 from screen. The exact point lies in
    # the empty gap between FP (0.2) and TP (0.6); where within it is set by
    # search.tau_tolerance (core.metrics._max_margin_threshold) —
    # this guards "re-estimates", not a specific value.
    assert 0.2 < out["tau_test"][1] <= 0.6
    assert 0.2 < out["tau_test"][2] <= 0.6
    assert out["micro_f_beta"] == pytest.approx(1.0)
    assert len(out["records"]) == len(test.images)
    # floor records keep FPs below the cut (input for below_threshold diagnosis)
    n_floor = sum(len(r["detections"]) for r in out["floor_records"])
    n_kept = sum(len(r["detections"]) for r in out["records"])
    assert n_floor > n_kept


def test_empty_setting_errors(dataset_root):
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    with pytest.raises(ValueError, match="empty setting"):
        evaluate_on_holdout(CFG, object(), ds.holdout, ds.ontology, _setting({}),
                         Materials())


# ------------------------------------------------------------------- gate

GT = ([10, 10, 30, 30], 1)
HIT = ([10, 10, 30, 30], 0.9, 1)
MISS = ([60, 60, 80, 80], 0.9, 1)


def _rec(dets, gts):
    return {"file": "x", "width": 100, "height": 100,
            "detections": [{"box": b, "score": s, "category_id": c}
                           for b, s, c in dets],
            "gts": [{"box": b, "category_id": c} for b, c in gts]}


def test_gate_adopts_clear_improvement():
    champ = [_rec([MISS], [GT]) for _ in range(20)]
    cand = [_rec([HIT], [GT]) for _ in range(20)]
    out = gate(champ, cand, cat_ids=[1], iou_thr=0.5, margin=0.0,
               n_boot=200, seed=1)
    assert out["adopted"] is True and out["delta"] == pytest.approx(1.0)


def test_gate_holds_champion_when_no_gain():
    recs = [_rec([HIT], [GT]) for _ in range(10)]
    out = gate(recs, recs, cat_ids=[1], iou_thr=0.5, margin=0.0,
               n_boot=200, seed=1)
    assert out["adopted"] is False and "keeping champion" in out["reason"]


def test_gate_respects_margin():
    """Not adopted if the CI lower bound is positive but below margin."""
    champ = [_rec([MISS], [GT]) for _ in range(20)]
    cand = [_rec([HIT], [GT]) for _ in range(20)]
    out = gate(champ, cand, cat_ids=[1], iou_thr=0.5, margin=2.0,
               n_boot=200, seed=1)
    assert out["adopted"] is False


def test_gate_requires_paired_records():
    recs = [_rec([HIT], [GT])]
    with pytest.raises(ValueError, match="records length mismatch"):
        gate(recs, recs * 2, cat_ids=[1], iou_thr=0.5, margin=0.0)


def test_beta_flows_into_tau_and_micro(dataset_root, monkeypatch):
    """cfg.beta flows through τ estimation and micro — at β=2, micro is F2."""
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    test = ds.holdout

    def fake_infer(ctx, client, config, queries, ontology, image_paths,
                   alias_to_name=None, **kw):
        # only class 1 GT is hit — overall TP1 FN1 → P=1, R=0.5
        return [[{"box": list(a.bbox_xyxy), "score": 0.6,
                  "category_id": a.category_id}
                 for a in e.annotations if a.category_id == 1]
                for e in test.images]

    monkeypatch.setattr(gateway, "infer_config", fake_infer)
    setting = _setting({1: {"config_id": 0, "modality": "text", "threshold": 0.5,
                            "f_beta": 0.0},
                        2: {"config_id": 0, "modality": "text", "threshold": 0.5,
                            "f_beta": 0.0}})
    cfg2 = config_from_dict({"dataset": "x", "beta": 2.0})
    out1 = evaluate_on_holdout(CFG, object(), test, ds.ontology, setting, Materials())
    out2 = evaluate_on_holdout(cfg2, object(), test, ds.ontology, setting, Materials())
    assert out1["micro_f_beta"] == pytest.approx(2 / 3)          # F1
    assert out2["micro_f_beta"] == pytest.approx(5 * 0.5 / 4.5)  # F2 ≈ 0.5556


def test_eval_artifact_keeps_candidate_records(tmp_path):
    """Candidate records are kept in the artifact — re-judging past runs under a changed
    gate needs more than champion records."""
    import inspect

    from agentic_detector_config.loop import runner
    src = inspect.getsource(runner.Loop.run_round)
    i = src.index('_write(rd / "eval.json"')
    block = src[i:i + 400]
    assert '"records"' in block, "eval.json has no candidate records"


def test_gate_boundary_requires_positive_delta():
    """Boundary rule: adopt when the lower bound is **>=** margin and the point estimate is a real gain.

    A one-sided lower bound of exactly 0 is common — on a small holdout many resamples
    give the same micro F(β) for both settings, so the Δ distribution piles up at 0.
    A strict `>` alone would reject a real gain (Δ>0) and no change (Δ=0) for the same reason."""
    import numpy as np

    from agentic_detector_config.core import metrics
    from agentic_detector_config.eval.evaluate import gate

    def fake_ci(a, b, stat, **kw):
        return dict(fake_ci.out, n_images=8)

    real = metrics.bootstrap_ci
    metrics.bootstrap_ci = fake_ci
    try:
        fake_ci.out = {"delta": 0.029, "lo": 0.0, "hi": None}
        assert gate([], [], cat_ids=[1], iou_thr=0.5, margin=0.0)["adopted"] is True
        fake_ci.out = {"delta": 0.0, "lo": 0.0, "hi": None}      # no change
        assert gate([], [], cat_ids=[1], iou_thr=0.5, margin=0.0)["adopted"] is False
        fake_ci.out = {"delta": 0.03, "lo": -0.004, "hi": None}  # lower bound short
        assert gate([], [], cat_ids=[1], iou_thr=0.5, margin=0.0)["adopted"] is False
        fake_ci.out = {"delta": -0.01, "lo": 0.001, "hi": None}  # regression
        assert gate([], [], cat_ids=[1], iou_thr=0.5, margin=0.0)["adopted"] is False
    finally:
        metrics.bootstrap_ci = real
