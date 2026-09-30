"""metrics.py: FP cause classification + F1-max threshold estimation."""

import pytest

from agentic_detector_config.core import metrics

IOU = 0.5


def _rec(dets, gts, w=100, h=100):
    return {"detections": dets, "gts": gts, "width": w, "height": h}


def _d(box, score, cid, sources=None):
    d = {"box": box, "score": score, "category_id": cid}
    if sources:
        d["sources"] = sources
    return d


def test_estimate_f_beta_threshold_finds_correct_cut():
    """A distribution whose correct cut can be computed by hand: at thr=0.6, TP2/FP0/FN0 -> F1=1.0."""
    per_image = [_rec(
        dets=[_d([10, 10, 30, 30], 0.9, 1),   # matches GT A
              _d([50, 50, 70, 70], 0.6, 1),   # matches GT B
              _d([0, 80, 10, 95], 0.3, 1),    # FP
              _d([80, 0, 95, 10], 0.2, 1)],   # FP
        gts=[{"box": [10, 10, 30, 30], "category_id": 1},
             {"box": [50, 50, 70, 70], "category_id": 1}])]
    thr, f1 = metrics.estimate_f_beta_threshold(per_image, 1, floor=0.03,
                                            max_candidates=50, iou_thr=IOU)
    assert thr == pytest.approx(0.6)
    assert f1 == pytest.approx(1.0)


def test_estimate_threshold_no_detections_keeps_floor():
    per_image = [_rec([], [{"box": [1, 1, 5, 5], "category_id": 1}])]
    thr, f1 = metrics.estimate_f_beta_threshold(per_image, 1, floor=0.03,
                                            max_candidates=50, iou_thr=IOU)
    assert thr == 0.03 and f1 == 0.0


def test_estimate_threshold_subsamples_candidates():
    dets = [_d([10, 10, 30, 30], 0.5 + i * 0.001, 1) for i in range(200)]
    per_image = [_rec(dets, [{"box": [10, 10, 30, 30], "category_id": 1}])]
    thr, _ = metrics.estimate_f_beta_threshold(per_image, 1, floor=0.03,
                                           max_candidates=10, iou_thr=IOU)
    assert 0.5 <= thr <= 0.7  # still valid after candidate subsampling


def test_fp_classification_kinds():
    gt1 = {"box": [10, 10, 30, 30], "category_id": 1}
    per_image = [_rec(
        dets=[
            _d([10, 10, 30, 30], 0.9, 1),                    # TP
            _d([11, 11, 31, 31], 0.8, 1),                    # duplicate (re-matches the same GT)
            _d([18, 18, 38, 38], 0.7, 1),                    # localization (IoU≈0.22)
            _d([10, 10, 30, 30], 0.6, 2),                    # class_confusion
            _d([70, 70, 90, 90], 0.5, 1),                    # background
            _d([70, 10, 90, 30], 0.5, 1, sources=["tile"]),  # tiling_artifact
        ],
        gts=[gt1])]
    result = metrics.classify_errors(per_image, {1: 0.4, 2: 0.4}, iou_thr=IOU)
    summary = result["fp_summary"]
    assert summary["duplicate"] == 1
    assert summary["localization"] == 1
    assert summary["class_confusion"] == 1
    assert summary["background"] == 1
    assert summary["tiling_artifact"] == 1


def test_iom_matrix_detects_containment():
    import numpy as np
    big = np.array([[0.0, 0.0, 100.0, 100.0]])
    small = np.array([[10.0, 10.0, 20.0, 20.0]])   # fully contained
    far = np.array([[200.0, 200.0, 210.0, 210.0]])
    assert metrics.iom_matrix(small, big)[0, 0] == pytest.approx(1.0)
    assert metrics.iou_matrix(small, big)[0, 0] == pytest.approx(0.01)  # IoU is the area ratio
    assert metrics.iom_matrix(small, far)[0, 0] == 0.0


def test_fp_nested_on_object_not_background():
    """A small box on a large GT (IoU<0.1, IoM 1.0) is not background —
    localization if same class, class_confusion if a different class."""
    per_image = [_rec(
        dets=[_d([0, 0, 100, 100], 0.9, 1),      # TP
              _d([40, 40, 55, 55], 0.8, 1),      # small, inside — IoU 0.023
              _d([40, 40, 55, 55], 0.7, 2)],     # small, inside, different class
        gts=[{"box": [0, 0, 100, 100], "category_id": 1}])]
    summary = metrics.classify_errors(per_image, {1: 0.4, 2: 0.4}, iou_thr=IOU)["fp_summary"]
    assert summary["localization"] == 1
    assert summary["class_confusion"] == 1
    assert summary["background"] == 0


def test_fn_classification_kinds():
    thresholds = {1: 0.5}
    # 3 GTs: one has a floor detection below threshold (below_threshold),
    # one has no detection and is small (missed_small), one has no detection (missed)
    gts = [{"box": [10, 10, 40, 40], "category_id": 1},   # 900px²
           {"box": [60, 60, 63, 63], "category_id": 1},   # 9px² (bottom quantile)
           {"box": [70, 10, 95, 35], "category_id": 1}]   # 625px²
    per_image = [_rec([], gts)]
    floor = [_rec([_d([10, 10, 40, 40], 0.2, 1)], gts)]   # detection below threshold
    result = metrics.classify_errors(per_image, thresholds, iou_thr=IOU,
                                     floor_detections=floor)
    summary = result["fn_summary"]
    assert summary["below_threshold"] == 1
    assert summary["missed_small"] == 1
    assert summary["missed"] == 1


def test_coco_map_perfect_predictions():
    per_image = [_rec([_d([10, 10, 30, 30], 0.95, 1)],
                      [{"box": [10, 10, 30, 30], "category_id": 1}])]
    result = metrics.coco_map(per_image, [1, 2], {1: "cat a", 2: "cat b"})
    assert result["map"] == pytest.approx(1.0, abs=0.01)


def test_match_greedy_prefers_higher_score():
    import numpy as np
    preds = np.array([[10, 10, 30, 30], [12, 12, 32, 32]], dtype=float)
    scores = np.array([0.5, 0.9])
    gts = np.array([[10, 10, 30, 30]], dtype=float)
    matches, fps, fns = metrics.match_greedy(preds, scores, gts, IOU)
    assert matches == [(1, 0)]  # the higher-scoring box takes the GT
    assert fps == [0] and fns == []


# ------------------------------------------------------------- F-beta

def test_prf_fbeta_hand_values():
    """Hand-computed pin at P=1, R=0.5: F1=2/3, F2=5/9, ... (β>1 = recall-weighted)."""
    from agentic_detector_config.core.metrics import prf

    base = prf(1, 0, 1)                      # P=1, R=0.5
    assert base["f_beta"] == pytest.approx(2 / 3)
    assert prf(1, 0, 1, beta=2.0)["f_beta"] == pytest.approx(5 * 0.5 / 4.5)   # 0.5556
    assert prf(1, 0, 1, beta=0.5)["f_beta"] == pytest.approx(1.25 * 0.5 / 0.75)  # 0.8333
    # β=1 is bit-identical to before
    assert prf(3, 2, 1, beta=1.0) == prf(3, 2, 1)


def test_threshold_estimation_follows_beta():
    """Recall-weighted (β=2) picks a low threshold (recovers recall); F1 picks a precise one."""
    # 10 GTs: thr 0.9 -> TP5/FP0 (P1, R.5) | thr 0.3 -> TP10/FP20 (P⅓, R1)
    gts = [{"box": [10 * i, 0, 10 * i + 8, 8], "category_id": 1} for i in range(10)]
    dets = [{"box": g["box"], "score": 0.9, "category_id": 1} for g in gts[:5]]
    dets += [{"box": g["box"], "score": 0.3, "category_id": 1} for g in gts[5:]]
    dets += [{"box": [500 + 9 * j, 0, 500 + 9 * j + 8, 8], "score": 0.3,
              "category_id": 1} for j in range(20)]
    recs = [{"detections": dets, "gts": gts, "width": 1000, "height": 100}]

    thr_f1, _ = metrics.estimate_f_beta_threshold(recs, 1, floor=0.03,
                                              max_candidates=50, iou_thr=0.5)
    thr_f2, _ = metrics.estimate_f_beta_threshold(recs, 1, floor=0.03,
                                              max_candidates=50, iou_thr=0.5,
                                              beta=2.0)
    assert thr_f1 == pytest.approx(0.9)      # F1: 0.667 > 0.5
    assert thr_f2 == pytest.approx(0.3)      # F2: 0.714 > 0.556


# --------------------------------------------------- τ selection: plateau and candidate grid
#
# Background: when reclassification makes scores bimodal, F(β) becomes nearly flat over a
# wide interval. τ could then land just above the noise cluster (margin ~0.002), so on unseen
# data the noise tail crossed it and FPs flooded through — a champion with holdout F(β) 0.98
# collapsed to 0.57. Moving τ to the middle of the same interval leaves holdout unchanged and
# restored it to 0.81.
#
# Two causes overlapped (the tests below pin the fixes):
#   (1) on ties/near-ties the **lowest** τ survived (recall-first rule).
#   (2) candidates were subsampled **rank-uniformly** (linspace over index). When unique scores
#       crowd the noise side, most of the 50 candidates fall in the noise interval and the
#       global optimum never enters the candidate set (e.g. all candidates: τ=0.0566/F=0.9886;
#       max_candidates=50: τ=0.0217/F=0.9831).


def test_threshold_unique_max_is_unchanged():
    """With a unique maximum, behavior must match the plain argmax — unchanged at the default tol."""
    per_image = [_rec(
        dets=[_d([10, 10, 30, 30], 0.9, 1),
              _d([50, 50, 70, 70], 0.6, 1),
              _d([0, 80, 10, 95], 0.3, 1)],
        gts=[{"box": [10, 10, 30, 30], "category_id": 1},
             {"box": [50, 50, 70, 70], "category_id": 1}])]
    thr, f1 = metrics.estimate_f_beta_threshold(per_image, 1, floor=0.03,
                                                max_candidates=50, iou_thr=IOU)
    assert thr == pytest.approx(0.6) and f1 == pytest.approx(1.0)


def _bimodal_records():
    """Of 21 GTs, 20 are hit at high scores and 1 at 0.20; noise FPs at 0.10–0.14.

    F(β) is 1.0 at τ=0.20 and 40/41≈0.9756 at τ=0.80 — a near-tie. There are no scores between
    the two candidates (0.20, 0.80): this empty gap is where τ can go, and holdout evidence
    cannot decide τ within it.
    """
    gts, dets = [], []
    for i in range(20):                                  # 20 high-score TPs
        box = [i * 5, 0, i * 5 + 4, 4]
        gts.append({"box": box, "category_id": 1})
        dets.append(_d(box, 0.80 + i * 0.005, 1))
    low = [0, 50, 4, 54]                                 # 1 low-score TP
    gts.append({"box": low, "category_id": 1})
    dets.append(_d(low, 0.20, 1))
    for i in range(5):                                   # 5 background FPs
        dets.append(_d([i * 6, 90, i * 6 + 4, 94], 0.10 + i * 0.01, 1))
    return [_rec(dets, gts, w=200, h=200)]


def test_threshold_picks_plateau_midpoint_under_tolerance():
    """On a near-tie (within tol), pick the **middle** of the empty gap — the only position biased
    toward neither failure. Flipping ties to "highest τ" merely moves the fragility to the other
    side (cutting real detections)."""
    recs = _bimodal_records()
    thr, _ = metrics.estimate_f_beta_threshold(recs, 1, floor=0.03,
                                               max_candidates=200, iou_thr=IOU,
                                               tol=0.03)
    assert 0.20 < thr < 0.80, f"must lie inside the empty gap (got {thr})"
    assert thr == pytest.approx(0.50, abs=0.02), "center of the widest gap"


def test_threshold_tolerance_zero_keeps_strict_argmax():
    """With tol=0 (default), near-ties are not treated as ties — compatible with existing runs."""
    recs = _bimodal_records()
    thr, f1 = metrics.estimate_f_beta_threshold(recs, 1, floor=0.03,
                                                max_candidates=200, iou_thr=IOU)
    assert thr == pytest.approx(0.20) and f1 == pytest.approx(1.0)


def test_threshold_candidates_are_not_rank_uniform():
    """With a rank-uniform candidate grid, the noise cluster eats the whole budget and the global
    optimum never enters the candidates. With 100 noise scores packed in [0.10,0.12] and the
    optimal τ in (0.25, 0.30], the optimum must be found even with a small budget."""
    gts, dets = [], []
    for i in range(20):
        box = [i * 5, 0, i * 5 + 4, 4]
        gts.append({"box": box, "category_id": 1})
        dets.append(_d(box, 0.80 + i * 0.005, 1))
    low = [0, 50, 4, 54]
    gts.append({"box": low, "category_id": 1})
    dets.append(_d(low, 0.30, 1))                        # the optimum keeps this one
    dets.append(_d([100, 90, 104, 94], 0.25, 1))         # this one must be cut
    for i in range(200):                                 # noise cluster
        dets.append(_d([(i % 40) * 4, 120 + (i // 40) * 6,
                        (i % 40) * 4 + 3, 123 + (i // 40) * 6],
                       0.10 + i * 0.0002, 1))
    recs = [_rec(dets, gts, w=300, h=300)]
    thr, f1 = metrics.estimate_f_beta_threshold(recs, 1, floor=0.03,
                                                max_candidates=20, iou_thr=IOU)
    # Rank-uniform subsampling would place τ inside the noise cluster (≈0.139) and stop at F=0.913.
    assert f1 == pytest.approx(1.0), f"missed the global optimum (F={f1}, τ={thr})"
    assert 0.25 < thr <= 0.30


def test_threshold_is_deterministic():
    """Same input, same τ — a precondition of the replay contract."""
    recs = _bimodal_records()
    a = metrics.estimate_f_beta_threshold(recs, 1, floor=0.03, max_candidates=37,
                                          iou_thr=IOU, tol=0.03)
    b = metrics.estimate_f_beta_threshold(recs, 1, floor=0.03, max_candidates=37,
                                          iou_thr=IOU, tol=0.03)
    assert a == b
