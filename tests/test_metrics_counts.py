"""Count reuse in the gate bootstrap and threshold search — results are **exactly**
the same as the previous computation.

The gate bootstrap re-matched every image and class on every resample (2000 by
default). Per-image TP/FP/FN do not depend on the resample, so they are counted
once and only the integer sums are redone (the screen's fit_dev comparison
already does this). The threshold search matches once per class, using the
prefix property of score-ordered greedy matching (a higher-scored match does
not depend on lower-scored detections).
"""

import bisect

import numpy as np
import pytest

from agentic_configuration_search.core import metrics
from agentic_configuration_search.eval.evaluate import gate, micro_f_beta_stat


def _records(rng, n_images, cats, *, tie_decimals=None):
    recs = []
    for _ in range(n_images):
        gts, dets = [], []
        for _ in range(int(rng.integers(0, 6))):
            x, y = rng.uniform(0, 80, 2)
            w, h = rng.uniform(5, 30, 2)
            gts.append({"box": [x, y, x + w, y + h], "category_id": int(rng.choice(cats))})
        for g in gts:                                   # hitting detections (jittered) and false ones
            if rng.random() < 0.7:
                j = rng.normal(0, 3, 4)
                dets.append({"box": list(np.array(g["box"]) + j),
                             "category_id": g["category_id"] if rng.random() < 0.85
                             else int(rng.choice(cats)),
                             "score": float(rng.uniform(0.2, 1.0))})
        for _ in range(int(rng.integers(0, 5))):
            x, y = rng.uniform(0, 80, 2)
            dets.append({"box": [x, y, x + 15, y + 15], "category_id": int(rng.choice(cats)),
                         "score": float(rng.uniform(0.0, 0.8))})
        if tie_decimals is not None:                    # create score ties on purpose
            for d in dets:
                d["score"] = round(d["score"], tie_decimals)
        recs.append({"detections": dets, "gts": gts})
    return recs


def _old_gate_ci(a, b, cats, iou, beta, n_boot, seed, alpha):
    """The gate before the change — re-matches the records on every resample."""
    return metrics.bootstrap_ci(a, b, micro_f_beta_stat(cats, iou, beta), n_boot=n_boot,
                                seed=seed, alpha=alpha, tail="lower")


@pytest.mark.parametrize("seed", range(6))
def test_gate_equals_rematching_bootstrap(seed):
    rng = np.random.default_rng(seed)
    cats = [1, 2, 3]
    a = _records(rng, 25, cats, tie_decimals=1 if seed % 2 else None)
    b = _records(np.random.default_rng(seed + 100), 25, cats)
    for beta in (1.0, 2.0):
        old = _old_gate_ci(a, b, cats, 0.5, beta, 300, seed, 0.1)
        new = gate(a, b, cat_ids=cats, iou_thr=0.5, margin=0.0, n_boot=300, seed=seed,
                   alpha=0.1, beta=beta)
        assert (new["delta"], new["lo"], new["hi"], new["n_images"]) == \
            (old["delta"], old["lo"], old["hi"], old["n_images"])


def test_gate_with_identical_records_and_empty_images():
    rng = np.random.default_rng(7)
    a = _records(rng, 10, [1]) + [{"detections": [], "gts": []}] * 3
    new = gate(a, a, cat_ids=[1], iou_thr=0.5, margin=0.0, n_boot=200, seed=1, alpha=0.1)
    assert new["delta"] == 0.0 and new["lo"] == 0.0 and new["adopted"] is False
    old = _old_gate_ci(a, a, [1], 0.5, 1.0, 200, 1, 0.1)
    assert new["lo"] == old["lo"]


def test_image_counts_are_micro_totals():
    rng = np.random.default_rng(3)
    recs = _records(rng, 15, [1, 2])
    rows = metrics.image_counts(recs, [1, 2], 0.5)
    assert rows.shape == (15, 3) and rows.dtype.kind == "i"
    for rec, row in zip(recs, rows):
        p = [metrics.class_prf_at([rec], c, 0.0, 0.5) for c in (1, 2)]
        assert list(row) == [sum(x["tp"] for x in p), sum(x["fp"] for x in p),
                             sum(x["fn"] for x in p)]
    assert metrics.micro_f_counts(rows) == metrics.records_micro_f_beta(recs, [1, 2], 0.5)
    assert metrics.micro_f_counts(rows[:0]) == 0.0


def _old_estimate(per_image, category_id, *, floor, max_candidates, iou_thr, beta=1.0,
                  tol=0.0):
    """The threshold search before the change — re-matches everything per candidate threshold."""
    scores = sorted({float(d["score"]) for rec in per_image for d in rec["detections"]
                     if d["category_id"] == category_id})
    if not scores:
        return float(floor), 0.0
    cand = metrics._threshold_candidates(scores, max_candidates)
    evaluated = []
    for thr in cand:
        p = metrics.class_prf_at(per_image, category_id, thr, iou_thr, beta)
        evaluated.append((float(thr), p["f_beta"]))
    best_f1 = max(f for _, f in evaluated)
    if tol <= 0.0:
        best_thr = next(thr for thr, f in evaluated if f == best_f1)
    else:
        plateau = [thr for thr, f in evaluated if f >= best_f1 - tol]
        best_thr, best_f1 = metrics._max_margin_threshold(plateau, scores, floor,
                                                          dict(evaluated), best_f1)
    return best_thr, max(best_f1, 0.0)


@pytest.mark.parametrize("seed", range(8))
def test_threshold_search_equals_rematching(seed):
    rng = np.random.default_rng(seed)
    recs = _records(rng, 30, [1, 2], tie_decimals=1 if seed % 2 else 2)
    for cid in (1, 2, 9):                               # 9: a class with no detections
        for max_c, tol, beta in ((50, 0.0, 1.0), (5, 0.0, 1.0), (50, 0.02, 2.0),
                                 (4, 0.05, 0.5)):
            kw = dict(floor=0.05, max_candidates=max_c, iou_thr=0.5, beta=beta, tol=tol)
            assert metrics.estimate_f_beta_threshold(recs, cid, **kw) == \
                _old_estimate(recs, cid, **kw)


def test_duplicate_category_ids_count_once_like_before():
    """records_micro_f_beta groups by class id (dict), counting a duplicate id once —
    the count rows must do the same so the gate result does not change."""
    rng = np.random.default_rng(11)
    a, b = _records(rng, 20, [1, 2]), _records(np.random.default_rng(12), 20, [1, 2])
    old = _old_gate_ci(a, b, [1, 1, 2], 0.5, 1.0, 200, 3, 0.1)
    new = gate(a, b, cat_ids=[1, 1, 2], iou_thr=0.5, margin=0.0, n_boot=200, seed=3,
               alpha=0.1)
    assert (new["delta"], new["lo"]) == (old["delta"], old["lo"])
    assert (metrics.image_counts(a, [1, 1, 2], 0.5) == metrics.image_counts(a, [1, 2], 0.5)).all()
