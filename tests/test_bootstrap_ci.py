"""metrics.bootstrap_ci / rank_auc — the decision tools behind every A/B gate."""

import math

import pytest

from agentic_configuration_search.core.metrics import bootstrap_ci, rank_auc, records_micro_f_beta


def _rec(dets, gts):
    return {"file": "x", "width": 100, "height": 100,
            "detections": [{"box": b, "score": s, "category_id": c} for b, s, c in dets],
            "gts": [{"box": b, "category_id": c} for b, c in gts]}


GT = ([10, 10, 30, 30], 1)
HIT = ([10, 10, 30, 30], 0.9, 1)
MISS = ([60, 60, 80, 80], 0.9, 1)


def _stat(records):
    return records_micro_f_beta(records, [1], iou_thr=0.5)


def test_identical_records_delta_zero_ci_contains_zero():
    recs = [_rec([HIT], [GT]) for _ in range(10)]
    out = bootstrap_ci(recs, recs, _stat, n_boot=200, seed=1)
    assert out["delta"] == 0.0
    assert out["lo"] <= 0.0 <= out["hi"]
    assert out["n_images"] == 10


def test_clear_improvement_ci_above_zero():
    a = [_rec([MISS], [GT]) for _ in range(20)]   # all FP+FN
    b = [_rec([HIT], [GT]) for _ in range(20)]    # all TP
    out = bootstrap_ci(a, b, _stat, n_boot=200, seed=1)
    assert out["delta"] == pytest.approx(1.0)
    assert out["lo"] > 0.0


def test_paired_requires_same_length():
    recs = [_rec([HIT], [GT])]
    with pytest.raises(ValueError, match="records length mismatch"):
        bootstrap_ci(recs, recs * 2, _stat)


def test_deterministic_with_seed():
    a = [_rec([MISS], [GT]), _rec([HIT], [GT])] * 5
    b = [_rec([HIT], [GT]), _rec([HIT], [GT])] * 5
    o1 = bootstrap_ci(a, b, _stat, n_boot=100, seed=7)
    o2 = bootstrap_ci(a, b, _stat, n_boot=100, seed=7)
    assert o1 == o2


def test_rank_auc_perfect_and_random_and_ties():
    assert rank_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert rank_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert rank_auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == pytest.approx(0.5)
    assert math.isnan(rank_auc([0.5, 0.6], [True, True]))  # one class absent


def test_one_sided_tail_is_less_conservative():
    """The gate asks a one-sided question (lower bound ≥ margin) — a two-sided 2.5% cut halves
    the effective significance level and is over-conservative. tail="lower" gives the alpha percentile."""
    import numpy as np

    from agentic_configuration_search.core.metrics import bootstrap_ci

    def stat(recs):
        return float(np.mean([r["v"] for r in recs]))

    rng = np.random.default_rng(0)
    a = [{"v": float(x)} for x in rng.normal(0.0, 1.0, 200)]
    b = [{"v": float(x) + 0.15} for x in rng.normal(0.0, 1.0, 200)]
    two = bootstrap_ci(a, b, stat, n_boot=500, seed=1, alpha=0.05)
    one = bootstrap_ci(a, b, stat, n_boot=500, seed=1, alpha=0.05, tail="lower")
    assert one["lo"] > two["lo"]           # one-sided is less conservative
    assert two["delta"] == one["delta"]    # same point estimate
    assert one["hi"] is None               # no upper bound in one-sided mode


def test_gate_uses_one_sided_bound():
    from agentic_configuration_search.eval.evaluate import gate
    import inspect
    src = inspect.getsource(gate)
    assert 'tail="lower"' in src, "the gate does not use the one-sided lower bound"
