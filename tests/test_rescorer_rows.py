"""rescorer training row cap — `rescorer.max_train_rows`.

Even after the image subset, many detections per image can push the rows over
the cap. TPs are rare and the core of the ranking signal, so all of them stay;
FPs are thinned evenly over (class × score quartile) strata — easy low-score FPs
shrink a lot, FPs near the threshold mostly stay. Kept rows carry
inverse-probability weights (stratum's original rows ÷ kept) so the GBDT and the
isotonic calibration see the original distribution (TP rate, score
distribution). fit_dev rows are not thinned.
"""

import numpy as np
import pytest

from agentic_configuration_search.fitting import rescorer as rs_mod
from agentic_configuration_search.fitting.verifier import fit_isotonic


def _rows(n_tp=30, n_fp=970, cids=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tp + n_fp):
        rows.append({"det_id": f"d{i}", "image": f"im{i % 50:02d}.png",
                     "cid": cids[i % len(cids)], "y": i < n_tp,
                     "score": float(rng.uniform(0.03, 1.0))})
    return rows


def test_under_cap_is_untouched():
    rows = _rows()
    kept, w, info = rs_mod.cap_train_rows(rows, 0)
    assert kept == rows and np.all(w == 1.0) and info["capped"] is False
    kept, w, info = rs_mod.cap_train_rows(rows, 5000)
    assert kept == rows and info["capped"] is False


def test_keeps_every_positive_and_restores_stratum_mass():
    rows = _rows()
    kept, w, info = rs_mod.cap_train_rows(rows, 200)
    assert len(kept) == 200 and info["capped"] is True
    assert sum(r["y"] for r in kept) == 30                    # every TP
    # inverse-probability weights — per class, the weighted TP/FP sums equal the original counts
    for cid in (1, 2):
        for y in (True, False):
            orig = sum(1 for r in rows if r["cid"] == cid and r["y"] == y)
            got = sum(wi for r, wi in zip(kept, w) if r["cid"] == cid and r["y"] == y)
            assert got == pytest.approx(orig)
    assert info["n_rows"] == 1000 and info["n_kept"] == 200


def test_hard_negatives_are_kept_more_than_easy_ones():
    rows = _rows(n_tp=0, n_fp=4000, cids=(1,))
    kept, _, _ = rs_mod.cap_train_rows(rows, 400)
    s = np.array([r["score"] for r in rows])
    q = np.quantile(s, [0.25, 0.75])
    low = sum(1 for r in kept if r["score"] <= q[0])
    high = sum(1 for r in kept if r["score"] >= q[1])
    assert low == pytest.approx(high, abs=2)                  # the same share per stratum


def test_capping_is_deterministic_and_order_free():
    rows = _rows()
    a, wa, _ = rs_mod.cap_train_rows(rows, 300)
    b, wb, _ = rs_mod.cap_train_rows(list(reversed(rows)), 300)
    key = lambda pair: pair[0]["det_id"]                       # noqa: E731
    assert sorted(zip(a, wa), key=key) == sorted(zip(b, wb), key=key)


def test_positives_are_thinned_when_they_alone_exceed_half():
    rows = _rows(n_tp=600, n_fp=400)
    kept, w, _ = rs_mod.cap_train_rows(rows, 200)
    assert len(kept) == 200
    assert sum(wi for r, wi in zip(kept, w) if r["y"]) == pytest.approx(600)


def test_unused_negative_budget_goes_to_positives():
    # many TPs and few FPs leave part of the FP share unused — the TPs take those
    # slots (the cap is used in full)
    rows = _rows(n_tp=900, n_fp=40)
    kept, w, _ = rs_mod.cap_train_rows(rows, 200)
    assert len(kept) == 200
    assert sum(1 for r in kept if not r["y"]) == 40                  # every FP
    assert sum(wi for r, wi in zip(kept, w) if r["y"]) == pytest.approx(900)


def test_weighted_isotonic_equals_repeated_rows():
    s = np.array([0.1, 0.2, 0.2, 0.4, 0.6, 0.9])
    y = np.array([0, 1, 0, 0, 1, 1], dtype=float)
    w = np.array([3, 1, 2, 1, 2, 1], dtype=float)
    rep = np.repeat(np.arange(len(s)), w.astype(int))
    assert fit_isotonic(s, y, w) == fit_isotonic(s[rep], y[rep])


def test_fit_rescorer_records_the_cap(tmp_path):
    from tests.test_rescorer import _fit, _rs
    rescorer, info = _fit(tmp_path, _rs(min_samples=50, min_samples_per_class=1,
                                        margin=0.005, ensemble_members=2,
                                        max_train_rows=60))
    assert rescorer is not None
    assert info["n_train_raw"] == 96 and info["n_train"] == 60
    assert info["train_rows_capped"] is True


def test_config_validates_max_train_rows():
    from agentic_configuration_search.config import ConfigError, config_from_dict
    with pytest.raises(ConfigError):
        config_from_dict({"dataset": "x", "rescorer": {"max_train_rows": -1}})
    with pytest.raises(ConfigError):     # below the tripwire the fit would always be skipped
        config_from_dict({"dataset": "x", "rescorer": {"max_train_rows": 10,
                                                       "min_samples": 64}})
