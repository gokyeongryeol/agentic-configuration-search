"""verifier — isotonic monotonicity, feature contract, and training / pickle round-trip /
dimension contract of the gbdt engine (hand-crafted + embedding)."""

from types import SimpleNamespace

import numpy as np
import pytest

from agentic_configuration_search.core.metrics import rank_auc

class _FakeModel:
    """Discriminator stub providing only predict_proba — pins the calibration path with a hand-made P̂.

    Returns sigmoid(10·(score−0.5)) on the first column (score): a deterministic P̂ without training.
    """

    def predict_proba(self, X):
        import numpy as _np
        s = _np.asarray(X, dtype=float)[:, 0]
        p = 1.0 / (1.0 + _np.exp(-10.0 * (s - 0.5)))
        return _np.stack([1 - p, p], axis=1)


def _stub_verifier(features, emb_dim=4):
    return GbdtVerifier(features, emb_dim, _FakeModel(), isotonic=[])

from agentic_configuration_search.fitting.verifier import GbdtVerifier, detection_features, fit_isotonic


def test_isotonic_pools_ties_and_is_order_invariant():
    """Identical raw scores must be merged before PAV — otherwise one x gets several y values and
    np.interp picks the last one, calibrating a tied group with true rate 0.1 to 1.0
    (the inflated P̂ then pushes that group's scores up wholesale in rescorer blending).
    Ties are common with ensemble clip saturation, constant fills for degenerate folds, and identical GBDT leaves."""
    from agentic_configuration_search.fitting.verifier import _iso_apply

    raw = np.array([0.1] + [0.5] * 20 + [0.9])
    got = []
    for pos in ([0] * 18 + [1, 1], [1, 1] + [0] * 18, [1, 0, 1, 0] + [0] * 16):
        y = np.array([0.0] + [float(v) for v in pos] + [1.0])
        p = _iso_apply(fit_isotonic(raw, y), np.array([0.5]))[0]
        got.append(round(float(p), 6))
    assert got == [0.1, 0.1, 0.1]          # true rate = 2/20, order-independent


def test_isotonic_monotone_and_bounded():
    """isotonic (PAV) is a shared utility used by gbdt's internal CV calibration — pins monotonicity and range."""
    points = fit_isotonic(np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9]),
                          np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]))
    ys = [p[1] for p in points]
    assert ys == sorted(ys)
    assert all(0.0 <= v <= 1.0 for v in ys)


def _separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x_neg = rng.normal(-1.0, 0.5, size=(n // 2, 2))
    x_pos = rng.normal(+1.0, 0.5, size=(n // 2, 2))
    X = np.vstack([x_neg, x_pos])
    y = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
    return X, y












def test_detection_features_signals():
    stats = {1: {"count": 10, "mean": 400.0, "std": 50.0}}
    d = {"det_id": "a", "box": [10, 10, 30, 30], "score": 0.6, "category_id": 1,
         "config_id": 5, "sources": ["tile"]}
    other = {"det_id": "b", "box": [10, 10, 30, 30], "score": 0.5, "category_id": 2,
             "config_id": 5, "sources": ["full"]}
    feats = detection_features(d, [d, other], stats, tiled_config_ids={5})
    assert feats["score"] == 0.6
    assert feats["overlap_margin"] == pytest.approx(0.1)   # score gap to an overlapping other class
    assert feats["tile_agree"] == 0.0                      # tiled config + not detected on full
    assert feats["area_z"] == pytest.approx(0.0)           # area 400 == mean

    feats2 = detection_features(d, [d], stats, tiled_config_ids=set())
    assert feats2["overlap_margin"] == 1.0                 # no overlap -> +1
    assert feats2["tile_agree"] == 1.0


# ------------------------------------------------- gbdt engine

def _emb_separable(n=200, dim=16, seed=0):
    """Hand-crafted features are uninformative, signal only in embedding — verifies gbdt actually uses emb."""
    rng = np.random.default_rng(seed)
    y = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
    X_hand = rng.normal(0, 1, size=(n, 2))                 # no signal
    E = rng.normal(0, 0.5, size=(n, dim))
    E[:, 0] += np.where(y > 0, 2.0, -2.0)                  # signal only in dimension 1
    return X_hand, E, y


def test_gbdt_fit_predict_pickle_roundtrip():
    """The rescorer pickles this object wholesale into rescorer.pkl — round-trip equality."""
    import pickle

    X, E, y = _emb_separable()
    v = GbdtVerifier.fit(["f_beta", "f2"], X, E, y)
    p = v.predict_proba(np.hstack([X, E]))
    assert rank_auc(p, y.astype(bool)) > 0.95              # recovers the emb signal
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert v.features == ["f_beta", "f2"] and v.emb_dim == E.shape[1]

    v2 = pickle.loads(pickle.dumps(v))
    assert isinstance(v2, GbdtVerifier)
    assert np.allclose(v2.predict_proba(np.hstack([X, E])), p)


def test_gbdt_isotonic_calibration_from_inner_cv():
    X, E, y = _emb_separable()
    v = GbdtVerifier.fit(["f_beta", "f2"], X, E, y)
    ys = [pt[1] for pt in v.isotonic]
    assert ys == sorted(ys)                                # monotone non-decreasing
    # internal CV calibration, so the curve must not be the saturated (all 0/1) train probabilities
    assert len(set(ys)) > 1


def test_gbdt_dimension_mismatch_raises():
    X, E, y = _emb_separable()
    v = GbdtVerifier.fit(["f_beta", "f2"], X, E, y)
    with pytest.raises(ValueError, match="feature width mismatch"):
        v.predict_proba(np.zeros((1, 3)))




# ------------------------------------------- overlap_margin vectorization equivalence (regression)

def _margin_reference(det, kept_dets):
    """The pairwise-loop implementation from before vectorization — the semantic reference."""
    from agentic_configuration_search.core.metrics import iou_matrix
    from agentic_configuration_search.fitting.verifier import OVERLAP_IOU

    score = float(det["score"])
    margin = 1.0
    det_box = np.array([det["box"]], dtype=float)
    for other in kept_dets:
        if other is det or other["category_id"] == det["category_id"]:
            continue
        iou = iou_matrix(det_box, np.array([other["box"]], dtype=float))[0, 0]
        if iou >= OVERLAP_IOU:
            margin = min(margin, score - float(other["score"]))
    return float(margin)


def _mk(box, cid, score):
    return {"box": list(box), "category_id": cid, "score": score,
            "config_id": 0, "sources": ["full"]}


def test_overlap_margin_matches_reference_on_edge_cases():
    """Covers no overlap / same category / self / threshold boundary / duplicate objects."""
    a = _mk((0, 0, 10, 10), 1, 0.8)
    same_cat = _mk((0, 0, 10, 10), 1, 0.9)          # same category — must be excluded
    overlap = _mk((0, 0, 10, 10), 2, 0.5)           # IoU 1.0, other class -> margin 0.3
    apart = _mk((100, 100, 110, 110), 2, 0.1)       # no overlap -> no effect
    # a box with IoU exactly 0.7 (pins whether the boundary is inclusive)
    boundary = _mk((0, 0, 10, 7.0), 3, 0.2)
    dup = a                                          # the same object inserted twice

    cases = [
        [a],                                         # alone
        [a, same_cat],
        [a, overlap],
        [a, apart],
        [a, overlap, apart, same_cat],
        [a, boundary],
        [a, dup, overlap],
        [],                                          # empty list (det not included)
    ]
    for kept in cases:
        assert (detection_features(a, kept, {}, set())["overlap_margin"]
                == _margin_reference(a, kept)), kept


def test_overlap_margin_matches_reference_random():
    """Random dense layout — checks that the vectorized reduction (min) is order-independent."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 40))
        dets = []
        for _ in range(n):
            x, y = rng.uniform(0, 50, size=2)
            w, h = rng.uniform(5, 30, size=2)
            dets.append(_mk((x, y, x + w, y + h), int(rng.integers(1, 4)),
                            float(rng.uniform(0.03, 1.0))))
        for d in dets:
            got = detection_features(d, dets, {}, set())["overlap_margin"]
            assert got == _margin_reference(d, dets), (n, d)
