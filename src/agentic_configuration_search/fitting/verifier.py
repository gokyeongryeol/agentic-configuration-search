"""verifier: calibrated P(TP|features) — an internal component of the rescorer.

- `GbdtVerifier` — the discriminator that produces the rescorer's P̂
- `detection_features` — feature extraction shared by training and P̂ computation
The rescorer's calibrated score comes from this P̂, so decision quality depends on this module.

Model: **GbdtVerifier** — HistGradientBoosting (hand-crafted features + a
per-detection embedding, i.e. clients.yoloe's per-anchor visual embedding) +
internal-CV isotonic calibration. The class score is a scalar projection of the
embedding onto the prompt direction, discarding context and geometry; the
nonlinear model recovers those dimensions. Requires sklearn (BSD-3, a declared
dependency); without embeddings (emb_dim=0) it fits on hand-crafted features only.

After calibration, P̂ reads as "the probability this detection is a TP".
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.metrics import iou_matrix

log = logging.getLogger(__name__)

# Feature order is a save/load contract.
FEATURES = ["score", "overlap_margin", "area_z", "tile_agree"]
OVERLAP_IOU = 0.7  # overlap criterion for overlap_margin


def detection_features(det: dict, context_dets: list[dict], area_stats: dict,
                       tiled_config_ids: set) -> dict:
    """Training and P̂ computation extract features with **the same function** — if
    they diverged, train and inference distributions would differ and calibration
    would be meaningless.

    context_dets: detections of the same image forming the feature context
    (including det itself) — the production caller (rescorer) passes **all
    candidates above floor**.
    """
    score = float(det["score"])

    # Minimum score gap to overlapping detections of other classes.
    # +1.0 if nothing overlaps (no ambiguity).
    #
    # The caller invokes this per detection, so calling iou_matrix per pair would
    # mean n² calls per image — candidates are stacked and computed in **one**
    # call (elementwise, so per-pair values are bit-identical).
    margin = 1.0
    others = [o for o in context_dets
              if o is not det and o["category_id"] != det["category_id"]]
    if others:
        ious = iou_matrix(np.array([det["box"]], dtype=float),
                          np.array([o["box"] for o in others], dtype=float))[0]
        hit = ious >= OVERLAP_IOU
        if hit.any():
            diffs = score - np.array([float(o["score"]) for o in others])[hit]
            margin = min(margin, float(diffs.min()))

    # |z-score| of the box area against the class's GT area distribution
    st = area_stats.get(det["category_id"], {"count": 0})
    if st.get("count", 0) >= 2 and st.get("std", 0.0) > 0:
        x1, y1, x2, y2 = det["box"]
        area_z = abs(((x2 - x1) * (y2 - y1) - st["mean"]) / st["std"])
    else:
        area_z = 0.0

    # 0 if the config is tiled and the full pass did not see this detection (tile-only)
    tile_agree = 1.0
    if det.get("config_id") in tiled_config_ids and "full" not in det.get("sources", ["full"]):
        tile_agree = 0.0

    return {"score": score, "overlap_margin": float(margin),
            "area_z": float(area_z), "tile_agree": tile_agree}




def fit_isotonic(scores: np.ndarray, y: np.ndarray,
                 weights: np.ndarray | None = None) -> list[list[float]]:
    """PAV (pool adjacent violators) — list of (score, calibrated_p) breakpoints.

    Corrects, via a monotone map, raw classifier probabilities that disagree
    with the observed TP rate. Prediction interpolates linearly between
    breakpoints (np.interp) and clamps to the end values outside the range.

    **Tied scores are merged before PAV.** If rows with the same x fall into
    different blocks, one x gets several y values and `np.interp` picks the
    last, so a tied group's calibration depends on row order. Ties are common
    (clip saturation, constant fill for degenerate folds, identical GBDT leaves).

    With `weights` (row weights) it is weighted PAV — an integer weight equals
    repeating that row as many times (the rescorer row cap's inverse-probability
    weights restore the original distribution).
    """
    s_all = np.asarray(scores, dtype=float)
    y_all = np.asarray(y, dtype=float)
    if s_all.size == 0:
        return []
    w_all = (np.ones_like(y_all) if weights is None
             else np.asarray(weights, dtype=float))
    order = np.argsort(s_all, kind="mergesort")
    s = s_all[order]
    t = y_all[order]
    w = w_all[order]
    xs, starts = np.unique(s, return_index=True)      # ascending unique values + first positions
    sums = np.add.reduceat(t * w, starts)
    cnts = np.add.reduceat(w, starts)
    # block: (x_lo, x_hi, sum, count) — merge with the previous block on monotonicity violation
    blocks: list[list[float]] = []
    for x, bsum, bcnt in zip(xs, sums, cnts):
        blocks.append([float(x), float(x), float(bsum), float(bcnt)])
        while len(blocks) > 1 and \
                blocks[-2][2] / blocks[-2][3] >= blocks[-1][2] / blocks[-1][3]:
            lo, hi, s2, c2 = blocks.pop()
            blocks[-1][1] = hi
            blocks[-1][2] += s2
            blocks[-1][3] += c2
    points: list[list[float]] = []
    for x_lo, x_hi, bsum, bcnt in blocks:
        p = bsum / bcnt
        points.append([x_lo, float(p)])
        points.append([x_hi, float(p)])
    return points




GBDT_SEED = 0


def fit_members(X: np.ndarray, y: np.ndarray, members: int,
                sample_weight: np.ndarray | None = None) -> list:
    """Members varying seed and an 80% subsample. members=1 is a single model on all data.
    `sample_weight` is the row weight (the rescorer row cap's inverse-probability
    weights) — None means uniform."""
    def w(idx):
        return None if sample_weight is None else np.asarray(sample_weight)[idx]

    from sklearn.ensemble import HistGradientBoostingClassifier

    def make(seed):
        return HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=15, random_state=seed)

    if members <= 1:
        return [make(GBDT_SEED).fit(X, y.astype(bool), sample_weight=w(slice(None)))]
    out = []
    for m in range(members):
        rng = np.random.default_rng(GBDT_SEED + m)
        sub = rng.choice(len(y), size=max(2, int(0.8 * len(y))), replace=False)
        if len(set(y[sub].tolist())) < 2:      # use all data if the subsample is single-class
            sub = np.arange(len(y))
        out.append(make(GBDT_SEED + m).fit(X[sub], y[sub].astype(bool),
                                           sample_weight=w(sub)))
    return out


def ensemble_raw(models: list, X: np.ndarray) -> np.ndarray:
    """**Logit mean** of member probabilities — the proper combiner for cross-entropy (Wood, JMLR 2023).

    Averaging probabilities overweights extremes and smears the pre-isotonic distribution.
    """
    if len(models) == 1:
        return models[0].predict_proba(X)[:, 1]
    zs = []
    for m in models:
        p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        zs.append(np.log(p / (1 - p)))
    return 1.0 / (1.0 + np.exp(-np.mean(zs, axis=0)))


def _iso_apply(isotonic: list[list[float]], raw: np.ndarray) -> np.ndarray:
    if not isotonic:
        return raw
    xs = np.array([p[0] for p in isotonic])
    ys = np.array([p[1] for p in isotonic])
    return np.interp(raw, xs, ys)


class GbdtVerifier:
    """GBDT (hand-crafted features + per-detection embedding) + internal-CV isotonic calibration.

    `features` holds **hand-crafted feature names only** — by contract the
    embedding columns follow the hand-crafted ones unnamed, `emb_dim` wide
    (possibly 0). The sole consumer is the rescorer, which passes the
    (hand-crafted + embedding) matrix to `predict_proba` to get P̂.

    Calibration: fitting isotonic directly on the GBDT's uncalibrated train
    probabilities is meaningless, as they saturate to 0/1. To calibrate without
    losing training data, isotonic is fit on **internal K-fold OOF
    probabilities** (as in sklearn CalibratedClassifierCV(ensemble=False)), and
    the final model is refit on everything. On small samples a held-out
    calibration split hurts AUC through isotonic staircasing.

    Ensemble (`members`>1): uses the **logit mean** of members varying seed and
    an 80% subsample (averaging probabilities overweights extremes and smears
    the pre-calibration distribution).

    Serialization: the rescorer pickles this whole object (rescorer.MODEL_FILE) —
    pickle is used only to exchange artifacts inside a run directory.
    """

    def __init__(self, features: list[str], emb_dim: int, models,
                 isotonic: list[list[float]], meta: dict | None = None):
        self.features = list(features)
        self.emb_dim = int(emb_dim)
        # also accepts a single model (members=1, test stubs)
        self.models = list(models) if isinstance(models, (list, tuple)) else [models]
        self.isotonic = [list(map(float, p)) for p in isotonic]
        self.meta = meta or {}

    @property
    def members(self) -> int:
        return len(self.models)

    @classmethod
    def fit(cls, features: list[str], X_hand: np.ndarray, E: np.ndarray,
            y: np.ndarray, *, inner_folds: int = 5,
            members: int = 1, sample_weight: np.ndarray | None = None) -> "GbdtVerifier":
        """X_hand: (n, len(features)) hand-crafted features, E: (n, emb_dim) embeddings.
        `sample_weight`: row weights — used by both the GBDT and the isotonic
        calibration (None means uniform)."""
        X = np.hstack([np.asarray(X_hand, dtype=float), np.asarray(E, dtype=float)])
        y = np.asarray(y, dtype=float)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
        rng = np.random.default_rng(GBDT_SEED)
        order = rng.permutation(len(y))
        oof = np.zeros(len(y))
        for te in np.array_split(order, inner_folds):
            tr = np.setdiff1d(order, te)
            if len(set(y[tr].tolist())) < 2:
                oof[te] = float(np.average(y[tr], weights=None if sample_weight is None
                                           else sample_weight[tr]))
                continue
            fold = fit_members(X[tr], y[tr], members,
                               None if sample_weight is None else sample_weight[tr])
            oof[te] = ensemble_raw(fold, X[te])
        iso = fit_isotonic(oof, y, sample_weight)
        models = fit_members(X, y, members, sample_weight)
        return cls(features, E.shape[1], models, iso,
                   meta={"n": int(len(y)), "pos": int(y.sum()), "members": len(models)})

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (n, len(features)+emb_dim) — hand-crafted columns followed by embeddings."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        want = len(self.features) + self.emb_dim
        if X.shape[1] != want:
            raise ValueError(f"feature width mismatch: {X.shape[1]} vs {want} "
                             f"({len(self.features)} hand-crafted + emb {self.emb_dim})")
        return _iso_apply(self.isotonic, ensemble_raw(self.models, X))

