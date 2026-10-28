"""rescorer — recalibrates detection scores with a discriminator.

Moves detection scores **in both directions** — raising the rank of true objects
below τ_c buys recall; lowering false positives above it buys precision.

Form (logit-space blending):

    s' = sigmoid( logit(s) + α_c · logit(P̂) )

- `P̂`: TP probability from the discriminator (hand-crafted features + YOLOE
  per-anchor visual embedding, 5-seed GBDT ensemble + internal-CV isotonic).
- `α_c`: **per-class** mixing coefficient. Degradation is local to classes, so a
  single dataset-wide α cannot guarantee no class degrades. The grid includes 0
  and α>0 requires passing a **non-degradation margin test**, so classes
  without proven gain stay exactly as-is.
- The score range [0,1] is preserved — downstream contracts (τ_c etc.) are unchanged.

**Not monotone**: P̂ is a function of the embedding and features, so within-class
ranking changes. This is the only path that moves the PR curve (a monotone
transform leaves F(β) unchanged once τ_c is re-estimated) and is what the recall
channel relies on.

**Dataset fingerprint**: the artifact carries a fingerprint of the training
dataset and ontology; a mismatch on load is an explicit error. The
discriminator is trained per dataset, so applying it to another dataset as-is
can make within-class ranking worse than the score alone.

Client-side — applied to above-floor candidates after the detector's
postprocessing (NMS/top-k).

**Discriminator fit on fit_train, α grid on fit_dev** — fit_dev is held out, so
P̂ is measured on it directly (fit_rescorer). oof_phat is kept for diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
# load sklearn (OpenMP) first — threadpool_limits changes only the OpenMP already
# loaded, so if it were first imported inside the discriminator-fit block the
# first fit would not get rescorer.fit_threads
import sklearn.ensemble  # noqa: F401
from threadpoolctl import threadpool_limits

from ..core import metrics
from ..core.metrics import iou_matrix, per_image_records
from .verifier import FEATURES as VERIFIER_FEATURES
from .verifier import detection_features
from .verifier import (GbdtVerifier, _iso_apply, ensemble_raw,
                       fit_isotonic, fit_members)

log = logging.getLogger(__name__)

EPS = 1e-6
MODEL_FILE = "rescorer.pkl"


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -30, 30)))


def blend(score, phat, alpha: float) -> np.ndarray:
    """s' = sigmoid(logit(s) + α·logit(P̂)). α=0 returns s unchanged."""
    if alpha == 0.0:
        return np.asarray(score, dtype=float)
    return sigmoid(logit(score) + float(alpha) * logit(phat))


def dataset_fingerprint(dataset: str, ontology) -> dict:
    """Fingerprint identifying which dataset/ontology an artifact came from.

    The ontology hash is built from the sorted (category_id, name) list — adding,
    removing, or renaming a class makes per-class α keys meaningless, so that
    must also register as a mismatch.
    """
    payload = "\n".join(f"{c.id}:{c.name}" for c in ontology.classes)
    return {"dataset": str(dataset),
            "ontology_sha256": hashlib.sha256(payload.encode()).hexdigest()[:16],
            "n_classes": len(ontology.classes)}


class Rescorer:
    """Discriminator + per-class α. A small per-dataset artifact that can be attached and detached."""

    def __init__(self, discriminator: GbdtVerifier, alphas: dict,
                 meta: dict | None = None, fingerprint: dict | None = None):
        self.discriminator = discriminator
        self.alphas = {int(k): float(v) for k, v in dict(alphas).items()}
        self.meta = meta or {}
        self.fingerprint = fingerprint or {}

    @property
    def active(self) -> bool:
        """Whether any class has α>0 — if all are 0, applying it is the identity."""
        return any(a != 0.0 for a in self.alphas.values())

    def alpha_for(self, cid: int) -> float:
        """Classes unseen in training get 0 (identity) — no guessing."""
        return self.alphas.get(int(cid), 0.0)

    # ------------------------------------------------------------- apply
    def phat(self, feats_list: list[dict], embeddings: list) -> np.ndarray:
        """Per-detection P̂. feats_list and embeddings must be in corresponding order."""
        if not feats_list:
            return np.zeros(0)
        names = self.discriminator.features
        X = np.array([[f[n] for n in names] for f in feats_list], dtype=float)
        if self.discriminator.emb_dim == 0:
            E = np.zeros((len(feats_list), 0), dtype=float)
        else:
            E = np.array([np.asarray(e, dtype=float).ravel() for e in embeddings],
                         dtype=float)
        if E.shape[1] != self.discriminator.emb_dim:
            raise ValueError(f"embedding dimension mismatch: {E.shape[1]} vs "
                             f"{self.discriminator.emb_dim}")
        return self.discriminator.predict_proba(np.hstack([X, E]))

    def rescore(self, feats_list: list[dict], embeddings: list,
                scores: list[float], category_ids: list[int]) -> np.ndarray:
        """Raw scores -> recalibrated scores. Classes with α_c=0 pass through unchanged.

        α is per class, so category_ids is required — same length as the rest.
        """
        scores = np.asarray(scores, dtype=float)
        if len(category_ids) != len(scores):
            raise ValueError(f"category_ids length mismatch: {len(category_ids)} vs "
                             f"{len(scores)}")
        if not self.active:
            return scores
        alphas = np.array([self.alpha_for(c) for c in category_ids], dtype=float)
        out = scores.copy()
        touch = alphas != 0.0
        if not touch.any():
            return out
        # compute P̂ only for rows with α>0 — the rest keep their input bit-exact
        idx = np.flatnonzero(touch)
        p = self.phat([feats_list[i] for i in idx], [embeddings[i] for i in idx])
        out[idx] = sigmoid(logit(scores[idx]) + alphas[idx] * logit(p))
        return out

    def rescore_from_phat(self, phat, scores: list[float],
                          category_ids: list[int]) -> np.ndarray:
        """Recalibrate from a precomputed P̂ — bit-exact with `rescore`.

        Avoids running the discriminator twice when the caller also uses P̂
        **for something else** (e.g. oof_phat diagnostics). The discriminator is
        row-independent (GBDT decision function -> logit mean -> isotonic
        interpolation), so a subset of P̂ computed over all rows equals P̂
        computed on that subset alone.
        """
        scores = np.asarray(scores, dtype=float)
        if len(category_ids) != len(scores):
            raise ValueError(f"category_ids length mismatch: {len(category_ids)} vs "
                             f"{len(scores)}")
        phat = np.asarray(phat, dtype=float)
        if len(phat) != len(scores):
            raise ValueError(f"phat length mismatch: {len(phat)} vs {len(scores)}")
        out = scores.copy()
        alphas = np.array([self.alpha_for(c) for c in category_ids], dtype=float)
        touch = alphas != 0.0
        if touch.any():
            out[touch] = sigmoid(logit(scores[touch])
                                 + alphas[touch] * logit(phat[touch]))
        return out

    # ------------------------------------------------------------- serialization
    def save(self, path: Path) -> None:
        path = Path(path)
        model_path = path.with_name(MODEL_FILE)
        model_path.write_bytes(pickle.dumps(self.discriminator))
        path.write_text(json.dumps({
            "alphas": {str(k): v for k, v in sorted(self.alphas.items())},
            "features": self.discriminator.features,
            "emb_dim": self.discriminator.emb_dim,
            "model_file": model_path.name,
            "fingerprint": self.fingerprint,
            "meta": self.meta,
        }, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path, expect: dict | None = None) -> "Rescorer":
        """If `expect` (a dataset_fingerprint result) is given, a mismatch is rejected with an explicit error."""
        path = Path(path)
        d = json.loads(path.read_text())
        if "alphas" not in d:
            raise ValueError(
                f"{path} has no per-class α (alphas) — re-fit the rescorer by "
                "rerunning the loop")
        got = d.get("fingerprint") or {}
        if expect is not None and got != expect:
            raise ValueError(
                f"{path} has a different dataset fingerprint: {got} vs {expect}. The rescorer is "
                "fit per dataset; applied to another dataset, its within-class "
                "ranking is worse than the raw score alone")
        with open(path.parent / d["model_file"], "rb") as fh:
            disc = pickle.load(fh)
        return cls(disc, d["alphas"], d.get("meta"), got)


def fit_discriminator(feature_names: list[str], X_hand: np.ndarray,
                      E: np.ndarray, y: np.ndarray, *, members: int = 5,
                      sample_weight: np.ndarray | None = None) -> GbdtVerifier:
    """Train the discriminator — thin wrapper over GbdtVerifier.fit (produces the rescorer's P̂)."""
    return GbdtVerifier.fit(feature_names, X_hand, E, y, members=members,
                            sample_weight=sample_weight)


SCORE_BINS = 4      # FP strata = (class × score quantile bin)


def _row_hash(r: dict) -> str:
    return hashlib.sha256(f"{r['image']}|{r['det_id']}".encode()).hexdigest()


def _proportional(sizes: list[int], budget: int) -> list[int]:
    """Split budget in proportion to sizes (largest remainder, deterministic) —
    each share is at most its size."""
    total = sum(sizes)
    if total <= budget:
        return list(sizes)
    raw = [budget * s / total for s in sizes]
    out = [int(x) for x in raw]
    for i in sorted(range(len(sizes)), key=lambda i: (out[i] - raw[i], i))[
            :budget - sum(out)]:
        out[i] += 1
    return out


def _even(sizes: list[int], budget: int) -> list[int]:
    """Split budget evenly — what a small bin cannot fill is shared by the rest."""
    out = [0] * len(sizes)
    left, open_ = budget, [i for i, s in enumerate(sizes) if s > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for i in list(open_):
            take = min(share, sizes[i] - out[i], left)
            out[i] += take
            left -= take
            if out[i] >= sizes[i]:
                open_.remove(i)
            if left == 0:
                break
    return out


def cap_train_rows(rows: list[dict], cap: int | None):
    """Thin the discriminator's training rows to cap — (kept rows, row weights, record).

    TPs are rare and the core of the ranking signal, so they are kept first — if
    the TPs alone exceed half of cap they are thinned in class proportion, but slots
    the FPs cannot fill (cap − the TP share) go to the TPs — rows are always at
    most cap, and when thinning, cap is used in full. FPs get a share per class
    proportional to the class's FP count, split evenly across score quartile
    bins — easy low-score FPs shrink a lot while FPs near the threshold mostly
    stay. Within a stratum, rows are taken in (image, det_id) hash order
    (deterministic, independent of input order). A kept row's weight is the
    stratum's original row count ÷ the number kept, so the GBDT and the
    isotonic calibration see the original distribution (TP rate, score
    distribution). Kept rows keep their original order.
    """
    n = len(rows)
    if cap is not None and cap < 1:
        raise ValueError(f"cap {cap} — None removes the cap (it is not 0 rows)")
    if cap is None or n <= cap:
        return rows, np.ones(n), {"capped": False, "n_rows": n, "n_kept": n}
    pos = [i for i, r in enumerate(rows) if r["y"]]
    neg = [i for i, r in enumerate(rows) if not r["y"]]
    strata: list[list[int]] = []
    budgets: list[int] = []

    def by_class(idx):
        out: dict = {}
        for i in idx:
            out.setdefault(rows[i]["cid"], []).append(i)
        return [out[c] for c in sorted(out)]

    pos_groups = by_class(pos)
    pos_take = _proportional([len(g) for g in pos_groups],
                             min(len(pos), max(cap // 2, cap - len(neg))))
    strata += pos_groups
    budgets += pos_take
    neg_groups = by_class(neg)
    neg_take = _proportional([len(g) for g in neg_groups], cap - sum(pos_take))
    for g, take in zip(neg_groups, neg_take):
        s = np.array([rows[i]["score"] for i in g])
        edges = np.quantile(s, np.linspace(0, 1, SCORE_BINS + 1)[1:-1])
        b = np.searchsorted(edges, s, side="right")
        bins = [[g[j] for j in np.flatnonzero(b == k)] for k in range(SCORE_BINS)]
        strata += bins
        budgets += _even([len(x) for x in bins], take)
    weight = np.zeros(n)
    for idx, k in zip(strata, budgets):
        if k <= 0:
            continue
        for i in sorted(idx, key=lambda i: _row_hash(rows[i]))[:k]:
            weight[i] = len(idx) / k
    keep = np.flatnonzero(weight > 0)
    return ([rows[i] for i in keep], weight[keep],
            {"capped": True, "n_rows": n, "n_kept": int(len(keep)),
             "n_pos": len(pos), "n_pos_kept": int(sum(pos_take))})


def oof_phat(feature_names: list[str], X_hand: np.ndarray, E: np.ndarray,
             y: np.ndarray, *, folds: int = 5, seed: int = 0,
             members: int = 5, groups: Optional[list] = None) -> np.ndarray:
    """Out-of-fold P̂ for α selection — keeps discriminator overfitting out of α selection.

    Trains members per outer fold, takes the **logit mean**, and fits isotonic
    once over the pooled OOF values (calling `GbdtVerifier.fit` per fold would
    nest the internal-CV isotonic and blow up the number of fits).

    With `groups` (per-row group key, usually the image filename), splits are
    **by group** — detections of one image share context features
    (overlap_margin etc.) and include near-duplicates from full/tile passes, so
    row-level shuffling would let a model trained on other detections of an
    image predict that image (biasing α selection optimistically).
    """
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None:
        keys = sorted(set(groups))
        rng.shuffle(keys)
        bucket = {k: i % folds for i, k in enumerate(keys)}
        g = np.array([bucket[x] for x in groups])
        test_sets = [np.where(g == f)[0] for f in range(folds)]
    else:
        test_sets = list(np.array_split(rng.permutation(n), folds))
    test_sets = [t for t in test_sets if len(t)]
    raw = np.zeros(n)
    X = np.hstack([np.asarray(X_hand, dtype=float), np.asarray(E, dtype=float)])
    all_idx = np.arange(n)
    for te in test_sets:
        tr = np.setdiff1d(all_idx, te)
        if len(tr) == 0 or len(set(y[tr].tolist())) < 2:
            # Single-class training part -> use a constant. It is exactly 0 or 1 and
            # creates a large tied group, but fit_isotonic pools ties, so the
            # calibration follows that group's true rate (without pooling P̂
            # would jump to 1.0 here)
            raw[te] = float(np.mean(y[tr])) if len(tr) else float(np.mean(y))
            continue
        raw[te] = ensemble_raw(fit_members(X[tr], y[tr], members), X[te])
    iso = fit_isotonic(raw, y)
    return _iso_apply(iso, raw)


# ------------------------- discriminator fit / apply (lifecycle owned by loop.rescore)

def _label_correct(det: dict, cid: int, gt_anns, iou_thr: float) -> bool:
    """Whether labeling this detection as class cid agrees with a GT.

    `iou_thr` is **passed by the caller from config** — with a default here, the
    F(β)/τ_c path would use `detector.iou_threshold` while P̂'s training target
    stayed pinned at 0.5, so raising the threshold would make the discriminator
    learn a different criterion.

    Note: no matching is done, so duplicate detections on the same GT are all
    labeled true — P̂ estimates "does this box sit on a real object"; duplicate
    suppression is handled by the detector's NMS and the gateway's within-class
    folding.
    """
    box = np.array([det["box"]], dtype=float)
    for a in gt_anns:
        if a.category_id != cid:
            continue
        if iou_matrix(box, np.array([a.bbox_xyxy], dtype=float))[0, 0] >= iou_thr:
            return True
    return False


def _rescorer_rows(preds: dict, train, tiled_ids: set, iou_thr: float,
                   area_stats: dict | None = None):
    """rescorer training rows — attaches features, embedding, and GT label to every above-floor candidate.

    The feature context (overlap_margin etc.) is **all above-floor candidates of
    the same image** — application sees the same context, so train and inference
    distributions match.

    `area_stats`: per-class area stats for the area_z feature. Defaults to this
    split's own, but the loop wiring passes **full-train** stats — fitting
    (fit_train/fit_dev) and application (fit_dev/holdout) must share one
    reference so feature distributions stay aligned.
    """
    entries = {e.file_name: e for e in train.images}
    area_stats = area_stats if area_stats is not None else train.area_stats()
    by_image: dict[str, list[dict]] = {}
    for r in preds.values():
        by_image.setdefault(r["image"], []).append(r)
    rows = []
    for image, dets in sorted(by_image.items()):
        anns = entries[image].annotations
        for d in dets:
            # Keep the row even without an embedding (e.g. predictions made without
            # return_embeddings). The discriminator can fit on hand-crafted
            # features alone, so instead of dropping the row, leave emb=None and
            # let fit_rescorer decide the width.
            feats = detection_features(d, dets, area_stats, tiled_ids)
            rows.append({"det_id": d["det_id"], "feats": feats,
                         "emb": d.get("embedding"), "score": float(d["score"]),
                         "y": bool(_label_correct(d, d["category_id"], anns, iou_thr)),
                         "cid": d["category_id"], "image": image, "box": d["box"]})
    return rows


def fit_rescorer(rs_cfg, *, train_preds: dict, train_split, dev_preds: dict,
                 dev_split, dev_image_paths, ontology, dataset_name: str,
                 tiled_ids: set | None = None, iou_thr: float = 0.5,
                 threshold_floor: float = 0.03, max_threshold_candidates: int = 50,
                 beta: float = 1.0, area_stats: dict | None = None):
    """Discriminator training (fit_train) + α grid selection (fit_dev).

    fit_dev is held out from discriminator training, so P̂ is applied directly to
    measure each class's F(α) curve, and α_c is picked by a margin gate. α=0 is
    on the grid, so without gain it naturally falls back to identity.
    info["fit_s"] holds the seconds of the discriminator fit and the α selection.

    Returns (rescorer|None, info).
    """
    tiled_ids = tiled_ids or set()
    rows_tr = _rescorer_rows(train_preds, train_split, tiled_ids, iou_thr, area_stats)
    rows_dev = _rescorer_rows(dev_preds, dev_split, tiled_ids, iou_thr, area_stats)
    # If some rows have embeddings, drop the ones missing them (width mismatch).
    # If none do, emb_dim=0 — fit on hand-crafted features only (the rescorer
    # axis must stay alive even when no embeddings are returned).
    has_emb = any(r["emb"] is not None for r in rows_tr)
    if has_emb:
        rows_tr = [r for r in rows_tr if r["emb"] is not None]
        rows_dev = [r for r in rows_dev if r["emb"] is not None]
    else:
        for r in rows_tr + rows_dev:
            r["emb"] = []
    feat_names = list(VERIFIER_FEATURES)
    # row cap — thinned and weighted only above it (otherwise the same uniform fit as before)
    n_raw = len(rows_tr)
    rows_tr, w_tr, cap_info = cap_train_rows(rows_tr, rs_cfg.max_train_rows)
    ys_tr = np.array([r["y"] for r in rows_tr], dtype=float)
    info = {"n_train": len(rows_tr), "n_dev": len(rows_dev),
            "pos_train": int(ys_tr.sum()) if len(ys_tr) else 0,
            "embeddings": bool(has_emb), "n_train_raw": n_raw,
            "train_rows_capped": cap_info["capped"]}
    if len(rows_tr) < rs_cfg.min_samples or not (0 < ys_tr.sum() < len(ys_tr)):
        # The total-count guard is a discriminator-fit tripwire (below it,
        # HistGradientBoosting's min_samples_leaf degenerates P̂ to a constant)
        log.warning("rescorer training skipped: %d fit_train candidates (< %d) or a single class",
                    len(rows_tr), rs_cfg.min_samples)
        info["skipped"] = True
        return None, info
    X_tr = np.array([[r["feats"][n] for n in feat_names] for r in rows_tr], dtype=float)
    E_tr = (np.array([r["emb"] for r in rows_tr], dtype=float) if has_emb
            else np.zeros((len(rows_tr), 0), dtype=float))
    if not has_emb:
        log.warning("rescorer: fitting without embeddings — using only %d hand-crafted features "
                    "(the detector does not provide embeddings)", len(feat_names))
    t0 = time.monotonic()
    # the entry point sets the OpenMP thread default to 1 — released only for the discriminator fit
    with threadpool_limits(limits=rs_cfg.fit_threads, user_api="openmp"):
        disc = fit_discriminator(feat_names, X_tr, E_tr, ys_tr,
                                 members=rs_cfg.ensemble_members,
                                 sample_weight=w_tr if cap_info["capped"] else None)
    t1 = time.monotonic()
    alphas, sel = _select_alphas(rs_cfg, disc, rows_dev, dev_split, dev_image_paths,
                                 iou_thr=iou_thr, threshold_floor=threshold_floor,
                                 max_threshold_candidates=max_threshold_candidates,
                                 beta=beta)
    info.update(sel)
    info["fit_s"] = {"discriminator": round(t1 - t0, 2),
                     "alpha": round(time.monotonic() - t1, 2)}
    info.update({"ensemble_members": disc.members, "features": feat_names})
    log.info("rescorer: train n=%d%s, dev n=%d, α>0 classes %d/%d (margin %.3f)",
             len(rows_tr), f" (capped — {n_raw} before)" if cap_info["capped"] else "",
             len(rows_dev), info["alpha_active_classes"], info["n_classes"],
             rs_cfg.margin)
    fp = dataset_fingerprint(dataset_name, ontology)
    # time (fit_s) is not artifact metadata — the caller keeps it as wall_s in rescorer_info.json
    meta = {k: v for k, v in info.items() if k != "fit_s"}
    return Rescorer(disc, alphas, meta=meta, fingerprint=fp), info


def refit_alphas(rs_cfg, discriminator: GbdtVerifier, *, dev_preds: dict, dev_split,
                 dev_image_paths, tiled_ids: set | None = None, iou_thr: float = 0.5,
                 threshold_floor: float = 0.03, max_threshold_candidates: int = 50,
                 beta: float = 1.0, area_stats: dict | None = None) -> Rescorer:
    """Keep the discriminator and re-pick only α on these fit_dev predictions (the
    same α selection as fit_rescorer). If the discriminator was fit with
    embeddings, rows without one are dropped."""
    rows_dev = _rescorer_rows(dev_preds, dev_split, tiled_ids or set(), iou_thr,
                              area_stats)
    if discriminator.emb_dim:
        rows_dev = [r for r in rows_dev if r["emb"] is not None]
    else:
        for r in rows_dev:
            r["emb"] = []
    alphas, sel = _select_alphas(rs_cfg, discriminator, rows_dev, dev_split,
                                 dev_image_paths, iou_thr=iou_thr,
                                 threshold_floor=threshold_floor,
                                 max_threshold_candidates=max_threshold_candidates,
                                 beta=beta)
    return Rescorer(discriminator, alphas, meta=sel)


def _select_alphas(rs_cfg, disc: GbdtVerifier, rows_dev: list[dict], dev_split,
                   dev_image_paths, *, iou_thr: float, threshold_floor: float,
                   max_threshold_candidates: int, beta: float) -> tuple[dict, dict]:
    """The per-class α grid on fit_dev rows — (alphas, a piece of info)."""
    info: dict = {}
    alphas: dict[int, float] = {}
    sel_f1: dict[int, dict[float, float]] = {}
    counts: dict[int, int] = {}
    if rows_dev:
        probe = Rescorer(disc, {})
        p_dev = probe.phat([r["feats"] for r in rows_dev],
                           [r["emb"] for r in rows_dev])
        for r in rows_dev:
            counts[r["cid"]] = counts.get(r["cid"], 0) + 1
        thin = {c for c, k in counts.items() if k < rs_cfg.min_samples_per_class}

        def recs_for(alpha):
            by: dict[str, list[dict]] = {}
            for r, ph in zip(rows_dev, p_dev):
                by.setdefault(r["image"], []).append(
                    {"box": r["box"], "category_id": r["cid"],
                     "score": float(blend([r["score"]], [ph], alpha)[0])})
            return per_image_records(dev_split, dev_image_paths,
                                     [by.get(p.name, []) for p in dev_image_paths])

        # only classes where comparison is meaningful — α moves F(β) only with both GT and detections
        base_recs = recs_for(0.0)
        with_det = {d["category_id"] for rec in base_recs for d in rec["detections"]}
        with_gt = {g["category_id"] for rec in base_recs for g in rec["gts"]}
        informative = with_det & with_gt

        cids = sorted(counts)
        sel_f1 = {cid: {} for cid in cids}
        for a in rs_cfg.alphas:
            rs = base_recs if a == 0.0 else recs_for(a)
            for cid in cids:
                if cid in thin or cid not in informative:
                    continue          # no curve — pinned to α=0 below
                _thr, f1 = metrics.estimate_f_beta_threshold(
                    rs, cid, floor=threshold_floor,
                    max_candidates=max_threshold_candidates, iou_thr=iou_thr,
                    beta=beta)
                sel_f1[cid][a] = f1
        margin = float(rs_cfg.margin)
        for cid in cids:
            curve = sel_f1[cid]
            if not curve:
                alphas[cid] = 0.0     # thin / not comparable — identity for this class; the dataset is not abandoned
                continue
            base_c = curve.get(0.0, 0.0)
            best_c = max((a for a in rs_cfg.alphas if a in curve),
                         key=lambda a: (curve[a], -a))
            # non-degradation margin: stay identity unless the fit_dev gain exceeds margin
            alphas[cid] = (float(best_c) if curve[best_c] - base_c > margin else 0.0)
        ys_dev = np.array([r["y"] for r in rows_dev], dtype=float)
        info.update({"auc_dev": metrics.rank_auc(p_dev, ys_dev.astype(bool)),
                     "auc_score_only_dev": metrics.rank_auc(
                         np.array([r["score"] for r in rows_dev]),
                         ys_dev.astype(bool)),
                     "brier_dev": metrics.brier(p_dev, ys_dev.astype(bool)),
                     "ece_dev": metrics.expected_calibration_error(
                         p_dev, ys_dev.astype(bool)),
                     "thin_classes": {str(c): counts[c] for c in sorted(thin)}})

    n_on = sum(1 for a in alphas.values() if a != 0.0)
    info.update({"alphas": {str(k): v for k, v in sorted(alphas.items())},
                 "alpha_active_classes": n_on, "n_classes": len(counts),
                 "margin": float(rs_cfg.margin),
                 "class_counts_dev": {str(c): counts[c] for c in sorted(counts)},
                 "sel_f_beta_by_class_alpha": {
                     str(cid): {str(a): round(v, 4) for a, v in sorted(c.items())}
                     for cid, c in sel_f1.items()}})
    return alphas, info


def _apply_rescorer(rescorer, preds: dict, train, tiled_ids,
                    area_stats: dict | None = None) -> int:
    """Replace scores in preds with recalibrated ones (originals kept as score_raw).

    Classes with α_c=0 are untouched — they stay exactly as-is.
    `train` is only the default source of area_stats; its GT is not read.
    """
    if not rescorer.active:
        return 0
    area_stats = area_stats if area_stats is not None else train.area_stats()
    by_image: dict[str, list[dict]] = {}
    for r in preds.values():
        by_image.setdefault(r["image"], []).append(r)
    n = 0
    for dets in by_image.values():
        usable = [d for d in dets if d.get("embedding") is not None]
        if not usable:
            continue
        feats = [detection_features(d, dets, area_stats, tiled_ids) for d in usable]
        new = rescorer.rescore(feats, [d["embedding"] for d in usable],
                              [d["score"] for d in usable],
                              [d["category_id"] for d in usable])
        for d, s in zip(usable, new):
            if float(s) == float(d["score"]):
                continue                # α_c=0 class
            d["score_raw"] = float(d["score"])
            d["score"] = float(s)
            n += 1
    return n
