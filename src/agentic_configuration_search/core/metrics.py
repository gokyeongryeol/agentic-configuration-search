"""mAP/F(β), F(β)-max threshold estimation, and FP/FN cause classification.

Threshold estimation is a pure function (score distribution + GT -> threshold),
so it lives here, separate from the stages, and is unit-tested.
"""

from __future__ import annotations

import bisect

import contextlib
import io
import logging

import numpy as np

from .coco import Split

log = logging.getLogger(__name__)

# IoM lower bound for "the box sits on a real object" (partial detections and
# loose boxes are not background even when IoU is low)
IOM_ON_OBJECT = 0.7


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    # Zero-area box pairs have union=0 — np.where evaluates both branches, so make
    # the division safe first (0/0 warns, or raises under np.seterr(all='raise'))
    return np.where(union > 0, inter / np.clip(union, 1e-9, None), 0.0)


def iom_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """intersection / min(area) — for detecting containment. A small box inside a
    large one has IoU reduced by the area ratio, but IoM is 1.0. **Never used for
    matching or evaluation** (partial detections would count as TP and break the
    metrics) — only for asking "does the box sit on a real object"."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    min_area = np.minimum(area_a[:, None], area_b[None, :])
    return np.where(min_area > 0, inter / np.clip(min_area, 1e-9, None), 0.0)


def match_greedy(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                 gt_boxes: np.ndarray, iou_thr: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy matching by descending score -> (matches[(pred_i, gt_j)], fp_pred_idx, fn_gt_idx).

    Ties are broken by **input order** (stable sort). Any sort is deterministic
    for a given input, so this is for interpretability, not reproducibility —
    tie order stays local, so an unrelated detection cannot reshuffle other
    tied pairs.
    """
    order = np.argsort(-pred_scores, kind="mergesort")
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_taken = np.zeros(len(gt_boxes), dtype=bool)
    matches, fps = [], []
    for i in order:
        if len(gt_boxes) == 0:
            fps.append(int(i))
            continue
        cand = np.where(~gt_taken, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            gt_taken[j] = True
            matches.append((int(i), j))
        else:
            fps.append(int(i))
    fns = [int(j) for j in range(len(gt_boxes)) if not gt_taken[j]]
    return matches, fps, fns


def prf(tp: int, fp: int, fn: int, beta: float = 1.0) -> dict:
    """The "f_beta" field is the F-beta set by beta (beta>1 weights recall, beta<1
    precision; beta=1 is F1). Threshold estimation, micro and the gate all read
    this key, so the metric changes through the single beta argument."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    b2 = float(beta) ** 2
    denom = b2 * precision + recall
    f1 = 0.0 if denom == 0 else (1 + b2) * precision * recall / denom
    return {"precision": precision, "recall": recall, "f_beta": f1, "tp": tp, "fp": fp, "fn": fn}


def micro_f_beta(per_class_prf: dict, beta: float = 1.0) -> float:
    """F(-beta) over TP/FP/FN summed across all classes — the adoption gate's metric.

    Not macro (mean of per-class F1): a single rare class swings the total and
    inflates variance. Micro corresponds directly to "overall label quality".
    """
    if not per_class_prf:
        return 0.0
    tp = sum(v["tp"] for v in per_class_prf.values())
    fp = sum(v["fp"] for v in per_class_prf.values())
    fn = sum(v["fn"] for v in per_class_prf.values())
    return prf(tp, fp, fn, beta)["f_beta"]


# --------------------------------------------------- F1-max threshold

def _class_counts(per_image: list[dict], category_id: int, threshold: float, iou_thr: float):
    tp = fp = fn = 0
    for rec in per_image:
        boxes, scores = [], []
        for d in rec["detections"]:
            if d["category_id"] == category_id and d["score"] >= threshold:
                boxes.append(d["box"])
                scores.append(d["score"])
        gts = [g["box"] for g in rec["gts"] if g["category_id"] == category_id]
        m, f, n = match_greedy(np.array(boxes, dtype=float).reshape(-1, 4),
                               np.array(scores, dtype=float),
                               np.array(gts, dtype=float).reshape(-1, 4), iou_thr)
        tp += len(m); fp += len(f); fn += len(n)
    return tp, fp, fn


def _threshold_candidates(scores: list[float], max_candidates: int) -> list[float]:
    """Candidate grid — **uniform in score**, not in rank.

    A rank-uniform grid (linspace over index) spends the whole candidate budget
    where unique scores cluster (e.g. low noise scores) and misses the global
    optimum.

    F(beta) only changes at observed scores, so each uniform grid point is snapped
    up to **the first observed score at or above it**. The lowest score (the
    keep-everything cut) is always kept.
    """
    if len(scores) <= max_candidates:
        return list(scores)
    out: list[float] = []
    j = 0
    for v in np.linspace(scores[0], scores[-1], max_candidates):
        j = bisect.bisect_left(scores, float(v), j)
        if j >= len(scores):
            break
        if not out or scores[j] != out[-1]:
            out.append(scores[j])
    if not out or out[0] != scores[0]:
        out.insert(0, scores[0])
    return out


def _max_margin_threshold(plateau: list[float], scores: list[float], floor: float,
                          f_of: dict, best_f1: float) -> tuple[float, float]:
    """Midpoint of the gap, within the plateau, with **the largest margin to the
    next-lower observed score**.

    When F(beta) is flat over a wide range, holdout evidence cannot pin τ inside
    it. The lowest τ sits right on top of the noise cluster and the highest just
    below true detections, both with no margin — each exposed to one failure mode.

    τ owns the interval (previous observed score, that score], so picking its
    midpoint yields F(τ) exactly equal to F(that score) — nothing leaks outside
    the plateau.
    """
    best_thr, best_gap, best_f = float(min(plateau)), -1.0, f_of[min(plateau)]
    for v in plateau:
        i = bisect.bisect_left(scores, v)
        prev = scores[i - 1] if i > 0 else float(floor)
        gap = v - prev
        if gap > best_gap:
            best_gap, best_thr, best_f = gap, (prev + v) / 2.0, f_of[v]
    return float(best_thr), float(best_f)

def estimate_f_beta_threshold(per_image: list[dict], category_id: int, *,
                          floor: float, max_candidates: int, iou_thr: float,
                          beta: float = 1.0, tol: float = 0.0) -> tuple[float, float]:
    """(best_threshold, best_f_beta). per_image entries:
        {"detections": [{"box","score","category_id"}], "gts": [{"box","category_id"}]}

    Algorithm: unique scores as candidates -> if too many, reduce to a
    **score-uniform** grid (_threshold_candidates) -> F(beta) per candidate ->
    among candidates within `tol` of the max, the point with the largest margin
    to observed scores (_max_margin_threshold). No candidates -> keep floor.

    tol=0 uses the first candidate reaching the max. tol>0 picks the safest
    point within the range F(beta) cannot distinguish (_max_margin_threshold).

    A class whose F(beta) never exceeds 0 at any candidate only gets a warning
    and its threshold is left unchanged — blocking it here would erase the class
    from later rounds' diagnosis and recovery signals.
    """
    scores = sorted({float(d["score"]) for rec in per_image for d in rec["detections"]
                     if d["category_id"] == category_id})
    if not scores:
        return float(floor), 0.0
    cand = _threshold_candidates(scores, max_candidates)

    evaluated = []
    for thr in cand:
        tp, fp, fn = _class_counts(per_image, category_id, thr, iou_thr)
        evaluated.append((float(thr), prf(tp, fp, fn, beta)["f_beta"]))
    best_f1 = max(f for _, f in evaluated)
    if tol <= 0.0:
        # tol=0 — first candidate reaching the max (lowest τ on ties).
        best_thr = next(thr for thr, f in evaluated if f == best_f1)
    else:
        # Candidates within tol are **indistinguishable**; pick the one with the
        # largest margin to the next-lower observed score (_max_margin_threshold).
        plateau = [thr for thr, f in evaluated if f >= best_f1 - tol]
        best_thr, best_f1 = _max_margin_threshold(plateau, scores, floor,
                                                  dict(evaluated), best_f1)
    if best_f1 <= 0.0 and any(g["category_id"] == category_id
                              for rec in per_image for g in rec["gts"]):
        log.warning("class %d: no threshold yields F(β)>0 (%d candidates, GT present) "
                    "— leaving τ_c at %.3f. Detections for this class are "
                    "unreliable",
                    category_id, len(scores), best_thr)
    return best_thr, max(best_f1, 0.0)


def brier(prob, label) -> float:
    """Brier score (=MSE). Lower is better. Grows with miscalibration even at identical ranking."""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(label, dtype=float)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(prob, label, bins: int = 10) -> float:
    """Equal-width-bin ECE — sample-weighted mean of |mean prob - observed rate|.

    Bins are equal-width over [0,1], not quantile bins, because P̂ is read as an
    absolute probability — the error must be measured on that axis to show its impact.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(label, dtype=float)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / p.size)


def class_prf_at(per_image: list[dict], category_id: int, threshold: float,
                 iou_thr: float, beta: float = 1.0) -> dict:
    tp, fp, fn = _class_counts(per_image, category_id, threshold, iou_thr)
    return prf(tp, fp, fn, beta)


# ------------------------------------------------------------- mAP (pycocotools)

def coco_map(per_image: list[dict], category_ids: list[int], id_to_name: dict[int, str]) -> dict:
    """Build COCO GT/DT in memory from per_image (schema above) and compute mAP/AP50."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = {"images": [], "annotations": [], "categories":
          [{"id": cid, "name": id_to_name[cid]} for cid in category_ids]}
    dt = []
    ann_id = 1
    for img_id, rec in enumerate(per_image, start=1):
        gt["images"].append({"id": img_id, "width": rec.get("width", 1), "height": rec.get("height", 1)})
        for g in rec["gts"]:
            x1, y1, x2, y2 = g["box"]
            gt["annotations"].append({"id": ann_id, "image_id": img_id,
                                      "category_id": g["category_id"],
                                      "bbox": [x1, y1, x2 - x1, y2 - y1],
                                      "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            ann_id += 1
        for d in rec["detections"]:
            x1, y1, x2, y2 = d["box"]
            dt.append({"image_id": img_id, "category_id": d["category_id"],
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d["score"]})

    if not gt["annotations"]:
        return {"map": 0.0, "ap50": 0.0, "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools output
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt) if dt else COCO()
        if not dt:
            coco_dt.dataset = {"images": gt["images"], "annotations": [], "categories": gt["categories"]}
            coco_dt.createIndex()
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        per_class = {}
        for k, cid in enumerate(category_ids):
            ev_c = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ev_c.params.catIds = [cid]
            ev_c.evaluate(); ev_c.accumulate(); ev_c.summarize()
            per_class[cid] = {"map": float(ev_c.stats[0]), "ap50": float(ev_c.stats[1])}
    return {"map": float(ev.stats[0]), "ap50": float(ev.stats[1]), "per_class": per_class}


# --------------------------------------------- paired bootstrap CI / rank AUC

def records_micro_f_beta(per_image: list[dict], category_ids: list[int],
                     iou_thr: float, beta: float = 1.0) -> float:
    """per_image records -> micro F(-beta). Used as a `bootstrap_ci` stat_fn."""
    per_class = {cid: class_prf_at(per_image, cid, 0.0, iou_thr) for cid in category_ids}
    return micro_f_beta(per_class, beta)


def bootstrap_ci(records_a: list[dict], records_b: list[dict], stat_fn, *,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                 tail: str = "two") -> dict:
    """Image-level paired bootstrap — **the basis of the adoption gate's decision**
    (eval.evaluate.gate compares this CI's lower bound to the margin).

    records_a/b use the `per_image_records` schema and must cover **the same
    images in the same order** (paired). The same indices are resampled on both
    sides to get the distribution of Δ = stat(B) - stat(A).
    Returns {"delta", "lo", "hi", "n_images"}.

    `tail`: "two" gives a two-sided interval (lower alpha/2, upper 1-alpha/2) —
    for reports and A/B. "lower" gives a **one-sided lower bound** (alpha
    percentile) with upper None. The gate asks only "is Δ above the margin", a
    one-sided test; a two-sided bound would halve the effective significance
    level and be overly conservative — so that path uses "lower". "upper" gives a
    **one-sided upper bound** (1-alpha percentile) with lower None — used by the
    tiling near-miss (eval.screen), which asks "can B not being better than A be
    ruled out".
    """
    if len(records_a) != len(records_b):
        raise ValueError(f"records length mismatch: {len(records_a)} vs {len(records_b)} — "
                         "a paired comparison requires the same image set")
    n = len(records_a)
    if tail not in ("two", "lower", "upper"):
        raise ValueError(f"tail: two|lower|upper ({tail!r})")
    if n == 0:
        return {"delta": 0.0, "lo": None if tail == "upper" else 0.0,
                "hi": None if tail == "lower" else 0.0, "n_images": 0}
    rng = np.random.default_rng(seed)
    point = stat_fn(records_b) - stat_fn(records_a)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_a = [records_a[i] for i in idx]
        sample_b = [records_b[i] for i in idx]
        deltas[b] = stat_fn(sample_b) - stat_fn(sample_a)
    if tail == "lower":
        lo, hi = float(np.quantile(deltas, alpha)), None
    elif tail == "upper":
        lo, hi = None, float(np.quantile(deltas, 1 - alpha))
    else:
        q = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
        lo, hi = float(q[0]), float(q[1])
    return {"delta": float(point), "lo": lo, "hi": hi, "n_images": n}


def rank_auc(scores, labels) -> float:
    """Rank AUC via Mann-Whitney U (ties get average rank). NaN if either class is empty.

    Measures how well a signal such as the discriminator's P̂ separates TP from FP."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # average rank over tied runs
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2
        i = j + 1
    ranks = np.empty(len(scores))
    ranks[order] = ranks_sorted
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


# ---------------------------------------------------- FP/FN cause classification

FP_KINDS = ["localization", "duplicate", "class_confusion", "background", "tiling_artifact"]
FN_KINDS = ["below_threshold", "missed_small", "missed"]

# missed_small cut — the object's size relative to the image,
# r = √(box area / image area); below this it is "small". The detector sees the
# image resized by its preprocessor, so this ratio, not the original px, sets how
# large an object looks. 0.04 = 4% of a side (0.16% of the area) — about 26px on a
# square image at YOLOE's 640px input, within COCO's small (< 32²).
# config: diagnose.small_object_ratio.
SMALL_OBJECT_RATIO = 0.04


def _size_ratio(box, rec: dict) -> float:
    """r = √(box area / image area) — the size the detector sees (the axis of
    the missed_small cut)."""
    x1, y1, x2, y2 = box
    return float(np.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
                         / max(rec["width"] * rec["height"], 1e-9)))


def classify_errors(per_image: list[dict], per_class_threshold: dict[int, float], *,
                    iou_thr: float, floor_detections: list[dict] | None = None,
                    small_ratio: float = SMALL_OBJECT_RATIO) -> dict:
    """Classify FPs among thresholded detections and FNs among GTs.

    per_image detections must be **already thresholded**. Distinguishing
    below_threshold FNs requires the detections between floor and threshold, so
    floor_detections (full record: per-rec detections in the same image order as
    per_image) is passed separately. If None, below_threshold folds into missed.

    FP classification uses an IoM axis besides IoU — small duplicate/partial
    detections on a large GT have IoU<0.1 and would be miscounted as background,
    but boxes on a real object (IoM>=0.7) are localization/class_confusion.

    missed_small is cut by **the object's size relative to the image** — small if
    r = √(box area / image area) is below small_ratio. A cut by rank within a
    class would produce "small" misses even on a dataset of only large objects.
    An FN record's size_ratio is that r — material for judging how much zoom is
    needed.

    Matched GT (TP) keep their r in `tp` too — the yardstick "objects this small
    are already detected" (s* of the tiling near-miss, planner.gates.tp_floor).
    """
    fp_records, fn_records, tp_records = [], [], []

    for img_idx, rec in enumerate(per_image):
        dets = rec["detections"]
        gts = rec["gts"]
        gt_boxes = np.array([g["box"] for g in gts], dtype=float).reshape(-1, 4)

        # per-class matching yields TP/FP/FN
        matched_gt: set[int] = set()
        fp_dets: list[dict] = []
        for cid in {d["category_id"] for d in dets} | {g["category_id"] for g in gts}:
            c_dets = [d for d in dets if d["category_id"] == cid]
            c_gt_idx = [j for j, g in enumerate(gts) if g["category_id"] == cid]
            m, f, _ = match_greedy(
                np.array([d["box"] for d in c_dets], dtype=float).reshape(-1, 4),
                np.array([d["score"] for d in c_dets], dtype=float),
                gt_boxes[c_gt_idx].reshape(-1, 4), iou_thr)
            for _, j in m:
                matched_gt.add(c_gt_idx[j])
            fp_dets += [c_dets[i] for i in f]

        # FP causes
        for d in fp_dets:
            box = np.array([d["box"]], dtype=float)
            ious = iou_matrix(box, gt_boxes)[0] if len(gts) else np.array([])
            ioms = iom_matrix(box, gt_boxes)[0] if len(gts) else np.array([])
            same = [ious[j] for j, g in enumerate(gts) if g["category_id"] == d["category_id"]]
            other = [ious[j] for j, g in enumerate(gts) if g["category_id"] != d["category_id"]]
            iom_same = max((ioms[j] for j, g in enumerate(gts)
                            if g["category_id"] == d["category_id"]), default=0.0)
            iom_other = max((ioms[j] for j, g in enumerate(gts)
                             if g["category_id"] != d["category_id"]), default=0.0)
            best_same = max(same) if same else 0.0
            best_other = max(other) if other else 0.0
            if best_same >= iou_thr:
                kind = "duplicate"          # duplicates an already-matched GT
            elif best_other >= iou_thr:
                kind = "class_confusion"
            elif best_same >= 0.1 or iom_same >= IOM_ON_OBJECT:
                kind = "localization"       # on a same-class object (incl. partial/loose boxes)
            elif iom_other >= IOM_ON_OBJECT:
                kind = "class_confusion"    # contains or is contained by an other-class object
            elif max(best_same, best_other) < 0.1:
                kind = ("tiling_artifact"
                        if d.get("sources") and "full" not in d["sources"] else "background")
            else:
                kind = "background"
            fp_records.append({"image_index": img_idx, "kind": kind, **{k: d[k] for k in ("box", "score", "category_id")}})

        # FN causes (matched GT only leave their TP size)
        floor_dets = (floor_detections[img_idx]["detections"]
                      if floor_detections is not None else [])
        for j, g in enumerate(gts):
            if j in matched_gt:
                tp_records.append({"category_id": g["category_id"],
                                   "size_ratio": round(_size_ratio(g["box"], rec), 4)})
                continue
            cid = g["category_id"]
            gt_box = np.array([g["box"]], dtype=float)
            below = [d for d in floor_dets
                     if d["category_id"] == cid
                     and d["score"] < per_class_threshold.get(cid, float("inf"))
                     and iou_matrix(np.array([d["box"]], dtype=float), gt_box)[0, 0] >= iou_thr]
            ratio = _size_ratio(g["box"], rec)
            if below:
                kind = "below_threshold"
            elif ratio < small_ratio:
                kind = "missed_small"
            else:
                kind = "missed"
            fn_records.append({"image_index": img_idx, "kind": kind,
                               "box": g["box"], "category_id": cid,
                               "size_ratio": round(ratio, 4)})

    def _summary(records, kinds):
        return {k: sum(1 for r in records if r["kind"] == k) for k in kinds}

    return {"fp": fp_records, "fn": fn_records, "tp": tp_records,
            "fp_summary": _summary(fp_records, FP_KINDS),
            "fn_summary": _summary(fn_records, FN_KINDS)}


def per_image_records(split: "Split", image_paths, detections_by_image) -> list[dict]:
    """Convert (split, inference results) into this module's per_image record schema."""
    by_name = {split.image_path(e): e for e in split.images}
    out = []
    for path, dets in zip(image_paths, detections_by_image):
        entry = by_name[path]
        out.append({"file": entry.file_name, "width": entry.width, "height": entry.height,
                    "detections": dets,
                    "gts": [{"box": a.bbox_xyxy, "category_id": a.category_id}
                            for a in entry.annotations]})
    return out
