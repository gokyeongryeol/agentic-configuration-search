"""Pure functions for negative-prompt emulation on the YOLOE detector.

YOLOE has no native negative prompts. Emulation: run the negative prompts
(text and crops) **as extra positive classes** in the same pass, apply batched
NMS over the union (per class — no cross-class suppression), then **drop
positives with IoU >= threshold against any negative detection** and discard
the negative detections themselves. The suppression-strength grid is this IoU
threshold grid, carried in the config's neg_eta field.

All CPU-only pure functions — a detection is a {"box": xyxy, "score": float,
"category_id": int} dict; negatives are passed as a separate list.
"""

from __future__ import annotations

import numpy as np

from . import metrics


def batched_nms(dets: list[dict], iou_thr: float) -> list[dict]:
    """Per-class (batched) greedy NMS — descending score, fold at IoU >= iou_thr.

    Suppresses only within the same category_id (class-independent) — the
    emulation contract is that merging negatives in as extra classes never
    touches positives during NMS. The sort tie-break (box, category_id) is
    fixed, so the result is deterministic.
    (Same greedy as infer.gateway.dedup_class_duplicates — this one takes an
    arbitrary detection list.)
    """
    if len(dets) < 2:
        return list(dets)
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: (-x["score"], x["box"], x["category_id"])):
        boxes = [k["box"] for k in kept if k["category_id"] == d["category_id"]]
        if boxes and metrics.iou_matrix(
                np.array([d["box"]], dtype=float),
                np.array(boxes, dtype=float)).max() >= iou_thr:
            continue
        kept.append(d)
    return kept


def suppress_by_negatives(positives: list[dict], negatives: list[dict],
                          iou_thr: float) -> list[dict]:
    """Drop positives with IoU >= iou_thr against a negative (inclusive — >=).

    The negative's category_id is ignored — the spec is "drop **every**
    positive overlapping any negative detection" (broader than per-target
    suppression).
    """
    if not positives or not negatives:
        return list(positives)
    neg_boxes = np.array([n["box"] for n in negatives], dtype=float)
    out = []
    for d in positives:
        ious = metrics.iou_matrix(np.array([d["box"]], dtype=float), neg_boxes)
        if ious.max() >= iou_thr:
            continue
        out.append(d)
    return out


def emulate_negatives(positives: list[dict], negatives: list[dict], *,
                      sup_iou: float, nms_iou: float) -> list[dict]:
    """Full emulation: batched NMS on the union -> IoU suppression -> drop negatives.

    - positives and negatives must use disjoint category_id spaces (the caller
      assigns negatives separate channel ids) — batched NMS class independence
      relies on it.
    - sup_iou: suppression IoU threshold (value carried by the neg_eta grid).
    - nms_iou: NMS fold threshold (detector.iou_threshold).
    """
    union = batched_nms([*positives, *negatives], nms_iou)
    neg_ids = {id(n) for n in negatives}
    pos_kept = [d for d in union if id(d) not in neg_ids]
    neg_kept = [d for d in union if id(d) in neg_ids]
    return suppress_by_negatives(pos_kept, neg_kept, sup_iou)
