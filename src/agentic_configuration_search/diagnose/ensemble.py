"""text×semantic modality ensemble merge — **diagnostic only**.

union is not a selectable option — a single τ_c over a mixed score
distribution breaks F(β) estimation. It is used only when screen records "the
F(β) union would have reached" in `modality_diagnostic`; a consistently large
union_margin is a reason to revisit per-modality τ.

Semantics match stage 1 of `tiling.merge` (class-wise IoU dedup): when both
modalities hit the same object, only the higher-scoring one is kept. No
geometric recovery like seam union (both passes see the same full image).
"""

from __future__ import annotations

import numpy as np

from ..core.metrics import iou_matrix

DEDUP_IOU = 0.5  # same value as tiling.NMS_IOU — same-object duplicate criterion


def merge_modalities(semantic_dets: list[dict], text_dets: list[dict],
                     iou: float = DEDUP_IOU) -> list[dict]:
    """Union of one image's semantic/text pass detections, with class-wise IoU dedup.

    - Scan in descending score; a detection is absorbed into an existing one of
      the same class with IoU ≥ iou (box and score of the higher one are kept).
    - Tags provenance in `d["modalities"]` (separate from tiling's `sources`) —
      a detection both modalities agree on becomes ["semantic", "text"].
    """
    pool = [{**d, "modalities": list(d.get("modalities", ["semantic"]))}
            for d in semantic_dets]
    pool += [{**d, "modalities": list(d.get("modalities", ["text"]))}
             for d in text_dets]
    pool.sort(key=lambda d: -d["score"])

    # Kept boxes are also held as an array and each detection is compared with
    # all of them at once — a 1×1 IoU per pair took seconds per image and class
    # at topk 300 × two modalities. The verdict is unchanged (the first kept box,
    # in order, with the same class and IoU ≥ iou) and so are the IoU values.
    kept: list[dict] = []
    boxes = np.empty((len(pool), 4))
    cats: list = []
    for det in pool:
        m = len(kept)
        target = None
        if m:
            same = np.fromiter((c == det["category_id"] for c in cats), bool, m)
            if same.any():
                overlap = iou_matrix(boxes[:m], np.array([det["box"]], dtype=float))[:, 0]
                idx = np.flatnonzero(same & (overlap >= iou))
                if len(idx):
                    target = kept[int(idx[0])]
        if target is None:
            boxes[m] = det["box"]
            cats.append(det["category_id"])
            kept.append(det)
        else:
            target["modalities"] = sorted(set(target["modalities"]) | set(det["modalities"]))
            # Merge tiling sources too — for auditing provenance of the merged
            # result (diagnostic-only path; nothing downstream consumes it).
            if "sources" in det or "sources" in target:
                target["sources"] = sorted(set(target.get("sources", []))
                                           | set(det.get("sources", [])))
    return kept
