"""text×semantic modality ensemble merge — **diagnostic only**.

union is not a selectable option — a single τ_c over a mixed score
distribution breaks F1 estimation. It is used only when screen records "the F1
union would have reached" in `modality_diagnostic`; a consistently large
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

    kept: list[dict] = []
    for det in pool:
        target = None
        for k in kept:
            if k["category_id"] != det["category_id"]:
                continue
            overlap = iou_matrix(np.array([k["box"]], dtype=float),
                                 np.array([det["box"]], dtype=float))[0, 0]
            if overlap >= iou:
                target = k
                break
        if target is None:
            kept.append(det)
        else:
            target["modalities"] = sorted(set(target["modalities"]) | set(det["modalities"]))
            # Merge tiling sources too — for auditing provenance of the merged
            # result (diagnostic-only path; nothing downstream consumes it).
            if "sources" in det or "sources" in target:
                target["sources"] = sorted(set(target.get("sources", []))
                                           | set(det.get("sources", [])))
    return kept
