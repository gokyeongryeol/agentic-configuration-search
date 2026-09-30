"""Detector response contract — per image {"boxes","scores","text"[,"embeddings"]}.

clients.yoloe responds in this format; infer.gateway converts it to the internal
detection list via parse_prediction.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def parse_prediction(pred: dict, name_to_id: dict,
                     alias_to_name: Optional[dict] = None) -> list[dict]:
    """Detector response dict → internal detection list (assigns category_id).

    **`text` (class name) is the only class-resolution path.** `labels` are flat
    prompt indices whose meaning depends on prompt layout, so they are unused. The
    backend (clients.yoloe) always fills `text`; a missing entry is treated as an
    out-of-contract response and raises explicitly.

    `alias_to_name`: when alias phrases were sent in the text modality, the
    response `text` may be the alias — map it back to the original class name.

    If the response carries `embeddings` (per-detection, when `return_embeddings`
    was requested), attach each as the detection's `embedding` key — features
    for the verifier (gbdt engine).
    """
    texts = pred.get("text") or []
    boxes, scores = pred.get("boxes", []), pred.get("scores", [])
    embs = pred.get("embeddings")
    dets = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if i >= len(texts):
            raise ValueError(
                f"response detection {i} has no text (class name) — response contract violation")
        name = texts[i]
        if alias_to_name:
            name = alias_to_name.get(name, name)
        cat = name_to_id.get(name)
        if cat is None:  # class not in ontology — drop and warn
            log.warning("response class %r not in the ontology — dropping detection", name)
            continue
        det = {"box": [float(v) for v in box], "score": float(score),
               "category_id": cat}
        if embs is not None and i < len(embs):
            det["embedding"] = [float(v) for v in embs[i]]
        dets.append(det)
    return dets
