"""Crop-classifier axis material — mining, training call, artifact assembly (from the split the caller passes).

Serving: the detector client (clients.yoloe) loads the artifact directory
(classifier.pt / classifier_config.json) and **reclassifies predicted boxes**
client-side — replacing the category or judging background (0), which the
client drops. It cannot create new detections, so it has two levers:
correcting class_confusion and removing background FPs. On a single-class
dataset only the latter remains, and it acts as a binary verifier (object vs
background).

Training data (from the split the caller passes — the loop uses fit_train, see
loop.adapt_round):
  positive   = GT crops (label = class idx+1, ascending category id)
  background = FPs from that split's pass that pass the threshold but match no GT (label = 0)

**Purity requirement**: the background-mining thresholds must be the **screen
thresholds estimated on fit_dev** — using tau_test (a legacy name; it is
estimated on holdout) would let the holdout pick training material and break
material purity. The contract is that the caller (loop.adapt_round) passes the
setting's per_class thresholds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.coco import Ontology, Split, crop_image
from ..core.io import promptable
from ..core.metrics import iou_matrix
from ..services import external

log = logging.getLogger(__name__)

# Model-reconstruction contract — must match the universal-best constants in
# workers/adapt.py (a constant-equality test in tests/test_crop_classifier.py
# guards drift). fitting cannot import workers (layering rule), so they are
# duplicated here.
MODEL_NAME = "google/siglip2-base-patch16-224"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_NAME = "qkvo"
IMAGE_SIZE = 224

MATCH_IOU = 0.5                  # overlapping a GT at least this much is not background
CONTAINMENT_THRESHOLD = 0.7      # containing / contained by a GT this much = "mislocalized object" — excluded


def _contained_in_any_gt(box, gt_boxes: np.ndarray, threshold: float) -> bool:
    """containment = inter / min(area). An FP that largely contains or is contained
    by a GT is a mislocalized object, not background, so it is excluded from
    background training."""
    if len(gt_boxes) == 0:
        return False
    b = np.asarray(box, dtype=float)
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    if area_b <= 0:
        return True   # degenerate boxes cannot become training crops anyway
    ix1 = np.maximum(gt_boxes[:, 0], b[0])
    iy1 = np.maximum(gt_boxes[:, 1], b[1])
    ix2 = np.minimum(gt_boxes[:, 2], b[2])
    iy2 = np.minimum(gt_boxes[:, 3], b[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    gt_areas = np.clip((gt_boxes[:, 2] - gt_boxes[:, 0])
                       * (gt_boxes[:, 3] - gt_boxes[:, 1]), 1e-6, None)
    containment = inter / np.clip(np.minimum(area_b, gt_areas), 1e-6, None)
    return bool(np.any(containment >= threshold))


def _is_background(det: dict, gt_boxes: np.ndarray) -> bool:
    """Against GTs of all classes: background only if IoU<MATCH_IOU with every GT and
    it passes the containment filter — detections on another class's GT must
    not be learned as background."""
    if len(gt_boxes):
        ious = iou_matrix(np.array([det["box"]], dtype=float), gt_boxes)[0]
        if ious.max() >= MATCH_IOU:
            return False
    return not _contained_in_any_gt(det["box"], gt_boxes, CONTAINMENT_THRESHOLD)


def _save_crop(image: Image.Image, box, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    crop_image(image, box).save(path, quality=92)


def mine_classifier_crops(train: Split, ontology: Ontology,
                          dets_by_image: list[list[dict]],
                          thresholds: dict[int, float],
                          out_dir: Path) -> tuple[Path, dict]:
    """Write crops + manifest.jsonl — returns (manifest_path, per-source counts).

    dets_by_image = floor detections from that split's pass (image order = train.images).
    thresholds = per-class thresholds **estimated on fit_dev** (purity requirement — see module docstring).
    """
    out_dir = Path(out_dir)
    sorted_ids = [c.id for c in ontology.classes]      # classifier contract: ascending id
    label_of = {cid: i + 1 for i, cid in enumerate(sorted_ids)}
    crops_dir = out_dir / "crops"
    records: list[dict] = []
    counts = {"train_gt": 0, "train_fp": 0, "skipped_degenerate": 0}

    for entry, dets in zip(train.images, dets_by_image):
        gt_boxes = np.array([a.bbox_xyxy for a in entry.annotations],
                            dtype=float).reshape(-1, 4)
        bg = [d for d in dets
              if d["score"] >= thresholds.get(d["category_id"], float("inf"))
              and promptable(d["box"]) and _is_background(d, gt_boxes)]
        if not entry.annotations and not bg:
            continue
        with Image.open(train.image_path(entry)) as im:
            im = im.convert("RGB")
            for a in entry.annotations:                # positive — GT crop
                if not promptable(a.bbox_xyxy):
                    counts["skipped_degenerate"] += 1
                    continue
                path = crops_dir / "positive" / f"gt_{a.id}.jpg"
                _save_crop(im, a.bbox_xyxy, path)
                records.append({"crop_path": str(path),
                                "label": label_of[a.category_id],
                                "source": "train_gt"})
                counts["train_gt"] += 1
            for d in bg:                               # background — FPs above threshold
                path = crops_dir / "background" / f"fp_{counts['train_fp']:05d}.jpg"
                _save_crop(im, d["box"], path)
                records.append({"crop_path": str(path), "label": 0,
                                "source": "train_fp"})
                counts["train_fp"] += 1

    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    log.info("crop classifier training data: %s (%d crops total)", counts, len(records))
    return manifest, counts


def assemble_artifacts(adapt_out: Path, ontology: Ontology) -> None:
    """Assemble the classifier_config.json that sits next to classifier.pt.

    Everything clients.yoloe needs to rebuild the same architecture
    (SigLIP2+LoRA) — values diverging from the workers/adapt.py constants break
    loading. Reclassification labels are tied to class_names indices
    (+1, 0=background).
    """
    adapt_out = Path(adapt_out)
    adapt_out.mkdir(parents=True, exist_ok=True)
    class_names = [c.name for c in ontology.classes]
    (adapt_out / "classifier_config.json").write_text(json.dumps({
        "classifier_model": "siglip",
        "classifier_model_name": MODEL_NAME,
        "classifier_lora_r": LORA_R,
        "classifier_lora_alpha": LORA_ALPHA,
        "classifier_lora_target": LORA_TARGET_NAME,
        "classifier_image_size": IMAGE_SIZE,
        "class_names": class_names,
    }, indent=2, ensure_ascii=False))


def train_crop_classifier(repo_root: Path, *, manifest: Path, ontology: Ontology,
                          out_dir: Path, device: str, log_path: Path) -> Path:
    """Worker training + artifact assembly — returns the directory to load into the client."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(ontology.classes) + 1            # +1 = background(0)
    external.run_adapt_worker(repo_root, manifest=manifest,
                              num_classes=num_classes,
                              out_path=out_dir / "classifier.pt",
                              device=device, log_path=log_path)
    assemble_artifacts(out_dir, ontology)
    return out_dir
