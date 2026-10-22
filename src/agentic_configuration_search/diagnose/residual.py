"""Structure of the final champion's residual misses — report only (not a planner input).

The diagnosis (diagnose.errors) splits FNs by cause (kind) alone. To decide
"what to collect more of", an operator needs to know **which objects** those
FNs were — size, crowding, frame truncation, aspect ratio — and the rate
against the GT in each bucket. This computes that from GT annotations and the
diagnosis rows alone (no inference).

- train: detail with facts attached to the diagnosis FN rows + per-class bucket
  counts + label quality flags (very small GT, duplicate GT).
- holdout: the champion's holdout records reclassified by the same rule
  (metrics.classify_errors), reporting **per-class bucket counts only** — no
  individual images or boxes (holdout stays aggregate-only).

The output (rounds/final/residual.json) feeds no decision of the loop.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import metrics

# Detector input side — converts size_ratio to input px. YOLOE letterboxes the
# long side to 640; size_ratio is √(box/image area), so this is exact for square
# images (the same assumption as planner.llm.INPUT_SIDE).
INPUT_SIDE = 640
LARGE_FACTOR = 3.0        # small < r ≤ medium < LARGE_FACTOR·r ≤ large
TINY_GT_PX = 3            # GT with a short side at or below this — annotation error suspected
DUP_GT_IOU = 0.95         # same-image same-class GT overlapping this much — duplicate suspected
BORDER_FRAC = 0.01        # a box within this fraction of an image side touches the border
ELONGATED = 3.0           # long side / short side at or above this is an elongated object
SIZE_BUCKETS = ("small", "medium", "large")
CROWD_BUCKETS = ((1, "1"), (5, "2-5"), (20, "6-20"), (math.inf, "21+"))


def size_bucket(ratio: float, small_ratio: float) -> str:
    if ratio < small_ratio:
        return "small"
    return "medium" if ratio < LARGE_FACTOR * small_ratio else "large"


def crowd_bucket(n_gt: int) -> str:
    return next(name for hi, name in CROWD_BUCKETS if n_gt <= hi)


def object_facts(box, width: float, height: float, *, n_gt: int,
                 small_ratio: float) -> dict:
    """Facts of one GT box — size bucket, input px, crowding bucket, border
    contact, elongation."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    ratio = math.sqrt(bw * bh / max(width * height, 1e-9))
    mx, my = BORDER_FRAC * width, BORDER_FRAC * height
    short = min(bw, bh)
    return {"size": size_bucket(ratio, small_ratio),
            "px": round(ratio * INPUT_SIDE, 1),
            "crowd": crowd_bucket(n_gt), "n_gt_image": n_gt,
            "border": bool(x1 <= mx or y1 <= my or x2 >= width - mx
                           or y2 >= height - my),
            "elongated": bool(short > 0 and max(bw, bh) / short >= ELONGATED)}


def image_rows(split) -> list[dict]:
    """Split → per_image rows without inference (file, width, height, gts) — the
    material for train facts."""
    return [{"file": e.file_name, "width": e.width, "height": e.height,
             "gts": [{"box": list(a.bbox_xyxy), "category_id": a.category_id}
                     for a in e.annotations]} for e in split.images]


def _empty_facets() -> dict:
    return {"gt": 0, "fn": 0, "size": {b: [0, 0] for b in SIZE_BUCKETS},
            "crowd": {name: [0, 0] for _, name in CROWD_BUCKETS},
            "border": [0, 0], "elongated": [0, 0], "kind_size": {}}


def _add(facets: dict, facts: dict, *, missed: bool, kind: str | None) -> None:
    i = 1 if missed else 0
    if not missed:
        facets["gt"] += 1
    else:
        facets["fn"] += 1
        ks = facets["kind_size"].setdefault(kind, {b: 0 for b in SIZE_BUCKETS})
        ks[facts["size"]] += 1
    facets["size"][facts["size"]][i] += 1
    facets["crowd"][facts["crowd"]][i] += 1
    if facts["border"]:
        facets["border"][i] += 1
    if facts["elongated"]:
        facets["elongated"][i] += 1


def _label_flags(images: list[dict], names: dict) -> dict:
    """Very small GT (short side ≤ TINY_GT_PX) and duplicate GT (same class,
    IoU ≥ DUP_GT_IOU)."""
    tiny, dup = [], []
    for rec in images:
        gts = rec["gts"]
        for g in gts:
            x1, y1, x2, y2 = g["box"]
            side = min(x2 - x1, y2 - y1)
            if side <= TINY_GT_PX:
                tiny.append({"image": rec["file"], "box": list(g["box"]),
                             "category": names.get(g["category_id"]),
                             "min_side": round(float(side), 1)})
        by_cls: dict = {}
        for g in gts:
            by_cls.setdefault(g["category_id"], []).append(g["box"])
        for cid, boxes in by_cls.items():
            if len(boxes) < 2:
                continue
            iou = metrics.iou_matrix(np.asarray(boxes, dtype=float),
                                     np.asarray(boxes, dtype=float))
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if iou[i, j] >= DUP_GT_IOU:
                        dup.append({"image": rec["file"], "box": list(boxes[j]),
                                    "category": names.get(cid),
                                    "iou": round(float(iou[i, j]), 3)})
    return {"tiny_gt": tiny, "duplicate_gt": dup}


def _split_facets(images: list[dict], fn_rows: list[dict], names: dict, *,
                  small_ratio: float) -> tuple[dict, list[dict]]:
    """Per-class bucket counts and FN rows (with facts attached). fn_rows carry
    image, box and kind."""
    per_class = {name: _empty_facets() for name in names.values()}
    meta = {}
    for rec in images:
        n = len(rec["gts"])
        meta[rec["file"]] = (rec["width"], rec["height"], n)
        for g in rec["gts"]:
            name = names.get(g["category_id"])
            if name is None:
                continue
            _add(per_class[name], object_facts(
                g["box"], rec["width"], rec["height"], n_gt=n,
                small_ratio=small_ratio), missed=False, kind=None)
    enriched = []
    for r in fn_rows:
        m = meta.get(r.get("image"))
        name = r.get("category") or names.get(r.get("category_id"))
        if m is None or name not in per_class or not r.get("box"):
            continue
        facts = object_facts(r["box"], m[0], m[1], n_gt=m[2],
                             small_ratio=small_ratio)
        _add(per_class[name], facts, missed=True, kind=r.get("kind"))
        enriched.append({**r, "category": name, **facts})
    return per_class, enriched


def residual(train_images: list[dict], diag_train: dict,
             holdout_records: list[dict], holdout_floor: list[dict] | None,
             thresholds: dict, ontology, *, iou_thr: float,
             small_ratio: float = metrics.SMALL_OBJECT_RATIO) -> dict:
    """Structure of residual misses — {"small_ratio", "input_side", "train", "holdout"}.

    train_images = image_rows(train) (GT only), diag_train = the final
    champion's train diagnosis (with detailed fn rows). holdout_records/floor =
    the champion's holdout evaluation records/floor_records, thresholds = its
    tau_test."""
    names = {c.id: c.name for c in ontology.classes}
    thresholds = {int(k): v for k, v in (thresholds or {}).items()}
    tr_per_class, tr_fn = _split_facets(
        train_images, diag_train.get("fn") or [], names, small_ratio=small_ratio)

    raw = metrics.classify_errors(holdout_records, thresholds, iou_thr=iou_thr,
                                  floor_detections=holdout_floor,
                                  small_ratio=small_ratio)
    files = [rec["file"] for rec in holdout_records]
    ho_fn = [{"image": files[r["image_index"]], "box": r["box"],
              "kind": r["kind"], "category_id": r["category_id"]}
             for r in raw["fn"]]
    ho_per_class, _ = _split_facets(holdout_records, ho_fn, names,
                                    small_ratio=small_ratio)
    ho_flags = _label_flags(holdout_records, names)
    return {
        "small_ratio": small_ratio, "input_side": INPUT_SIDE,
        "train": {"n_images": len(train_images), "per_class": tr_per_class,
                  "fn": tr_fn, "label_flags": _label_flags(train_images, names)},
        # holdout: aggregates and counts only — no filenames or boxes
        "holdout": {"n_images": len(holdout_records), "per_class": ho_per_class,
                    "label_flags": {k: len(v) for k, v in ho_flags.items()}},
    }
