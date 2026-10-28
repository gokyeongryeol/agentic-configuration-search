"""COCO I/O + Ontology + Split.

All three read the same `_annotations.coco.json`, so they share one file.
Key constraints:
- Roboflow-style dummy supercategories (entries with supercategory=="none"
  that other categories reference as their supercategory) are excluded from
  the ontology.
- Detector responses are mapped to classes by `text` (name)
  (`clients.wire.parse_prediction`).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ANNOTATIONS_FILENAME = "_annotations.coco.json"


# ---------------------------------------------------------------- bbox utils

def xywh_to_xyxy(box) -> list[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ------------------------------------------------------------------ Ontology

@dataclass
class ClassDef:
    id: int
    name: str
    slug: str


class Ontology:
    """Class list + id<->name<->slug mappings."""

    def __init__(self, classes: list[ClassDef]):
        self.classes = sorted(classes, key=lambda c: c.id)
        self.id_to_name = {c.id: c.name for c in self.classes}
        self.name_to_id = {c.name: c.id for c in self.classes}
        self.id_to_slug = {c.id: c.slug for c in self.classes}

    @classmethod
    def from_categories(cls, categories: list[dict]) -> "Ontology":
        """Exclude Roboflow-style dummy supercategories.

        Rule: an entry is a dummy if supercategory=="none" and another category
        references it as its supercategory. If there is none, keep everything.
        """
        referenced = {c.get("supercategory") for c in categories}
        kept = []
        for c in categories:
            if c.get("supercategory") == "none" and c["name"] in referenced:
                log.info("excluding dummy supercategory from the ontology: id=%s name=%r", c["id"], c["name"])
                continue
            kept.append(ClassDef(id=int(c["id"]), name=str(c["name"]), slug=slugify(str(c["name"]))))
        if not kept:
            raise ValueError("ontology is empty — check categories")
        return cls(kept)

    def to_dict(self) -> dict:
        return {"classes": [{"id": c.id, "name": c.name, "slug": c.slug} for c in self.classes]}

    @classmethod
    def from_dict(cls, data: dict) -> "Ontology":
        return cls([ClassDef(**c) for c in data["classes"]])


# --------------------------------------------------------------------- Split

@dataclass
class Annotation:
    id: int
    image_id: int
    category_id: int
    bbox_xyxy: list[float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class ImageEntry:
    id: int
    file_name: str
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)


class Split:
    """One COCO split (train / valid / test)."""

    def __init__(self, root: Path, raw: dict, ontology: Ontology):
        self.root = Path(root)
        self.raw = raw
        self.ontology = ontology
        self.images: list[ImageEntry] = []
        by_id: dict[int, ImageEntry] = {}
        for im in raw.get("images", []):
            entry = ImageEntry(id=int(im["id"]), file_name=str(im["file_name"]),
                               width=int(im["width"]), height=int(im["height"]))
            self.images.append(entry)
            by_id[entry.id] = entry
        self.images.sort(key=lambda e: e.file_name)  # deterministic order
        valid_cats = set(ontology.id_to_name)
        for ann in raw.get("annotations", []):
            img = by_id.get(int(ann["image_id"]))
            cat = int(ann["category_id"])
            if img is None or cat not in valid_cats:
                continue
            bbox = xywh_to_xyxy(ann["bbox"])
            img.annotations.append(Annotation(id=int(ann["id"]), image_id=img.id,
                                              category_id=cat, bbox_xyxy=bbox))

    @classmethod
    def load(cls, root: Path, ontology: Optional[Ontology] = None) -> "Split":
        path = Path(root) / ANNOTATIONS_FILENAME
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        onto = ontology or Ontology.from_categories(raw.get("categories", []))
        return cls(Path(root), raw, onto)

    def image_path(self, entry: ImageEntry) -> Path:
        return self.root / entry.file_name

    def annotations_by_class(self) -> dict[int, list[Annotation]]:
        out: dict[int, list[Annotation]] = {cid: [] for cid in self.ontology.id_to_name}
        for img in self.images:
            for ann in img.annotations:
                out[ann.category_id].append(ann)
        return out

    def area_stats(self) -> dict[int, dict]:
        """Per-class GT area stats — for the rescorer feature area_z."""
        stats = {}
        for cid, anns in self.annotations_by_class().items():
            areas = np.array([a.area for a in anns], dtype=np.float64)
            if len(areas) == 0:
                stats[cid] = {"count": 0, "mean": 0.0, "std": 0.0}
            else:
                stats[cid] = {"count": int(len(areas)),
                              "mean": float(areas.mean()),
                              "std": float(areas.std())}
        return stats


def crop_bounds(size, bbox_xyxy, padding: float = 1.0) -> tuple[int, int, int, int]:
    """Region `crop_image` cuts out, as (left, top, right, bottom).

    Separate function because drawing the original bbox on the crop requires
    knowing the crop's origin.
    """
    width, height = size
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * padding, (y2 - y1) / 2 * padding
    left = max(0, int(round(cx - hw)))
    top = max(0, int(round(cy - hh)))
    right = min(width, int(round(cx + hw)))
    bottom = min(height, int(round(cy + hh)))
    if right <= left or bottom <= top:  # guard against degenerate bboxes
        right, bottom = min(width, left + 1), min(height, top + 1)
    return left, top, right, bottom


def crop_image(image: Image.Image, bbox_xyxy, padding: float = 1.0) -> Image.Image:
    """Crop a bbox (expanded by the padding factor, clamped to image bounds)."""
    return image.crop(crop_bounds(image.size, bbox_xyxy, padding))


MIN_VLM_CROP_PX = 96     # lower bound on the short side of a crop sent to the VLM —
                         # error and GT crops are often tens of px, too small as is
                         # for the VLM to make out the shape


def upscale_for_vlm(crop: Image.Image, min_px: int = MIN_VLM_CROP_PX) -> Image.Image:
    """Upscale keeping the aspect ratio when the short side is below min_px — shared by every crop sent to a VLM."""
    w, h = crop.size
    short = min(w, h)
    if short <= 0 or short >= min_px:
        return crop
    scale = min_px / short
    return crop.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                       Image.LANCZOS)
