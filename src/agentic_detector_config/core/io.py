"""Shared loader leaf — jsonl, prompt boxes, prompt safety guard.

Run/split loading lives in core/data.py.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

from .prompts import PromptBox


def data_uri(image: Image.Image) -> str:
    """PIL image -> JPEG base64 data URI (OpenAI multimodal image_url format)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# Minimum prompt-box side length (px) — smaller boxes resize to a 0-pixel
# crop and fail. A safety guard applied always, independent of quality filters.
MIN_PROMPT_PX = 2.0


def promptable(bbox_xyxy) -> bool:
    """Whether this box can be used as a visual prompt (i.e. is croppable)."""
    x1, y1, x2, y2 = bbox_xyxy
    return (x2 - x1) >= MIN_PROMPT_PX and (y2 - y1) >= MIN_PROMPT_PX


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_prompt_boxes(path: Path) -> dict[str, list[PromptBox]]:
    data = json.loads(Path(path).read_text())
    return {name: [PromptBox(class_name=name, image=e["image"], bbox_xyxy=e["bbox_xyxy"])
                   for e in entries]
            for name, entries in data.items()}
