"""Detector query builders — consumed by clients.yoloe.parse_queries.

Two format pitfalls:
- A query with zero positives cannot form a channel -> never build a
  negatives-only combination.
- If every prompt's `box` is empty the query is treated as text modality ->
  classes with zero exemplars are left out of semantic queries.

prompt_order (first appearance of each category) is the ordering contract of
query construction. Class resolution of responses is by `text` (name) alone,
so it does not depend on this order (`clients.wire.parse_prediction`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PromptBox:
    """One exemplar/negative for a query: which box in which image."""
    class_name: str
    image: str            # image path
    bbox_xyxy: list


def build_queries(
    exemplars: dict[str, list[PromptBox]],
    visual_negatives: Optional[dict[str, list[PromptBox]]],
    box_threshold: float,
    multimodal_threshold: float,
) -> tuple[list[dict], list[str], list[str]]:
    """Return (queries, prompt_order, excluded_classes).

    - exemplars: class_name -> list of PromptBox. Empty classes are excluded with a warning.
    - visual_negatives: None means negatives off — the query structure itself changes.
      Negatives of a class with no positives are dropped (no matching positive).
    - prompt_order: first-appearance order of category text over the query
      traversal (query order -> prompt order, negatives excluded).
    """
    active = {name: boxes for name, boxes in exemplars.items() if boxes}
    excluded = sorted(set(exemplars) - set(active))
    if excluded:
        log.warning("classes with 0 exemplars %s — excluded from the query (prevents falling back to the text modality)", excluded)
    if not active:
        raise ValueError("no positive exemplars at all — cannot build a query")

    # One query per image. Deterministic order: iterate exemplars by sorted class
    # name (to fix first-appearance order); images in order of appearance.
    image_order: list[str] = []
    per_image_prompts: dict[str, list[dict]] = {}

    def _image_slot(image: str) -> list[dict]:
        if image not in per_image_prompts:
            per_image_prompts[image] = []
            image_order.append(image)
        return per_image_prompts[image]

    for name in sorted(active):
        for box in active[name]:
            _image_slot(box.image).append({
                "text": name,
                "box": [float(v) for v in box.bbox_xyxy],
                "box_threshold": float(box_threshold),
                "multimodal_threshold": float(multimodal_threshold),
            })

    if visual_negatives:
        for name in sorted(visual_negatives):
            if name not in active:
                log.warning("class %r: dropped %d negatives — no matching positive",
                            name, len(visual_negatives[name]))
                continue
            for box in visual_negatives[name]:
                _image_slot(box.image).append({
                    "text": name,
                    "box": [float(v) for v in box.bbox_xyxy],
                    "negative_target": name,
                    "box_threshold": float(box_threshold),
                    "multimodal_threshold": float(multimodal_threshold),
                })

    queries = [{"prompt_image": image, "prompts": per_image_prompts[image]}
               for image in image_order]

    # compute prompt_order in query traversal order
    prompt_order: list[str] = []
    for q in queries:
        for p in q["prompts"]:
            if p.get("negative_target"):
                continue
            if p["text"] not in prompt_order:
                prompt_order.append(p["text"])
    return queries, prompt_order, excluded


def build_text_queries(
    class_names: list[str],
    aliases: dict[str, str],
    box_threshold: float,
    multimodal_threshold: float,
    visual_negatives: Optional[dict[str, list[str]]] = None,
) -> tuple[list[dict], list[str], dict[str, str]]:
    """Text-modality queries. Returns (queries, prompt_order, alias_to_name).

    - Every prompt's `box` must be an empty list to be treated as text modality
      (clients.yoloe.parse_queries). The box key itself must be present.
    - No `prompt_image` — the text branch does not read it.
    - Aliases are **added alongside** the class-name prompt — the detector
      treats each phrase as its own channel, so name and alias are independent
      channels. Two phrases of the same class can detect the same object twice,
      and the detector's NMS is per phrase with no cross-suppression —
      **infer.gateway folds within-class duplicates when alias_to_name has
      multiplicity**.
    - Each negative is **replicated with every positive phrase of the class
      (name + aliases) as target** — attaching it to the name only would let
      alias channels bypass per-target suppression (clients.yoloe folds by
      phrase, so the replication is harmless).
    - The detector echoes our phrase back as the response `text`, so the
      reverse map `alias_to_name` (phrase -> original class name, including the
      name itself) is returned too. Without passing it to `parse_prediction`,
      ontology lookup fails.
    - prompt_order is by **original class name** (one per class) — class
      resolution goes solely through response `text`, so no alignment with a
      flat index is needed.
    """
    if not class_names:
        raise ValueError("class_names is empty — cannot build a text query")
    prompts = []
    prompt_order: list[str] = []
    alias_to_name: dict[str, str] = {name: name for name in class_names}

    def _prompt(text: str) -> dict:
        return {"text": text, "box": [],
                "box_threshold": float(box_threshold),
                "multimodal_threshold": float(multimodal_threshold)}

    for name in class_names:
        prompt_order.append(name)
        prompts.append(_prompt(name))
        phrase = str(aliases.get(name) or "").strip()
        if not phrase or phrase == name:
            continue                         # identity alias — the name prompt suffices
        if phrase in alias_to_name and alias_to_name[phrase] != name:
            raise ValueError(f"text alias collision: {phrase!r} maps to both "
                             f"{alias_to_name[phrase]!r} and {name!r}")
        alias_to_name[phrase] = name
        prompts.append(_prompt(phrase))
    # Text negatives — suppress by phrase; the `negative_target` key marks a negative.
    # Symmetric with semantic negatives under the same hygiene rule: negatives of a
    # class with no positives are dropped (nothing to suppress).
    for name in sorted(visual_negatives or {}):
        # all positive phrases of this class — name first, then aliases (insertion order)
        targets = [p for p, cls in alias_to_name.items() if cls == name]
        if not targets:
            log.warning("class %r: dropped %d text negatives — no matching positive",
                        name, len(visual_negatives[name]))
            continue
        for phrase in visual_negatives[name]:
            phrase = str(phrase).strip()
            if not phrase or phrase in alias_to_name:
                # a phrase equal to a positive phrase would self-suppress — drop it
                if phrase in alias_to_name:
                    log.warning("dropped text negative %r — identical to a positive phrase", phrase)
                continue
            for target in targets:
                prompts.append({
                    "text": phrase,
                    "box": [],
                    "negative_target": target,
                    "box_threshold": float(box_threshold),
                    "multimodal_threshold": float(multimodal_threshold),
                })
    return [{"prompts": prompts}], prompt_order, alias_to_name
