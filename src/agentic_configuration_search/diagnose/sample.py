"""Select representative FP/FN samples for the prompt.

Diagnosis detail can run to thousands of rows, too many for the prompt. The
small sample chosen here is **every individual error the planner sees**, and
VLM crop descriptions target the same sample (diagnose.describe) — if they
diverged, notes would describe detections absent from the prompt or some
samples would lack notes.

It lives in diagnose rather than planner because of layering: describe
(diagnose) needs it, and diagnose cannot import planner.
"""

from __future__ import annotations

MAX_DETAIL = 12          # samples per split·side (sized for 2–3 active kinds)
KIND_CAP = 4             # per-kind cap — beyond 4, samples just repeat the aggregate
IMAGE_CAP = 2            # per-image cap within a kind — keeps one pathological image from dominating


def severity(r: dict) -> float:
    """Sort key — FP by descending score, FN by descending box area (both
    returned negated).

    Severity differs per side: a high-scoring FP does more harm, and missing a
    large GT does more harm."""
    if "score" in r:
        return -float(r["score"])
    b = r.get("box") or (0.0, 0.0, 0.0, 0.0)
    return -(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def pick_detail(detail: list[dict]) -> list[dict]:
    """Pick representative samples — round-robin over kinds (cause coverage) →
    round-robin over classes within a kind (avoid skew) → severity order, with
    IMAGE_CAP per image and KIND_CAP per kind. Fully deterministic (tie-break:
    severity → filename → box)."""
    per_kind: dict = {}
    for r in detail:
        per_kind.setdefault(r.get("kind"), []).append(r)
    queues: dict = {}
    for kind, rows in per_kind.items():
        rows = sorted(rows, key=lambda r: (severity(r), str(r.get("image", "")),
                                           tuple(r.get("box") or ())))
        by_class: dict = {}
        class_order = []                 # starting with the class of the most severe row
        for r in rows:
            cid = r.get("category_id")
            if cid not in by_class:
                class_order.append(cid)
                by_class[cid] = []
            by_class[cid].append(r)
        picked: list[dict] = []
        img_count: dict = {}
        progressed = True
        while len(picked) < KIND_CAP and progressed:
            progressed = False
            for cid in class_order:
                while by_class[cid]:
                    r = by_class[cid].pop(0)
                    img = str(r.get("image", ""))
                    if img_count.get(img, 0) >= IMAGE_CAP:
                        continue         # drop rows past the image cap
                    img_count[img] = img_count.get(img, 0) + 1
                    picked.append(r)
                    progressed = True
                    break
                if len(picked) >= KIND_CAP:
                    break
        queues[kind] = picked
    sample: list[dict] = []
    while len(sample) < MAX_DETAIL and any(queues.values()):
        for k in sorted(queues):
            if queues[k] and len(sample) < MAX_DETAIL:
                sample.append(queues[k].pop(0))
    return sample
