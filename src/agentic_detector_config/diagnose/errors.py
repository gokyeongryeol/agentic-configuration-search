"""FP/FN diagnosis — classification by cause and per-class aggregation.

core.metrics.classify_errors (a tested leaf) does the classification; this
module aggregates it into the shape the planner consumes. Train diagnosis
includes detailed rows with image filenames; **holdout diagnosis emits only
anonymized aggregates** — structurally preventing the planner from making
proposals overfit to individual holdout images.
"""

from __future__ import annotations

from ..core import metrics


def diagnose(records: list[dict], floor_records: list[dict] | None,
             thresholds: dict[int, float], ontology, *, iou_thr: float,
             anonymize: bool,
             small_ratio: float = metrics.SMALL_OBJECT_RATIO) -> dict:
    """records/floor_records use the metrics.per_image_records schema (as
    produced by evaluate_on_holdout / screen). thresholds = the applied τ_c.
    small_ratio = the missed_small cut (object size relative to the image).

    Returns: {"n_images", "summary": {fp, fn}, "per_class": {name: {...}},
              plus "fp"/"fn" detail lists (with filenames) only when
              anonymize=False}.
    """
    raw = metrics.classify_errors(records, thresholds, iou_thr=iou_thr,
                                  floor_detections=floor_records,
                                  small_ratio=small_ratio)
    per_class: dict[str, dict] = {}
    gt_counts: dict[int, int] = {}
    for rec in records:
        for g in rec["gts"]:
            gt_counts[g["category_id"]] = gt_counts.get(g["category_id"], 0) + 1
    for c in ontology.classes:
        fp_kinds: dict[str, int] = {}
        for r in raw["fp"]:
            if r["category_id"] == c.id:
                fp_kinds[r["kind"]] = fp_kinds.get(r["kind"], 0) + 1
        fn_kinds: dict[str, int] = {}
        for r in raw["fn"]:
            if r["category_id"] == c.id:
                fn_kinds[r["kind"]] = fn_kinds.get(r["kind"], 0) + 1
        n_gt = gt_counts.get(c.id, 0)
        n_fn = sum(fn_kinds.values())
        per_class[c.name] = {
            "gt": n_gt, "fp": fp_kinds, "fn": fn_kinds,
            "miss_rate": round(n_fn / n_gt, 4) if n_gt else 0.0,
        }
    out = {"n_images": len(records),
           "summary": {"fp": raw["fp_summary"], "fn": raw["fn_summary"]},
           "per_class": per_class,
           "anonymized": anonymize}
    if not anonymize:
        files = [rec["file"] for rec in records]
        names = {c.id: c.name for c in ontology.classes}

        def _named(rows: list[dict]) -> list[dict]:
            # **Replace** image_index with the filename — carrying both gives two
            # identifiers for one image, and since the prompt substitutes image
            # with an alias (planner.llm), the numbers would disagree too.
            #
            # Attach the category name — per_class is keyed by name but detail
            # rows carry only the id, so consumers (planner, VLM descriptions)
            # could not attribute individual errors to a class.
            return [{**{k: v for k, v in r.items() if k != "image_index"},
                     "category": names.get(r.get("category_id")),
                     "image": files[r["image_index"]]} for r in rows]

        out["fp"] = _named(raw["fp"])
        out["fn"] = _named(raw["fn"])
    return out
