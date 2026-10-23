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
            # **Replace** image_index with the filename — one identifier per
            # image. Attach the category name — per_class is keyed by name, so
            # consumers (planner, VLM descriptions) attribute errors to a class.
            return [{**{k: v for k, v in r.items() if k != "image_index"},
                     "category": names.get(r.get("category_id")),
                     "image": files[r["image_index"]]} for r in rows]

        out["fp"] = _named(raw["fp"])
        out["fn"] = _named(raw["fn"])
    return out


def size_profile(floor_records: list[dict], thresholds: dict[int, float], ontology, *,
                 iou_thr: float,
                 small_ratio: float = metrics.SMALL_OBJECT_RATIO) -> dict:
    """Per-class size ratios r (ascending) of TP and missed_small GT — the material
    for s* and m of the tiling near-miss (planner.gates.tp_floor).

    thresholds are the champion's **fit_dev screen thresholds**
    (setting.per_class[*].threshold), applied directly to the floor records.
    Using the diagnosis classified at tau_test (estimated on holdout) would make
    s* → tiling_shortfall → the fold gate's exception (a screen decision) lean on
    holdout. Detections of classes without a threshold (not adopted by screen)
    are dropped. The returned "thresholds" marks the source."""
    thr = {int(k): float(v) for k, v in thresholds.items()}
    kept = [{**rec, "detections": [d for d in rec["detections"]
                                   if d["category_id"] in thr
                                   and d["score"] >= thr[d["category_id"]]]}
            for rec in floor_records]
    raw = metrics.classify_errors(kept, thr, iou_thr=iou_thr,
                                  floor_detections=floor_records,
                                  small_ratio=small_ratio)
    names = {c.id: c.name for c in ontology.classes}

    def _by_class(rows) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in rows:
            name = names.get(r["category_id"])
            if name is not None:
                out.setdefault(name, []).append(float(r["size_ratio"]))
        return {k: sorted(v) for k, v in sorted(out.items())}
    return {"thresholds": "fit_dev",
            "tp": _by_class(raw["tp"]),
            "missed_small": _by_class(r for r in raw["fn"]
                                      if r["kind"] == "missed_small")}
