"""screen — score a candidate's config grid on fit_dev and fix its setting.

Grid evaluation, shared config, per-class modality pick. Two boundaries:
(1) The evaluation split is **fit_dev**, not all of train (search runs on
    fit_dev; τ_c, adoption, and saturation on the holdout).
(2) This only **fixes the setting and ranks** candidates — τ_c is
    re-estimated on the holdout (eval.evaluate); the thresholds and F(β) here
    exist only to rank candidates in screen.

union (merging both modalities) is not a selectable option — a single τ_c over
a mixed score distribution breaks F(β) estimation. "The F(β) union would have
reached" is recorded as a diagnostic only.
"""

from __future__ import annotations

import logging

from ..core import metrics
from ..core.metrics import per_image_records
from ..diagnose import ensemble
from ..fitting.materials import Materials
from ..infer import gateway
from ..infer.queries import build_config_queries, config_grid, text_query_bundle
from .runner import cfg_ctx, combined_detections

log = logging.getLogger(__name__)


def select_shared_config(rows: list[dict], grid: list[dict]) -> dict[int, dict]:
    """From the config×class evaluation, pick **one config shared by all
    classes** plus per-class thresholds.

    Picks the config with the largest sum of per-class F(β). Picking per class
    multiplies inference by the number of distinct configs; sharing costs one
    pass regardless of class count. Thresholds are not shared — they are
    applied post hoc at no extra inference cost.
    """
    by_class: dict[int, dict[int, dict]] = {}
    for r in rows:
        by_class.setdefault(r["category_id"], {})[r["config_id"]] = r
    if not by_class:
        return {}
    common = set.intersection(*(set(m) for m in by_class.values()))
    if not common:
        raise ValueError(
            "no common config evaluated for every class — grid evaluation is incomplete "
            f"(evaluated configs per class: { {cid: sorted(m) for cid, m in by_class.items()} })")
    # ties go to the smaller id — config 0 is untiled with negatives off, the cheapest
    pick = max(sorted(common),
               key=lambda i: sum(m[i]["f_beta"] for m in by_class.values()))
    return {cid: m[pick] for cid, m in by_class.items()}


def _estimate_class(cfg, recs, cid):
    return metrics.estimate_f_beta_threshold(
        recs, cid, floor=cfg.query.threshold_floor,
        max_candidates=cfg.search.max_threshold_candidates,
        iou_thr=cfg.detector.iou_threshold, beta=cfg.beta,
        tol=cfg.search.tau_tolerance)


def evaluate_grid(cfg, client, split, ontology, materials: Materials,
                  grid: list[dict]) -> tuple[list[dict], dict[int, list]]:
    """Evaluate every grid config on `split` — (rows, dets per config). One real
    inference pass per config."""
    image_paths = [split.image_path(e) for e in split.images]
    rows: list[dict] = []
    dets_by_config: dict[int, list] = {}
    ctx = cfg_ctx(cfg)
    for config in grid:
        queries, _, _ = build_config_queries(cfg.query, config,
                                                 materials.exemplars,
                                                 materials.visual_negatives)
        dets = gateway.infer_config(ctx, client, config, queries,
                                    ontology, image_paths)
        dets_by_config[config["config_id"]] = dets
        recs = per_image_records(split, image_paths, dets)
        for c in ontology.classes:
            thr, f1 = _estimate_class(cfg, recs, c.id)
            rows.append({"config_id": config["config_id"], "category_id": c.id,
                         "threshold": thr, "f_beta": f1})
    return rows, dets_by_config


def screen_candidate(cfg, client, fit_dev, ontology, materials: Materials, *,
                     neg_eta_grid=(), tiling_variants=()) -> dict:
    """Fix one candidate's setting and score it on fit_dev.

    Returns: {"configs", "per_class": {cid: {config_id, threshold, f1, modality}},
           "fit_dev_micro_f_beta", "modality_diagnostic"}.
    """
    grid = config_grid(neg_eta_grid, tiling_variants)
    image_paths = [fit_dev.image_path(e) for e in fit_dev.images]
    ctx = cfg_ctx(cfg)

    have_semantic = any(materials.exemplars.values())
    rows: list[dict] = []
    dets_by_config: dict[int, list] = {}
    if have_semantic:
        rows, dets_by_config = evaluate_grid(cfg, client, fit_dev, ontology,
                                             materials, grid)
        best = select_shared_config(rows, grid)
    else:
        best = {}
    per_class: dict[int, dict] = {
        cid: {"config_id": r["config_id"], "threshold": r["threshold"],
              "f_beta": r["f_beta"], "modality": "semantic"}
        for cid, r in best.items()}

    # --- text pass + per-class modality pick (one of {semantic, text}) ---
    modality_diagnostic: dict[str, dict] = {}
    text_recs_by_config: dict[int, list] = {}
    text_dets_by_config: dict[int, list] = {}
    if "text" in cfg.search.modalities:
        grid_by_id = {c["config_id"]: c for c in grid}
        tq, _, alias = text_query_bundle(cfg.query, ontology,
                                              materials.aliases,
                                              materials.text_negatives)
        # A text-only candidate (no semantic best) sweeps the whole grid with
        # the text pass — this is the neg_eta sweep path for text negatives.
        # The text pass keeps the neg axis: emulation uses neg_eta as the
        # suppression IoU for text negatives, so the strength screen picks
        # must apply to the text pass too for screen to match eval.
        wanted = (sorted({r["config_id"] for r in best.values()}) if best
                  else sorted(grid_by_id))
        for config_id in wanted:
            txt = gateway.infer_config(
                ctx, client, grid_by_id[config_id], tq,
                ontology, image_paths, alias_to_name=alias)
            text_dets_by_config[config_id] = txt
            text_recs_by_config[config_id] = per_image_records(fit_dev, image_paths, txt)

        for cid, r in list(per_class.items()):
            config_id = r["config_id"]
            text_thr, text_f1 = _estimate_class(cfg, text_recs_by_config[config_id], cid)
            pick, thr, f1 = "semantic", r["threshold"], r["f_beta"]
            if text_f1 > f1:  # ties go to semantic
                pick, thr, f1 = "text", text_thr, text_f1
            per_class[cid] = {"config_id": config_id, "threshold": thr,
                              "f_beta": f1, "modality": pick}
            union_dets = [ensemble.merge_modalities(a, b)
                          for a, b in zip(dets_by_config[config_id],
                                          text_dets_by_config[config_id])]
            union_thr, union_f1 = _estimate_class(
                cfg, per_image_records(fit_dev, image_paths, union_dets), cid)
            modality_diagnostic[str(cid)] = {
                "chosen": pick, "chosen_f_beta": f1,
                "semantic": {"f_beta": r["f_beta"], "threshold": r["threshold"]},
                "text": {"f_beta": text_f1, "threshold": text_thr},
                "union": {"f_beta": union_f1, "threshold": union_thr},
                "union_margin": union_f1 - f1,
            }

        # Rescue classes left out of semantic (zero exemplars, or a text-only
        # candidate) via the text pass.
        if per_class:
            config_ids = [e["config_id"] for e in per_class.values()]
            shared_id = max(set(config_ids), key=config_ids.count)
        else:
            # Shared config for a text-only candidate — apply the semantic rule
            # (max sum of per-class F(β)) to the text scoring rows.
            rows_t = []
            for config_id in wanted:
                for c in ontology.classes:
                    thr, f1 = _estimate_class(cfg, text_recs_by_config[config_id],
                                              c.id)
                    rows_t.append({"config_id": config_id, "category_id": c.id,
                                   "threshold": thr, "f_beta": f1})
            best_t = select_shared_config(rows_t, grid)
            shared_id = (next(iter(best_t.values()))["config_id"] if best_t
                         else 0)
        for c in ontology.classes:
            if c.id in per_class:
                continue
            thr, f1 = _estimate_class(cfg, text_recs_by_config[shared_id], c.id)
            if f1 > 0.0:
                per_class[c.id] = {"config_id": shared_id, "threshold": thr,
                                   "f_beta": f1, "modality": "text"}

    # --- fit_dev micro F(β) of the adopted setting (reuses cached passes — no
    # extra inference) ---
    micro = 0.0
    if per_class:
        groups: dict[tuple[int, str], list[int]] = {}
        for cid, e in per_class.items():
            groups.setdefault((e["config_id"], e["modality"]), []).append(cid)
        passes = {}
        for (config_id, modality) in groups:
            passes[(config_id, modality)] = (
                dets_by_config[config_id] if modality == "semantic"
                else text_dets_by_config[config_id])
        thresholds = {cid: e["threshold"] for cid, e in per_class.items()}
        _, combined = combined_detections(len(image_paths), passes, groups, thresholds)
        recs = per_image_records(fit_dev, image_paths, combined)
        micro = float(metrics.records_micro_f_beta(
            recs, [c.id for c in ontology.classes],
            iou_thr=cfg.detector.iou_threshold, beta=cfg.beta))

    return {"configs": grid, "per_class": per_class,
            "fit_dev_micro_f_beta": micro,
            "modality_diagnostic": modality_diagnostic}
