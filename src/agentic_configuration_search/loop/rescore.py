"""Round wiring for the rescorer axis.

The rescorer is a post-processing layer that sits on top of any setting: under
the setting fixed by the candidate's other axes (axes.rescorer=True), run
inference on fit_train/fit_dev with embeddings, fit the discriminator
(fit_train) and α grid (fit_dev); that candidate's fit_dev scoring and holdout
evaluation then **re-estimate thresholds on the rescored scores**.

area_stats is computed on **all of train** on every path — fit and apply must
see the same feature distribution, and using holdout GT statistics would break
material purity.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..core import metrics
from ..core.metrics import per_image_records
from ..eval.runner import chosen_groups, infer_for_groups
from ..eval.screen import count_rows
from ..fitting.materials import Materials
from ..fitting.rescorer import Rescorer, _apply_rescorer, fit_rescorer, refit_alphas

log = logging.getLogger(__name__)


def preds_dict(split, passes: dict, groups: dict, prefix: str) -> dict:
    """Group passes -> preds keyed by det_id (the data shape rescorer fit/apply use).

    det_id includes (config, modality) — semantic and text groups of the same
    config can coexist, so file name + index alone is not unique.
    """
    preds: dict[str, dict] = {}
    for key in sorted(groups):
        config_id, modality = key
        cids = set(groups[key])
        for entry, dets in zip(split.images, passes[key]):
            for k, d in enumerate(dets):
                if d["category_id"] not in cids:
                    continue
                det_id = f"{prefix}:{config_id}:{modality}:{entry.file_name}:{k}"
                preds[det_id] = {"det_id": det_id, "image": entry.file_name,
                                 "box": list(d["box"]), "score": float(d["score"]),
                                 "category_id": d["category_id"],
                                 "embedding": d.get("embedding"),
                                 "sources": d.get("sources", ["full"])}
    return preds


def dets_by_image(split, preds: dict) -> list[list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in preds.values():
        by.setdefault(r["image"], []).append(
            {"box": r["box"], "score": r["score"],
             "category_id": r["category_id"], "sources": r["sources"]})
    return [by.get(e.file_name, []) for e in split.images]


def _tiled_ids(setting: dict) -> set[int]:
    return {c["config_id"] for c in setting["configs"] if c["tiling"] is not None}


def split_preds(cfg, client, ds, setting, materials: Materials, split, prefix):
    grid = {c["config_id"]: c for c in setting["configs"]}
    groups = chosen_groups(setting["per_class"])
    paths = [split.image_path(e) for e in split.images]
    passes = infer_for_groups(cfg, client, groups, grid, ds.ontology, paths,
                              materials, return_embeddings=True)
    return preds_dict(split, passes, groups, prefix), paths


def fit_for_setting(cfg, client, ds, setting: dict, materials: Materials,
                    out_dir: Path):
    """Fit discriminator (fit_train) and α (fit_dev) under `setting` —
    (rescorer|None, info, dev_preds). dev_preds is reused for scoring.
    info["wall_s"] holds seconds per stage — inference (fit_train, fit_dev, with
    embeddings) and the discriminator fit and α selection (absent when the
    discriminator is skipped)."""
    area_stats = ds.train.area_stats()
    t0 = time.monotonic()
    train_preds, _ = split_preds(cfg, client, ds, setting, materials,
                                 ds.fit_train, "tr")
    t1 = time.monotonic()
    dev_preds, dev_paths = split_preds(cfg, client, ds, setting, materials,
                                       ds.fit_dev, "dv")
    wall = {"infer_train": round(t1 - t0, 2),
            "infer_dev": round(time.monotonic() - t1, 2)}
    rescorer, info = fit_rescorer(
        cfg.rescorer, train_preds=train_preds, train_split=ds.fit_train,
        dev_preds=dev_preds, dev_split=ds.fit_dev, dev_image_paths=dev_paths,
        ontology=ds.ontology, dataset_name=ds.root.name,
        tiled_ids=_tiled_ids(setting), iou_thr=cfg.detector.iou_threshold,
        threshold_floor=cfg.query.threshold_floor,
        max_threshold_candidates=cfg.search.max_threshold_candidates,
        beta=cfg.beta, area_stats=area_stats)
    info["wall_s"] = wall | info.pop("fit_s", {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescorer_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str))
    if rescorer is not None:
        rescorer.save(out_dir / "rescorer.json")
    return rescorer, info, dev_preds


def champion_scores(cfg, ds, setting: dict, champion: Rescorer, dev_preds: dict,
                    dev_split) -> dict:
    """Two micro F(β) of the candidate's fit_dev predictions under the champion's
    discriminator — a record of how scoring without refitting the discriminator
    would have gone (never used for a decision).

    champion_rescorer: the champion's discriminator and α as they are.
    champion_disc_alpha: the champion's discriminator with α re-picked on this
    candidate's fit_dev."""
    paths = [dev_split.image_path(e) for e in dev_split.images]
    realpha = refit_alphas(
        cfg.rescorer, champion.discriminator, dev_preds=dev_preds, dev_split=dev_split,
        dev_image_paths=paths, tiled_ids=_tiled_ids(setting),
        iou_thr=cfg.detector.iou_threshold, threshold_floor=cfg.query.threshold_floor,
        max_threshold_candidates=cfg.search.max_threshold_candidates, beta=cfg.beta,
        area_stats=ds.train.area_stats())
    return {name: rescored_split_score(cfg, ds, setting, r, dev_preds,
                                       dev_split)["micro_f_beta"]
            for name, r in (("champion_rescorer", champion),
                            ("champion_disc_alpha", realpha))}


def apply_to_preds(cfg, ds, setting, rescorer, preds: dict) -> dict:
    preds = {k: dict(v) for k, v in preds.items()}
    _apply_rescorer(rescorer, preds, ds.train, _tiled_ids(setting),
                    area_stats=ds.train.area_stats())
    return preds


def rescored_split_score(cfg, ds, setting: dict, rescorer, preds: dict,
                         split) -> dict:
    """Re-estimate thresholds on rescored scores -> micro (same shape as
    eval.evaluate.score_setting, but reuses preds instead of re-running inference)."""
    preds = apply_to_preds(cfg, ds, setting, rescorer, preds)
    paths = [split.image_path(e) for e in split.images]
    floor = dets_by_image(split, preds)
    floor_recs = per_image_records(split, paths, floor)
    thresholds: dict[int, float] = {}
    per_class = {}
    for cid, e in setting["per_class"].items():
        thr, f1 = metrics.estimate_f_beta_threshold(
            floor_recs, cid, floor=cfg.query.threshold_floor,
            max_candidates=cfg.search.max_threshold_candidates,
            iou_thr=cfg.detector.iou_threshold, beta=cfg.beta,
            tol=cfg.search.tau_tolerance)
        thresholds[cid] = thr
        per_class[cid] = {**e, "threshold": thr, "f_beta": f1}
    # thresholds are estimated only for classes in per_class — the rest stay at floor
    kept = [[d for d in dets
             if d["category_id"] not in thresholds
             or d["score"] >= thresholds[d["category_id"]]]
            for dets in floor]
    records = per_image_records(split, paths, kept)
    cat_ids = [c.id for c in ds.ontology.classes]
    iou = cfg.detector.iou_threshold
    micro = float(metrics.records_micro_f_beta(records, cat_ids, iou_thr=iou,
                                           beta=cfg.beta))
    return {"per_class": per_class, "thresholds": thresholds,
            "micro_f_beta": micro, "records": records, "floor_records": floor_recs,
            "per_class_prf": {cid: metrics.class_prf_at(records, cid, 0.0, iou)
                              for cid in cat_ids}}


def rescored_screen(cfg, ds, setting: dict, rescorer, dev_preds: dict) -> dict:
    """The setting with the candidate's fit_dev scoring updated to rescored scores.

    tiling_f (the tiled/untiled comparison and gap_ub) stays the pre-rescoring
    screen's — that screen is why the tile config was picked. Dropping it would
    lose tiling_f from the history row of an adopted tiling+rescorer candidate."""
    out = rescored_split_score(cfg, ds, setting, rescorer, dev_preds, ds.fit_dev)
    # fit_dev_counts is the rescored (served) channel, and the pre-rescorer channel
    # is kept as _raw — the lost-value prescreen of a candidate inheriting the
    # rescorer compares pre-rescorer channels **before** the refit
    # (loop.runner._prescreen)
    raw = setting.get("fit_dev_counts_raw") or setting.get("fit_dev_counts")
    counts = ({"fit_dev_counts": count_rows(out["records"],
                                            [c.id for c in ds.ontology.classes],
                                            cfg.detector.iou_threshold)}
              if out.get("records") is not None else {})
    return {"configs": setting["configs"], "per_class": out["per_class"],
            "fit_dev_micro_f_beta": out["micro_f_beta"],
            "modality_diagnostic": setting.get("modality_diagnostic", {})} \
        | counts | ({"fit_dev_counts_raw": raw} if raw is not None else {}) \
        | ({"tiling_f": setting["tiling_f"]} if setting.get("tiling_f") else {})


def rescored_evaluate(cfg, client, ds, setting: dict, materials: Materials,
                      rescorer) -> dict:
    """Holdout evaluation — estimates tau_test (legacy name; holdout thresholds) on rescored scores."""
    hold_preds, _ = split_preds(cfg, client, ds, setting, materials,
                                ds.holdout, "ho")
    out = rescored_split_score(cfg, ds, setting, rescorer, hold_preds, ds.holdout)
    return {"tau_test": out["thresholds"], "micro_f_beta": out["micro_f_beta"],
            "per_class_prf": out["per_class_prf"], "records": out["records"],
            "floor_records": out["floor_records"]}
