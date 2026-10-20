"""Holdout evaluation — τ_c estimation, micro F(β), adoption gate.

τ_c is the **single channel estimated on the holdout** (the explicit exception
to the materials-purity guard — tests/test_materialize_purity.py). The field is
named `tau_test` for artifact compatibility; the split is the holdout.
Adoption and saturation are decided by a paired bootstrap CI over the records
produced here.

The holdout is read only **after** the setting has been fixed by screen
(fit_dev). It changes only τ and the adoption decision, never the materials.
"""

from __future__ import annotations

import logging

from ..core import metrics
from ..fitting.materials import Materials
from .runner import chosen_groups, infer_for_groups, setting_records

log = logging.getLogger(__name__)


def score_setting(cfg, client, split, ontology, setting: dict,
                  materials: Materials, *,
                  return_embeddings: bool = False) -> dict:
    """Infer the setting on `split` → re-estimate thresholds on that split → micro F(β).

    The thresholds carried in per_class (used for fit_dev scoring) are
    **discarded**; F(β)-max thresholds are re-estimated on the target split.
    Applied to the holdout, these are `tau_test`.
    """
    per_class = setting["per_class"]
    if not per_class:
        raise ValueError("empty setting — screen adopted no classes")
    grid = {c["config_id"]: c for c in setting["configs"]}
    image_paths = [split.image_path(e) for e in split.images]
    groups = chosen_groups(per_class)
    passes = infer_for_groups(cfg, client, groups, grid, ontology,
                              image_paths, materials,
                              return_embeddings=return_embeddings)
    floor_recs, _ = setting_records(split, image_paths, passes, groups,
                                    thresholds=None)
    thresholds: dict[int, float] = {}
    for cid in per_class:
        thr, _f1 = metrics.estimate_f_beta_threshold(
            floor_recs, cid, floor=cfg.query.threshold_floor,
            max_candidates=cfg.search.max_threshold_candidates,
            iou_thr=cfg.detector.iou_threshold, beta=cfg.beta,
            tol=cfg.search.tau_tolerance)
        thresholds[cid] = thr
    _, records = setting_records(split, image_paths, passes, groups,
                                 thresholds=thresholds)
    cat_ids = [c.id for c in ontology.classes]
    iou = cfg.detector.iou_threshold
    micro = float(metrics.records_micro_f_beta(records, cat_ids, iou_thr=iou,
                                           beta=cfg.beta))
    per_class_prf = {cid: metrics.class_prf_at(records, cid, 0.0, iou, cfg.beta)
                     for cid in cat_ids}
    return {"thresholds": thresholds, "micro_f_beta": micro,
            "per_class_prf": per_class_prf,
            "records": records, "floor_records": floor_recs}


def evaluate_on_holdout(cfg, client, test_split, ontology, setting: dict,
                     materials: Materials) -> dict:
    """Infer the fixed setting on the holdout (the argument name test_split is
    legacy) → estimate `tau_test` → micro F(β)."""
    out = score_setting(cfg, client, test_split, ontology, setting, materials)
    out["tau_test"] = out.pop("thresholds")
    return out


def micro_f_beta_stat(cat_ids: list[int], iou_thr: float, beta: float = 1.0):
    """Statistic factory for bootstrap_ci — the single metric for adoption and
    saturation (micro F-beta when beta ≠ 1)."""
    def stat(records: list[dict]) -> float:
        return float(metrics.records_micro_f_beta(records, cat_ids, iou_thr=iou_thr,
                                              beta=beta))
    return stat


def gate(champion_records: list[dict], candidate_records: list[dict], *,
         cat_ids: list[int], iou_thr: float, margin: float,
         n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
         beta: float = 1.0) -> dict:
    """Adoption gate — adopt only if the paired-bootstrap one-sided lower bound
    of Δ micro F(β) is ≥ margin and Δ > 0. Otherwise the champion stays, unchanged.

    Both record lists must cover **the same holdout images in the same order**
    (paired) — metrics.bootstrap_ci raises on a length mismatch.
    """
    # One-sided — the gate only asks "Δ > margin". A two-sided bound (alpha/2)
    # halves the effective significance level and rejects real gains.
    out = metrics.bootstrap_ci(champion_records, candidate_records,
                               micro_f_beta_stat(cat_ids, iou_thr, beta),
                               n_boot=n_boot, seed=seed, alpha=alpha,
                               tail="lower")
    # Adopt when the lower bound is **≥** margin and the point estimate is a
    # real improvement. A one-sided lower bound of exactly margin (0) is common:
    # on a small holdout many resamples give both settings the same micro F(β),
    # so the Δ distribution piles up at 0. A strict `>` alone would reject a
    # real gain (Δ>0) and no change (Δ=0) for the same reason, so delta breaks
    # the tie at the boundary.
    adopted = out["lo"] >= margin and out["delta"] > 0
    return {**out, "margin": margin, "adopted": adopted,
            "reason": (f"Δ micro F(β) {out['delta']:+.4f} > 0 and lower bound "
                       f"{out['lo']:+.4f} ≥ margin {margin}"
                       if adopted else
                       f"Δ micro F(β) {out['delta']:+.4f} / lower bound {out['lo']:+.4f} "
                       f"— fails margin {margin}, keeping champion")}
