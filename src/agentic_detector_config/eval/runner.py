"""Grouped inference for an adopted setting — the runner shared by screen
(fit_dev) and holdout evaluation.

All inference goes through gateway (the seam convention).
"""

from __future__ import annotations

from pathlib import Path

from ..core.metrics import per_image_records
from ..fitting.materials import Materials
from ..infer import gateway
from ..infer.queries import build_config_queries, text_query_bundle


def chosen_groups(per_class: dict[int, dict]) -> dict[tuple[int, str], list[int]]:
    """Group per_class picks as (config_id, modality) → [category_id...].

    modality is only semantic | text — union is not selectable (a single τ_c
    over a mixed score distribution). It is recorded as a diagnostic only:
    eval.screen."""
    groups: dict[tuple[int, str], list[int]] = {}
    for cid, chosen in per_class.items():
        modality = chosen["modality"]
        if modality not in ("semantic", "text"):
            raise ValueError(f"unknown modality {modality!r} (class {cid})")
        groups.setdefault((chosen["config_id"], modality), []).append(int(cid))
    return groups


def infer_for_groups(cfg, client, groups: dict[tuple[int, str], list[int]],
                     grid: dict[int, dict], ontology, image_paths: list[Path],
                     materials: Materials,
                     return_embeddings: bool = False) -> dict[tuple[int, str], list[list[dict]]]:
    """Detections per (config_id, modality) group; each base pass runs once.
    Holdout evaluation and every re-inference of an adopted setting go through
    this function, so they share the exact same pass construction."""
    emb_kw = {"return_embeddings": True} if return_embeddings else {}
    base: dict[tuple[int, str], list[list[dict]]] = {}
    for config_id, modality in sorted(groups):
        config = grid[config_id]
        if modality == "semantic":
            queries, _, _ = build_config_queries(cfg.query, config,
                                                     materials.exemplars,
                                                     materials.visual_negatives)
            base[(config_id, modality)] = gateway.infer_config(
                cfg_ctx(cfg), client, config, queries, ontology,
                image_paths, **emb_kw)
        else:
            tq, _, alias = text_query_bundle(cfg.query, ontology,
                                                  materials.aliases,
                                                  materials.text_negatives)
            # The text pass keeps the neg axis — neg_eta is the suppression IoU
            # for text negatives (same convention as eval.screen's text pass).
            base[(config_id, modality)] = gateway.infer_config(
                cfg_ctx(cfg), client, config, tq, ontology,
                image_paths, alias_to_name=alias, **emb_kw)
    return base


class _Ctx:
    """Minimal ctx that gateway expects — just cfg."""

    def __init__(self, cfg):
        self.cfg = cfg


def cfg_ctx(cfg) -> _Ctx:
    return _Ctx(cfg)


def combined_detections(image_count: int, passes: dict, groups: dict,
                        thresholds: dict[int, float] | None = None
                        ) -> tuple[list[list[dict]], list[list[dict]]]:
    """Merge group passes by class assignment → (all floor dets, thresholded).

    With thresholds=None the second result equals the floor. The floor set
    feeds τ estimation and the FN (below_threshold) diagnosis."""
    floor_all: list[list[dict]] = [[] for _ in range(image_count)]
    combined: list[list[dict]] = [[] for _ in range(image_count)]
    for key in sorted(groups):
        cids = set(groups[key])
        for i, dets in enumerate(passes[key]):
            for d in dets:
                if d["category_id"] not in cids:
                    continue  # keep only classes that adopted this (config, modality)
                floor_all[i].append(d)
                if thresholds is None or d["score"] >= thresholds[d["category_id"]]:
                    combined[i].append(d)
    return floor_all, combined


def setting_records(split, image_paths, passes, groups,
                    thresholds: dict[int, float] | None):
    """(floor records, thresholded records) — for metrics."""
    floor_all, combined = combined_detections(len(image_paths), passes, groups,
                                              thresholds)
    return (per_image_records(split, image_paths, floor_all),
            per_image_records(split, image_paths, combined))
