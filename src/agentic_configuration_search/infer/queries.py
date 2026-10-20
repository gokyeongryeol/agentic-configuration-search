"""Config grid and query-bundle assembly.

The assembly rules live here in one place so screen (fit_dev) and test
evaluation use **identical query construction** — if construction diverged
between paths, thresholds and decisions would drift apart.

Adopted aliases and phrases arrive as **materials arguments** (material merging
and manual-first precedence are already handled by fitting.mine).
"""

from __future__ import annotations

from typing import Optional

from ..core.prompts import build_queries, build_text_queries


def config_grid(neg_eta_grid=(), tiling_variants=()) -> list[dict]:
    """(negatives off + N neg_eta values) × (tiling off + M variants).

    Tiling-off is always in the grid, so disabling an axis leaves the decision
    path unchanged. Which axes are on is set by the candidate setting — an
    empty sequence means that axis is off.
    """
    neg_variants = [{"visual_negatives": False, "neg_eta": None}]
    neg_variants += [{"visual_negatives": True, "neg_eta": float(e)} for e in neg_eta_grid]
    tile_variants = [None, *tiling_variants]
    grid = []
    for nv in neg_variants:
        for tv in tile_variants:
            grid.append({"config_id": len(grid), **nv, "tiling": tv})
    return grid


def build_config_queries(qcfg, config: dict, exemplars, visual_negatives):
    """Semantic query construction depends only on negatives on/off — unified in
    this one function so screen and holdout evaluation build identically.
    qcfg = Config.query."""
    return build_queries(
        exemplars,
        visual_negatives if config["visual_negatives"] else None,
        box_threshold=qcfg.box_threshold,
        multimodal_threshold=qcfg.threshold_floor,
    )


def text_query_bundle(qcfg, ontology, aliases: Optional[dict[str, str]] = None,
                      text_negatives: Optional[dict[str, list[str]]] = None):
    """Text-modality query bundle: (queries, prompt_order, alias_to_name).

    `aliases` is the already-merged final set (manual first). Empty dict/None =
    class names only."""
    return build_text_queries(
        [c.name for c in ontology.classes],
        dict(aliases or {}),
        box_threshold=qcfg.box_threshold,
        multimodal_threshold=qcfg.threshold_floor,
        visual_negatives=(text_negatives or None) or None,
    )
