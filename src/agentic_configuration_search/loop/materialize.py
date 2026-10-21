"""Material generation for candidate settings — **reads only dataset.train**
(material-purity invariant).

No holdout/test argument is accepted at all — that materials cannot depend on
held-out data is enforced at the signature level, and
tests/test_materialize_purity.py re-checks it via output fingerprints.

Every GPU dependency is an injection point: encode_fn (embedding), caption_fn
(VLM captions), refine_* (vLLM refinement), client (train pass for FN/negative
mining). None raises an explicit error — silently disabling an axis is the
worst failure mode.

Materials for the rescorer, lora and crop_classifier axes (discriminator,
weights, artifact) are not built here — they need inference passes and
training subprocesses, so they belong to the runner's round orchestration.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from pathlib import Path

from ..core.prompts import PromptBox
from ..fitting import mine
from ..fitting.materials import Materials
from ..infer import gateway
from ..infer.queries import build_config_queries
from ..planner.axes import validate_axes

log = logging.getLogger(__name__)

# Mining passes always use the safe-point config (no negatives, no tiling, floor threshold)
_MINE_CONFIG = {"config_id": -1, "visual_negatives": False, "neg_eta": None, "tiling": None}


class _Ctx:
    def __init__(self, cfg):
        self.cfg = cfg


def _require(value, name: str, axis: str):
    if value is None:
        raise ValueError(f"axes.{axis} requires {name}, but it was not injected "
                         "(refusing to silently disable the axis)")
    return value


# Material kind -> the axes keys that determine it. If all these keys equal the
# parent's, the parent's material is inherited as-is instead of re-mined.
_INHERIT_KEYS: dict[str, tuple] = {
    # exemplars also bind fn_k — fn_arm mining needs the base selection
    # (chosen_ann_ids), which is lost if only the base is inherited
    "exemplars": ("exemplar", "exemplar_k", "fn_k"),
    "visual_negatives": ("visual_negatives", "visual_negative_k", "exemplar",
                         "exemplar_k", "fn_k"),
    "aliases": ("alias",),
    "text_negatives": ("text_negatives", "text_negative_k", "exemplar",
                       "exemplar_k", "fn_k"),
}


def inherit_plan(axes: dict, parent_axes: dict | None) -> set[str]:
    """Material kinds to inherit unchanged from the parent.

    The champion's score was achieved **with those materials**. Re-mining an
    inherited axis can yield different materials for the same axes, and then
    the Δ the gate measures is "increment + material swap" rather than a pure
    increment. If refreshing materials helps, that too must be proposed as an
    explicit increment and judged by the gate.

    negative/text_negative also depend on exemplars — the FP pool comes from
    the train pass of the serving configuration, so it changes when exemplars
    change.
    """
    if not parent_axes:
        return set()
    return {kind for kind, keys in _INHERIT_KEYS.items()
            if all(axes.get(k) == parent_axes.get(k) for k in keys)}


def materialize(cfg, dataset, axes: dict, out_dir: Path, *, client=None,
                parent=None, parent_axes: dict | None = None,
                encode_fn=None, caption_fn=None, vlm_caption_fn=None,
                refine_alias_fn=None,
                refine_negatives_fn=None, seed: int = 0) -> Materials:
    """Mine materials for the axes enabled in `axes` from train; save and return Materials."""
    axes = validate_axes(axes)
    train, onto = dataset.train, dataset.ontology
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # Quantity axes (None = config default) override the mining quotas
    quota = {}
    if axes.get("exemplar_k") is not None:
        quota["exemplars_per_class"] = int(axes["exemplar_k"])
    if axes.get("fn_k") is not None:
        quota["fn_exemplars_per_class"] = int(axes["fn_k"])
    if axes.get("visual_negative_k") is not None:
        quota["visual_negatives_per_class"] = int(axes["visual_negative_k"])
    if axes.get("text_negative_k") is not None:
        quota["text_negatives_max_per_class"] = int(axes["text_negative_k"])
    params = dataclasses.replace(cfg.mine, **quota) if quota else cfg.mine
    iou = cfg.detector.iou_threshold
    image_paths = [train.image_path(e) for e in train.images]
    class_names = [c.name for c in onto.classes]
    reports: dict = {}
    # Inherited axes take the parent's materials as-is — so re-mining cannot perturb the champion's score
    inherited = inherit_plan(axes, parent_axes) if parent else set()
    parent_mat = Materials.load(parent) if inherited else None
    if inherited:
        reports["parent"] = str(parent)
        reports["inherited"] = sorted(inherited)

    def _train_pass(ex):
        c = _require(client, "client", "exemplar/negative")
        queries, _, _ = build_config_queries(cfg.query, _MINE_CONFIG, ex, None)
        return gateway.infer_config(_Ctx(cfg), c, _MINE_CONFIG, queries,
                                    onto, image_paths)

    # --- visual exemplar (base → fn_arm) ---
    exemplars: dict[str, list[PromptBox]] = {}
    chosen: dict[int, set[int]] = {}
    cache = None
    gt_selection: dict[str, list[PromptBox]] | None = None  # mining source (not served)

    def _need_encoder(axis):
        return (_require(encode_fn, "encode_fn", axis)
                if params.exemplar_selection != "random" else encode_fn)

    def _gt_selection(axis) -> dict[str, list[PromptBox]]:
        """Mining source for candidates without served exemplars — GT k-center
        selection. Not stored in Materials (a text-only candidate serves text only)."""
        nonlocal cache, gt_selection
        if gt_selection is None:
            cache = cache or mine.CropCache(train, onto, out_dir / "gt_crops")
            gt_selection, _ = mine.gt_exemplars(train, onto, cache, params,
                                                _need_encoder(axis), rng)
        return {n: bs for n, bs in gt_selection.items() if bs}

    if "exemplars" in inherited:
        # exemplar=none while the parent's materials hold exemplars means a
        # relabelled champion (screen did not use exemplar) — screen runs the
        # semantic pass whenever exemplars exist, so inheriting them would split
        # the label from what is served
        if axes["exemplar"] != "none":
            exemplars = dict(parent_mat.exemplars)
    elif axes["exemplar"] != "none":
        cache = mine.CropCache(train, onto, out_dir / "gt_crops")
        exemplars, chosen = mine.gt_exemplars(train, onto, cache, params,
                                              _need_encoder("exemplar"), rng)
    if axes["exemplar"] == "fn_arm" and "exemplars" not in inherited:
        base_dets = _train_pass(exemplars)
        _fp, fn_anns, _sk, _up = mine.mine_pools(train, onto, base_dets,
                                                 iou_thr=iou, params=params)
        fn_boxes, fn_report = mine.fn_exemplar_candidates(
            train, onto, fn_anns, cache, chosen, params,
            encode_fn, rng)
        for name, boxes in fn_boxes.items():
            exemplars.setdefault(name, []).extend(boxes)
        reports["fn_mining"] = fn_report

    # --- negative pool (material for the visual axis or text_negatives) ---
    neg_entries: dict[str, list[dict]] = {}
    fp_pool_all: dict[str, list[dict]] = {}
    # the gt_crop source needs no FP pass — it uses GT crops only
    need_fp = ((axes["visual_negatives"] and "visual_negatives" not in inherited)
               or (axes["text_negatives"] in ("fp_crop", "background")
                   and "text_negatives" not in inherited))
    if need_fp:
        # FP mining uses **the pass of the configuration to be served
        # (including adopted FN exemplars)**. A text-only candidate without
        # exemplars mines with a GT k-center selection as exemplars — an
        # approximation that differs from the served configuration (text pass).
        src_exemplars = exemplars
        if not any(src_exemplars.values()):
            src_exemplars = _gt_selection("negative/text_negatives")
        pass_dets = _train_pass(src_exemplars)
        fp_pool_all, _fn, skipped, _up = mine.mine_pools(train, onto, pass_dets,
                                                         iou_thr=iou, params=params)
        neg_entries = mine.top_negatives(fp_pool_all,
                                         params.visual_negatives_per_class)
        reports["negatives_excluded_on_object"] = skipped

    visual_negatives: dict[str, list[PromptBox]] = {}
    if "visual_negatives" in inherited:
        visual_negatives = dict(parent_mat.visual_negatives)
    elif axes["visual_negatives"]:
        visual_negatives = {name: [PromptBox(class_name=name, image=e["image"],
                                      bbox_xyxy=e["bbox_xyxy"]) for e in entries]
                     for name, entries in neg_entries.items() if entries}

    # --- text alias ---
    aliases: dict[str, str] = dict(params.text_aliases)
    if "aliases" in inherited:
        aliases = {**parent_mat.aliases, **params.text_aliases}
    elif axes["alias"]:
        source = {n: bs for n, bs in exemplars.items() if bs}
        if not source:
            # alias for a text-only candidate: no served exemplars, but a GT
            # selection is still needed as the caption source (captioner
            # output = vocabulary sample for text prompts)
            source = _gt_selection("alias")
        # the crop->text converter is chosen by the axes value (currently only vlm)
        cap = _require(vlm_caption_fn, "vlm_caption_fn", f"alias={axes['alias']}")
        reports["alias_captioner"] = axes["alias"]
        pools = mine.collect_caption_pools(source, cap, normalize=True)
        generated, alias_report = mine.alias_candidates(
            pools, class_names, params.text_aliases,
            _require(refine_alias_fn, "refine_alias_fn", "alias"))
        aliases = {**generated, **params.text_aliases}   # manual always wins
        reports["alias"] = alias_report

    # --- text negative phrases ---
    # The source is chosen by the axes value (fp_crop | gt_crop | background):
    #   fp_crop    regions mistaken for the target — suppress the false detections themselves
    #   gt_crop    GT crops of other classes — suppress confused classes
    #   background FPs overlapping no GT of any class — suppress background false detections
    text_negatives: dict[str, list[str]] = {}
    src = axes["text_negatives"]
    if "text_negatives" in inherited:
        text_negatives = dict(parent_mat.text_negatives)
    elif src:
        if src == "gt_crop":
            # a text-only candidate without exemplars has no cache yet
            cache = cache or mine.CropCache(train, onto, out_dir / "gt_crops")
            tn_src = mine.other_class_gt(cache, onto, train,
                                         params.text_negatives_max_per_class * 3)
        elif src == "background":
            bg = mine.background_only(fp_pool_all, train, onto)
            tn_src = mine.top_negatives(bg, params.text_negatives_max_per_class * 3)
        else:
            tn_src = neg_entries
        reports["text_negative_source"] = {
            "kind": src, "per_class": {n: len(v) for n, v in tn_src.items() if v}}
        pools = mine.collect_caption_pools(
            {n: es for n, es in tn_src.items() if es},
            _require(caption_fn, "caption_fn", "text_negatives"), normalize=False)
        text_negatives, tn_report = mine.text_negative_candidates(
            pools, class_names,
            _require(refine_negatives_fn, "refine_negatives_fn", "text_negatives"),
            params.text_negatives_max_per_class)
        reports["text_negatives"] = tn_report

    materials = Materials(exemplars=exemplars, visual_negatives=visual_negatives,
                          aliases=aliases, text_negatives=text_negatives)
    materials.save(out_dir)
    (out_dir / "mine_reports.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    return materials
