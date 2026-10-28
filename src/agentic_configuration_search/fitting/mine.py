"""Prompt material mining — **reads train only**.

**No adoption gate here** — adoption is decided by the loop's screen (fit_dev)
+ holdout CI gate. The only filters left here are hygiene rules, not
measurements — the promptable guard, FN pre-filters (headroom, miss rate,
size), negative hygiene (score floor, IoM), and the alias cross-pool filter.

All GPU dependencies sit behind injection points:
- `encode_fn(cid, crop_paths) -> np.ndarray` — crop embeddings (real impl: embedder.encode_crops)
- `caption_fn(rows) -> {id: [caption, ...]}` — crop captions (real impl: infer.phrases.vlm_caption)
- `refine_alias_fn` / `refine_negatives_fn` — vLLM refinement (real impl: infer.phrases)
Tests swap in fakes for all of them.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..core import metrics
from ..core.coco import Annotation, Ontology, Split, crop_image
from ..core.io import MIN_PROMPT_PX, promptable
from ..core.prompts import PromptBox
from ..infer import embedder, phrases

log = logging.getLogger(__name__)


# ------------------------------------------------------------- GT crop cache

class CropCache:
    """Crop files and embedding-row positions for croppable GTs.

    Files are named by **annotation id** (`a{id}.jpg`) — keying by list position
    would make the cache return another annotation's crop when the candidate
    set changes.
    """

    def __init__(self, train: Split, ontology: Ontology, pool_dir: Path):
        self.pool_dir = Path(pool_dir)
        img_by_id = {e.id: e for e in train.images}
        self.prompt_pool: dict[int, list[Annotation]] = {}
        self.paths: dict[int, list[Path]] = {}
        self.row_of: dict[int, dict[int, int]] = {}   # cid -> {ann.id: embedding row}
        degenerate: dict[str, int] = {}
        for cid, anns in sorted(train.annotations_by_class().items()):
            ok = [a for a in anns if promptable(a.bbox_xyxy)]
            if len(ok) != len(anns):
                degenerate[ontology.id_to_name[cid]] = len(anns) - len(ok)
            self.prompt_pool[cid] = ok
            slug = ontology.id_to_slug[cid]
            (self.pool_dir / slug).mkdir(parents=True, exist_ok=True)
            paths = []
            for ann in ok:
                p = self.pool_dir / slug / f"a{ann.id}.jpg"
                if not p.exists():
                    entry = img_by_id[ann.image_id]
                    with Image.open(train.image_path(entry)) as im:
                        crop_image(im.convert("RGB"), ann.bbox_xyxy).save(p, quality=92)
                paths.append(p)
            self.paths[cid] = paths
            self.row_of[cid] = {a.id: j for j, a in enumerate(ok)}
        if degenerate:
            log.warning("excluding exemplar candidate: GT too small to crop (side < %.0fpx) %s",
                        MIN_PROMPT_PX, degenerate)


def make_encoder(cache: CropCache, *, repo_root: Path, model: str, device: str,
                 log_dir: Path, slug_of: dict[int, str]):
    """Real encode_fn — embeds each class once (memoized). GPU path."""
    memo: dict[int, np.ndarray] = {}

    def encode_fn(cid: int, crop_paths) -> np.ndarray:
        if cid not in memo:
            memo[cid] = embedder.encode_crops(
                repo_root, list(crop_paths), model, device,
                log_dir / f"embed_{slug_of[cid]}.log")
        return memo[cid]

    return encode_fn


# ------------------------------------------------------------ GT exemplar

def gt_exemplars(train: Split, ontology: Ontology, cache: CropCache, params,
                 encode_fn, rng) -> tuple[dict[str, list[PromptBox]], dict[int, set[int]]]:
    """Per-class exemplar selection — kcenter/mean (embedding-based) or random.

    Returns (exemplars, chosen_ann_ids) — the latter prevents duplicates in FN mining.
    """
    img_by_id = {e.id: e for e in train.images}
    exemplars: dict[str, list[PromptBox]] = {}
    chosen_ann_ids: dict[int, set[int]] = {}
    prefetch = getattr(encode_fn, "prefetch", None)
    if prefetch is not None and params.exemplar_selection != "random":
        # all classes before the per-class calls — the real implementation embeds
        # c1..cN in one worker run, and encode_fn(cid, …) below reads the cache
        # (same crops, same result)
        prefetch({cid: cache.paths[cid]
                  for cid, anns in sorted(cache.prompt_pool.items()) if anns})
    for cid, anns in sorted(cache.prompt_pool.items()):
        name = ontology.id_to_name[cid]
        k = min(params.exemplar_base_per_class, len(anns))
        if not anns:
            exemplars[name] = []
            log.warning("class %r: no promptable train GT — 0 exemplars", name)
            continue
        if params.exemplar_selection == "random":
            idx = embedder.select_indices("random", k, len(anns), rng=rng)
        else:
            emb = encode_fn(cid, cache.paths[cid])
            idx = embedder.select_indices(params.exemplar_selection, k, len(anns),
                                          embeddings=emb,
                                          outlier_mad=params.kcenter_outlier_mad)
        exemplars[name] = [
            PromptBox(class_name=name,
                      image=str(train.image_path(img_by_id[anns[j].image_id])),
                      bbox_xyxy=anns[j].bbox_xyxy)
            for j in idx]
        chosen_ann_ids[cid] = {anns[j].id for j in idx}
    return exemplars, chosen_ann_ids


# -------------------------------------------------------- FP/FN pool mining

def mine_pools(train: Split, ontology: Ontology, dets_by_image: list, *,
               iou_thr: float, params) -> tuple[dict, dict, dict, dict]:
    """Match one pass's detections to GT -> (FP pool, FN anns, IoM-excluded, uncroppable-excluded).

    Matching uses **all detections of the pass** — applying
    `visual_negative_min_score` before matching would drop detections that hit a
    GT below that score, counting actually-detected GTs as FN. The score floor is
    a negative-candidate hygiene rule, so it applies only to the FP branch.
    """
    fp_pool: dict[str, list] = {ontology.id_to_name[c.id]: [] for c in ontology.classes}
    fn_anns: dict[int, list] = {cid: [] for cid in ontology.id_to_name}
    skipped: dict[str, int] = {}          # excluded for overlapping a same-class object (IoM)
    unpromptable: dict[str, int] = {}     # excluded as uncroppable (side < MIN_PROMPT_PX)
    for entry, dets in zip(train.images, dets_by_image):
        for cid in ontology.id_to_name:
            c_dets = [d for d in dets if d["category_id"] == cid]
            gt_anns = [a for a in entry.annotations if a.category_id == cid]
            gt_np = np.array([a.bbox_xyxy for a in gt_anns], dtype=float).reshape(-1, 4)
            _, fps, fns = metrics.match_greedy(
                np.array([d["box"] for d in c_dets], dtype=float).reshape(-1, 4),
                np.array([d["score"] for d in c_dets], dtype=float),
                gt_np, iou_thr)
            for i in fps:
                name = ontology.id_to_name[cid]
                if c_dets[i]["score"] < params.visual_negative_min_score:
                    continue              # negative-candidate score floor (FP branch only)
                if not promptable(c_dets[i]["box"]):
                    unpromptable[name] = unpromptable.get(name, 0) + 1
                    continue
                # Even if judged FP by IoU<thr, a box containing or contained by a
                # same-class object is not background — as a negative it would
                # suppress the class itself
                if (params.visual_negative_exclude_gt_iom > 0 and len(gt_np)
                        and metrics.iom_matrix(
                            np.array([c_dets[i]["box"]], dtype=float), gt_np).max()
                        >= params.visual_negative_exclude_gt_iom):
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                fp_pool[name].append(
                    {"image": str(train.image_path(entry)),
                     "score": c_dets[i]["score"],
                     "bbox_xyxy": c_dets[i]["box"], "source": "fp_mining"})
            for j in fns:
                fn_anns[cid].append(gt_anns[j])
    if unpromptable:
        log.info("negative mining: excluding uncroppable candidates %s", unpromptable)
    return fp_pool, fn_anns, skipped, unpromptable


def background_only(fp_pool: dict[str, list], train: Split, ontology: Ontology,
                    *, iom_thr: float = 0.3) -> dict[str, list]:
    """Only FPs that **overlap no GT of any class** — true background.

    mine_pools' IoM exclusion only checks same-class GT. An FP on another class's
    object is a misclassification, not background, so it must be excluded from
    the source of background-suppression phrases.
    """
    gt_by_image: dict[str, np.ndarray] = {}
    for entry in train.images:
        boxes = [a.bbox_xyxy for a in entry.annotations]
        gt_by_image[str(train.image_path(entry))] = \
            np.array(boxes, dtype=float).reshape(-1, 4)
    out: dict[str, list] = {}
    for name, pool in fp_pool.items():
        keep = []
        for e in pool:
            gt = gt_by_image.get(e["image"])
            if gt is None or not len(gt):
                keep.append(e)
                continue
            iom = metrics.iom_matrix(
                np.array([e["bbox_xyxy"]], dtype=float), gt).max()
            if iom < iom_thr:
                keep.append(e)
        out[name] = keep
    return out


def other_class_gt(cache: "CropCache", ontology: Ontology, train: Split,
                   per_class: int) -> dict[str, list]:
    """Per class, **GT crops of the other classes** — source of confuser-suppression phrases.

    Draws evenly from each other class (if one class dominated the pool, the
    phrases would suppress only that class). Deterministic selection in
    annotation-id order.
    """
    img_by_id = {e.id: e for e in train.images}
    by_name: dict[str, list] = {}
    for cid, anns in sorted(cache.prompt_pool.items()):
        name = ontology.id_to_name[cid]
        by_name[name] = [
            {"image": str(train.image_path(img_by_id[a.image_id])),
             "score": 1.0, "bbox_xyxy": list(a.bbox_xyxy), "source": "gt_crop"}
            for a in sorted(anns, key=lambda a: a.id)]
    out: dict[str, list] = {}
    for name in by_name:
        others = [(n, rows) for n, rows in sorted(by_name.items())
                  if n != name and rows]
        picked: list = []
        i = 0
        while len(picked) < per_class and others:
            progressed = False
            for _n, rows in others:
                if i < len(rows) and len(picked) < per_class:
                    picked.append(rows[i])
                    progressed = True
            if not progressed:
                break
            i += 1
        out[name] = picked
    return out


def top_negatives(fp_pool: dict[str, list], per_class: int) -> dict[str, list]:
    """Top per_class by descending score — the negative pool."""
    return {name: sorted(pool, key=lambda r: -r["score"])[:per_class]
            for name, pool in fp_pool.items()}


# ------------------------------------------------------ FN exemplar candidates

def fn_exemplar_candidates(train: Split, ontology: Ontology, fn_anns: dict,
                           cache: CropCache, chosen_ann_ids: dict, params,
                           encode_fn, rng) -> tuple[dict[str, list[PromptBox]], dict]:
    """Missed GTs (FN) from the base pass -> extra exemplar candidates — the recall
    channel symmetric to FP->negative mining (GT, so zero mislabel risk). GTs
    already used as exemplars are excluded.

    Three pre-filters (all hygiene — excluding harmful classes is the loop gate's job):
    - headroom: skip if recoverable misses are fewer than fn_min_candidates.
    - miss-rate floor: skip classes with essentially no holes.
    - size: cut tiny candidates below median GT area × ratio (if only tiny
      candidates remain and they can fill the quota, refill largest first).
    """
    img_by_id = {e.id: e for e in train.images}
    anns_by_cid = train.annotations_by_class()      # miss rate and median use all GT
    fn_boxes: dict[str, list[PromptBox]] = {}
    report: dict[str, dict] = {}
    if params.exemplar_fn_per_class <= 0:
        return fn_boxes, report
    for cid, missed in sorted(fn_anns.items()):
        name = ontology.id_to_name[cid]
        total_gt = len(anns_by_cid.get(cid, []))
        if not missed or not total_gt:
            continue
        miss_rate = len(missed) / total_gt
        report[name] = {"miss_rate": round(miss_rate, 4),
                        "candidates": len(missed), "added": 0}
        if len(missed) < params.fn_min_candidates:
            report[name]["gate_skipped"] = "headroom"
            continue
        if miss_rate < params.fn_min_miss_rate:
            report[name]["gate_skipped"] = "miss_rate"
            continue
        min_area = params.fn_min_area_ratio * float(
            np.median([a.area for a in anns_by_cid[cid]]))
        pool = [a for a in missed
                if a.id not in chosen_ann_ids.get(cid, set())
                and promptable(a.bbox_xyxy)]
        cand = [a for a in pool if a.area >= min_area]
        small = [a for a in pool if a.area < min_area]
        if small:
            report[name]["tiny_removed"] = len(small)
        if not cand and len(small) >= params.exemplar_fn_per_class:
            cand = sorted(small, key=lambda a: a.area,
                          reverse=True)[:params.exemplar_fn_per_class]
            report[name]["area_refilled"] = len(cand)
        if not cand:
            continue
        k = min(params.exemplar_fn_per_class, len(cand))
        if len(cand) > k:
            if params.exemplar_selection == "random":
                pick = rng.sample(range(len(cand)), k)
            else:
                # kcenter to cover the missed appearances as widely as possible.
                # The FN set concentrates truncation, occlusion, and label errors,
                # so it is noisier than base; trim lightly (fn_kcenter_outlier_mad) —
                # keep diversity but drop extremes that cannot serve as prompts.
                emb = encode_fn(cid, cache.paths[cid])
                cand_emb = emb[[cache.row_of[cid][a.id] for a in cand]]
                pick = embedder.kcenter_greedy(cand_emb, k,
                                               outlier_mad=params.fn_kcenter_outlier_mad)
        else:
            pick = list(range(len(cand)))
        fn_boxes[name] = [
            PromptBox(class_name=name,
                      image=str(train.image_path(img_by_id[cand[i].image_id])),
                      bbox_xyxy=cand[i].bbox_xyxy)
            for i in pick]
        report[name]["added"] = len(pick)
    return fn_boxes, report


# ------------------------------------------------------- captions -> phrase candidates

def caption_rows(boxes_by_name: dict[str, list]) -> list[dict]:
    """Manifest rows for caption_fn — accepts both PromptBox and negative dicts."""
    rows = []
    for name, entries in sorted(boxes_by_name.items()):
        for i, e in enumerate(entries):
            image = e.image if isinstance(e, PromptBox) else e["image"]
            bbox = e.bbox_xyxy if isinstance(e, PromptBox) else e["bbox_xyxy"]
            rows.append({"id": f"{name}#{i}", "image": str(image),
                         "bbox_xyxy": list(bbox)})
    return rows


def collect_caption_pools(boxes_by_name: dict[str, list], caption_fn, *,
                          normalize: bool) -> dict[str, list[str]]:
    """Crop caption pools. Alias mining uses normalize=True (for aggregation and
    cross-pool comparison); negative mining uses False (refine normalizes at
    aggregation — idempotent, so same result, and the report keeps the raw text)."""
    rows = caption_rows(boxes_by_name)
    if not rows:
        return {}
    caps_by_id = caption_fn(rows)
    pools: dict[str, list[str]] = {}
    for name, entries in sorted(boxes_by_name.items()):
        caps = [c for i in range(len(entries))
                for c in caps_by_id.get(f"{name}#{i}", [])]
        if normalize:
            caps = [p for c in caps if (p := phrases.normalize_caption(c))]
        pools[name] = caps
    return pools


def alias_candidates(pools: dict[str, list[str]], class_names: list[str],
                     manual: dict[str, str], refine_fn) -> tuple[dict[str, str], dict]:
    """Normalized caption pools -> per-class alias **candidates** (adoption is the loop gate's call).

    - Cross-pool filter: a phrase also produced by another class's crops has no
      discriminative power — empirical confusion evidence name comparison
      cannot see, so it is filtered deterministically before the LLM.
    - Classes with a manual alias are not mined (manual always wins), but their
      pools still serve as contrast evidence when judging other classes.
    - refine_fn(name, class_names, pool, others) -> {"alias", "reason"} —
      verbatim and conflict checks are already done by phrases.refine_alias.
    """
    report: dict = {"pools": {}, "rejected": {}, "cross_dropped": {}}
    aliases: dict[str, str] = {}
    taken = {a.lower() for a in manual.values()}
    for name in sorted(pools):
        if name in manual:
            report["rejected"][name] = "manual alias takes precedence"
            continue
        pool = pools.get(name, [])
        if not pool:
            continue
        report["pools"][name] = dict(Counter(pool).most_common())
        others = {n: p for n, p in pools.items() if n != name and p}
        foreign = {c for p in others.values() for c in p}
        cross = sorted(set(pool) & foreign)
        if cross:
            report["cross_dropped"][name] = {
                c: sorted(n for n, p in others.items() if c in p) for c in cross}
            pool = [c for c in pool if c not in foreign]
        if not pool:
            report["rejected"][name] = "candidates exhausted by cross-pool conflicts"
            continue
        refined = refine_fn(name, class_names, pool, others)
        alias = refined.get("alias")
        if not alias:
            report["rejected"][name] = refined.get("reason", "")
            continue
        if alias.lower() in taken or alias in class_names:
            report["rejected"][name] = f"{alias!r}: conflicts with another class's prompt"
            continue
        taken.add(alias.lower())
        aliases[name] = alias
    return aliases, report


def text_negative_candidates(pools: dict[str, list[str]], class_names: list[str],
                             refine_fn, max_keep: int) -> tuple[dict[str, list[str]], dict]:
    """Negative-crop caption pools -> per-class suppression phrase **candidates** (adoption is the loop gate's call).

    refine_fn(name, class_names, pool, max_keep) -> {"keep", "rejected"} —
    safety filters (rules 1-4) are applied by phrases.refine_negatives.
    """
    report: dict = {"pools": {}, "keep": {}, "rejected": {}}
    mined: dict[str, list[str]] = {}
    for name in sorted(pools):
        pool = pools[name]
        if not pool:
            continue
        report["pools"][name] = dict(Counter(pool).most_common())
        refined = refine_fn(name, class_names, pool, max_keep)
        report["keep"][name] = refined.get("keep", [])
        report["rejected"][name] = refined.get("rejected", [])
        if refined.get("keep"):
            mined[name] = refined["keep"]
    return mined, report
