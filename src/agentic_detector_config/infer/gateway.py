"""Single path for all detector inference.

All of infer_config (including tiling) goes through this module. Tests block
inference at **one place**, `monkeypatch.setattr(gateway, "infer_config", …)` —
consumers must call it module-qualified as `gateway.infer_config(...)`.

chosen_groups/infer_for_groups (group inference orchestration) live in
eval/runner.py; this module holds only the seam.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import numpy as np

from ..clients.wire import parse_prediction
from ..core import metrics, tiling

BATCH = 8  # images per predict_images call


def _chunks(items, size: int):
    chunk = []
    for it in items:
        chunk.append(it)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _prefetched(items, size: int = BATCH):
    """Image iterable → BATCH-sized lists, with the next chunk pulled on a worker
    thread while the caller runs the current one on the detector.

    Pulling a chunk is CPU work (decoding the image, cutting tile crops); without
    the overlap the GPU waits for it between every call. One thread suffices —
    only one chunk ahead is needed — and keeps the source iterator single-consumer."""
    it = _chunks(items, size)
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(next, it, None)
        while (chunk := nxt.result()) is not None:
            nxt = pool.submit(next, it, None)
            yield chunk


def _tile_pieces(image_paths, tile_cfg: dict, plans: list):
    """Yield [full] + tile crops per image, flat, so pieces fill BATCH across
    image boundaries. Per-image merge geometry is appended to plans as (tiles,
    degenerate, size). The detector runs each image of a call on its own, so
    how pieces are grouped does not change results (same as per-image run_tiled)."""
    for path in image_paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
        tiles, degenerate = tiling.tile_plan(im.width, im.height, tile_cfg)
        plans.append((tiles, degenerate, (im.width, im.height)))
        yield im
        if not degenerate:
            for t in tiles:
                yield im.crop(t)


def _split_by_plan(flat: list, plans: list):
    """Flat piece results → per-image (piece results, plan)."""
    pos = 0
    for plan in plans:
        tiles, degenerate, _ = plan
        n = 1 + (0 if degenerate else len(tiles))
        yield flat[pos:pos + n], plan
        pos += n


def _has_alias_multiplicity(alias_to_name) -> bool:
    """Does any class have 2+ phrases (name + alias)? Triggers duplicate folding."""
    return bool(alias_to_name) and len(alias_to_name) > len(set(alias_to_name.values()))


def dedup_class_duplicates(dets: list[dict], iou_thr: float) -> list[dict]:
    """Fold near-duplicate detections of the same class, highest score first.

    With name + alias phrases, one object can be detected once per phrase, and
    the detector's NMS is per phrase (channel), so they do not suppress each
    other — fold **after** alias_to_name has merged them into one class.
    Greedy by descending score, hence deterministic.
    """
    if len(dets) < 2:
        return dets
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: (-x["score"], x["box"])):
        boxes = [k["box"] for k in kept if k["category_id"] == d["category_id"]]
        if boxes and metrics.iou_matrix(
                np.array([d["box"]], dtype=float),
                np.array(boxes, dtype=float)).max() >= iou_thr:
            continue
        kept.append(d)
    return kept


def make_predictor(ctx, client, config: dict, queries, ontology,
                   alias_to_name=None, return_embeddings: bool = False):
    """Image list → per-image detection lists. The single gate for all detector
    inference; infer_config (including tiling) wraps it.

    neg_eta is passed as the negative-suppression IoU threshold (emulated in
    clients.yoloe). With return_embeddings=True, per-detection embeddings are
    requested and attached under `embedding` (input to the rescorer)."""
    z = ctx.cfg.detector
    neg_kwargs = {}
    if config["visual_negatives"]:
        neg_kwargs = {"neg_eta": config["neg_eta"]}
    if return_embeddings:
        neg_kwargs["return_embeddings"] = True

    def predict(images) -> list[list[dict]]:
        out: list[list[dict]] = []
        for batch in _prefetched(images):
            preds = client.predict_images(batch, queries,
                                          apply_topk=z.apply_topk,
                                          iou_threshold=z.iou_threshold, **neg_kwargs)
            # Class resolution uses text (name) only — labels (flat prompt
            # indices) are unused (see clients.wire.parse_prediction)
            parsed = [parse_prediction(p, ontology.name_to_id,
                                       alias_to_name=alias_to_name) for p in preds]
            if _has_alias_multiplicity(alias_to_name):
                parsed = [dedup_class_duplicates(d, z.iou_threshold) for d in parsed]
            out += parsed
        return out

    return predict


def infer_config(ctx, client, config: dict, queries, ontology,
                 image_paths: list[Path], alias_to_name=None,
                 return_embeddings: bool = False) -> list[list[dict]]:
    """Run one config over images → per-image detection lists
    ({"box","score","category_id","sources"}). Classes come from response text.

    With return_embeddings=True each detection carries `embedding`; under tiling
    merge the dedup/union winner's embedding is kept (the merge copies `{**d}`,
    so the key survives)."""
    predict = make_predictor(ctx, client, config, queries, ontology,
                             alias_to_name=alias_to_name,
                             return_embeddings=return_embeddings)

    def _emb(d: dict) -> dict:
        return {"embedding": d["embedding"]} if d.get("embedding") is not None else {}

    results: list[list[dict]] = []
    if config["tiling"] is None:
        for dets in predict([str(p) for p in image_paths]):
            results.append([{**d, "sources": ["full"]} for d in dets])
    else:
        plans: list = []
        flat = predict(_tile_pieces(image_paths, config["tiling"], plans))
        for piece_dets, (tiles, degenerate, size) in _split_by_plan(flat, plans):
            # the merge compares on the label key → pass category_id as label
            label_form = [[{"box": d["box"], "score": d["score"],
                            "label": d["category_id"], **_emb(d)} for d in dets]
                          for dets in piece_dets]
            merged = tiling.merge_from_results(label_form, tiles, degenerate,
                                               size[0], size[1], config["tiling"])
            results.append([{"box": d["box"], "score": d["score"],
                             "category_id": d["label"], "sources": d["sources"],
                             **_emb(d)}
                            for d in merged])
    return results
