"""Single path for all detector inference.

All of infer_config (including tiling) goes through this module. Tests block
inference at **one place**, `monkeypatch.setattr(gateway, "infer_config", …)` —
consumers must call it module-qualified as `gateway.infer_config(...)`.

chosen_groups/infer_for_groups (group inference orchestration) live in
eval/runner.py; this module holds only the seam.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageOps

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
        # opened the way the detector opens a path (Pillow + EXIF orientation,
        # Ultralytics' handling of a source list) — tiles and merge geometry
        # share the frame of the untiled pass and of the full piece below
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        tiles, degenerate = tiling.tile_plan(im.width, im.height, tile_cfg)
        plans.append((tiles, degenerate, (im.width, im.height)))
        # the full image goes in as the file path, the same input as the
        # untiled pass — the detector gets the same pixels, and the pass cache
        # reuses the untiled inference
        yield str(path)
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


# ---------------------------------------------------------------- pass cache
# Reuses inference for the same input under the same request conditions. Exact
# only while the detector runs each image of a call on its own (the YOLOE
# backend predicts with batch=1); batched forwards could shift scores with the
# batch composition. Keys hash what the detector actually receives (file bytes
# for a path, the pixel buffer for an in-memory image), so a changed file under
# the same path is a different key. When the detector state (weights, artifact)
# changes, invalidate_pass_cache() empties it — the runner wraps the
# detector_hooks controls. Embedding requests are not cached.
PASS_CACHE_MAX = 8192            # entry cap (pieces) — responses kept as JSON strings
_pass_cache: OrderedDict | None = None


@contextmanager
def pass_cache(max_entries: int = PASS_CACHE_MAX):
    """Reuse infer_config results inside this block (outside it, nothing changes)."""
    global _pass_cache, PASS_CACHE_MAX
    if max_entries < 0:
        raise ValueError(f"pass_cache max_entries must be >= 0 (got {max_entries})")
    prev, prev_max = _pass_cache, PASS_CACHE_MAX
    _pass_cache, PASS_CACHE_MAX = OrderedDict(), max_entries
    try:
        yield
    finally:
        _pass_cache, PASS_CACHE_MAX = prev, prev_max


def invalidate_pass_cache() -> None:
    """The detector state changed — the same input is a different inference."""
    if _pass_cache is not None:
        _pass_cache.clear()


def _content_key(image) -> str:
    """Hash of what the detector receives for one image argument."""
    h = hashlib.sha256()
    if isinstance(image, Image.Image):
        h.update(f"pil:{image.mode}:{image.size}:".encode())
        h.update(image.tobytes())
    else:
        h.update(b"file:")
        h.update(Path(image).read_bytes())
    return h.hexdigest()


def _request_signature(queries, kw: dict) -> str | None:
    """Hash of the request conditions — the whole argument bundle passed to the
    detector (kw) and the queries, with exemplar prompt images counted by
    content. None if something does not serialize to JSON (not cached — so no
    repr with a memory address ends up in a key)."""
    try:
        enc = [{**q, "prompt_image": _content_key(q["prompt_image"])}
               if q.get("prompt_image") is not None else q for q in (queries or [])]
        blob = json.dumps([enc, kw], sort_keys=True)
    except (TypeError, OSError):
        return None
    return hashlib.sha256(blob.encode()).hexdigest()


def _cached_predict_images(client, batch, queries, sig, **kw) -> list:
    """Send only the pieces of a batch missing from the cache (a duplicate within
    the batch once).

    This batch's responses are gathered in a local dict first, and entries over
    the cap are evicted at the end — evicting while storing could drop a hit of
    the same batch and raise KeyError on read."""
    cache = _pass_cache
    keys = [sig + _content_key(im) for im in batch]
    got = {}
    for k in dict.fromkeys(keys):
        if k in cache:
            cache.move_to_end(k)
            got[k] = cache[k]
    todo = [k for k in dict.fromkeys(keys) if k not in got]
    if todo:
        first = {k: im for k, im in zip(reversed(keys), reversed(batch))}
        for k, p in zip(todo, client.predict_images([first[k] for k in todo],
                                                    queries, **kw)):
            got[k] = cache[k] = json.dumps(p)
    while len(cache) > PASS_CACHE_MAX:
        cache.popitem(last=False)
    return [json.loads(got[k]) for k in keys]


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

    # the argument bundle sent to the detector — the cache key is built from it
    # too (so a new argument cannot be left out of the key)
    kw = dict(apply_topk=z.apply_topk, iou_threshold=z.iou_threshold, **neg_kwargs)
    sig = (_request_signature(queries, kw)
           if _pass_cache is not None and not return_embeddings else None)

    def predict(images) -> list[list[dict]]:
        out: list[list[dict]] = []
        for batch in _prefetched(images):
            if sig is not None and _pass_cache is not None:
                preds = _cached_predict_images(client, batch, queries, sig, **kw)
            else:
                preds = client.predict_images(batch, queries, **kw)
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
