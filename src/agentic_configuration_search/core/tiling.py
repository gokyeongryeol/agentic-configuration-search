"""In-house SAHI: tile split / offset / merge.

The detector has no tiled inference, so we build it here.
Internal detection representation is a dict:
    {"box": [x1,y1,x2,y2], "score": float, "label": int, "sources": ["full"|"tile", ...]}
`label` is the category_id — the caller (gateway) converts it at parse time.
"""

from __future__ import annotations

import math

import numpy as np

NMS_IOU = 0.5
# tolerance (px) for a box edge to count as "touching" a tile seam
EDGE_EPS = 2.0


def offset_detections(dets: list[dict], dx: float, dy: float) -> list[dict]:
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        out.append({**d, "box": [x1 + dx, y1 + dy, x2 + dx, y2 + dy]})
    return out


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    # Make the denominator safe first, as in core.metrics.iou_matrix —
    # np.where evaluates both branches, so zero-area pairs would warn on 0/0
    return np.where(union > 0, inter / np.clip(union, 1e-9, None), 0.0)


def interior_seams(tiles, width: int, height: int) -> tuple[set, set]:
    """Tile boundaries lying **inside the image** (x seams, y seams).

    Image borders are excluded — a box touching them was not cut by tiling.
    """
    seams_x, seams_y = set(), set()
    for x1, y1, x2, y2 in tiles:
        for v in (x1, x2):
            if EDGE_EPS < v < width - EDGE_EPS:
                seams_x.add(float(v))
        for v in (y1, y2):
            if EDGE_EPS < v < height - EDGE_EPS:
                seams_y.add(float(v))
    return seams_x, seams_y


def _seam_adjacent(a, b, seams_x, seams_y) -> bool:
    """Whether a and b abut the same interior seam **from opposite sides**.

    This defines "two pieces of one object cut by a tile boundary". Plain
    containment would also swallow **distinct instances** inside a large box.
    """
    for v in seams_x:
        a_ends, a_starts = abs(a[2] - v) <= EDGE_EPS, abs(a[0] - v) <= EDGE_EPS
        b_ends, b_starts = abs(b[2] - v) <= EDGE_EPS, abs(b[0] - v) <= EDGE_EPS
        if ((a_ends and b_starts) or (a_starts and b_ends)) and \
                min(a[3], b[3]) - max(a[1], b[1]) > 0:  # overlap vertically
            return True
    for v in seams_y:
        a_ends, a_starts = abs(a[3] - v) <= EDGE_EPS, abs(a[1] - v) <= EDGE_EPS
        b_ends, b_starts = abs(b[3] - v) <= EDGE_EPS, abs(b[1] - v) <= EDGE_EPS
        if ((a_ends and b_starts) or (a_starts and b_ends)) and \
                min(a[2], b[2]) - max(a[0], b[0]) > 0:  # overlap horizontally
            return True
    return False


def _seam_adjacent_many(a: np.ndarray, b, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """_seam_adjacent for every row of a at once — (N,) bool. Same rule."""
    out = np.zeros(len(a), bool)
    if len(sx):
        a_ends = np.abs(a[:, 2:3] - sx) <= EDGE_EPS          # (N, S)
        a_starts = np.abs(a[:, 0:1] - sx) <= EDGE_EPS
        b_ends = np.abs(b[2] - sx) <= EDGE_EPS                # (S,)
        b_starts = np.abs(b[0] - sx) <= EDGE_EPS
        touch = (a_ends & b_starts) | (a_starts & b_ends)
        overlap = (np.minimum(a[:, 3], b[3]) - np.maximum(a[:, 1], b[1])) > 0
        out |= touch.any(axis=1) & overlap
    if len(sy):
        a_ends = np.abs(a[:, 3:4] - sy) <= EDGE_EPS
        a_starts = np.abs(a[:, 1:2] - sy) <= EDGE_EPS
        b_ends = np.abs(b[3] - sy) <= EDGE_EPS
        b_starts = np.abs(b[1] - sy) <= EDGE_EPS
        touch = (a_ends & b_starts) | (a_starts & b_ends)
        overlap = (np.minimum(a[:, 2], b[2]) - np.maximum(a[:, 0], b[0])) > 0
        out |= touch.any(axis=1) & overlap
    return out


def merge(full_dets: list[dict], tile_dets: list[dict], mode: str = "nmm",
          tiles=None, width: int = 0, height: int = 0) -> list[dict]:
    """Merge full + tile detections, in two stages.

    1. **Class-wise IoU dedup** — collapse the same object detected in both full
       and tile. Same semantics as the detector's own NMS (IoU>=0.5), so
       full-only results pass through unchanged.
    2. **Seam union** (`mode="nmm"`) — union only the pieces cut by interior
       tile boundaries. Boxes that do not abut a seam are **never merged.**

    Do NOT union on containment (inter/min(area)) >= threshold: a small
    detection inside a large box has containment 1.0, so distinct instances get
    absorbed into one giant box. The seam condition prevents that.

    Results keep their sources tags — read by the rescorer feature tile_agree.
    """
    pool = [{**d, "sources": list(d.get("sources", ["full"]))} for d in full_dets]
    pool += [{**d, "sources": list(d.get("sources", ["tile"]))} for d in tile_dets]
    pool.sort(key=lambda d: -d["score"])

    seams_x, seams_y = (interior_seams(tiles, width, height) if tiles else (set(), set()))
    sx = np.array(sorted(seams_x), dtype=float)
    sy = np.array(sorted(seams_y), dtype=float)

    # Kept boxes are also held as an array, and each detection is compared with
    # all of them at once — building a 1×1 IoU per pair took tens of seconds per
    # image at topk 300 × (1+g²) pieces. The verdict is unchanged: the first kept
    # box, in order, with the same label and IoU ≥ NMS_IOU (dedup) or touching
    # it across a seam (union). IoU comes from _iou_matrix as before, so the
    # values are the same.
    kept: list[dict] = []
    boxes = np.empty((len(pool), 4))
    labels: list = []
    for det in pool:
        m = len(kept)
        hit = None
        if m:
            same = np.fromiter((lab == det["label"] for lab in labels), bool, m)
            if same.any():
                kb = boxes[:m]
                iou = _iou_matrix(kb, np.array([det["box"]], dtype=float))[:, 0]
                dup = same & (iou >= NMS_IOU)
                cut = (same & _seam_adjacent_many(kb, det["box"], sx, sy)
                       if mode == "nmm" else np.zeros(m, bool))
                idx = np.flatnonzero(dup | cut)
                if len(idx):
                    i = int(idx[0])
                    hit = (("dedup" if dup[i] else "union"), i)
        if hit is None:
            boxes[m] = det["box"]
            labels.append(det["label"])
            kept.append(det)
            continue
        kind, i = hit
        k = kept[i]
        if kind == "union":
            k["box"] = [min(k["box"][0], det["box"][0]), min(k["box"][1], det["box"][1]),
                        max(k["box"][2], det["box"][2]), max(k["box"][3], det["box"][3])]
            boxes[i] = k["box"]
        k["sources"] = sorted(set(k["sources"]) | set(det["sources"]))
    return kept


def _grid_positions(size: int, n: int, overlap_ratio: float) -> tuple[int, list[int]]:
    """(tile side, start positions) covering one axis with n tiles.

    side = ceil(size / ((n-1)(1-o)+1)) — the zoom is resolution-independent, and
    positions are spread evenly between both ends, so the actual overlap is at
    least o (the ceiling leaves slack)."""
    side = min(size, math.ceil(size / ((n - 1) * (1 - overlap_ratio) + 1)))
    span = size - side
    return side, sorted({round(i * span / (n - 1)) for i in range(n)})


def grid_tiles(width: int, height: int, grid: int,
               overlap_ratio: float) -> list[tuple[int, int, int, int]]:
    """Image-relative g×g tiles — tile count (= inference cost) and zoom are fixed
    regardless of resolution. Each axis is split separately, so non-square images
    are g×g too."""
    sw, xs = _grid_positions(width, grid, overlap_ratio)
    sh, ys = _grid_positions(height, grid, overlap_ratio)
    return [(x, y, x + sw, y + sh) for y in ys for x in xs]


def tile_plan(width: int, height: int, tile_cfg: dict):
    """(tiles, degenerate) — the tiles run_tiled will cut (deterministic).

    tile_cfg is an image-relative g×g grid (`grid`) — cost and zoom do not depend
    on resolution. Absolute px (`slice`) is not accepted — its tile count grows
    quadratically with resolution.
    """
    if "grid" not in tile_cfg:
        raise ValueError(f"tiling config has no grid ({tile_cfg!r}) — the absolute "
                         "px slice form is not supported")
    tiles = grid_tiles(width, height, int(tile_cfg["grid"]), tile_cfg["overlap"])
    degenerate = len(tiles) == 1 and tiles[0] == (0, 0, width, height)
    return tiles, degenerate


def tile_count(width: int, height: int, tile_cfg: dict) -> int:
    """Tile inferences per image (excluding the full pass; 0 if degenerate)."""
    tiles, degenerate = tile_plan(width, height, tile_cfg)
    return 0 if degenerate else len(tiles)


def merge_from_results(results: list[list[dict]], tiles, degenerate: bool,
                       width: int, height: int, tile_cfg: dict) -> list[dict]:
    """Inference results in [full]+crops order -> offset + merge. Second half of run_tiled."""
    full_dets = [{**d, "sources": ["full"]} for d in results[0]]
    tile_dets: list[dict] = []
    for (x1, y1, _x2, _y2), dets in zip(tiles if not degenerate else [], results[1:]):
        tile_dets += [{**d, "sources": ["tile"]} for d in offset_detections(dets, x1, y1)]
    return merge(full_dets, tile_dets, tile_cfg.get("merge", "nmm"),
                 tiles=tiles, width=width, height=height)


def run_tiled(predict_fn, image, tile_cfg: dict) -> list[dict]:
    """Run inference once on the full image + N tiles, then merge.

    predict_fn(images: list[PIL.Image]) -> list[list[dict]] — detections per image.
    image: PIL.Image (loaded by the caller).
    tile_cfg: {"grid": int, "overlap": float, "merge": "nmm"|"nms"}
    """
    tiles, degenerate = tile_plan(image.width, image.height, tile_cfg)
    crops = [] if degenerate else [image.crop(t) for t in tiles]
    results = predict_fn([image] + crops)
    return merge_from_results(results, tiles, degenerate,
                              image.width, image.height, tile_cfg)
