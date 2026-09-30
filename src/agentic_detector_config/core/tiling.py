"""In-house SAHI: slice / offset / merge.

The detector has no tiled inference, so we build it here.
Internal detection representation is a dict:
    {"box": [x1,y1,x2,y2], "score": float, "label": int, "sources": ["full"|"tile", ...]}
`label` is the category_id — the caller (gateway) converts it at parse time.
"""

from __future__ import annotations

import numpy as np

NMS_IOU = 0.5
# tolerance (px) for a box edge to count as "touching" a tile seam
EDGE_EPS = 2.0


def slice_grid(width: int, height: int, slice_wh, overlap_ratio: float) -> list[tuple[int, int, int, int]]:
    """List of tiles (x1,y1,x2,y2). The last tile is clamped inward so every pixel
    is covered by at least one tile."""
    sw, sh = int(slice_wh[0]), int(slice_wh[1])
    step_x = max(1, int(sw * (1 - overlap_ratio)))
    step_y = max(1, int(sh * (1 - overlap_ratio)))

    def _positions(size: int, slice_size: int, step: int) -> list[int]:
        if size <= slice_size:
            return [0]
        pos = list(range(0, size - slice_size, step))
        pos.append(size - slice_size)  # clamp the border tile
        return sorted(set(pos))

    return [
        (x, y, min(x + sw, width), min(y + sh, height))
        for y in _positions(height, sh, step_y)
        for x in _positions(width, sw, step_x)
    ]


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

    kept: list[dict] = []
    for det in pool:
        target = None
        for k in kept:
            if k["label"] != det["label"]:
                continue
            iou = _iou_matrix(np.array([k["box"]]), np.array([det["box"]]))[0, 0]
            if iou >= NMS_IOU:          # duplicate of the same object -> keep the higher score
                target = ("dedup", k)
                break
            if mode == "nmm" and _seam_adjacent(k["box"], det["box"], seams_x, seams_y):
                target = ("union", k)   # piece cut at a seam -> union
                break
        if target is None:
            kept.append(det)
            continue
        kind, k = target
        if kind == "union":
            k["box"] = [min(k["box"][0], det["box"][0]), min(k["box"][1], det["box"][1]),
                        max(k["box"][2], det["box"][2]), max(k["box"][3], det["box"][3])]
        k["sources"] = sorted(set(k["sources"]) | set(det["sources"]))
    return kept


def tile_plan(width: int, height: int, tile_cfg: dict):
    """(tiles, degenerate) — the tiles run_tiled will cut (deterministic)."""
    tiles = slice_grid(width, height, tile_cfg["slice"], tile_cfg["overlap"])
    degenerate = len(tiles) == 1 and tiles[0] == (0, 0, width, height)
    return tiles, degenerate


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
    tile_cfg: {"slice": [w,h], "overlap": float, "merge": "nmm"|"nms"}
    """
    tiles, degenerate = tile_plan(image.width, image.height, tile_cfg)
    crops = [] if degenerate else [image.crop(t) for t in tiles]
    results = predict_fn([image] + crops)
    return merge_from_results(results, tiles, degenerate,
                              image.width, image.height, tile_cfg)
