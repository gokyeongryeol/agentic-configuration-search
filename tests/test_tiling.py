"""tiling.py: grid coverage, offsets, merging of boundary boxes."""

import numpy as np
import pytest
from PIL import Image

from agentic_configuration_search.core import tiling


def test_offset_detections():
    dets = [{"box": [1, 2, 3, 4], "score": 0.9, "label": 0}]
    out = tiling.offset_detections(dets, 10, 20)
    assert out[0]["box"] == [11, 22, 13, 24]
    assert dets[0]["box"] == [1, 2, 3, 4]  # original unchanged


TILES_2x1 = [(0, 0, 60, 100), (40, 0, 100, 100)]  # internal seams: x=40, x=60


def test_interior_seams_excludes_image_border():
    sx, sy = tiling.interior_seams(TILES_2x1, 100, 100)
    assert sx == {40.0, 60.0}   # 0 and 100 (borders) excluded
    assert sy == set()


def test_nmm_unions_seam_cut_pieces():
    """Only two pieces abutting the seam (x=60) from both sides are unioned."""
    a = {"box": [30, 20, 60, 50], "score": 0.9, "label": 1, "sources": ["tile"]}
    b = {"box": [60, 20, 85, 50], "score": 0.8, "label": 1, "sources": ["tile"]}
    merged = tiling.merge([], [a, b], mode="nmm", tiles=TILES_2x1, width=100, height=100)
    assert len(merged) == 1
    assert merged[0]["box"] == [30, 20, 85, 50]
    assert merged[0]["score"] == 0.9


def test_nested_distinct_instances_never_unioned():
    """Regression guard: a small detection inside a large box has containment 1.0 but is a
    different instance — they must never be merged."""
    big = {"box": [0, 0, 100, 100], "score": 0.9, "label": 1, "sources": ["full"]}
    small = {"box": [40, 40, 50, 50], "score": 0.8, "label": 1, "sources": ["full"]}
    merged = tiling.merge([big, small], [], mode="nmm",
                          tiles=TILES_2x1, width=100, height=100)
    assert len(merged) == 2
    assert sorted(d["box"] for d in merged) == [[0, 0, 100, 100], [40, 40, 50, 50]]


def test_seam_touching_but_same_side_not_unioned():
    """If both end on the same side of the seam, they are not cut pieces."""
    a = {"box": [10, 10, 60, 30], "score": 0.9, "label": 1}
    b = {"box": [20, 50, 60, 70], "score": 0.8, "label": 1}
    merged = tiling.merge([], [a, b], mode="nmm", tiles=TILES_2x1, width=100, height=100)
    assert len(merged) == 2


def test_seam_adjacent_but_no_perpendicular_overlap_not_unioned():
    """Abutting the seam but not overlapping vertically means different objects."""
    a = {"box": [30, 10, 60, 20], "score": 0.9, "label": 1}
    b = {"box": [60, 70, 85, 90], "score": 0.8, "label": 1}
    merged = tiling.merge([], [a, b], mode="nmm", tiles=TILES_2x1, width=100, height=100)
    assert len(merged) == 2


def test_iou_dedup_collapses_same_object_from_full_and_tile():
    full = [{"box": [10, 10, 50, 50], "score": 0.7, "label": 1}]
    tile = [{"box": [12, 12, 48, 48], "score": 0.9, "label": 1}]
    merged = tiling.merge(full, tile, mode="nmm", tiles=TILES_2x1, width=100, height=100)
    assert len(merged) == 1
    assert set(merged[0]["sources"]) == {"full", "tile"}
    assert merged[0]["score"] == 0.9        # higher score kept
    assert merged[0]["box"] == [12, 12, 48, 48]  # dedup does not grow the box


def test_merge_different_labels_not_merged():
    a = {"box": [10, 10, 50, 50], "score": 0.9, "label": 1}
    b = {"box": [10, 10, 50, 50], "score": 0.8, "label": 2}
    assert len(tiling.merge([a], [b], mode="nmm", tiles=TILES_2x1,
                            width=100, height=100)) == 2


def test_tile_only_detection_tagged():
    tile = [{"box": [70, 70, 90, 90], "score": 0.9, "label": 1}]
    merged = tiling.merge([], tile, mode="nmm", tiles=TILES_2x1, width=100, height=100)
    assert merged[0]["sources"] == ["tile"]  # source for the rescorer feature (tile_agree)


def test_run_tiled_offsets_and_merges():
    im = Image.new("RGB", (100, 80))
    calls = []

    def predict_fn(images):
        calls.append(len(images))
        # assume a detection near (80,60) appears only on the full image,
        # and each tile yields a tile-local (0,0,10,10) detection
        out = []
        for i, img in enumerate(images):
            if i == 0:
                out.append([{"box": [70, 50, 90, 70], "score": 0.9, "label": 1}])
            else:
                out.append([{"box": [0, 0, 10, 10], "score": 0.5, "label": 2}])
        return out

    merged = tiling.run_tiled(predict_fn, im, {"grid": 2, "overlap": 0.25, "merge": "nmm"})
    assert sum(calls) == 1 + 4
    label1 = [d for d in merged if d["label"] == 1]
    assert label1 and label1[0]["sources"] == ["full"]
    # tile detections are offset into global coordinates
    label2 = [d for d in merged if d["label"] == 2]
    assert any(d["box"][0] > 0 for d in label2)


def test_run_tiled_degenerate_tile_is_identity():
    """For a tiny image (1px) the tile equals the full image — inference must not be
    duplicated and results must match tiling-off."""
    im = Image.new("RGB", (1, 1))
    dets = [{"box": [0, 0, 1, 1], "score": 0.9, "label": 1},
            {"box": [0, 0, 1, 1], "score": 0.8, "label": 2}]
    calls = []

    def predict_fn(images):
        calls.append(len(images))
        return [list(dets) for _ in images]

    merged = tiling.run_tiled(predict_fn, im,
                              {"grid": 2, "overlap": 0.5, "merge": "nmm"})
    assert calls == [1]                       # only 1 full pass — no duplicate inference
    assert len(merged) == len(dets)
    assert all(d["sources"] == ["full"] for d in merged)


# --- grid: image-relative tiles -------------------------------------------------
# Absolute px slices grew the tile count quadratically with resolution (320 at
# 640px → 9 tiles, at 4000×3000 → 221 tiles). A grid is g×g at any resolution —
# fixed cost and zoom.

def _covers(tiles, w, h):
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        mask[y1:y2, x1:x2] = True
    return mask.all()


def test_grid_tile_count_is_resolution_independent():
    for w, h in [(325, 325), (640, 640), (640, 480), (1920, 1080),
                 (1700, 2200), (4000, 3000), (1001, 997)]:
        for g in (2, 3):
            for overlap in (0.25, 0.5):
                tiles, degenerate = tiling.tile_plan(
                    w, h, {"grid": g, "overlap": overlap, "merge": "nmm"})
                assert not degenerate
                assert len(tiles) == g * g, (w, h, g, overlap)
                assert _covers(tiles, w, h)


def test_grid_zoom_and_overlap_hold_per_axis():
    """tile side = side/((g-1)(1-o)+1) (rounded up) — the zoom does not depend on
    resolution and the actual overlap is at least the requested one. Non-square
    images are split per axis."""
    w, h, g, o = 1920, 1080, 3, 0.5
    tiles, _ = tiling.tile_plan(w, h, {"grid": g, "overlap": o, "merge": "nmm"})
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    sw, sh = tiles[0][2] - tiles[0][0], tiles[0][3] - tiles[0][1]
    assert sw == 960 and sh == 540           # 2.0x zoom
    for pos, s in ((xs, sw), (ys, sh)):
        for a, b in zip(pos, pos[1:]):
            assert (a + s - b) / s >= o       # adjacent tiles overlap >= overlap


def test_default_grids_match_old_640_slices():
    """The default grids (grid 2 and 3, overlap 0.5) match the old slice 416 / 320 at
    640px in tile count (4 / 9) and zoom (1.5x / 2.0x)."""
    t2, _ = tiling.tile_plan(640, 640, {"grid": 2, "overlap": 0.5})
    t3, _ = tiling.tile_plan(640, 640, {"grid": 3, "overlap": 0.5})
    assert len(t2) == 4 and len(t3) == 9
    assert t3[0] == (0, 0, 320, 320)
    assert abs((t2[0][2] - t2[0][0]) - 416) <= 16


def test_yoloe_grid_matches_old_384_slice():
    """The yoloe configs' grid 2 / overlap 0.33 keeps the old 384px 2×2 tiles at 640px."""
    tiles, _ = tiling.tile_plan(640, 640, {"grid": 2, "overlap": 0.33})
    assert len(tiles) == 4 and tiles[0] == (0, 0, 384, 384)


def test_slice_form_rejected():
    with pytest.raises(ValueError, match="grid"):
        tiling.tile_plan(640, 640, {"slice": [320, 320], "overlap": 0.25})


def test_tile_count_helper():
    assert tiling.tile_count(4000, 3000, {"grid": 3, "overlap": 0.5}) == 9
    assert tiling.tile_count(500, 400, {"grid": 2, "overlap": 0.25}) == 4


def test_run_tiled_grid_calls_full_plus_g_squared():
    calls = []

    def predict_fn(images):
        calls.append([im.size for im in images])
        return [[] for _ in images]

    tiling.run_tiled(predict_fn, Image.new("RGB", (4000, 3000)),
                     {"grid": 3, "overlap": 0.5, "merge": "nmm"})
    assert len(calls) == 1 and len(calls[0]) == 1 + 9
    assert calls[0][0] == (4000, 3000)
    assert set(calls[0][1:]) == {(2000, 1500)}




# --- merge vectorization: results must be bit-identical to the old implementation (a 1×1 IoU per detection pair) ---

def _reference_merge(full_dets, tile_dets, mode="nmm", tiles=None, width=0, height=0):
    """merge as it was before vectorization — the equivalence reference."""
    import numpy as np
    from agentic_configuration_search.core.tiling import NMS_IOU, _iou_matrix, _seam_adjacent, interior_seams
    pool = [{**d, "sources": list(d.get("sources", ["full"]))} for d in full_dets]
    pool += [{**d, "sources": list(d.get("sources", ["tile"]))} for d in tile_dets]
    pool.sort(key=lambda d: -d["score"])
    seams_x, seams_y = (interior_seams(tiles, width, height) if tiles else (set(), set()))
    kept = []
    for det in pool:
        target = None
        for k in kept:
            if k["label"] != det["label"]:
                continue
            iou = _iou_matrix(np.array([k["box"]]), np.array([det["box"]]))[0, 0]
            if iou >= NMS_IOU:
                target = ("dedup", k)
                break
            if mode == "nmm" and _seam_adjacent(k["box"], det["box"], seams_x, seams_y):
                target = ("union", k)
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


def _random_pieces(rng, width, height, grid, n_per_piece, n_labels):
    """Full + tile detections (tile coordinates). Some boxes touch a tile border
    exactly to take the seam-union path, some overlap the same object in full and
    tile to take the dedup path."""
    cfg = {"grid": grid, "overlap": 0.5, "merge": "nmm"}
    tiles, _ = tiling.tile_plan(width, height, cfg)

    def box(w, h):
        x, y = rng.uniform(0, w - 8), rng.uniform(0, h - 8)
        s = rng.uniform(4, 60)
        return [x, y, min(w, x + s), min(h, y + rng.uniform(4, 60))]

    full = [{"box": box(width, height), "score": round(rng.uniform(0.03, 0.95), 3),
             "label": rng.randrange(n_labels)} for _ in range(n_per_piece)]
    per_tile = []
    for (x1, y1, x2, y2) in tiles:
        tw, th = x2 - x1, y2 - y1
        ds = []
        for _ in range(n_per_piece):
            b = box(tw, th)
            r = rng.random()
            if r < 0.15:            # touches the right border — seam-union candidate with the neighbor tile
                b[2] = float(tw)
            elif r < 0.30:          # touches the bottom border
                b[3] = float(th)
            elif r < 0.45 and full:  # nearly the same object as a full detection
                fb = rng.choice(full)["box"]
                b = [fb[0] - x1 + 1, fb[1] - y1, fb[2] - x1 + 1, fb[3] - y1]
            ds.append({"box": b, "score": round(rng.uniform(0.03, 0.95), 3),
                       "label": rng.randrange(n_labels)})
        per_tile.append(ds)
    return [full] + per_tile, tiles


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("mode", ["nmm", "nms"])
def test_merge_matches_pairwise_reference(seed, mode):
    import random
    rng = random.Random(seed)
    width, height = rng.choice([(640, 640), (800, 600), (333, 517)])
    grid = rng.choice([2, 3])
    results, tiles = _random_pieces(rng, width, height, grid,
                                    n_per_piece=rng.choice([0, 3, 40]),
                                    n_labels=rng.choice([1, 3]))
    cfg = {"grid": grid, "overlap": 0.5, "merge": mode}
    full = [{**d, "sources": ["full"]} for d in results[0]]
    tile = []
    for (x1, y1, _x2, _y2), dets in zip(tiles, results[1:]):
        tile += [{**d, "sources": ["tile"]} for d in tiling.offset_detections(dets, x1, y1)]
    got = tiling.merge(full, tile, mode, tiles=tiles, width=width, height=height)
    ref = _reference_merge(full, tile, mode, tiles=tiles, width=width, height=height)
    assert got == ref
    assert tiling.merge_from_results(results, tiles, False, width, height, cfg) == ref


def test_merge_is_fast_on_topk_sized_inputs():
    """apply_topk (300) per piece × (full + 4 tiles) — the old implementation took
    seconds to tens of seconds per image (a 1×1 numpy IoU per detection pair).
    It must stay under a second per image."""
    import random
    import time
    rng = random.Random(0)
    results, tiles = _random_pieces(rng, 640, 640, 2, n_per_piece=300, n_labels=1)
    t0 = time.perf_counter()
    tiling.merge_from_results(results, tiles, False, 640, 640,
                              {"grid": 2, "overlap": 0.5, "merge": "nmm"})
    assert time.perf_counter() - t0 < 1.0
