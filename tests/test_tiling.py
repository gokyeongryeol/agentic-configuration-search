"""tiling.py: grid coverage, offsets, merging of boundary boxes."""

import numpy as np
from PIL import Image

from agentic_detector_config.core import tiling


def test_slice_grid_covers_every_pixel():
    w, h = 100, 80
    tiles = tiling.slice_grid(w, h, (64, 64), 0.25)
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        mask[y1:y2, x1:x2] = True
    assert mask.all(), "every pixel must be covered by at least one tile"


def test_slice_grid_small_image_single_tile():
    assert tiling.slice_grid(50, 40, (64, 64), 0.25) == [(0, 0, 50, 40)]


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

    merged = tiling.run_tiled(predict_fn, im, {"slice": [64, 64], "overlap": 0.25, "merge": "nmm"})
    assert sum(calls) == 1 + len(tiling.slice_grid(100, 80, (64, 64), 0.25))
    label1 = [d for d in merged if d["label"] == 1]
    assert label1 and label1[0]["sources"] == ["full"]
    # tile detections are offset into global coordinates
    label2 = [d for d in merged if d["label"] == 2]
    assert any(d["box"][0] > 0 for d in label2)


def test_run_tiled_degenerate_tile_is_identity():
    """If slice ≥ image size, the tile equals the full image — inference must not be duplicated
    and results must match tiling-off."""
    im = Image.new("RGB", (640, 640))
    dets = [{"box": [0, 0, 600, 600], "score": 0.9, "label": 1},
            {"box": [100, 100, 120, 120], "score": 0.8, "label": 1},
            {"box": [300, 300, 320, 320], "score": 0.7, "label": 1}]
    calls = []

    def predict_fn(images):
        calls.append(len(images))
        return [list(dets) for _ in images]

    merged = tiling.run_tiled(predict_fn, im,
                              {"slice": [640, 640], "overlap": 0.25, "merge": "nmm"})
    assert calls == [1]                       # only 1 full pass — no duplicate inference
    assert len(merged) == len(dets)           # nested detections preserved
    assert all(d["sources"] == ["full"] for d in merged)
