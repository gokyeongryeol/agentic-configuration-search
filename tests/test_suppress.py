"""core/suppress — pure functions emulating negatives (CPU, synthetic boxes)."""

from agentic_configuration_search.core.suppress import (batched_nms, emulate_negatives,
                                                   suppress_by_negatives)


def det(box, score=0.9, cid=1):
    return {"box": [float(v) for v in box], "score": float(score),
            "category_id": int(cid)}


# ------------------------------------------------------------ batched NMS

def test_nms_folds_same_class_duplicates():
    a = det([0, 0, 10, 10], 0.9, cid=1)
    b = det([1, 0, 11, 10], 0.5, cid=1)          # IoU≈0.8 — folded
    kept = batched_nms([b, a], 0.5)
    assert kept == [a]


def test_nms_is_class_independent():
    """Different classes (notably the negative channel) do not suppress each other in NMS."""
    a = det([0, 0, 10, 10], 0.9, cid=1)
    n = det([0, 0, 10, 10], 0.95, cid=101)       # same spot, different channel
    kept = batched_nms([a, n], 0.5)
    assert len(kept) == 2                        # both survive — suppression is the IoU stage's job


def test_nms_keeps_below_threshold_overlap():
    a = det([0, 0, 10, 10], 0.9, cid=1)
    b = det([8, 0, 18, 10], 0.5, cid=1)          # IoU≈0.11 < 0.5 — kept
    assert len(batched_nms([a, b], 0.5)) == 2


# ------------------------------------------------------------- suppression rule

def test_overlapping_positive_removed_at_threshold():
    pos = [det([0, 0, 10, 10], 0.9, cid=1)]
    neg = [det([0, 0, 10, 10], 0.3, cid=101)]    # IoU 1.0 ≥ 0.5 — removed
    assert suppress_by_negatives(pos, neg, 0.5) == []


def test_non_overlapping_positive_kept():
    pos = [det([0, 0, 10, 10], 0.9, cid=1)]
    neg = [det([50, 50, 60, 60], 0.9, cid=101)]
    assert suppress_by_negatives(pos, neg, 0.5) == pos


def test_boundary_iou_exactly_at_threshold_removed():
    # [0,0,10,10] vs [0,5,10,15]: intersection 50, union 150 -> IoU exactly 1/3
    pos = [det([0, 0, 10, 10], 0.9, cid=1)]
    neg = [det([0, 5, 10, 15], 0.9, cid=101)]
    assert suppress_by_negatives(pos, neg, 1 / 3) == []          # ≥ threshold — removed
    assert suppress_by_negatives(pos, neg, 1 / 3 + 1e-9) == pos  # threshold above — kept


def test_multiple_negatives_any_hit_removes():
    pos = [det([0, 0, 10, 10], 0.9, cid=1),
           det([100, 100, 110, 110], 0.8, cid=2)]
    neg = [det([50, 50, 60, 60], 0.9, cid=101),      # overlaps nothing
           det([1, 0, 11, 10], 0.9, cid=102)]        # IoU≈0.8 with the first positive
    kept = suppress_by_negatives(pos, neg, 0.5)
    assert kept == [pos[1]]


# --------------------------------------------------------- full emulation

def test_emulate_full_pipeline():
    """NMS (class-independent) -> suppression -> negative discard, in one pass."""
    p1 = det([0, 0, 10, 10], 0.9, cid=1)
    p1_dup = det([1, 0, 11, 10], 0.4, cid=1)         # folded by NMS
    p2 = det([100, 100, 110, 110], 0.8, cid=2)
    n1 = det([0, 0, 10, 10], 0.99, cid=101)          # on top of p1 — coexists in NMS (class-independent)
    n1_dup = det([0, 1, 10, 11], 0.5, cid=101)       # negatives also fold among themselves via NMS
    out = emulate_negatives([p1, p1_dup, p2], [n1, n1_dup],
                            sup_iou=0.5, nms_iou=0.5)
    assert out == [p2]                               # p1 suppressed, negatives discarded, p2 survives


def test_emulate_high_threshold_keeps_positive():
    p1 = det([0, 0, 10, 10], 0.9, cid=1)
    n1 = det([0, 5, 10, 15], 0.9, cid=101)           # IoU 1/3 < 0.75 — kept
    out = emulate_negatives([p1], [n1], sup_iou=0.75, nms_iou=0.5)
    assert out == [p1]


def test_emulate_without_negatives_is_nms_only():
    p1 = det([0, 0, 10, 10], 0.9, cid=1)
    p2 = det([1, 0, 11, 10], 0.5, cid=1)
    out = emulate_negatives([p1, p2], [], sup_iou=0.25, nms_iou=0.5)
    assert out == [p1]
