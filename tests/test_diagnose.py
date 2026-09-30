"""diagnose/errors — per-class aggregation and holdout (`test` key) anonymization.

Note: classify_errors' missed_small cut is the lower quartile of the class's GT
areas — for a class with a single GT the cut equals its own area, so it is always
classified missed_small (pinned as current behavior). The fixture reflects this.
"""

import pytest

from agentic_detector_config.core.coco import Ontology
from agentic_detector_config.diagnose.errors import diagnose
from tests.conftest import CATS

ONTO = Ontology.from_categories(CATS)
IOU = 0.5

GT_SMALL = ([10, 10, 30, 30], 1)      # 400px² — class 1 lower quartile
GT_BIG = ([10, 10, 50, 50], 1)        # 1600px² — above the cut


def _rec(file, dets, gts):
    return {"file": file, "width": 100, "height": 100,
            "detections": [{"box": b, "score": s, "category_id": c}
                           for b, s, c in dets],
            "gts": [{"box": b, "category_id": c} for b, c in gts]}


def _records():
    """3 images: TP+background FP / below_threshold FN (large GT) / single-GT-class FN."""
    recs = [
        _rec("a.png", [([10, 10, 30, 30], 0.9, 1), ([60, 60, 80, 80], 0.7, 1)],
             [GT_SMALL]),
        _rec("b.png", [], [GT_BIG]),
        _rec("c.png", [], [([40, 40, 70, 70], 2)]),
    ]
    floor = [
        _rec("a.png", [], [GT_SMALL]),
        _rec("b.png", [([10, 10, 50, 50], 0.2, 1)], [GT_BIG]),  # hit below the cut
        _rec("c.png", [], []),
    ]
    return recs, floor


def test_per_class_aggregation_and_details():
    recs, floor = _records()
    out = diagnose(recs, floor, {1: 0.5, 2: 0.5}, ONTO, iou_thr=IOU,
                   anonymize=False)
    assert out["n_images"] == 3
    a = out["per_class"]["cat a"]
    assert a["gt"] == 2 and a["fp"] == {"background": 1}
    assert a["fn"] == {"below_threshold": 1}
    assert a["miss_rate"] == pytest.approx(0.5)
    b = out["per_class"]["cat b"]
    assert b["fn"] == {"missed_small": 1} and b["miss_rate"] == pytest.approx(1.0)
    # Details are keyed by **file name**, not image index (for humans and the planner).
    # The index is replaced, not kept alongside — with two identifiers, numbers drift
    # when the prompt substitutes image aliases, which only confuses.
    assert {r["image"] for r in out["fn"]} == {"b.png", "c.png"}
    assert out["fp"][0]["image"] == "a.png"
    assert all("image_index" not in r for r in out["fp"] + out["fn"])


def test_anonymize_drops_details():
    recs, floor = _records()
    out = diagnose(recs, floor, {1: 0.5, 2: 0.5}, ONTO, iou_thr=IOU,
                   anonymize=True)
    assert out["anonymized"] is True
    assert "fp" not in out and "fn" not in out          # aggregates only
    assert out["summary"]["fn"]["below_threshold"] == 1
    assert out["summary"]["fn"]["missed_small"] == 1
    assert out["summary"]["fp"]["background"] == 1


def test_without_floor_below_threshold_becomes_missed():
    recs, _ = _records()
    out = diagnose(recs, None, {1: 0.5, 2: 0.5}, ONTO, iou_thr=IOU,
                   anonymize=True)
    assert out["summary"]["fn"]["below_threshold"] == 0
    assert out["summary"]["fn"]["missed"] == 1          # large GT counts as missed
