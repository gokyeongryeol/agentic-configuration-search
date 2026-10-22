"""diagnose/residual — structure of the final champion's residual misses (size,
crowding, border, aspect) and label quality flags. A report-only artifact; it
never reaches the planner.

Holdout yields aggregates only — no individual images or boxes may remain.
"""

import json

import pytest

from agentic_configuration_search.core.coco import Ontology
from agentic_configuration_search.diagnose import residual
from agentic_configuration_search.diagnose.errors import diagnose
from tests.conftest import CATS

ONTO = Ontology.from_categories(CATS)
IOU = 0.5


def _img(file, gts, w=100, h=100, dets=()):
    return {"file": file, "width": w, "height": h,
            "detections": [{"box": b, "score": s, "category_id": c}
                           for b, s, c in dets],
            "gts": [{"box": b, "category_id": c} for b, c in gts]}


def test_buckets():
    # small_ratio 0.04: small < 0.04 ≤ medium < 0.12 ≤ large
    assert residual.size_bucket(0.03, 0.04) == "small"
    assert residual.size_bucket(0.04, 0.04) == "medium"
    assert residual.size_bucket(0.119, 0.04) == "medium"
    assert residual.size_bucket(0.12, 0.04) == "large"
    assert [residual.crowd_bucket(n) for n in (1, 2, 5, 6, 20, 21)] == \
        ["1", "2-5", "2-5", "6-20", "6-20", "21+"]
    f = residual.object_facts([0, 40, 30, 50], 100, 100, n_gt=3, small_ratio=0.04)
    assert f["border"] is True and f["elongated"] is True     # x1=0, 30×10
    assert f["crowd"] == "2-5"
    assert f["px"] == pytest.approx(640 * (300 / 10000) ** 0.5, abs=1)
    g = residual.object_facts([40, 40, 60, 60], 100, 100, n_gt=1, small_ratio=0.04)
    assert g["border"] is False and g["elongated"] is False and g["size"] == "large"


def _train():
    """3 train images: a misses the small border one of its 2 GT, b misses its
    1 GT (a large one), c has a 1px-wide GT (label error suspected) and the same
    box twice (duplicate GT)."""
    return [
        _img("a.png", [([40, 40, 60, 60], 1), ([0, 50, 3, 53], 1)],
             dets=[([40, 40, 60, 60], 0.9, 1)]),
        _img("b.png", [([10, 10, 60, 60], 2)]),
        _img("c.png", [([20, 20, 21, 40], 1), ([50, 50, 70, 70], 2),
                       ([50, 50, 70, 70], 2)],
             dets=[([50, 50, 70, 70], 0.9, 2), ([20, 20, 21, 40], 0.9, 1)]),
    ]


def test_residual_train_detail_and_holdout_aggregate():
    train = _train()
    d_train = diagnose(train, None, {1: 0.5, 2: 0.5}, ONTO, iou_thr=IOU,
                       anonymize=False)
    hold = [_img("h1.png", [([5, 5, 45, 45], 1)]),
            _img("h2.png", [([0, 0, 2, 2], 2), ([30, 30, 50, 50], 2)],
                 dets=[([30, 30, 50, 50], 0.9, 2)])]
    out = residual.residual(train, d_train, hold, None, {1: 0.5, 2: 0.5}, ONTO,
                            iou_thr=IOU, small_ratio=0.04)
    assert out["small_ratio"] == 0.04 and out["input_side"] == 640

    tr = out["train"]
    # detailed FN rows are the diagnosis rows with facts attached — image, box, kind as-is
    by_img = {r["image"]: r for r in tr["fn"]}
    assert set(by_img) == {"a.png", "b.png", "c.png"}
    a = by_img["a.png"]
    assert a["kind"] == "missed_small" and a["size"] == "small"
    assert a["border"] is True and a["n_gt_image"] == 2 and a["crowd"] == "2-5"
    assert by_img["b.png"]["size"] == "large" and by_img["b.png"]["crowd"] == "1"

    pa = tr["per_class"]["cat a"]
    assert pa["gt"] == 3 and pa["fn"] == 1
    assert pa["size"] == {"small": [1, 1], "medium": [1, 0], "large": [1, 0]}  # [GT, FN]
    assert pa["border"] == [1, 1]
    assert pa["kind_size"]["missed_small"]["small"] == 1

    flags = tr["label_flags"]
    assert [r["image"] for r in flags["tiny_gt"]] == ["a.png", "c.png"]   # 3px and 1px sides
    assert flags["duplicate_gt"][0]["image"] == "c.png"
    assert flags["duplicate_gt"][0]["category"] == "cat b"

    ho = out["holdout"]
    assert ho["n_images"] == 2
    assert ho["per_class"]["cat a"]["fn"] == 1
    assert ho["per_class"]["cat b"]["size"]["small"] == [1, 1]
    assert ho["label_flags"] == {"tiny_gt": 1, "duplicate_gt": 0}   # counts only
    # no filename or box anywhere on the holdout side
    blob = json.dumps(ho)
    assert "h1.png" not in blob and "h2.png" not in blob and "box" not in blob


def test_residual_holdout_matches_anonymized_diagnosis():
    """Holdout reclassification follows the same rule as diagnosis_test — the same FN count per class."""
    hold = _train()
    d_test = diagnose(hold, None, {1: 0.5, 2: 0.5}, ONTO, iou_thr=IOU,
                      anonymize=True)
    out = residual.residual([], {"fn": []}, hold, None, {1: 0.5, 2: 0.5}, ONTO,
                            iou_thr=IOU, small_ratio=0.04)
    for name, s in d_test["per_class"].items():
        assert out["holdout"]["per_class"][name]["fn"] == sum(s["fn"].values())


def test_image_rows_from_split(dataset_root):
    from agentic_configuration_search.core.coco import Split
    split = Split.load(dataset_root / "train")
    rows = residual.image_rows(split)
    assert len(rows) == len(split.images)
    assert set(rows[0]) == {"file", "width", "height", "gts"}
