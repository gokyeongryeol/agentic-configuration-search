"""coco.py: dummy category exclusion, bbox round-trip, split loading.

Uses only the synthetic fixture (train/valid/test) — no demo_data real-data tests
(hermetic; real-data checks happen in the GPU smoke run).
"""

from pathlib import Path

import pytest

from agentic_configuration_search.core.coco import Ontology, Split, xywh_to_xyxy
from tests.conftest import CATS


def test_dummy_supercategory_excluded():
    onto = Ontology.from_categories(CATS)
    assert [c.id for c in onto.classes] == [1, 2]
    assert 0 not in onto.id_to_name


def test_no_dummy_keeps_all():
    cats = [{"id": 1, "name": "a", "supercategory": "none"},
            {"id": 2, "name": "b", "supercategory": "none"}]
    # nobody references these as a supercategory → all kept
    onto = Ontology.from_categories(cats)
    assert len(onto.classes) == 2


def test_bbox_xywh_to_xyxy():
    assert xywh_to_xyxy([10.0, 20.0, 30.0, 40.0]) == [10.0, 20.0, 40.0, 60.0]


def test_slug():
    onto = Ontology.from_categories(CATS)
    assert onto.id_to_slug[1] == "cat_a"


def test_split_load_and_area_stats(dataset_root):
    split = Split.load(dataset_root / "train")
    assert len(split.images) == 10
    assert split.images[0].file_name == "tr1.png"  # sorted by file name (deterministic)
    by_class = split.annotations_by_class()
    assert len(by_class[1]) == 5 and len(by_class[2]) == 5
    stats = split.area_stats()
    assert stats[1]["count"] == 5 and stats[1]["mean"] == pytest.approx(400.0)
