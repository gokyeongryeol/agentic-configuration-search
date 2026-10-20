"""prompts.py: negative_target attachment, positive guarantee, exclusion of classes with 0 exemplars."""

import pytest

from agentic_configuration_search.core.coco import Ontology
from agentic_configuration_search.core.prompts import PromptBox, build_queries
from tests.conftest import CATS


def _box(name, image, xyxy=(1, 2, 3, 4)):
    return PromptBox(class_name=name, image=image, bbox_xyxy=list(xyxy))


def test_positive_and_negative_entries():
    exemplars = {"cat a": [_box("cat a", "img1.png")]}
    visual_negatives = {"cat a": [_box("cat a", "img2.png")]}
    queries, order, excluded = build_queries(exemplars, visual_negatives, 0.03, 0.03)
    assert excluded == []
    all_prompts = [p for q in queries for p in q["prompts"]]
    positives = [p for p in all_prompts if not p.get("negative_target")]
    negs = [p for p in all_prompts if p.get("negative_target")]
    assert len(positives) == 1 and len(negs) == 1
    assert negs[0]["negative_target"] == "cat a"
    assert positives[0]["box_threshold"] == 0.03
    assert positives[0]["box"] == [1.0, 2.0, 3.0, 4.0]  # semantic — box required


def test_zero_exemplar_class_excluded():
    """Classes with 0 exemplars must be dropped — with empty boxes the detector silently
    switches to the text modality."""
    exemplars = {"cat a": [_box("cat a", "img1.png")], "cat b": []}
    queries, order, excluded = build_queries(exemplars, None, 0.03, 0.03)
    assert excluded == ["cat b"]
    assert order == ["cat a"]


def test_all_empty_raises():
    with pytest.raises(ValueError):
        build_queries({"cat a": [], "cat b": []}, None, 0.03, 0.03)


def test_negative_without_positive_dropped():
    exemplars = {"cat a": [_box("cat a", "img1.png")], "cat b": []}
    visual_negatives = {"cat b": [_box("cat b", "img2.png")]}
    queries, order, _ = build_queries(exemplars, visual_negatives, 0.03, 0.03)
    assert all(not p.get("negative_target") == "cat b"
               for q in queries for p in q["prompts"])


def test_one_query_per_image_and_prompt_order():
    exemplars = {
        "cat b": [_box("cat b", "shared.png")],
        "cat a": [_box("cat a", "shared.png"), _box("cat a", "solo.png")],
    }
    queries, order, _ = build_queries(exemplars, None, 0.03, 0.03)
    assert len(queries) == 2  # one query per image
    # iterate classes in sorted order -> cat a appears first in shared.png
    assert order == ["cat a", "cat b"]
    name_to_id = Ontology.from_categories(CATS).name_to_id
    assert [name_to_id[n] for n in order] == [1, 2]  # consistent with the ontology


def test_negative_only_config_matches_positive_order():
    """Toggling negatives must not change prompt_order — the query-construction order contract
    must be identical in both configs."""
    exemplars = {"cat a": [_box("cat a", "i1.png")], "cat b": [_box("cat b", "i2.png")]}
    visual_negatives = {"cat b": [_box("cat b", "i1.png")]}
    _, order_off, _ = build_queries(exemplars, None, 0.03, 0.03)
    _, order_on, _ = build_queries(exemplars, visual_negatives, 0.03, 0.03)
    assert order_off == order_on
