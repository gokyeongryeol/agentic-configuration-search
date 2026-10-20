"""build_text_queries — schema invariants for text-modality queries.

Core contract: every prompt's box must be an empty list for the detector to take the text-modality
branch. If any box is filled it is silently treated as semantic — pinned via the schema.
"""

import pytest

from agentic_configuration_search.core.prompts import build_text_queries


def test_all_boxes_empty_and_no_prompt_image():
    queries, order, alias = build_text_queries(
        ["cat a", "cat b"], {}, box_threshold=0.03, multimodal_threshold=0.05)
    assert len(queries) == 1
    q = queries[0]
    assert "prompt_image" not in q            # the text branch does not read it — so do not include it
    assert all(p["box"] == [] for p in q["prompts"])
    assert [p["text"] for p in q["prompts"]] == ["cat a", "cat b"]
    assert order == ["cat a", "cat b"]        # prompt_order uses the original class names
    assert alias == {"cat a": "cat a", "cat b": "cat b"}


def test_alias_is_additive_with_reverse_map():
    """Aliases are added alongside the class-name prompt — the name prompt is always kept."""
    queries, order, alias = build_text_queries(
        ["cable-damage"], {"cable-damage": "damaged cable"},
        box_threshold=0.03, multimodal_threshold=0.05)
    assert [p["text"] for p in queries[0]["prompts"]] \
        == ["cable-damage", "damaged cable"]                    # name + alias
    assert order == ["cable-damage"]                            # order has one entry per class
    assert alias == {"cable-damage": "cable-damage",
                     "damaged cable": "cable-damage"}           # for reverse mapping (includes itself)


def test_negatives_duplicated_per_positive_phrase():
    """Negatives are duplicated targeting both the name and the alias — the detector's suppression
    gate only matches positives whose category id matches the negative_target string, so attaching
    only to the name would let the alias channel bypass suppression."""
    queries, _, _ = build_text_queries(
        ["cable-damage"], {"cable-damage": "damaged cable"},
        box_threshold=0.03, multimodal_threshold=0.05,
        visual_negatives={"cable-damage": ["intact cable"]})
    negs = [(p["text"], p["negative_target"]) for p in queries[0]["prompts"]
            if p.get("negative_target")]
    assert negs == [("intact cable", "cable-damage"),
                    ("intact cable", "damaged cable")]


def test_negatives_single_target_without_alias():
    queries, _, _ = build_text_queries(
        ["cat a", "cat b"], {}, box_threshold=0.03, multimodal_threshold=0.05,
        visual_negatives={"cat a": ["metal pole"]})
    negs = [(p["text"], p["negative_target"]) for p in queries[0]["prompts"]
            if p.get("negative_target")]
    assert negs == [("metal pole", "cat a")]


def test_identity_alias_adds_nothing():
    queries, _, alias = build_text_queries(
        ["cat a"], {"cat a": "cat a"}, box_threshold=0.03, multimodal_threshold=0.05)
    assert [p["text"] for p in queries[0]["prompts"]] == ["cat a"]
    assert alias == {"cat a": "cat a"}


def test_alias_equal_to_other_class_name_rejected():
    with pytest.raises(ValueError, match="text alias collision"):
        build_text_queries(["a", "b"], {"b": "a"},
                           box_threshold=0.03, multimodal_threshold=0.05)


def test_alias_collision_rejected():
    with pytest.raises(ValueError, match="text alias collision"):
        build_text_queries(["a", "b"], {"a": "same", "b": "same"},
                           box_threshold=0.03, multimodal_threshold=0.05)


def test_empty_class_names_rejected():
    with pytest.raises(ValueError):
        build_text_queries([], {}, box_threshold=0.03, multimodal_threshold=0.05)


def test_parse_prediction_reverses_alias():
    from agentic_configuration_search.clients.wire import parse_prediction
    pred = {"text": ["damaged cable"], "boxes": [[1, 2, 3, 4]], "scores": [0.7],
            "labels": [0]}
    dets = parse_prediction(pred, {"cable-damage": 1},
                            alias_to_name={"damaged cable": "cable-damage"})
    assert dets == [{"box": [1.0, 2.0, 3.0, 4.0], "score": 0.7, "category_id": 1}]
