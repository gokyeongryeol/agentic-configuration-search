"""visual_negatives takes the same mining sources as text_negatives (fp_crop · gt_crop · background)."""

from pathlib import Path

import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.diagnose.story import strategy_label
from agentic_configuration_search.loop.materialize import materialize
from agentic_configuration_search.planner.axes import (AXES, NEGATIVE_AXES, DatasetFacts,
                                                       axis_values, validate_axes)
from tests.test_materialize_purity import _fakes, _shuffle_holdout_gt

VN_GT = {"exemplar": "base", "visual_negatives": "gt_crop"}


def test_negative_axes_share_the_same_sources():
    assert AXES["visual_negatives"] == AXES["text_negatives"] == \
        (False, "fp_crop", "gt_crop", "background")


@pytest.mark.parametrize("axis", NEGATIVE_AXES)
def test_single_class_drops_gt_crop_for_both_negative_axes(axis):
    facts = DatasetFacts(n_classes=1)
    assert "gt_crop" not in axis_values(facts)[axis]
    axes = {axis: "gt_crop", **({"exemplar": "base"} if axis == "visual_negatives" else {})}
    validate_axes(axes)                                   # passes without facts
    with pytest.raises(ValueError, match="2\\+ classes"):
        validate_axes(axes, facts=facts)


def test_visual_negatives_still_need_exemplar():
    with pytest.raises(ValueError, match="requires an exemplar"):
        validate_axes({"visual_negatives": "background"})


def _materialize(root: Path, out: Path, axes: dict):
    cfg = config_from_dict({"dataset": str(root)})
    ds = load_dataset(root, seed=0, fit_train_ratio=0.8)
    return ds, materialize(cfg, ds, axes, out, seed=0, **_fakes())


def test_gt_crop_visual_negatives_are_other_classes_gt_without_a_mining_pass(
        dataset_root, tmp_path):
    """gt_crop uses GT crops only — materials come out without a mining pass (client),
    and each class's negatives are other classes' GT boxes."""
    ds, mat = _materialize(dataset_root, tmp_path / "m", VN_GT)   # client=None
    assert mat.visual_negatives
    gt_class = {(str(ds.train.image_path(e)), tuple(round(v) for v in a.bbox_xyxy)):
                ds.ontology.id_to_name[a.category_id]
                for e in ds.train.images for a in e.annotations}
    for name, boxes in mat.visual_negatives.items():
        for b in boxes:
            owner = gt_class[(b.image, tuple(round(v) for v in b.bbox_xyxy))]
            assert owner != name


def test_gt_crop_visual_negatives_are_holdout_pure(dataset_root, tmp_path):
    """The same contract as the purity guard — shuffling holdout GT leaves the material fingerprint unchanged."""
    _, a = _materialize(dataset_root, tmp_path / "a", VN_GT)
    _shuffle_holdout_gt(dataset_root)
    _, b = _materialize(dataset_root, tmp_path / "b", VN_GT)
    assert a.fingerprint() == b.fingerprint()


def test_report_label_names_the_source():
    assert strategy_label({"visual_negatives": "gt_crop"}) == \
        "Show look-alikes in photos (from other kinds)"
    assert strategy_label({"text_negatives": "background"}) == \
        "Name look-alikes in words (from the background)"


def test_prompt_explains_only_open_sources():
    from agentic_configuration_search.planner.board import _source_line
    two = _source_line(axis_values(DatasetFacts(n_classes=2)))
    one = _source_line(axis_values(DatasetFacts(n_classes=1)))
    assert "gt_crop" in two and "visual_negatives" in two
    assert "gt_crop" not in one and "fp_crop" in one
    assert _source_line(axis_values(), disabled=NEGATIVE_AXES) == ""


def test_remedy_covers_the_cause_specific_visual_sources():
    """Cause-matched visual sources are remedies too — background FPs get background, class confusion gt_crop."""
    from agentic_configuration_search.planner.deterministic import REMEDY
    assert ("visual_negatives", "background") in REMEDY["background"]
    assert ("visual_negatives", "gt_crop") in REMEDY["class_confusion"]


def test_deterministic_proposes_the_cause_specific_visual_source():
    """With the general fp_crop as the champion, the cause-matched source is the next variant."""
    from agentic_configuration_search.planner import deterministic
    champ = validate_axes({"exemplar": "base", "visual_negatives": "fp_crop"})
    hist = [{"round": 0, "champion": {"axes": champ}, "candidates": []}]
    out = deterministic.propose(hist, {"train": {"summary": {"fp": {"background": 9}}}},
                                max_candidates=10)
    vis = {c["axes"]["visual_negatives"] for c in out["candidates"]
           if c["remedy"].startswith("background:visual_negatives")}
    assert vis == {"background"}
    out = deterministic.propose(hist, {"train": {"summary": {"fp": {"class_confusion": 9}}}},
                                max_candidates=10)
    vis = {c["axes"]["visual_negatives"] for c in out["candidates"]
           if c["remedy"].startswith("class_confusion:visual_negatives")}
    assert vis == {"gt_crop"}
