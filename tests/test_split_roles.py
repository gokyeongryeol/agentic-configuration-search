"""Guard — split-role discipline.

Contract pinned here:
- train is split into fit_train/fit_dev 80/20, stratified per image (by class-combination group).
  Fixed seed; the boundary is unchanged across repeated calls and rounds.
- fit_train ∪ fit_dev = train (all of it), no overlap.
- No path reads test — loading must succeed even if the directory is missing.
- valid (holdout) must share train's ontology — a mismatch is an explicit error.
"""

import json
import shutil

import pytest

from agentic_configuration_search.core.coco import ANNOTATIONS_FILENAME
from agentic_configuration_search.core.data import load_dataset


def test_split_is_stratified_fixed_and_exhaustive(dataset_root):
    ds1 = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    ds2 = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)

    ft1 = sorted(im.file_name for im in ds1.fit_train.images)
    fd1 = sorted(im.file_name for im in ds1.fit_dev.images)
    assert ft1 == sorted(im.file_name for im in ds2.fit_train.images)  # seed reproducibility
    assert not set(ft1) & set(fd1)                                     # no overlap
    train = sorted(im.file_name for im in ds1.train.images)
    assert sorted(ft1 + fd1) == train                                # exhaustive
    assert len(ft1) == 8 and len(fd1) == 2                             # 80/20 (10 train images)
    # stratified: the two classes (5:5) get one each in fit_dev
    dev_cats = sorted(a.category_id
                      for im in ds1.fit_dev.images for a in im.annotations)
    assert dev_cats == [1, 2]


def test_different_seed_moves_boundary(dataset_root):
    names = set()
    for seed in range(4):
        ds = load_dataset(dataset_root, seed=seed, fit_train_ratio=0.8)
        names.add(tuple(sorted(im.file_name for im in ds.fit_dev.images)))
    assert len(names) > 1  # the seed really moves the boundary (not a fixed constant)


def test_test_is_never_read_when_valid_exists(dataset_root):
    """If valid exists, the loop does not read test.

    A split rewritten every round cannot be used for the final report, so the larger one (valid)
    goes to selection and test is kept only for measuring the confirmed champion once
    (scripts/predict.py --split test). Test is promoted as a substitute **only when valid is
    absent**, which test_test_substitutes_when_valid_absent pins.
    """
    shutil.rmtree(dataset_root / "test")   # loading must succeed even without test
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    assert not hasattr(ds, "test")
    assert ds.holdout_split == "valid"
    assert len(ds.holdout.images) == 2


def test_valid_is_preferred_when_both_exist(dataset_root):
    """If both exist, valid wins — test stays sealed."""
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    assert ds.holdout_split == "valid"
    assert sorted(im.file_name for im in ds.holdout.images) == ["v1.png", "v2.png"]


def test_test_substitutes_when_valid_absent(dataset_root):
    """If valid is absent, test stands in as the holdout.

    No sealed split remains then — selection and reporting happen on the same split, so holdout
    numbers are optimistically biased. Which split was used must therefore be recorded in the
    output, and holdout_split is that record.
    """
    shutil.rmtree(dataset_root / "valid")
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    assert ds.holdout_split == "test"
    assert [im.file_name for im in ds.holdout.images] == ["t1.png"]


def test_missing_both_holdout_splits_errors(dataset_root):
    """With neither valid nor test there is nowhere to judge adoption — explicit error."""
    shutil.rmtree(dataset_root / "valid")
    shutil.rmtree(dataset_root / "test")
    with pytest.raises(FileNotFoundError, match="valid"):
        load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)


def test_malformed_valid_errors_instead_of_substituting(dataset_root):
    """If valid/ **exists** but lacks annotations, it is corruption, not absence.

    Silently falling back to test would turn a loud failure into a silent protocol change — the run
    would proceed without knowing the sealed split was lost. The fallback must trigger only when
    valid/ is entirely absent.
    """
    (dataset_root / "valid" / ANNOTATIONS_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match=ANNOTATIONS_FILENAME):
        load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)


def test_malformed_test_errors_when_it_is_the_substitute(dataset_root):
    """A test/ promoted as a substitute gets the same rule — present but empty is an error."""
    shutil.rmtree(dataset_root / "valid")
    (dataset_root / "test" / ANNOTATIONS_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match=ANNOTATIONS_FILENAME):
        load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)


def test_substitute_ontology_mismatch_errors(dataset_root):
    """The ontology consistency check applies equally to a test promoted as a substitute."""
    shutil.rmtree(dataset_root / "valid")
    p = dataset_root / "test" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    coco["categories"] = [{"id": 1, "name": "other", "supercategory": "none"}]
    p.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="ontology differs from train"):
        load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)


def test_ontology_mismatch_errors(dataset_root):
    p = dataset_root / "valid" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    coco["categories"] = [{"id": 1, "name": "other", "supercategory": "none"}]
    p.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="ontology differs from train"):
        load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)


def test_missing_split_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope", seed=0, fit_train_ratio=0.8)
