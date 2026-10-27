"""fit_train subset — an image cap on the training material (lora, rescorer, crop classifier).

The larger fit_train is, the longer all three trainings take. When fit_train is
larger than `subsample.max_images`, the DINOv3 embeddings of GT objects are
clustered per class with k-center, and images holding the most not-yet-covered
centers are picked first (greedy set cover). Contract:
- at or under the cap, or 0 (off), nothing changes
- at least one image per class (rarer classes first), then not-yet-covered
  appearances (centers) first — a large cluster cannot take every slot the way
  it does in a random sample
- background images without GT keep their original share (FP material for the
  rescorer and classifier)
- deterministic — same input, same subset (resume, training reuse cache)
- the loop changes only fit_train — train, fit_dev and holdout stay as they are
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_configuration_search.fitting import subsample


def _split(spec):
    """spec: [[cid, ...] per image] — annotation ids from 1 in image order."""
    images, aid = [], 1
    for i, cids in enumerate(spec):
        anns = []
        for c in cids:
            anns.append(SimpleNamespace(id=aid, category_id=c))
            aid += 1
        images.append(SimpleNamespace(file_name=f"im{i:02d}.png", annotations=anns))
    return SimpleNamespace(images=images)


def _emb(split, look):
    """look: {ann_id: appearance number} — the same number is the same embedding,
    different numbers are orthogonal."""
    by: dict = {}
    for img in split.images:
        for a in img.annotations:
            by.setdefault(a.category_id, []).append(a.id)
    dim = max(look.values()) + 1
    out = {}
    for cid, ids in by.items():
        E = np.zeros((len(ids), dim))
        for r, a in enumerate(ids):
            E[r, look[a]] = 1.0
        out[cid] = (ids, E)
    return out


def test_small_or_disabled_is_untouched():
    sp = _split([[1], [1], [1]])
    emb = _emb(sp, {1: 0, 2: 0, 3: 0})
    for cap in (None, 3, 10):                       # None = no cap (off)
        keep, rep = subsample.select_fit_train(sp, emb, max_images=cap, center_ratio=1.0)
        assert keep == [0, 1, 2] and rep["applied"] is False


def test_zero_cap_is_not_off():
    """A cap of 0 reads as "0 images" and is confused with "off" — off is None (null in config)."""
    sp = _split([[1], [1], [1]])
    for cap in (0, -1):
        with pytest.raises(ValueError, match="None"):
            subsample.select_fit_train(sp, {}, max_images=cap, center_ratio=1.0)


def test_greedy_cover_takes_unseen_appearances_first():
    # class 1 has appearances A(0), B(1), C(2). im00 has many A, im01 has A and B,
    # im02 a single C. With 2 images it must be im01 and im02, which cover A, B and
    # C — not im00 with the most A.
    sp = _split([[1, 1, 1], [1, 1], [1], [1]])
    look = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 0}
    keep, rep = subsample.select_fit_train(sp, _emb(sp, look), max_images=2,
                                           center_ratio=1.0)
    assert keep == [1, 2] and rep["applied"] is True
    assert rep["n_selected"] == 2 and rep["first_pass_coverage"] == 1.0


def test_rare_class_is_always_kept():
    # class 2 is only in im03 — it is always covered on the first pass
    sp = _split([[1, 1], [1, 1], [1], [2]])
    look = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 2}
    keep, _ = subsample.select_fit_train(sp, _emb(sp, look), max_images=2,
                                         center_ratio=1.0)
    assert 3 in keep


def test_rare_class_survives_an_unfinished_first_pass():
    """Even when the first pass does not finish within the budget, every class
    gets at least one image — the cover score does not tell classes apart, so
    images of a large class with many centers could take every slot."""
    sp = _split([[1, 1, 1]] * 30 + [[2]])
    look = {i: i - 1 for i in range(1, 91)} | {91: 90}     # class 1's appearances all differ
    keep, rep = subsample.select_fit_train(sp, _emb(sp, look), max_images=10,
                                           center_ratio=1.0)
    assert 30 in keep and len(keep) == 10
    assert rep["first_pass_coverage"] < 1.0                 # the first pass did not finish


def test_background_cannot_take_every_slot():
    # even when background dominates, the GT images keep a share (one per class,
    # at most the number of GT images)
    sp = _split([[]] * 9 + [[1]])
    keep, rep = subsample.select_fit_train(sp, _emb(sp, {1: 0}), max_images=2,
                                           center_ratio=1.0)
    assert 9 in keep and rep["n_background"] == 1


def test_all_background_fit_train():
    sp = _split([[]] * 6)
    keep, rep = subsample.select_fit_train(sp, {}, max_images=3, center_ratio=0.1)
    assert len(keep) == 3 and rep["n_background"] == 3


def test_background_images_keep_their_share():
    # 4 of 10 images have no GT → with a cap of 5, 2 background images
    sp = _split([[1], [1], [], [1], [], [1], [], [1], [], [1]])
    look = {i: i - 1 for i in range(1, 7)}
    keep, rep = subsample.select_fit_train(sp, _emb(sp, look), max_images=5,
                                           center_ratio=1.0)
    bg = [i for i in keep if not sp.images[i].annotations]
    assert len(keep) == 5 and len(bg) == 2 and rep["n_background"] == 2


def test_second_pass_and_fill_when_budget_exceeds_cover():
    # with a single appearance the first pass ends after 1 image — the next passes fill the rest
    sp = _split([[1], [1], [1], [1], [1]])
    look = {i: 0 for i in range(1, 6)}
    keep, rep = subsample.select_fit_train(sp, _emb(sp, look), max_images=3,
                                           center_ratio=1.0)
    assert len(keep) == 3 and rep["passes"] >= 2


def test_images_without_embeddable_gt_only_fill():
    # im02's GT cannot be cropped, so it has no embedding — with no center to
    # cover, it only gets in if slots are left
    sp = _split([[1], [1], [1]])
    emb = {1: ([1, 2], np.eye(2))}
    keep, _ = subsample.select_fit_train(sp, emb, max_images=2, center_ratio=1.0)
    assert keep == [0, 1]


def test_selection_is_deterministic_and_order_free():
    sp = _split([[1, 2], [1], [2], [1, 1], [2, 2], [1]])
    look = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 2, 7: 3, 8: 1, 9: 4}
    emb = _emb(sp, look)
    a = subsample.select_fit_train(sp, emb, max_images=3, center_ratio=0.5)
    b = subsample.select_fit_train(sp, dict(reversed(list(emb.items()))),
                                   max_images=3, center_ratio=0.5)
    assert a == b


def test_center_count_follows_ratio():
    sp = _split([[1]] * 20)
    look = {i: i - 1 for i in range(1, 21)}
    _, rep = subsample.select_fit_train(sp, _emb(sp, look), max_images=5,
                                        center_ratio=0.1)
    assert rep["centers"] == {"1": 2}


# --------------------------------------------------------------- loop wiring

def _fake_encoder(calls):
    def encode_fn(cid, paths):
        calls.append(cid)
        # file name a{ann_id}.jpg — a different appearance per annotation
        ids = [int(p.stem[1:]) for p in paths]
        E = np.zeros((len(ids), 64))
        for r, a in enumerate(ids):
            E[r, a % 64] = 1.0
        return E
    return encode_fn


def _loop(root, out, encode_fn, **sub):
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop.runner import Loop
    cfg = config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(root),
                            "subsample": {"max_images": 4, **sub}})
    return Loop(cfg, root, out, encode_fn=encode_fn)


def test_loop_trains_on_the_subset_only(dataset_root, tmp_path):
    from agentic_configuration_search.core.data import load_dataset
    full = load_dataset(dataset_root)
    calls = []
    loop = _loop(dataset_root, tmp_path / "run", _fake_encoder(calls))
    names = {e.file_name for e in loop.ds.fit_train.images}
    assert len(names) == 4 and calls
    assert names <= {e.file_name for e in full.fit_train.images}
    # only the training material shrinks — mining (train), verdicts (fit_dev)
    # and adoption (holdout) stay as they are
    for attr in ("train", "fit_dev", "holdout"):
        assert ([e.file_name for e in getattr(loop.ds, attr).images]
                == [e.file_name for e in getattr(full, attr).images])
    rec = json.loads((tmp_path / "run" / subsample.SUBSET_FILE).read_text())
    assert sorted(rec["images"]) == sorted(names) and rec["report"]["applied"]


def test_selection_reads_only_fit_train_gt(dataset_root, tmp_path):
    """Material purity — the only crops embedded are fit_train GT (no fit_dev or holdout boxes)."""
    from agentic_configuration_search.core.data import load_dataset
    full = load_dataset(dataset_root)
    allowed = {a.id for e in full.fit_train.images for a in e.annotations}
    seen = []

    def enc(cid, paths):
        seen.extend(int(p.stem[1:]) for p in paths)
        return _fake_encoder([])(cid, paths)
    _loop(dataset_root, tmp_path / "run", enc)
    assert seen and set(seen) <= allowed


def test_run_summary_records_the_subset(dataset_root, tmp_path):
    loop = _loop(dataset_root, tmp_path / "run", _fake_encoder([]))
    assert loop.fit_train_summary() == {
        "images": 4, "images_before_subset": 8,
        "subset": json.loads((tmp_path / "run" / subsample.SUBSET_FILE).read_text())["report"]}
    small = _loop(dataset_root, tmp_path / "run2", None, max_images=100)
    assert small.fit_train_summary() == {"images": 8, "images_before_subset": 8,
                                         "subset": None}


def test_resume_reuses_the_recorded_subset(dataset_root, tmp_path):
    first = _loop(dataset_root, tmp_path / "run", _fake_encoder([]))

    def boom(cid, paths):
        raise AssertionError("a recorded subset is not embedded again")
    again = _loop(dataset_root, tmp_path / "run", boom)
    assert ([e.file_name for e in again.ds.fit_train.images]
            == [e.file_name for e in first.ds.fit_train.images])


def test_changed_settings_reselect(dataset_root, tmp_path):
    _loop(dataset_root, tmp_path / "run", _fake_encoder([]))
    calls = []
    loop = _loop(dataset_root, tmp_path / "run", _fake_encoder(calls), max_images=3)
    assert calls and len(loop.ds.fit_train.images) == 3


def test_changed_embedder_reselects(dataset_root, tmp_path):
    _loop(dataset_root, tmp_path / "run", _fake_encoder([]))
    calls = []
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop.runner import Loop
    cfg = config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(dataset_root),
                            "subsample": {"max_images": 4},
                            "mine": {"embedder_model": "other/model"}})
    Loop(cfg, dataset_root, tmp_path / "run", encode_fn=_fake_encoder(calls))
    assert calls, "a changed embedding model picks the subset again"


def test_record_with_missing_images_reselects(dataset_root, tmp_path):
    _loop(dataset_root, tmp_path / "run", _fake_encoder([]))
    path = tmp_path / "run" / subsample.SUBSET_FILE
    rec = json.loads(path.read_text())
    rec["images"][0] = "gone.png"                          # the record no longer matches fit_train
    path.write_text(json.dumps(rec))
    calls = []
    loop = _loop(dataset_root, tmp_path / "run", _fake_encoder(calls))
    assert calls and len(loop.ds.fit_train.images) == 4


def test_large_fit_train_without_encoder_is_an_error(dataset_root, tmp_path):
    with pytest.raises(ValueError, match="encode_fn"):
        _loop(dataset_root, tmp_path / "run", None)


def test_small_fit_train_needs_no_encoder(dataset_root, tmp_path):
    loop = _loop(dataset_root, tmp_path / "run", None, max_images=100)
    assert len(loop.ds.fit_train.images) == 8
    assert not (tmp_path / "run" / subsample.SUBSET_FILE).exists()


def test_config_validates_subsample():
    from agentic_configuration_search.config import ConfigError, config_from_dict
    for bad in ({"max_images": 0}, {"max_images": -1}, {"center_ratio": 0.0},
                {"center_ratio": 1.5}):
        with pytest.raises(ConfigError):
            config_from_dict({"dataset": "x", "subsample": bad})
    with pytest.raises(ConfigError, match="null"):
        config_from_dict({"dataset": "x", "subsample": {"max_images": 0}})
    cfg = config_from_dict({"dataset": "x", "subsample": {"max_images": None}})
    assert cfg.subsample.max_images is None


def test_null_cap_turns_subsampling_off(dataset_root, tmp_path):
    from agentic_configuration_search.core.data import load_dataset
    n = len(load_dataset(dataset_root).fit_train.images)
    loop = _loop(dataset_root, tmp_path / "run", None, max_images=None)
    assert len(loop.ds.fit_train.images) == n
    assert not (tmp_path / "run" / subsample.SUBSET_FILE).exists()


def test_yaml_null_and_wrong_types(tmp_path):
    """A real YAML null is None (off). Non-integers (a quoted "null", a float, true) are
    stopped at config time with ConfigError — not later as a TypeError in arithmetic."""
    from agentic_configuration_search.config import ConfigError, config_from_dict, load_config
    p = tmp_path / "c.yaml"
    p.write_text("dataset: x\nsubsample:\n  max_images: null\nrescorer:\n  max_train_rows: null\n")
    cfg = load_config(p)
    assert cfg.subsample.max_images is None and cfg.rescorer.max_train_rows is None
    for bad in ("null", 400.0, True):
        with pytest.raises(ConfigError):
            config_from_dict({"dataset": "x", "subsample": {"max_images": bad}})
        with pytest.raises(ConfigError):
            config_from_dict({"dataset": "x", "rescorer": {"max_train_rows": bad}})
