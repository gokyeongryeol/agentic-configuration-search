"""Alias caption source — not the served exemplars but plenty of large-enough GT crops per class.

Using the served exemplars as the caption source leaves a champion with
exemplar_base_k=1 a single crop per class, and for small objects that crop is
tens of px. The VLM returned an empty caption list for such crops (the response
schema allowed an empty array) and mining quietly came out with no material.

- Source: among each class's GT crops with a short side of at least
  mine.caption_min_side px, mine.caption_crops_per_class are picked by the same
  selection rule as exemplars
- A class for which the captioner gave no phrase gets a reason in the mining
  report, and the VLM captioner warns when crops get no caption
"""

import logging
import random

import numpy as np
import pytest

from agentic_configuration_search.config import ConfigError, config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.fitting import mine
from agentic_configuration_search.infer import phrases
from tests.conftest import make_split


def _params(**over):
    return config_from_dict({"dataset": "x", "mine": over}).mine


@pytest.fixture
def sized(tmp_path):
    """Class 1: 6 large boxes + 4 small ones; class 2: small boxes only."""
    ds = tmp_path / "ds"
    imgs = []
    for i in range(10):
        side = 60 if i < 6 else 12
        imgs.append({"name": f"a{i}.png", "w": 100, "h": 80,
                     "anns": [(1, 5, 5, side, side), (2, 70, 50, 10, 10)]})
    make_split(ds / "train", imgs)
    make_split(ds / "valid", [{"name": "v.png", "w": 100, "h": 80,
                               "anns": [(1, 5, 5, 30, 30)]}])
    make_split(ds / "test", [{"name": "t.png", "w": 100, "h": 80,
                              "anns": [(1, 5, 5, 30, 30)]}])
    split = load_dataset(ds, seed=0, fit_train_ratio=1.0).train
    return split, split.ontology, tmp_path / "crops"


def _encode(cid, paths):
    return np.array([[float(i), 0.0] for i in range(len(paths))])


def test_caption_source_keeps_large_crops_up_to_the_quota(sized):
    train, onto, crops = sized
    cache = mine.CropCache(train, onto, crops)
    params = _params(caption_crops_per_class=4, caption_min_side=32)
    src, report = mine.caption_sources(train, onto, cache, params, _encode,
                                       random.Random(0))
    big = src[onto.id_to_name[1]]
    assert len(big) == 4
    assert all(min(b.bbox_xyxy[2] - b.bbox_xyxy[0], b.bbox_xyxy[3] - b.bbox_xyxy[1]) >= 32
               for b in big)
    assert onto.id_to_name[2] not in src                  # no large crop
    assert report[onto.id_to_name[2]] == {"pool": 10, "large_enough": 0, "chosen": 0}
    assert report[onto.id_to_name[1]]["large_enough"] == 6


def test_caption_source_does_not_depend_on_served_exemplars(sized):
    """With a single served exemplar the caption source still fills its quota — they are different materials."""
    train, onto, crops = sized
    cache = mine.CropCache(train, onto, crops)
    one = _params(exemplar_base_per_class=1, caption_crops_per_class=5)
    src, _ = mine.caption_sources(train, onto, cache, one, _encode, random.Random(0))
    assert len(src[onto.id_to_name[1]]) == 5


def test_config_validates_caption_quota():
    with pytest.raises(ConfigError, match="caption_crops_per_class"):
        _params(caption_crops_per_class=0)
    with pytest.raises(ConfigError, match="caption_min_side"):
        _params(caption_min_side=1)


def test_empty_caption_pool_is_reported_not_silent():
    pools = {"cat a": [], "cat b": ["own phrase"]}
    _aliases, report = mine.alias_candidates(
        pools, ["cat a", "cat b"], {}, lambda *a: {"alias": None, "reason": "x"})
    assert "empty caption pool" in report["rejected"]["cat a"]


def test_vlm_warns_when_crops_get_no_caption(tmp_path, caplog):
    from PIL import Image

    img = tmp_path / "i.png"
    Image.new("RGB", (64, 64)).save(img)
    rows = [{"id": "a#0", "image": str(img), "bbox_xyxy": [0, 0, 40, 40]},
            {"id": "b#0", "image": str(img), "bbox_xyxy": [10, 10, 50, 50]}]
    with caplog.at_level(logging.WARNING, logger=phrases.log.name):
        out = phrases.vlm_caption(rows, lambda m, **kw: {"captions": []})
    assert out == {}
    assert any("no caption" in r.getMessage() and "2" in r.getMessage()
               for r in caplog.records)


def test_vlm_caption_sends_the_system_prompt(tmp_path):
    from PIL import Image

    img = tmp_path / "i.png"
    Image.new("RGB", (64, 64)).save(img)
    seen = []

    def chat(messages, **kw):
        seen.append(messages)
        return {"captions": [{"crop": 0, "phrases": ["red cap"]}]}

    out = phrases.vlm_caption([{"id": "a#0", "image": str(img), "bbox_xyxy": [0, 0, 40, 40]}],
                              chat)
    assert out == {"a#0": ["red cap"]}
    assert seen[0][0] == {"role": "system", "content": phrases.VLM_CAPTION_SYSTEM}


def test_caption_source_prefetches_all_classes_once(sized):
    """The embedding worker runs once, not per class — same cache key (all crops) as gt_exemplars."""
    train, onto, crops = sized
    cache = mine.CropCache(train, onto, crops)
    calls = []

    def encode_fn(cid, paths):
        return _encode(cid, paths)
    encode_fn.prefetch = lambda by_cid: calls.append(sorted(by_cid))
    mine.caption_sources(train, onto, cache, _params(caption_crops_per_class=2),
                         encode_fn, random.Random(0))
    assert calls == [[1]]          # only classes with a large crop, once, with all crop paths


def test_caption_source_random_selection_is_seeded(sized):
    train, onto, crops = sized
    cache = mine.CropCache(train, onto, crops)
    params = _params(exemplar_selection="random", caption_crops_per_class=3)
    a, _ = mine.caption_sources(train, onto, cache, params, None, random.Random(7))
    b, _ = mine.caption_sources(train, onto, cache, params, None, random.Random(7))
    assert a == b and len(a[onto.id_to_name[1]]) == 3


def test_class_without_caption_source_has_a_reason(dataset_root, tmp_path):
    """A class that never reached the captioner (no large crop) also gets a reason in the alias report."""
    import json

    from agentic_configuration_search.loop.materialize import materialize
    from tests.test_materialize_purity import _fakes

    cfg = config_from_dict({"dataset": str(dataset_root), "mine": {"caption_min_side": 64}})
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    mat = materialize(cfg, ds, {"exemplar": "base", "alias": "vlm"}, tmp_path / "m",
                      seed=0, **_fakes())
    assert mat.aliases == {}
    rep = json.loads((tmp_path / "m" / "mine_reports.json").read_text())
    assert all("caption_min_side" in r for r in rep["alias"]["rejected"].values())
    assert set(rep["alias"]["rejected"]) == set(rep["alias_caption_source"])
