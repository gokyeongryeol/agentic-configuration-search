"""fitting/mine — mining hygiene rules and determinism (the gate belongs to the loop, not here).

All GPU injection points (encode/caption/refine) are fakes.
"""

import random

import numpy as np
import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.coco import Split
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.core.prompts import PromptBox
from agentic_configuration_search.fitting import mine

PARAMS = config_from_dict({"dataset": "x"}).mine


def _params(**over):
    base = config_from_dict({"dataset": "x", "mine": over})
    return base.mine


@pytest.fixture
def train(dataset_root):
    return load_dataset(dataset_root, seed=0, fit_train_ratio=0.8).train


@pytest.fixture
def onto(train):
    return train.ontology


# ------------------------------------------------------------- CropCache

def test_crop_cache_keys_by_annotation_id(train, onto, tmp_path):
    cache = mine.CropCache(train, onto, tmp_path / "gt_crops")
    for cid, anns in cache.prompt_pool.items():
        slug = onto.id_to_slug[cid]
        for ann, p in zip(anns, cache.paths[cid]):
            assert p.name == f"a{ann.id}.jpg" and p.exists()
        assert cache.row_of[cid] == {a.id: j for j, a in enumerate(anns)}


def test_crop_cache_excludes_unpromptable(dataset_root, tmp_path):
    import json
    p = dataset_root / "train" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    # add a 1px-wide annotation — cannot be cropped (below MIN_PROMPT_PX)
    coco["annotations"].append({"id": 999, "image_id": 1, "category_id": 1,
                                "bbox": [5, 5, 1, 30], "area": 30,
                                "segmentation": [], "iscrowd": 0})
    p.write_text(json.dumps(coco))
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cache = mine.CropCache(ds.train, ds.ontology, tmp_path / "c")
    assert all(a.id != 999 for a in cache.prompt_pool[1])


# ---------------------------------------------------------- GT exemplar

def _spread_encode(cache):
    """Row i = [i, 0] — an embedding where kcenter picks both ends."""
    def encode_fn(cid, paths):
        return np.array([[float(i), 0.0] for i in range(len(paths))])
    return encode_fn


def test_gt_exemplars_kcenter_and_chosen_ids(train, onto, tmp_path):
    cache = mine.CropCache(train, onto, tmp_path / "c")
    params = _params(exemplars_per_class=2, kcenter_outlier_mad=0.0)
    ex, chosen = mine.gt_exemplars(train, onto, cache, params,
                                   _spread_encode(cache), random.Random(0))
    for cid in cache.prompt_pool:
        name = onto.id_to_name[cid]
        assert len(ex[name]) == 2
        assert all(isinstance(b, PromptBox) for b in ex[name])
        assert len(chosen[cid]) == 2
        # on embedding [0..n-1]×[0], kcenter with 2 = seed (near the middle) + farthest point
        rows = {cache.row_of[cid][a] for a in chosen[cid]}
        assert len(rows) == 2


def test_gt_exemplars_prefetches_all_classes_in_one_call(train, onto, tmp_path):
    """If encode_fn offers prefetch, every class with crops is handed over at once
    before the per-class calls — the real implementation embeds c1..cN in one
    worker run (one model load). The selection equals the one without prefetch."""
    cache = mine.CropCache(train, onto, tmp_path / "c")
    params = _params(exemplars_per_class=2, kcenter_outlier_mad=0.0)
    base = _spread_encode(cache)
    prefetched = []

    def encode_fn(cid, paths):
        return base(cid, paths)
    encode_fn.prefetch = lambda groups: prefetched.append(dict(groups))

    ex, chosen = mine.gt_exemplars(train, onto, cache, params, encode_fn,
                                   random.Random(0))
    want = {cid: cache.paths[cid] for cid, anns in sorted(cache.prompt_pool.items())
            if anns}
    assert prefetched == [want] and list(prefetched[0]) == sorted(want)
    ex0, chosen0 = mine.gt_exemplars(train, onto, cache, params, base, random.Random(0))
    assert chosen == chosen0


def test_gt_exemplars_random_skips_prefetch(train, onto, tmp_path):
    cache = mine.CropCache(train, onto, tmp_path / "c")
    params = _params(exemplar_selection="random", exemplars_per_class=2)

    def encode_fn(cid, paths):
        raise AssertionError("the random path does not call the embedder")
    encode_fn.prefetch = encode_fn

    mine.gt_exemplars(train, onto, cache, params, encode_fn, random.Random(0))


def test_gt_exemplars_random_is_seeded(train, onto, tmp_path):
    cache = mine.CropCache(train, onto, tmp_path / "c")
    params = _params(exemplar_selection="random", exemplars_per_class=2)

    def no_encode(cid, paths):
        raise AssertionError("the random path must not call the embedder")

    ex1, _ = mine.gt_exemplars(train, onto, cache, params, no_encode, random.Random(7))
    ex2, _ = mine.gt_exemplars(train, onto, cache, params, no_encode, random.Random(7))
    assert {n: [b.bbox_xyxy for b in bs] for n, bs in ex1.items()} == \
           {n: [b.bbox_xyxy for b in bs] for n, bs in ex2.items()}


# ------------------------------------------------------------ mine_pools

def _det(box, score, cid):
    return {"box": box, "score": score, "category_id": cid}


def test_mine_pools_separates_fp_fn_with_hygiene(train, onto):
    # image 0: GT(10,10,30,30) hit + FPs caught by hygiene filters + a valid FP
    # image 1: no detections -> its GT is an FN
    dets = [[] for _ in train.images]
    cid0 = train.images[0].annotations[0].category_id
    dets[0] = [
        _det([10, 10, 30, 30], 0.9, cid0),   # TP
        _det([60, 60, 80, 80], 0.04, cid0),  # below score floor -> excluded
        _det([60, 60, 61.5, 80], 0.5, cid0),  # 1.5px wide -> cannot be cropped, excluded
        _det([12, 12, 28, 28], 0.5, cid0),   # inside GT (IoM 1.0) -> real-object overlap, excluded
        _det([60, 10, 80, 30], 0.5, cid0),   # valid FP
    ]
    fp, fn, skipped, unpromptable = mine.mine_pools(
        train, onto, dets, iou_thr=0.5, params=PARAMS)
    name0 = onto.id_to_name[cid0]
    assert len(fp[name0]) == 1 and fp[name0][0]["bbox_xyxy"] == [60, 10, 80, 30]
    assert skipped == {name0: 1}
    assert unpromptable == {name0: 1}
    # image 0's GT was a TP so it is not an FN; the GTs of the other 9 images are all FNs
    assert sum(len(v) for v in fn.values()) == 9


def test_mine_pools_low_score_tp_still_matches(train, onto):
    """The score floor applies only to the FP branch — a below-floor detection that hits a GT is not an FN."""
    dets = [[] for _ in train.images]
    cid0 = train.images[0].annotations[0].category_id
    dets[0] = [_det([10, 10, 30, 30], 0.01, cid0)]   # below the floor (0.05) but a TP
    _, fn, _, _ = mine.mine_pools(train, onto, dets, iou_thr=0.5, params=PARAMS)
    assert all(a.image_id != train.images[0].id for anns in fn.values() for a in anns)


def test_top_negatives_sorted_and_capped():
    pool = {"c": [{"score": 0.2}, {"score": 0.9}, {"score": 0.5}]}
    out = mine.top_negatives(pool, 2)
    assert [r["score"] for r in out["c"]] == [0.9, 0.5]


# ------------------------------------------------- FN exemplar candidate filter

def _fn_setup(train, onto, tmp_path, **params_over):
    cache = mine.CropCache(train, onto, tmp_path / "c")
    params = _params(**params_over)
    return cache, params


def test_fn_headroom_filter(train, onto, tmp_path):
    cache, params = _fn_setup(train, onto, tmp_path,
                              fn_min_candidates=3, fn_exemplars_per_class=2)
    cid = 1
    missed = cache.prompt_pool[cid][:2]           # 2 candidates < 3
    boxes, report = mine.fn_exemplar_candidates(
        train, onto, {cid: missed}, cache, {}, params,
        _spread_encode(cache), random.Random(0))
    assert boxes == {}
    assert report[onto.id_to_name[cid]]["gate_skipped"] == "headroom"


def test_fn_miss_rate_filter(train, onto, tmp_path):
    cache, params = _fn_setup(train, onto, tmp_path,
                              fn_min_candidates=1, fn_min_miss_rate=0.5)
    cid = 1
    missed = cache.prompt_pool[cid][:1]           # 1/5 = 0.2 < 0.5
    boxes, report = mine.fn_exemplar_candidates(
        train, onto, {cid: missed}, cache, {}, params,
        _spread_encode(cache), random.Random(0))
    assert boxes == {}
    assert report[onto.id_to_name[cid]]["gate_skipped"] == "miss_rate"


def test_fn_excludes_already_chosen(train, onto, tmp_path):
    cache, params = _fn_setup(train, onto, tmp_path,
                              fn_min_candidates=0, fn_min_miss_rate=0.0,
                              fn_exemplars_per_class=5)
    cid = 1
    missed = list(cache.prompt_pool[cid])
    chosen = {cid: {missed[0].id}}
    boxes, _ = mine.fn_exemplar_candidates(
        train, onto, {cid: missed}, cache, chosen, params,
        _spread_encode(cache), random.Random(0))
    name = onto.id_to_name[cid]
    assert len(boxes[name]) == len(missed) - 1


def test_fn_candidates_report_added(train, onto, tmp_path):
    cache, params = _fn_setup(train, onto, tmp_path,
                              fn_min_candidates=0, fn_min_miss_rate=0.0,
                              fn_exemplars_per_class=2)
    cid = 1
    missed = list(cache.prompt_pool[cid])         # 5 -> 2 via kcenter
    boxes, report = mine.fn_exemplar_candidates(
        train, onto, {cid: missed}, cache, {}, params,
        _spread_encode(cache), random.Random(0))
    name = onto.id_to_name[cid]
    assert len(boxes[name]) == 2
    assert report[name]["added"] == 2


# --------------------------------------------------- captions -> alias/negative

def test_caption_pools_and_rows(tmp_path):
    boxes = {"cat a": [PromptBox("cat a", "i1.png", [1, 2, 3, 4])],
             "cat b": [{"image": "i2.png", "bbox_xyxy": [5, 6, 7, 8], "score": 0.5}]}
    rows = mine.caption_rows(boxes)
    assert rows == [{"id": "cat a#0", "image": "i1.png", "bbox_xyxy": [1, 2, 3, 4]},
                    {"id": "cat b#0", "image": "i2.png", "bbox_xyxy": [5, 6, 7, 8]}]

    def caption_fn(rows_):
        return {"cat a#0": ["The Red Cap.", "a red cap"], "cat b#0": ["blue ring"]}

    pools = mine.collect_caption_pools(boxes, caption_fn, normalize=True)
    assert pools == {"cat a": ["red cap", "red cap"], "cat b": ["blue ring"]}


def test_alias_cross_pool_dropped_before_llm():
    """Phrases that also appear in another class's pool are removed mechanically before the LLM call —
    if a class runs out of candidates, refine is not even called for it."""
    pools = {"cat a": ["shared phrase"], "cat b": ["shared phrase", "own phrase"]}
    called = []

    def refine_fn(name, class_names, pool, others):
        called.append((name, tuple(pool)))
        return {"alias": pool[0], "reason": "ok"}

    aliases, report = mine.alias_candidates(pools, ["cat a", "cat b"], {}, refine_fn)
    assert [c[0] for c in called] == ["cat b"]     # cat a exhausted — not called
    assert called[0][1] == ("own phrase",)         # pool with shared phrases removed
    assert aliases == {"cat b": "own phrase"}
    assert report["rejected"]["cat a"] == "candidates exhausted by cross-pool conflicts"


def test_alias_manual_priority_and_conflict():
    pools = {"cat a": ["nice phrase"], "cat b": ["cool phrase"]}

    def refine_fn(name, class_names, pool, others):
        return {"alias": "cat a", "reason": "proposal that collides with a class name"}  # induce a conflict

    aliases, report = mine.alias_candidates(
        pools, ["cat a", "cat b"], {"cat a": "manual phrase"}, refine_fn)
    assert aliases == {}
    assert report["rejected"]["cat a"] == "manual alias takes precedence"
    assert "conflicts with another class's prompt" in report["rejected"]["cat b"]


def test_text_negative_candidates_keep_flow():
    pools = {"cat a": ["red herring", "red herring"], "cat b": []}

    def refine_fn(name, class_names, pool, max_keep):
        assert max_keep == 5
        return {"keep": ["red herring"], "rejected": []}

    mined, report = mine.text_negative_candidates(pools, ["cat a", "cat b"],
                                                  refine_fn, max_keep=5)
    assert mined == {"cat a": ["red herring"]}
    assert "cat b" not in report["pools"]          # empty pool is skipped


def test_fn_arm_uses_weaker_trim():
    """The FN arm must also trim outliers — the FN set concentrates truncation, occlusion and label
    errors, so it is noisier than base. But since its purpose is to broadly cover missed
    appearances, it trims less than base (a larger MAD multiple)."""
    from agentic_configuration_search.config import config_from_dict

    cfg = config_from_dict({"dataset": "x"})
    assert (cfg.mine.kcenter_outlier_mad, cfg.mine.fn_kcenter_outlier_mad) == (3.0, 3.5)
    assert cfg.mine.fn_kcenter_outlier_mad > cfg.mine.kcenter_outlier_mad
    import inspect

    from agentic_configuration_search.fitting import mine
    src = inspect.getsource(mine.fn_exemplar_candidates)
    assert "fn_kcenter_outlier_mad" in src, "fn arm kcenter does not receive the trim criterion"
