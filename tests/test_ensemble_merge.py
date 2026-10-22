"""ensemble.merge_modalities vectorized — results must equal the old implementation (a 1×1 IoU per pair)."""

import random
import time

import numpy as np
import pytest

from agentic_configuration_search.core.metrics import iou_matrix
from agentic_configuration_search.diagnose import ensemble


def _reference(semantic_dets, text_dets, iou=ensemble.DEDUP_IOU):
    pool = [{**d, "modalities": list(d.get("modalities", ["semantic"]))}
            for d in semantic_dets]
    pool += [{**d, "modalities": list(d.get("modalities", ["text"]))}
             for d in text_dets]
    pool.sort(key=lambda d: -d["score"])
    kept = []
    for det in pool:
        target = None
        for k in kept:
            if k["category_id"] != det["category_id"]:
                continue
            overlap = iou_matrix(np.array([k["box"]], dtype=float),
                                 np.array([det["box"]], dtype=float))[0, 0]
            if overlap >= iou:
                target = k
                break
        if target is None:
            kept.append(det)
        else:
            target["modalities"] = sorted(set(target["modalities"]) | set(det["modalities"]))
            if "sources" in det or "sources" in target:
                target["sources"] = sorted(set(target.get("sources", []))
                                           | set(det.get("sources", [])))
    return kept


def _dets(rng, n, n_cls, base=None, tag_sources=False):
    out = []
    for _ in range(n):
        if base and rng.random() < 0.4:
            b = list(rng.choice(base)["box"])
            b = [b[0] + rng.uniform(-3, 3), b[1], b[2] + rng.uniform(-3, 3), b[3]]
        else:
            x, y = rng.uniform(0, 600), rng.uniform(0, 600)
            b = [x, y, x + rng.uniform(4, 80), y + rng.uniform(4, 80)]
        d = {"box": b, "score": round(rng.uniform(0.03, 0.95), 3),
             "category_id": rng.randrange(1, n_cls + 1)}
        if tag_sources and rng.random() < 0.5:
            d["sources"] = [rng.choice(["full", "tile"])]
        out.append(d)
    return out


@pytest.mark.parametrize("seed", range(12))
def test_merge_modalities_matches_pairwise_reference(seed):
    rng = random.Random(seed)
    n_cls = rng.choice([1, 3])
    sem = _dets(rng, rng.choice([0, 5, 60]), n_cls, tag_sources=True)
    txt = _dets(rng, rng.choice([0, 5, 60]), n_cls, base=sem, tag_sources=True)
    assert ensemble.merge_modalities(sem, txt) == _reference(sem, txt)


def test_merge_modalities_is_fast_on_topk_sized_inputs():
    rng = random.Random(0)
    sem = _dets(rng, 300, 1)
    txt = _dets(rng, 300, 1, base=sem)
    t0 = time.perf_counter()
    ensemble.merge_modalities(sem, txt)
    assert time.perf_counter() - t0 < 0.5
