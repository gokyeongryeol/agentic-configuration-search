"""rescorer axis round-wiring e2e — a stub whose only signal is in the embedding.

Scenario: scores alone do not separate TP/FP (F1-max 2/3), while the embedding carries a TP
signal -> below_threshold/background diagnosis -> rescorer candidate -> fit discriminator (fit_train)
and α (fit_dev) -> re-evaluate with calibrated scores -> gate accepts. Once the champion carries
the rescorer, the next round's diagnosis also runs on the calibrated channel.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.coco import Split
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from tests.conftest import make_split


@pytest.fixture
def root(tmp_path: Path) -> Path:
    size = lambda i: 20 + (i % 3) * 10  # noqa: E731 — size variety (see test_loop)
    ds = tmp_path / "ds"
    # 45 images -> fit_train 36 × 2 detections = 72 rows ≥ 64 — below that, HistGradientBoosting's
    # min_samples_leaf=20 degenerates P̂ to a constant (see the RescorerCfg.min_samples
    # comment)
    make_split(ds / "train", [
        {"name": f"tr{i:02d}.png", "w": 100, "h": 80,
         "anns": [(1 if i % 2 else 2, 10, 10, size(i), size(i))]}
        for i in range(45)])
    make_split(ds / "valid", [        # the loop's holdout
        {"name": f"t{i}.png", "w": 100, "h": 80,
         "anns": [(1 if i < 4 else 2, 10, 10, size(i), size(i))]}
        for i in range(8)])
    make_split(ds / "test", [{"name": "v.png", "w": 100, "h": 80,   # report-only — the loop never reads it
                              "anns": [(1, 10, 10, 20, 20)]}])
    return ds


def _fake_infer(root):
    splits = {n: Split.load(root / n) for n in ("train", "valid")}
    by_path = {str(s.image_path(e)): e for s in splits.values() for e in s.images}

    def fake(ctx, client, config, queries, ontology, image_paths,
             alias_to_name=None, return_embeddings=False, **kw):
        out = []
        for p in image_paths:
            entry = by_path[str(p)]
            idx = int(re.sub(r"\D", "", entry.file_name) or 0)
            dets = []
            for a in entry.annotations:
                # TP — scores alternate 0.4/0.6 (uninformative), +signal in embedding
                d = {"box": list(a.bbox_xyxy), "score": 0.4 + 0.2 * (idx % 2),
                     "category_id": a.category_id, "sources": ["full"]}
                if return_embeddings:
                    d["embedding"] = [2.0, 0.0, 0.0, 0.0]
                dets.append(d)
                # FP — score 0.5 (middle of the TP distribution), −signal in embedding
                d = {"box": [60.0, 50.0, 80.0, 70.0], "score": 0.5,
                     "category_id": a.category_id, "sources": ["full"]}
                if return_embeddings:
                    d["embedding"] = [-2.0, 0.0, 0.0, 0.0]
                dets.append(d)
            out.append(dets)
        return out
    return fake


def _cfg(root):
    return config_from_dict({
        "dataset": str(root),
        "rescorer": {"min_samples": 64, "min_samples_per_class": 1,
                     "margin": 0.005, "ensemble_members": 2},
    })


def _fakes():
    return dict(
        encode_fn=lambda cid, paths: np.array(
            [[float(i), 0.0] for i in range(len(paths))]),
        caption_fn=lambda rows: {r["id"]: [f"{r['id'].split('#')[0]} look"]
                                 for r in rows},
        vlm_caption_fn=lambda rows: {r["id"]: [f"{r['id'].split('#')[0]} look"]
                                 for r in rows},
        refine_negatives_fn=lambda name, cn, pool, mk: {"keep": [],
                                                        "rejected": []},
        refine_alias_fn=lambda name, cn, pool, others: {"alias": None,
                                                        "reason": "none"},
        client=object())


def test_rescorer_candidate_adopted_end_to_end(root, tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root))
    out = tmp_path / "run"
    result = run_rounds(root, out_dir=out, planner="deterministic", rounds=2,
                        cfg=_cfg(root), **_fakes())

    base_micro = result.decisions[0]["micro_f_beta"]
    assert base_micro < 0.75                              # scores alone cannot separate them
    r1 = result.decisions[1]
    assert r1["adopted"] is True, r1
    assert result.champion["axes"]["rescorer"] is True
    # per-class α — only classes with proven gain are calibrated, so the improvement is partial
    assert result.champion["eval"]["micro_f_beta"] > base_micro + 0.1
    # tau_test (legacy name; estimated on holdout) was re-estimated on the calibrated channel
    assert set(result.champion["eval"]["tau_test"]) == {1, 2}
    # rescorer artifact is attached to the champion and loadable
    rp = Path(result.champion["rescorer_path"])
    assert rp.exists()
    from agentic_configuration_search.fitting.rescorer import Rescorer
    assert Rescorer.load(rp).active
    # round 2 diagnosis ran on the calibrated channel (rescorer_path branch — recorded without crashing)
    assert (out / "rounds" / "round_002" / "diagnosis_train.json").exists()
