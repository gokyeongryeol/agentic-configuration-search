"""Guard — material purity.

Contract pinned here: the crops and phrases carried into prompts (material) are determined by train
alone. Perturbing **holdout** GT must leave the content fingerprint of materialize output unchanged.
The only exception is `tau_test` (the decision threshold estimated on holdout — the field name is
legacy, kept for artifact compatibility), which is an output of eval.evaluate, not material.

materialize does not take holdout at the signature level (it reads only dataset.train), but this
test is a regression defense — if someone passes the whole Dataset and starts reading holdout,
it fails here.

**The perturbed split must be the one the loop actually loads.** A guard that perturbs a split
the loop never opens (e.g. test/) passes vacuously no matter what changes, so the split name is
not hardcoded but derived from the split the Dataset actually selected.
"""

import json

import numpy as np

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.coco import ANNOTATIONS_FILENAME
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.loop.materialize import materialize

AXES = {"exemplar": "base", "alias": "vlm"}


def _shuffle_holdout_gt(dataset_root):
    """Perturb GT by cyclically permuting annotation classes in the holdout split."""
    name = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8).holdout_split
    p = dataset_root / name / ANNOTATIONS_FILENAME
    coco = json.loads(p.read_text())
    for a in coco["annotations"]:
        a["category_id"] = 1 if a["category_id"] == 2 else 2
    p.write_text(json.dumps(coco))
    return name


def _fakes():
    def encode_fn(cid, paths):
        return np.array([[float(i), 0.0] for i in range(len(paths))])

    def caption_fn(rows):
        # per-class unique phrases — they pass the cross-pool filter
        return {r["id"]: [f"{r['id'].split('#')[0]} look"] for r in rows}

    def refine_alias_fn(name, class_names, pool, others):
        return {"alias": pool[0], "reason": "t"}

    return dict(encode_fn=encode_fn, caption_fn=caption_fn, vlm_caption_fn=caption_fn,
                refine_alias_fn=refine_alias_fn)


def _materialize(dataset_root, out_dir):
    # the fixture GT boxes are 20px — the minimum side goes below that so the alias
    # caption source is not empty
    cfg = config_from_dict({"dataset": str(dataset_root), "mine": {"caption_min_side": 16}})
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    return materialize(cfg, ds, AXES, out_dir, seed=0, **_fakes())


def test_materials_do_not_depend_on_holdout(dataset_root, tmp_path):
    fp_a = _materialize(dataset_root, tmp_path / "a").fingerprint()
    name = _shuffle_holdout_gt(dataset_root)
    assert name == "valid"     # the fixture has 3 splits — is the perturbed one really the holdout?
    fp_b = _materialize(dataset_root, tmp_path / "b").fingerprint()
    assert fp_a == fp_b, "materials depend on holdout GT — purity violation"
    assert fp_a["exemplars"] and fp_a["aliases"]      # prevents a vacuous pass on empty output


def test_materials_do_depend_on_train(dataset_root, tmp_path):
    """Positive control — perturbing train GT must change the material."""
    fp_a = _materialize(dataset_root, tmp_path / "a").fingerprint()
    p = dataset_root / "train" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    for a in coco["annotations"]:
        a["bbox"] = [a["bbox"][0] + 5, a["bbox"][1] + 5, 24, 24]
    p.write_text(json.dumps(coco))
    fp_b = _materialize(dataset_root, tmp_path / "b").fingerprint()
    assert fp_a != fp_b, "train changed but materials are identical — mining does not read train"
