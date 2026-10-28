"""Material inheritance — materials for axes carried over from the champion are not re-mined.

Re-mining a carried-over axis would make the gate's Δ "increment + material swap"
rather than the pure increment. The champion's score was achieved with those
materials, so they are inherited from the parent unchanged.
"""

import numpy as np
import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.fitting.materials import Materials
from agentic_configuration_search.loop.materialize import inherit_plan, materialize
from agentic_configuration_search.planner.axes import BASELINE, validate_axes


def _ax(**kw):
    return validate_axes({**BASELINE, **kw})


def test_inherit_plan_keeps_unchanged_axes():
    """Same axis value → inherit; changed → re-mine. No parent → re-mine everything."""
    champ = _ax(exemplar="base", exemplar_base_k=1, alias="vlm")
    # candidate that only turns on alias → exemplar inherited
    cand = _ax(exemplar="base", exemplar_base_k=1, alias="vlm", rescorer=True)
    assert inherit_plan(cand, champ) == {"exemplars", "aliases",
                                         "visual_negatives", "text_negatives"}
    # candidate raising exemplar → only exemplar re-mined, alias inherited
    cand2 = _ax(exemplar="base", exemplar_base_k=2, alias="vlm")
    plan2 = inherit_plan(cand2, champ)
    assert "exemplars" not in plan2 and "aliases" in plan2
    # raising to fn_arm also re-mines exemplar (needs chosen_ann_ids)
    cand3 = _ax(exemplar="fn_arm", exemplar_base_k=1, alias="vlm")
    assert "exemplars" not in inherit_plan(cand3, champ)
    # no parent → no inheritance
    assert inherit_plan(cand, None) == set()


def test_inherited_materials_are_byte_identical(dataset_root, tmp_path):
    """Carried-over materials are byte-identical to the parent, and no re-mining happens."""
    cfg = config_from_dict({"dataset": str(dataset_root)})
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    calls = {"encode": 0, "caption": 0}

    def encode_fn(cid, paths):
        calls["encode"] += 1
        return np.array([[float(i), 0.0] for i in range(len(paths))])

    def caption_fn(rows):
        calls["caption"] += 1
        return {r["id"]: [f"{r['id'].split('#')[0]} look"] for r in rows}

    fakes = dict(encode_fn=encode_fn, caption_fn=caption_fn,
                 vlm_caption_fn=caption_fn,
                 refine_alias_fn=lambda n, cn, p, o: {"alias": p[0], "reason": "t"})
    champ_axes = _ax(exemplar="base", alias="vlm")
    parent = tmp_path / "parent"
    m0 = materialize(cfg, ds, champ_axes, parent, seed=0, **fakes)
    n_enc, n_cap = calls["encode"], calls["caption"]
    assert m0.exemplars and m0.aliases

    # candidate adding only rescorer — exemplar/alias carried over → must be inherited
    cand = _ax(exemplar="base", alias="vlm", rescorer=True)
    m1 = materialize(cfg, ds, cand, tmp_path / "child", seed=0,
                     parent=parent, parent_axes=champ_axes, **fakes)
    assert (tmp_path / "child" / "exemplars.json").read_text() == \
           (parent / "exemplars.json").read_text()
    assert (tmp_path / "child" / "text_aliases.json").read_text() == \
           (parent / "text_aliases.json").read_text()
    assert calls["encode"] == n_enc and calls["caption"] == n_cap, "re-mining happened"
    # provenance is recorded in the artifact
    import json
    rep = json.loads((tmp_path / "child" / "mine_reports.json").read_text())
    assert set(rep["inherited"]) >= {"exemplars", "aliases"}
    assert rep["parent"].endswith("parent")


def test_changed_axis_is_remined(dataset_root, tmp_path):
    """Materials for a changed axis are not inherited — that axis is the candidate's increment."""
    cfg = config_from_dict({"dataset": str(dataset_root)})
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    fakes = dict(encode_fn=lambda cid, p: np.array(
                     [[float(i), 0.0] for i in range(len(p))]),
                 caption_fn=lambda rows: {r["id"]: ["x look"] for r in rows},
                 vlm_caption_fn=lambda rows: {r["id"]: ["x look"] for r in rows},
                 refine_alias_fn=lambda n, cn, p, o: {"alias": p[0], "reason": "t"})
    champ_axes = _ax(exemplar="base", exemplar_base_k=1)
    parent = tmp_path / "p"
    m0 = materialize(cfg, ds, champ_axes, parent, seed=0, **fakes)
    cand = _ax(exemplar="base", exemplar_base_k=2)
    m1 = materialize(cfg, ds, cand, tmp_path / "c", seed=0,
                     parent=parent, parent_axes=champ_axes, **fakes)
    n0 = sum(len(v) for v in m0.exemplars.values())
    n1 = sum(len(v) for v in m1.exemplars.values())
    assert n1 > n0                                   # re-mined with k=2


def test_relabeled_exemplar_none_does_not_inherit_exemplars(dataset_root, tmp_path):
    """When screen did not use exemplar and the champion label was corrected to
    exemplar=none, the parent's materials still hold exemplars. An exemplar=none
    candidate on top must not inherit them — screen would run the semantic pass
    again and the label would split from what is served."""
    cfg = config_from_dict({"dataset": str(dataset_root)})
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    fakes = dict(encode_fn=lambda cid, p: np.array(
                     [[float(i), 0.0] for i in range(len(p))]),
                 caption_fn=lambda rows: {r["id"]: ["x look"] for r in rows},
                 vlm_caption_fn=lambda rows: {r["id"]: ["x look"] for r in rows},
                 refine_alias_fn=lambda n, cn, p, o: {"alias": p[0], "reason": "t"})
    parent = tmp_path / "p"
    m0 = materialize(cfg, ds, _ax(exemplar="base"), parent, seed=0, **fakes)
    assert any(m0.exemplars.values())
    relabeled = _ax()                                # the corrected champion label
    cand = _ax(alias="vlm")
    m1 = materialize(cfg, ds, cand, tmp_path / "c", seed=0,
                     parent=parent, parent_axes=relabeled, **fakes)
    assert not any(m1.exemplars.values())
