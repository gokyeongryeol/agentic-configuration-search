"""eval/screen — shared config selection, modality adoption and rescue.

Inference is blocked via the gateway seam monkeypatch.
"""

import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.core.prompts import PromptBox
from agentic_configuration_search.eval.screen import screen_candidate, select_shared_config
from agentic_configuration_search.fitting.materials import Materials
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.infer.queries import config_grid

CFG = config_from_dict({"dataset": "x"})


@pytest.fixture
def ds(dataset_root):
    return load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)


def _gt_hits(split, score=0.9):
    """Detections that exactly hit every GT of each image in the split."""
    return [[{"box": list(a.bbox_xyxy), "score": score, "category_id": a.category_id,
              "sources": ["full"]} for a in e.annotations] for e in split.images]


def _exemplars(onto):
    return {c.name: [PromptBox(c.name, "img.png", [1, 1, 8, 8])] for c in onto.classes}


# ---------------------------------------------------- select_shared_config

def test_select_shared_config_maximizes_f1_sum():
    grid = config_grid(neg_eta_grid=[1.0])          # config 0(off), 1(on)
    rows = [{"config_id": 0, "category_id": 1, "f_beta": 0.9, "threshold": 0.5},
            {"config_id": 1, "category_id": 1, "f_beta": 0.2, "threshold": 0.5},
            {"config_id": 0, "category_id": 2, "f_beta": 0.3, "threshold": 0.4},
            {"config_id": 1, "category_id": 2, "f_beta": 0.8, "threshold": 0.4}]
    best = select_shared_config(rows, grid)
    # totals: config0 = 1.2, config1 = 1.0 -> shared config 0
    assert best[1]["config_id"] == 0 and best[2]["config_id"] == 0


def test_select_shared_config_incomplete_grid_errors():
    grid = config_grid(neg_eta_grid=[1.0])
    rows = [{"config_id": 0, "category_id": 1, "f_beta": 0.9, "threshold": 0.5},
            {"config_id": 1, "category_id": 2, "f_beta": 0.8, "threshold": 0.4}]
    with pytest.raises(ValueError, match="no common config evaluated for every class"):
        select_shared_config(rows, grid)


# ------------------------------------------------------- screen_candidate

def test_text_only_candidate_rescues_all_classes(ds, monkeypatch):
    """Candidate without exemplars — skips the semantic grid and every class is rescued via text."""
    fit_dev = ds.fit_dev

    def fake_infer(ctx, client, config, queries, ontology, image_paths,
                   alias_to_name=None, **kw):
        assert alias_to_name is not None, "there must be only the text pass"
        return _gt_hits(fit_dev)

    monkeypatch.setattr(gateway, "infer_config", fake_infer)
    out = screen_candidate(CFG, object(), fit_dev, ds.ontology, Materials())
    assert set(out["per_class"]) == {1, 2}
    assert all(e["modality"] == "text" for e in out["per_class"].values())
    assert out["fit_dev_micro_f_beta"] == pytest.approx(1.0)


def _tiling_aware_infer(fit_dev, calls, *, tiles_help):
    """Text pass only — hits GT only on a tile config (tiles_help) or always."""
    def fake_infer(ctx, client, config, queries, ontology, image_paths,
                   alias_to_name=None, **kw):
        assert alias_to_name is not None, "there must be only the text pass"
        calls.append(config["tiling"])
        if tiles_help and config["tiling"] is None:
            return [[] for _ in fit_dev.images]
        return _gt_hits(fit_dev)
    return fake_infer


TILE = {"grid": 2, "overlap": 0.5, "merge": "nmm"}


def test_text_only_candidate_screens_each_tiling_variant(ds, monkeypatch):
    """A text-only candidate (no exemplars) runs the text pass on each tiling
    variant and shares the config that wins."""
    calls: list = []
    monkeypatch.setattr(gateway, "infer_config",
                        _tiling_aware_infer(ds.fit_dev, calls, tiles_help=True))
    out = screen_candidate(CFG, object(), ds.fit_dev, ds.ontology, Materials(),
                           tiling_variants=(TILE,))
    assert calls == [None, TILE]
    cfgs = {c["config_id"]: c for c in out["configs"]}
    assert all(cfgs[e["config_id"]]["tiling"] == TILE for e in out["per_class"].values())
    assert out["fit_dev_micro_f_beta"] == pytest.approx(1.0)


def test_text_only_tiling_tie_prefers_untiled(ds, monkeypatch):
    """On a tie the untiled config wins — same score, cheaper."""
    calls: list = []
    monkeypatch.setattr(gateway, "infer_config",
                        _tiling_aware_infer(ds.fit_dev, calls, tiles_help=False))
    out = screen_candidate(CFG, object(), ds.fit_dev, ds.ontology, Materials(),
                           tiling_variants=(TILE,))
    cfgs = {c["config_id"]: c for c in out["configs"]}
    assert all(cfgs[e["config_id"]]["tiling"] is None for e in out["per_class"].values())


def test_text_only_without_tiling_runs_one_pass(ds, monkeypatch):
    calls: list = []
    monkeypatch.setattr(gateway, "infer_config",
                        _tiling_aware_infer(ds.fit_dev, calls, tiles_help=False))
    screen_candidate(CFG, object(), ds.fit_dev, ds.ontology, Materials())
    assert calls == [None]


def test_modality_pick_per_class(ds, monkeypatch):
    """Only classes semantic cannot find move to text — ties go to semantic."""
    fit_dev = ds.fit_dev

    def fake_infer(ctx, client, config, queries, ontology, image_paths,
                   alias_to_name=None, **kw):
        hits = _gt_hits(fit_dev)
        if alias_to_name is None:                       # semantic: misses class 2
            return [[d for d in dets if d["category_id"] == 1] for dets in hits]
        return hits                                     # text: hits both
    monkeypatch.setattr(gateway, "infer_config", fake_infer)

    cfg = config_from_dict({"dataset": "x"})
    out = screen_candidate(cfg, object(), fit_dev, ds.ontology,
                           Materials(exemplars=_exemplars(ds.ontology)))
    assert out["per_class"][1]["modality"] == "semantic"   # tie -> semantic
    assert out["per_class"][2]["modality"] == "text"
    assert out["fit_dev_micro_f_beta"] == pytest.approx(1.0)
    diag = out["modality_diagnostic"]
    assert diag["2"]["semantic"]["f_beta"] == pytest.approx(0.0)
    assert diag["2"]["text"]["f_beta"] == pytest.approx(1.0)
