"""YOLOE loop integration surface — rescorer with emb_dim=0, screen's text-negative
IoU sweep (neg_eta reinterpreted), hooks client wiring. All CPU."""

import numpy as np

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.eval.screen import screen_candidate
from agentic_configuration_search.fitting.materials import Materials
from agentic_configuration_search.fitting.rescorer import Rescorer
from agentic_configuration_search.fitting.verifier import GbdtVerifier
from agentic_configuration_search.infer import gateway


# ---------------------------------------------------- rescorer emb_dim=0

def test_verifier_and_rescorer_run_with_zero_dim_embeddings():
    """Pins the emb_dim=0 path: with empty embeddings ([]) the discriminator fits and applies
    on hand-crafted features alone (for detections that carry no query embedding)."""
    rng = np.random.default_rng(0)
    n = 80
    X = rng.random((n, 4))
    y = (X[:, 0] > 0.5).astype(float)
    E = np.array([[] for _ in range(n)], dtype=float)      # (n, 0)
    disc = GbdtVerifier.fit(["score", "overlap_margin", "area_z", "tile_agree"],
                            X, E, y, members=1)
    assert disc.emb_dim == 0
    feats = [{"score": 0.9, "overlap_margin": 1.0, "area_z": 0.0,
              "tile_agree": 1.0},
             {"score": 0.1, "overlap_margin": 1.0, "area_z": 0.0,
              "tile_agree": 1.0}]
    r = Rescorer(disc, {1: 1.0})
    p = r.phat(feats, [[], []])                            # empty embedding rows
    assert p.shape == (2,) and np.all((0 <= p) & (p <= 1))
    out = r.rescore(feats, [[], []], [0.9, 0.1], [1, 1])
    assert out.shape == (2,)


# ------------------------------------------- screen: text negative IoU sweep

def test_text_only_candidate_sweeps_neg_grid(dataset_root, monkeypatch):
    """Text-only candidate: the text pass sweeps the whole neg grid and
    picks the best config (suppression IoU) as the shared config."""
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cfg = config_from_dict({"dataset": str(dataset_root)})
    seen = []

    def fake_infer(ctx, client, config, queries, onto, paths, **kw):
        seen.append((config["config_id"], config["visual_negatives"], config["neg_eta"]))
        good = config["neg_eta"] == 0.75          # only this config hits GT
        out = []
        for e in ds.fit_dev.images:
            dets = []
            if good:
                dets = [{"box": list(a.bbox_xyxy), "score": 0.9,
                         "category_id": a.category_id, "sources": ["full"]}
                        for a in e.annotations]
            out.append(dets)
        return out

    monkeypatch.setattr(gateway, "infer_config", fake_infer)
    materials = Materials(exemplars={}, visual_negatives={}, aliases={},
                          text_negatives={"cat a": ["shadow"]})
    setting = screen_candidate(cfg, object(), ds.fit_dev, ds.ontology, materials,
                               neg_eta_grid=[0.25, 0.75])
    # the whole grid (0=off, 1=0.25, 2=0.75) is evaluated by the text pass and the neg axis stays alive
    assert {(cid, neg_eta) for cid, neg, neg_eta in seen} == {
        (0, None), (1, 0.25), (2, 0.75)}
    assert all(e["config_id"] == 2 and e["modality"] == "text"
               for e in setting["per_class"].values())
    assert setting["fit_dev_micro_f_beta"] > 0.9


# ------------------------------------------------------------ hooks wiring

def test_build_hooks_wires_yoloe_client(tmp_path):
    from agentic_configuration_search.clients.yoloe import YoloeClient
    from agentic_configuration_search.loop.hooks import build_hooks
    cfg = config_from_dict({"dataset": "x"})
    hooks = build_hooks(cfg, tmp_path, tmp_path / "out")
    client = hooks["client"]
    assert isinstance(client, YoloeClient)
    # in-process state-control wiring — arm material for the lora (tail) and crop_classifier axes
    deps = hooks["detector_hooks"]
    assert deps["weights_ctl"] == client.set_weights
    assert deps["adapt_ctl"] == client.set_adapt_root
    assert deps["base_weights"] == tmp_path / cfg.detector.model


def test_build_hooks_yoloe_caption_fn_requires_llm(tmp_path):
    from agentic_configuration_search.loop.hooks import build_hooks
    cfg = config_from_dict({"dataset": "x"})
    hooks = build_hooks(cfg, tmp_path, tmp_path / "out")
    assert hooks["caption_fn"] is None           # no llm_base -> explicit-error path
    hooks2 = build_hooks(cfg, tmp_path, tmp_path / "out",
                         llm_base_url="http://127.0.0.1:9")
    assert callable(hooks2["caption_fn"])        # wired to the vLLM caption source
