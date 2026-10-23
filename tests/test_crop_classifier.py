"""crop classifier axis — mining hygiene, artifact contract, worker launcher, arm decision."""

import json
from pathlib import Path
import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.coco import Ontology
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.fitting import crop_classifier as cc
from agentic_configuration_search.services import external
from tests.conftest import CATS

CFG = config_from_dict({"dataset": "x"})
ONTO = Ontology.from_categories(CATS)


def test_constants_match_worker():
    """fitting cannot import workers (layering rule), so it duplicates the constants —
    if they drift the client cannot rebuild the checkpoint, so drift is blocked here."""
    from agentic_configuration_search.workers import adapt as worker

    assert cc.MODEL_NAME == worker.MODEL_NAME
    assert cc.LORA_R == worker.LORA_R
    assert cc.LORA_ALPHA == worker.LORA_ALPHA
    assert worker.LORA_TARGET == ["q_proj", "k_proj", "v_proj", "out_proj"]
    assert cc.LORA_TARGET_NAME == "qkvo"


# ------------------------------------------------------------------- mining

def test_mining_labels_and_background_hygiene(dataset_root, tmp_path):
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    train = ds.train
    dets = [[] for _ in train.images]
    e0 = train.images[0]                       # GT: cat? at [10,10,30,30]
    cid0 = e0.annotations[0].category_id
    dets[0] = [
        {"box": [10, 10, 30, 30], "score": 0.9, "category_id": cid0},   # on GT — not background
        {"box": [12, 12, 28, 28], "score": 0.9, "category_id": cid0},   # contained — not background
        {"box": [60, 40, 80, 60], "score": 0.9, "category_id": cid0},   # valid background
        {"box": [60, 10, 80, 30], "score": 0.1, "category_id": cid0},   # below threshold — excluded
    ]
    manifest, counts = cc.mine_classifier_crops(
        train, ds.ontology, dets, thresholds={1: 0.5, 2: 0.5},
        out_dir=tmp_path)
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    assert counts["train_gt"] == 10 and counts["train_fp"] == 1
    # label = idx+1 in ascending category id (reclassifier contract), background = 0
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    assert {r["label"] for r in by_src["train_fp"]} == {0}
    gt_labels = {r["label"] for r in by_src["train_gt"]}
    assert gt_labels == {1, 2}
    assert all(Path(r["crop_path"]).exists() for r in rows)


# ------------------------------------------------------------- artifact

def test_assemble_artifacts_contract(tmp_path):
    cc.assemble_artifacts(tmp_path, ONTO)
    conf = json.loads((tmp_path / "classifier_config.json").read_text())
    assert conf["classifier_model_name"] == cc.MODEL_NAME
    assert conf["classifier_lora_r"] == cc.LORA_R
    assert conf["classifier_lora_target"] == "qkvo"
    assert conf["class_names"] == ["cat a", "cat b"]     # ascending id


# --------------------------------------------------------------- launcher

def test_run_adapt_worker_command(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "VLLM_VENV", Path("external/vllm-venv"))
    repo = tmp_path
    (repo / "external/vllm-venv/bin").mkdir(parents=True)
    (repo / "external/vllm-venv/bin/python").touch()
    monkeypatch.chdir(tmp_path)
    rel_manifest = Path("m.jsonl")
    rel_manifest.write_text("{}\n")
    captured = {}

    class FakeProc:
        def wait(self):
            Path(captured["cmd"][captured["cmd"].index("--out") + 1]).touch()
            return 0

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw["env"]
        return FakeProc()

    import os
    external.run_adapt_worker(repo, manifest=rel_manifest, num_classes=3,
                              out_path=Path("cls.pt"), device="cuda:1",
                              log_path=Path("w.log"), popen=fake_popen)
    cmd = captured["cmd"]
    assert cmd[1:3] == ["-m", "agentic_configuration_search.workers.adapt"]
    assert cmd[cmd.index("--num-classes") + 1] == "3"
    assert os.path.isabs(cmd[cmd.index("--manifest") + 1])   # independent of worker cwd
    # the vLLM venv does not have this package installed — PYTHONPATH must be the
    # directory that *contains* the package (src/), not the repo root
    from agentic_configuration_search.services.paths import PACKAGE_PARENT
    assert captured["env"]["PYTHONPATH"] == str(PACKAGE_PARENT)
    assert (PACKAGE_PARENT / "agentic_configuration_search" / "workers").is_dir()


# ------------------------------------------------------- arm decision (unit)

def test_arm_tripwire_skips_training(dataset_root, tmp_path, monkeypatch):
    """Too few crops → skip without training; train_fn is never called."""
    from agentic_configuration_search.infer import gateway
    from agentic_configuration_search.loop.adapt_round import run_classifier_candidate

    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    monkeypatch.setattr(gateway, "infer_config",
                        lambda *a, **k: [[] for _ in a[5]])   # a[5] = image_paths
    cfg = config_from_dict({"dataset": "x",
                            "crop_classifier": {"min_samples": 10_000}})
    calls = []

    def fake_screen(materials, axes):
        return {"configs": [{"config_id": 0, "visual_negatives": False,
                             "neg_eta": None, "tiling": None}],
                "per_class": {1: {"config_id": 0, "modality": "text",
                                  "threshold": 0.5, "f_beta": 0.5}},
                "fit_dev_micro_f_beta": 0.5}

    from agentic_configuration_search.fitting.materials import Materials

    out = run_classifier_candidate(
        cfg, ds, {"exemplar": "none"}, Materials(),
        tmp_path / "cand", screen_fn=fake_screen,
        adapt_ctl=lambda p: None, repo_root=tmp_path, restore_root=None,
        seed=0, client=object(), train_fn=lambda *a, **k: calls.append(1))
    assert out["decision"]["adopted"] is False
    assert "training skipped" in out["decision"]["reason"]
    assert calls == []                               # training was not invoked


# --- the checkpoint keeps only the trained parameters ---------------------------------

def _tiny_peft(seed: int):
    """A small module with the same parameter naming as the SigLIP classifier
    (attention q/k/v/out projections + a `classifier` head) under peft LoRA."""
    import torch
    from peft import LoraConfig, get_peft_model

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
            self.v_proj, self.out_proj = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.out_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()
            self.classifier = torch.nn.Linear(8, 3)

        def forward(self, x):
            return self.classifier(self.attn(x))

    torch.manual_seed(seed)
    model = get_peft_model(Tiny(), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    return model


def test_worker_saves_only_trainable_params(tmp_path):
    """The saved keys are exactly the parameters training changes (requires_grad
    after freeze_to_trainable), with the format marker."""
    import torch
    from agentic_configuration_search.workers import adapt as worker

    model = _tiny_peft(0)
    worker.freeze_to_trainable(model)
    worker.save_checkpoint(model, 3, tmp_path / "classifier.pt")
    ckpt = torch.load(tmp_path / "classifier.pt", weights_only=True)
    assert ckpt["state_dict_format"] == "trainable" and ckpt["num_classes"] == 3
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(ckpt["state_dict"]) == trainable
    assert any("lora_" in k for k in trainable) and any("classifier" in k for k in trainable)
    assert not any("base_layer" in k for k in ckpt["state_dict"])   # no backbone weights


def test_loader_round_trips_trainable_checkpoint(tmp_path):
    """A trainable checkpoint put onto a freshly rebuilt model gives the trained
    model's outputs exactly; a missing key is refused; an old full checkpoint
    still loads strictly."""
    import torch
    from agentic_configuration_search.clients.yoloe import _load_classifier_state
    from agentic_configuration_search.workers import adapt as worker

    trained = _tiny_peft(0)
    worker.freeze_to_trainable(trained)
    with torch.no_grad():                        # stand in for training
        for n, p in trained.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p))
    worker.save_checkpoint(trained, 3, tmp_path / "c.pt")
    ckpt = torch.load(tmp_path / "c.pt", weights_only=True)

    # the backbone (base_layer) is the same pretrained weights on both sides —
    # here the same seed; the adapter and head differ until loaded
    fresh = _tiny_peft(0)
    x = torch.randn(4, 8)
    assert not torch.equal(fresh(x), trained(x))
    _load_classifier_state(fresh, ckpt, tmp_path)
    assert torch.equal(fresh(x), trained(x))

    lora_b = next(k for k in ckpt["state_dict"] if "lora_B" in k)
    broken = {**ckpt, "state_dict": {k: v for k, v in ckpt["state_dict"].items()
                                     if k != lora_b}}
    with pytest.raises(ValueError, match="missing"):
        _load_classifier_state(_tiny_peft(0), broken, tmp_path)

    full = {"state_dict": trained.state_dict(), "num_classes": 3,
            "classifier_model": "siglip"}            # an older run's format
    old = _tiny_peft(1)
    _load_classifier_state(old, full, tmp_path)
    assert torch.equal(old(x), trained(x))


def test_trainable_key_rule_matches_loader():
    from agentic_configuration_search.clients.yoloe import _is_trainable_key
    from agentic_configuration_search.workers import adapt as worker

    names = [n for n in _tiny_peft(0).state_dict()] + [
        "base_model.model.classifier.weight", "vision_model.encoder.layers.0.mlp.fc1.weight"]
    assert [worker.is_trainable_key(n) for n in names] == [_is_trainable_key(n) for n in names]
