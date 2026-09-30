"""crop classifier axis — mining hygiene, artifact contract, worker launcher, arm decision."""

import json
from pathlib import Path
import pytest

from agentic_detector_config.config import config_from_dict
from agentic_detector_config.core.coco import Ontology
from agentic_detector_config.core.data import load_dataset
from agentic_detector_config.fitting import crop_classifier as cc
from agentic_detector_config.services import external
from tests.conftest import CATS

CFG = config_from_dict({"dataset": "x"})
ONTO = Ontology.from_categories(CATS)


def test_constants_match_worker():
    """fitting cannot import workers (layering rule), so it duplicates the constants —
    if they drift the client cannot rebuild the checkpoint, so drift is blocked here."""
    from agentic_detector_config.workers import adapt as worker

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
    assert cmd[1:3] == ["-m", "agentic_detector_config.workers.adapt"]
    assert cmd[cmd.index("--num-classes") + 1] == "3"
    assert os.path.isabs(cmd[cmd.index("--manifest") + 1])   # independent of worker cwd
    # the vLLM venv does not have this package installed — PYTHONPATH must be the
    # directory that *contains* the package (src/), not the repo root
    from agentic_detector_config.services.paths import PACKAGE_PARENT
    assert captured["env"]["PYTHONPATH"] == str(PACKAGE_PARENT)
    assert (PACKAGE_PARENT / "agentic_detector_config" / "workers").is_dir()


# ------------------------------------------------------- arm decision (unit)

def test_arm_tripwire_skips_training(dataset_root, tmp_path, monkeypatch):
    """Too few crops → skip without training; train_fn is never called."""
    from agentic_detector_config.infer import gateway
    from agentic_detector_config.loop.adapt_round import run_classifier_candidate

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

    from agentic_detector_config.fitting.materials import Materials

    out = run_classifier_candidate(
        cfg, ds, {"exemplar": "none"}, Materials(),
        tmp_path / "cand", screen_fn=fake_screen,
        adapt_ctl=lambda p: None, repo_root=tmp_path, restore_root=None,
        seed=0, client=object(), train_fn=lambda *a, **k: calls.append(1))
    assert out["decision"]["adopted"] is False
    assert "training skipped" in out["decision"]["reason"]
    assert calls == []                               # training was not invoked
