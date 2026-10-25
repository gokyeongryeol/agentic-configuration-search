"""YOLOE tail fine-tune axis (lora) — material (fitting.tail), arm round (loop.tail_round),
weight swap (clients.yoloe.set_weights). Training is stubbed; all CPU.

e2e setup: a stub whose detections depend on the weight state pins λ-grid scoring,
the margin gate, and persistence of the champion weights (descriptor).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from agentic_configuration_search.clients.yoloe import YoloeClient, read_arm
from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.coco import Split
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.fitting.tail import build_tail_arms, export_yolo_dataset
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop import tail_round
from agentic_configuration_search.loop.runner import run_rounds
from tests.conftest import make_split


# ------------------------------------------------------- fitting.tail material

def test_export_yolo_dataset_box_polygons(dataset_root, tmp_path):
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    data_yaml = export_yolo_dataset(ds.fit_train, ds.ontology, tmp_path / "yd")
    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["train"] == "images/train" and cfg["val"] == "images/train"
    assert list(cfg["names"].values()) == [c.name for c in ds.ontology.classes]
    entry = ds.fit_train.images[0]
    label = (tmp_path / "yd" / "labels" / "train"
             / (Path(entry.file_name).stem + ".txt")).read_text().strip()
    parts = label.split("\n")[0].split()
    assert len(parts) == 9                       # cls + 8 coordinates of a rectangle polygon
    a = entry.annotations[0]
    cls_index = {c.id: i for i, c in enumerate(ds.ontology.classes)}
    assert int(parts[0]) == cls_index[a.category_id]
    xs = [float(parts[i]) for i in (1, 3, 5, 7)]
    ys = [float(parts[i]) for i in (2, 4, 6, 8)]
    assert min(xs) == pytest.approx(a.bbox_xyxy[0] / entry.width, abs=1e-5)
    assert max(ys) == pytest.approx(a.bbox_xyxy[3] / entry.height, abs=1e-5)
    # images are symlinks (no copy)
    assert (tmp_path / "yd" / "images" / "train" / entry.file_name).is_symlink()


def test_build_tail_arms_descriptors(tmp_path):
    tuned = tmp_path / "tuned_tail.pt"
    tuned.write_bytes(b"stub")
    arms = build_tail_arms(tuned, [0.0, 0.5, 1.0], tmp_path / "arms")
    assert sorted(arms) == [0.5, 1.0]            # λ=0 is base — no descriptor
    arm = read_arm(arms[0.5])
    assert arm["lam"] == 0.5 and arm["tuned"] == tuned.resolve()
    with pytest.raises(ValueError, match="descriptor"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"kind": "other"}))
        read_arm(bad)


# ------------------------------------------------ clients.yoloe weight swap

class SwapBackend:
    def __init__(self):
        self.loaded = []
        self.vocab_calls = 0

    def load_weights(self, weights):
        self.loaded.append(weights)

    def set_vocab(self, channels):
        self.vocab_calls += 1

    def infer(self, images, *, conf, iou, max_det, return_embeddings=False):
        return [[] for _ in images]


def test_set_weights_swaps_and_invalidates_vocab():
    from agentic_configuration_search.core.prompts import build_text_queries
    queries, _o, _a = build_text_queries(["cat a"], {}, box_threshold=0.01,
                                         multimodal_threshold=0.01)
    backend = SwapBackend()
    client = YoloeClient(backend=backend)
    client.predict_images(["im"], queries)
    assert backend.vocab_calls == 1
    client.set_weights("arm_lam0.5.json")
    assert backend.loaded == ["arm_lam0.5.json"]
    client.predict_images(["im"], queries)       # same queries — forces vocab recomputation
    assert backend.vocab_calls == 2
    client.set_weights("arm_lam0.5.json")        # same state — no-op
    assert backend.loaded == ["arm_lam0.5.json"]
    client.set_weights(None)                     # base restored
    assert backend.loaded == ["arm_lam0.5.json", None]


def test_pending_weights_kept_until_backend_exists():
    """set_weights before a backend exists — stores the state and applies it once a backend
    appears (the path resume uses to restore champion weights)."""
    client = YoloeClient()
    client.set_weights("arm.json")               # no backend — record only
    assert client._weights == "arm.json"
    backend = SwapBackend()
    client._backend = backend
    client.set_weights("arm2.json")              # now applied immediately
    assert backend.loaded == ["arm2.json"]


# ------------------------------------------------- tail arm round e2e

@pytest.fixture
def root(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    make_split(ds / "train", [
        {"name": f"tr{i:02d}.png", "w": 100, "h": 80,
         "anns": [(1 if i % 2 else 2, 10, 10, 20 + (i % 3) * 10,
                   20 + (i % 3) * 10)]} for i in range(12)])
    make_split(ds / "valid", [        # the loop's holdout
        {"name": f"t{i}.png", "w": 100, "h": 80,
         "anns": [(1 if i < 4 else 2, 10, 10, 20, 20)]} for i in range(8)])
    make_split(ds / "test", [{"name": "v.png", "w": 100, "h": 80,  # sealed — the loop never reads it
                              "anns": [(1, 10, 10, 20, 20)]}])
    return ds


class FakeDetector:
    def __init__(self):
        self.weights = None
        self.calls: list = []

    def ctl(self, weights):
        self.weights = weights
        self.calls.append(weights)

    @property
    def adapted(self) -> bool:
        return self.weights is not None and "arm_lam" in str(self.weights)


def _fake_infer(root, server):
    splits = {n: Split.load(root / n) for n in ("train", "valid")}
    by_path = {str(s.image_path(e)): e for s in splits.values() for e in s.images}

    def fake(ctx, client, config, queries, ontology, image_paths,
             alias_to_name=None, **kw):
        out = []
        for p in image_paths:
            entry = by_path[str(p)]
            dets = [{"box": list(a.bbox_xyxy), "score": 0.9,
                     "category_id": a.category_id, "sources": ["full"]}
                    for a in entry.annotations
                    if server.adapted or a.category_id == 1]
            out.append(dets)
        return out
    return fake


def _fakes():
    return dict(
        encode_fn=lambda cid, paths: np.array(
            [[float(i), 0.0] for i in range(len(paths))]),
        caption_fn=lambda rows: {r["id"]: [f"{r['id'].split('#')[0]} look"]
                                 for r in rows},
        vlm_caption_fn=lambda rows: {r["id"]: [f"{r['id'].split('#')[0]} look"]
                                 for r in rows},
        refine_alias_fn=lambda name, cn, pool, others: {"alias": pool[0],
                                                        "reason": "t"},
        refine_negatives_fn=lambda name, cn, pool, mk: {"keep": pool[:1],
                                                        "rejected": []},
        client=object())


def test_tail_candidate_adopted_end_to_end(root, tmp_path, monkeypatch):
    """lora-axis candidate -> routed to tail_round -> λ arms scored ->
    accepted -> champion.weights carries the arm descriptor path."""
    server = FakeDetector()
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, server))
    trained = []

    def fake_train(cfg, *, fit_train, ontology, base_weights, out_dir, seed,
                   log_path, **kw):
        trained.append(len(fit_train.images))
        p = Path(out_dir) / "tuned_tail.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
        return p

    monkeypatch.setattr(tail_round, "train_tail", fake_train)

    cfg = config_from_dict({
        "planner": {"kind": "deterministic"},
        "dataset": str(root),
        "lora": {"lambdas": [0.0, 0.5, 1.0], "margin": 0.01, "min_images": 4},
    })
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text("- id: adapt\n  axes: {lora: true}\n  rationale: t\n")
    result = run_rounds(
        root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
        cfg=cfg, candidates_file=cand_file,
        detector_hooks={"weights_ctl": server.ctl, "adapt_ctl": lambda r: None,
                   "repo_root": tmp_path,
                   "base_weights": tmp_path / "yoloe.pt"},
        **_fakes())

    # training uses fit_train only (material purity). There is one accepted lora candidate —
    # if the planner's own proposal and the human injection are the same combo, combo dedup folds them.
    n_ft = len(load_dataset(root, seed=0, fit_train_ratio=0.8).fit_train.images)
    assert trained and all(n == n_ft for n in trained)
    r1 = result.decisions[1]
    assert r1["adopted"] is True
    assert result.champion["axes"]["lora"] is True
    # tie (λ 0.5 vs 1.0) — smaller λ (conservative choice, reuses select_candidate)
    assert "arm_lam0.5" in str(result.champion["weights"])
    assert server.adapted and "arm_lam0.5" in str(server.weights)
    meta = json.loads((tmp_path / "run" / "champion.json").read_text())
    assert "arm_lam0.5" in str(meta["weights"])
    arm = read_arm(result.champion["weights"])
    assert arm["lam"] == 0.5


def test_tail_candidate_restores_on_rejection(root, tmp_path, monkeypatch):
    """An arm with no gain (stub behaves independently of weights) — not accepted + current (None) restored."""
    server = FakeDetector()
    splits = {n: Split.load(root / n) for n in ("train", "valid")}
    by_path = {str(s.image_path(e)): e
               for s in splits.values() for e in s.images}

    def flat_infer(ctx, client, config, queries, ontology, image_paths,
                   alias_to_name=None, **kw):
        return [[{"box": list(a.bbox_xyxy), "score": 0.9,
                  "category_id": a.category_id, "sources": ["full"]}
                 for a in by_path[str(p)].annotations] for p in image_paths]

    monkeypatch.setattr(gateway, "infer_config", flat_infer)

    def fake_train(cfg, *, out_dir, **kw):
        p = Path(out_dir) / "tuned_tail.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
        return p

    monkeypatch.setattr(tail_round, "train_tail", fake_train)
    cfg = config_from_dict({
        "planner": {"kind": "deterministic"},
        "dataset": str(root),
        "lora": {"lambdas": [0.0, 1.0], "margin": 0.01, "min_images": 4},
    })
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text("- id: adapt\n  axes: {lora: true}\n  rationale: t\n")
    result = run_rounds(
        root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
        cfg=cfg, candidates_file=cand_file,
        detector_hooks={"weights_ctl": server.ctl, "adapt_ctl": lambda r: None,
                   "repo_root": tmp_path, "base_weights": tmp_path / "y.pt"},
        **_fakes())
    assert result.decisions[1]["adopted"] is False   # no gain — rejected by the gate
    # key point: the detector is restored to base (None)
    assert server.weights is None
    assert result.champion.get("weights") in (None, "None")


# --- trainer checkpoints are not kept ------------------------------------------------

class _FakeProc:
    def __init__(self, rc):
        self.rc = rc

    def wait(self):
        return self.rc


def _fake_training(rc=0, write_subset=True):
    """A popen double that does what the worker does on disk: the ultralytics
    trainer leaves full checkpoints in <project>/train/weights/, and the worker
    saves the tail subset to --out."""
    def popen(cmd, env=None, stdout=None, stderr=None):
        args = dict(zip(cmd[::1], cmd[1::1]))
        wdir = Path(args["--project"]) / "train" / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "last.pt").write_bytes(b"x" * 1000)
        (wdir / "best.pt").write_bytes(b"x" * 1000)
        (wdir.parent / "args.yaml").write_text("epochs: 1\n")
        if write_subset:
            Path(args["--out"]).write_bytes(b"subset")
        return _FakeProc(rc)
    return popen


def test_train_tail_drops_trainer_checkpoints(dataset_root, tmp_path):
    """Only the tail subset is used, so the trainer's full-model checkpoints
    (written at the final epoch whatever `save` says) are deleted once the
    subset exists; the small run record stays."""
    from agentic_configuration_search.fitting.tail import train_tail
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cfg = config_from_dict({"dataset": str(dataset_root)})
    out = tmp_path / "tail"
    tuned = train_tail(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
                       base_weights=tmp_path / "base.pt", out_dir=out, seed=0,
                       log_path=out / "train.log", popen=_fake_training())
    assert tuned.read_bytes() == b"subset"
    assert not (out / "ultralytics" / "train" / "weights").exists()
    assert (out / "ultralytics" / "train" / "args.yaml").exists()


def test_train_tail_uses_expandable_segments(dataset_root, tmp_path, monkeypatch):
    """Tail training shares the GPU with the in-process detector — the training
    process gets the fragmentation-reducing allocator setting by default."""
    from agentic_configuration_search.fitting.tail import train_tail
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cfg = config_from_dict({"dataset": str(dataset_root)})
    seen = {}
    fake = _fake_training()

    def popen(cmd, env=None, **kw):
        seen["env"] = env
        return fake(cmd, env=env, **kw)

    train_tail(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
               base_weights=tmp_path / "base.pt", out_dir=tmp_path / "tail", seed=0,
               log_path=tmp_path / "tail" / "train.log", popen=popen)
    assert seen["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_train_tail_keeps_operator_alloc_conf(dataset_root, tmp_path, monkeypatch):
    from agentic_configuration_search.fitting.tail import train_tail
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cfg = config_from_dict({"dataset": str(dataset_root)})
    seen = {}
    fake = _fake_training()

    def popen(cmd, env=None, **kw):
        seen["env"] = env
        return fake(cmd, env=env, **kw)

    train_tail(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
               base_weights=tmp_path / "base.pt", out_dir=tmp_path / "tail", seed=0,
               log_path=tmp_path / "tail" / "train.log", popen=popen)
    assert seen["env"]["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:128"


def test_train_tail_keeps_checkpoints_when_training_fails(dataset_root, tmp_path):
    """A failed run keeps whatever the trainer left, for diagnosis."""
    from agentic_configuration_search.fitting.tail import train_tail
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cfg = config_from_dict({"dataset": str(dataset_root)})
    for i, popen in enumerate((_fake_training(rc=1), _fake_training(write_subset=False))):
        out = tmp_path / f"tail{i}"
        with pytest.raises(RuntimeError):
            train_tail(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
                       base_weights=tmp_path / "base.pt", out_dir=out, seed=0,
                       log_path=out / "train.log", popen=popen)
        assert (out / "ultralytics" / "train" / "weights" / "last.pt").exists()
