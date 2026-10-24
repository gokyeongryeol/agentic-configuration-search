"""lora (tail) training reuse — within a run, equal training inputs reuse the tuned subset.

Ported from zero-agent: retraining for every lora candidate cost one validation
run 8 trainings (~232 min). The training inputs (fit_train GT, base weights,
seed, training hyperparameters) do not depend on the champion within a run —
training always starts from the base weights. So the tuned subset is cached per
run, and later lora candidates rebuild the λ arm descriptors and re-score them
without training.
"""

import json
from pathlib import Path

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.fitting.tail import train_key
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop import tail_round
from agentic_configuration_search.loop.runner import run_rounds
from tests.test_tail import FakeDetector, _fake_infer, _fakes
from tests.test_tail import root  # noqa: F401 — fixture reuse


def _cfg(**lora):
    return config_from_dict({"planner": {"kind": "deterministic"}, "dataset": "x",
                             "lora": {"lambdas": [0.0, 0.5, 1.0], "margin": 0.01,
                                      "min_images": 1, **lora}})


def _key(cfg, ds, **kw):
    args = dict(fit_train=ds.fit_train, ontology=ds.ontology,
                base_weights=Path("/w/yoloe.pt"), seed=0) | kw
    return train_key(cfg, **args)


# ------------------------------------------------------------------ key

def test_train_key_covers_every_training_input(dataset_root):
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cfg = _cfg()
    k0 = _key(cfg, ds)
    assert _key(cfg, ds) == k0                         # deterministic
    names = lambda d: [e.file_name for e in d.fit_train.images]  # noqa: E731
    other = next(o for o in (load_dataset(dataset_root, seed=s, fit_train_ratio=0.8)
                             for s in range(1, 8)) if names(o) != names(ds))
    changed = [
        _key(cfg, ds, seed=1), _key(cfg, ds, base_weights=Path("/w/other.pt")),
        _key(cfg, other),                              # different fit_train boundary
        _key(_cfg(epochs=3), ds), _key(_cfg(lr=1e-4), ds),
        _key(_cfg(batch_size=4), ds)]
    assert all(k != k0 for k in changed)


def test_train_key_ignores_post_training_knobs(dataset_root):
    """The λ grid, margin, min_images and device do not change the training output."""
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    k0 = _key(_cfg(), ds)
    for kw in ({"lambdas": [0.0, 1.0]}, {"margin": 0.5}, {"min_images": 3},
               {"device": "cuda:7"}):
        assert _key(_cfg(**kw), ds) == k0, kw


# ----------------------------------------------------- run_tail_candidate unit

def _counting_train(monkeypatch):
    """train_tail stub — writes out_dir/tuned_tail.pt like the real one and counts calls."""
    calls = []

    def fake_train(cfg, *, out_dir, **kw):
        calls.append(out_dir)
        p = Path(out_dir) / "tuned_tail.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
        return p

    monkeypatch.setattr(tail_round, "train_tail", fake_train)
    return calls


def _screen_by_lam(det, gains):
    """fit_dev score from the loaded arm's λ — base 0.5, plus gains[λ]."""
    def screen(materials, axes):
        lam = (json.loads(Path(det.weights).read_text())["lam"]
               if det.weights else 0.0)
        return {"fit_dev_micro_f_beta": 0.5 + gains.get(lam, 0.0)}
    return screen


def _run(cfg, ds, out, det, cache, origin, gains=None):
    loaded = []
    screen = _screen_by_lam(det, gains or {0.5: 0.05, 1.0: 0.02})
    res = tail_round.run_tail_candidate(
        cfg, ds, {}, None, out,
        screen_fn=lambda m, a: (loaded.append(det.weights), screen(m, a))[1],
        weights_ctl=det.ctl, base_weights=Path("/w/yoloe.pt"),
        restore_weights=None, seed=0, cache_dir=cache, origin=origin)
    return res, loaded


def test_second_candidate_reuses_the_subset(dataset_root, tmp_path, monkeypatch):
    """The second lora candidate of a run does not train — it rebuilds the λ arms
    from the cached subset and scores each again. The reuse and its origin are
    recorded on the decision."""
    calls = _counting_train(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cache, det = tmp_path / "run" / "lora_cache", FakeDetector()
    first, _ = _run(_cfg(), ds, tmp_path / "run" / "c1", det, cache,
                    {"round": 1, "id": "r1_a"})
    assert len(calls) == 1 and str(calls[0]).startswith(str(cache))
    assert "delta_reused" not in first["decision"]
    second, loaded = _run(_cfg(), ds, tmp_path / "run" / "c2", det, cache,
                          {"round": 2, "id": "r2_b"})
    assert len(calls) == 1, "retrained under the same key"
    # the λ>0 arms were rebuilt in this candidate's directory and each scored
    arms = [w for w in loaded if w is not None]
    assert len(arms) == 2 and all("c2" in str(w) for w in arms)
    assert all(json.loads(Path(w).read_text())["tuned"].startswith(str(cache))
               for w in arms)
    assert second["decision"]["adopted"] is True
    assert second["decision"]["delta_reused"] == {"tail": {"round": 1, "id": "r1_a"}}
    rec = json.loads((tmp_path / "run" / "c2" / "tail_decision.json").read_text())
    assert rec["decision"]["delta_reused"]["tail"]["round"] == 1


def test_key_change_forces_retrain(dataset_root, tmp_path, monkeypatch):
    calls = _counting_train(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cache, det = tmp_path / "run" / "lora_cache", FakeDetector()
    _run(_cfg(), ds, tmp_path / "run" / "c1", det, cache, {"round": 1})
    res, _ = _run(_cfg(epochs=3), ds, tmp_path / "run" / "c2", det, cache,
                  {"round": 2})
    assert len(calls) == 2 and "delta_reused" not in res["decision"]


def test_interrupted_training_is_not_reused(dataset_root, tmp_path, monkeypatch):
    """A cache entry whose training did not finish (no key mark) is not reused — it is retrained."""
    calls = _counting_train(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cache, det = tmp_path / "run" / "lora_cache", FakeDetector()
    _run(_cfg(), ds, tmp_path / "run" / "c1", det, cache, {"round": 1})
    for meta in cache.rglob(tail_round.CACHE_META):
        meta.unlink()                                  # as if it died before the mark
    _run(_cfg(), ds, tmp_path / "run" / "c2", det, cache, {"round": 2})
    assert len(calls) == 2


def test_no_cache_dir_trains_in_the_candidate_directory(dataset_root, tmp_path,
                                                        monkeypatch):
    calls = _counting_train(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    det = FakeDetector()
    for c in ("c1", "c2"):
        _run(_cfg(), ds, tmp_path / c, det, None, None)
    assert calls == [tmp_path / "c1" / "tail", tmp_path / "c2" / "tail"]


# ------------------------------------------------------------- runner e2e

class _Base:
    adapted = False

def test_runner_reuses_within_a_round_and_across_resume(root, tmp_path,  # noqa: F811
                                                        monkeypatch):
    """Runner wiring — the second lora candidate of the same round and the lora
    candidate of the next round after resume both use the first training.
    Training lives in the run's lora_cache/."""
    det = FakeDetector()
    # a detector that ignores the arm — no lora arm wins, so the champion keeps
    # lora off and the next round's lora candidate trains (or reuses) again
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, _Base()))
    calls = _counting_train(monkeypatch)
    cfg = config_from_dict({
        "planner": {"kind": "deterministic", "max_candidates": 1},
        "dataset": str(root), "saturation": {"rounds": 5, "empty_rounds": 0},
        "lora": {"lambdas": [0.0, 1.0], "min_images": 4}})
    hooks = {"weights_ctl": det.ctl, "adapt_ctl": lambda r: None,
             "repo_root": tmp_path, "base_weights": tmp_path / "yoloe.pt"}
    cands = tmp_path / "candidates.yaml"
    cands.write_text("- id: a\n  axes: {lora: true}\n  rationale: x\n"
                     "- id: b\n  axes: {lora: true, rescorer: true}\n  rationale: x\n")
    run = tmp_path / "run"
    run_rounds(root, out_dir=run, planner="deterministic", rounds=1, cfg=cfg,
               candidates_file=cands, detector_hooks=hooks, **_fakes())
    assert len(calls) == 1 and str(calls[0]).startswith(str(run / "lora_cache"))
    hist = [json.loads(ln) for ln in (run / "history.jsonl").read_text().splitlines()]
    rows = [c for c in hist[1]["candidates"] if c["id"] == "r1_human_a"]
    assert rows and all("delta_reused" not in c for c in rows)
    rows = [c for c in hist[1]["candidates"] if c["id"] == "r1_human_b"]
    assert rows and all(c["delta_reused"] == {
        "tail": {"round": 1, "id": "r1_human_a"}} for c in rows)

    # resume — a new process (a new Loop) proposes a lora candidate with the same key next round
    cands.write_text("- id: c\n  axes: {lora: true, tiling: 2}\n  rationale: x\n")
    run_rounds(root, out_dir=run, planner="deterministic", rounds=2, cfg=cfg,
               candidates_file=cands, detector_hooks=hooks, resume=True, **_fakes())
    hist = [json.loads(ln) for ln in (run / "history.jsonl").read_text().splitlines()]
    rows = [c for c in hist[2]["candidates"] if c["id"] == "r2_human_c"]
    assert len(calls) == 1, "retrained under the same key after resume"
    assert rows and rows[0]["delta_reused"]["tail"]["round"] == 1
