"""crop classifier arm — trained once per run, a blend λ grid per proposal.

- The arm loads the artifact once and screens fit_dev for each λ in
  crop_classifier.lambdas; the best ΔF(β) > margin wins (ties: smaller λ). The
  baseline (no classifier) stays on the grid as lam=None.
- Training is cached per run (<run>/classifier_cache/) — later proposals reuse
  the artifact without mining or training (classifier_reused). An entry without
  its key.json mark is retrained; a tripwire is not cached; reuse=false trains
  every time.
- The runner keeps (artifact, λ) as the detector state: champion.json, the
  history summary, and the end-of-round state all carry λ.
"""

import json
from pathlib import Path

import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop import adapt_round
from agentic_configuration_search.loop.runner import run_rounds
from tests.test_loop_adapt import _fakes, _human_candidates_only, root  # noqa: F401


@pytest.fixture(autouse=True)
def _blend_off():
    gateway.set_classifier_blend(None)
    yield
    gateway.set_classifier_blend(None)


def _cfg(**cc):
    return config_from_dict({"dataset": "x", "planner": {"kind": "deterministic"},
                             "crop_classifier": {"min_samples": 4, "min_background": 0,
                                                 "margin": 0.01, **cc}})


def _stub_mining(monkeypatch, n_gt=10, n_fp=3):
    """Skip the fit_train pass and crop mining — record how often mining ran."""
    mined = []
    monkeypatch.setattr(adapt_round, "chosen_groups", lambda per_class: [])
    monkeypatch.setattr(adapt_round, "infer_for_groups", lambda *a, **k: [])
    monkeypatch.setattr(adapt_round, "combined_detections",
                        lambda n, passes, groups, thresholds=None: ([], None))

    def mine(train, ontology, floor, thresholds, out_dir):
        mined.append(out_dir)
        return Path(out_dir) / "manifest.jsonl", {"train_gt": n_gt, "train_fp": n_fp}
    monkeypatch.setattr(adapt_round, "mine_classifier_crops", mine)
    return mined


def _train_fn(trained):
    def train(repo_root, *, manifest, ontology, out_dir, device, log_path):
        trained.append(out_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "classifier.pt").write_bytes(b"fake")
        return out_dir
    return train


def _run(cfg, ds, out, gains, *, cache=None, origin=None, trained=None,
         restore=(None, None)):
    """One arm. The fit_dev score is 0.5 + gains[λ] for the loaded λ (0.5 without a classifier)."""
    state = {"root": None, "lam": None}
    calls = []

    def adapt_ctl(path, lam=None):
        calls.append((None if path is None else str(path), lam))
        state["root"], state["lam"] = path, (lam if path is not None else None)

    def screen(materials, axes):
        f = 0.5 if state["root"] is None else 0.5 + gains.get(state["lam"], 0.0)
        return {"fit_dev_micro_f_beta": f, "per_class": {}, "configs": []}

    res = adapt_round.run_classifier_candidate(
        cfg, ds, {}, None, out, screen_fn=screen, adapt_ctl=adapt_ctl,
        repo_root=out.parent, restore_root=restore[0], restore_lambda=restore[1],
        seed=0, client=object(), train_fn=_train_fn(trained if trained is not None else []),
        cache_dir=cache, origin=origin)
    return res, calls, state


def test_grid_picks_the_best_lambda_and_keeps_the_baseline(dataset_root, tmp_path,
                                                           monkeypatch):
    _stub_mining(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    res, calls, state = _run(_cfg(), ds, tmp_path / "c1",
                             {0.0: 0.02, 0.25: 0.08, 0.5: 0.08, 0.75: 0.05, 1.0: -0.1})
    assert res["decision"]["adopted"] is True
    assert res["adapt_lambda"] == 0.25                     # tie with 0.5 → smaller λ
    assert res["grid"][0] == {"lam": None, "f_beta": 0.5, "delta": 0.0}
    assert [g["lam"] for g in res["grid"][1:]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert state == {"root": res["adapt_root"], "lam": 0.25}   # left on the adopted arm
    rec = json.loads((tmp_path / "c1" / "classifier_decision.json").read_text())
    assert rec["grid"][0]["lam"] is None and rec["decision"]["lam"] == 0.25


def test_rejection_restores_the_champion_state(dataset_root, tmp_path, monkeypatch):
    _stub_mining(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    res, calls, state = _run(_cfg(), ds, tmp_path / "c1", {0.5: 0.005},
                             restore=("/champ/artifact", 0.75))
    assert res["decision"]["adopted"] is False and res["adapt_root"] is None
    assert res["decision"]["lam"] == 0.5                   # the grid's best is still recorded
    assert calls[0] == ("/champ/artifact", 0.75) and calls[-1] == ("/champ/artifact", 0.75)


def test_relabel_only_arm_can_win(dataset_root, tmp_path, monkeypatch):
    _stub_mining(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    res, _, _ = _run(_cfg(), ds, tmp_path / "c1", {0.0: 0.1, 0.25: 0.02})
    assert res["decision"]["adopted"] is True and res["adapt_lambda"] == 0.0


def test_second_proposal_reuses_the_artifact(dataset_root, tmp_path, monkeypatch):
    mined = _stub_mining(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cache, trained = tmp_path / "run" / "classifier_cache", []
    first, _, _ = _run(_cfg(), ds, tmp_path / "run" / "c1", {0.5: 0.1}, cache=cache,
                       origin={"round": 1, "id": "r1_a"}, trained=trained)
    assert len(trained) == 1 and str(trained[0]).startswith(str(cache))
    assert "classifier_reused" not in first["decision"]
    second, _, _ = _run(_cfg(), ds, tmp_path / "run" / "c2", {1.0: 0.1}, cache=cache,
                        origin={"round": 2, "id": "r2_b"}, trained=trained)
    assert len(trained) == 1 and len(mined) == 1, "retrained or re-mined under the same key"
    assert second["decision"]["classifier_reused"] == {"round": 1, "id": "r1_a"}
    assert second["adapt_root"] == first["adapt_root"] and second["adapt_lambda"] == 1.0


def test_interrupted_training_and_reuse_off_retrain(dataset_root, tmp_path, monkeypatch):
    _stub_mining(monkeypatch)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cache, trained = tmp_path / "run" / "classifier_cache", []
    _run(_cfg(), ds, tmp_path / "run" / "c1", {0.5: 0.1}, cache=cache, trained=trained)
    for meta in cache.rglob(adapt_round.CACHE_META):
        meta.unlink()                                      # as if it died before the mark
    _run(_cfg(), ds, tmp_path / "run" / "c2", {0.5: 0.1}, cache=cache, trained=trained)
    assert len(trained) == 2
    _run(_cfg(reuse=False), ds, tmp_path / "run" / "c3", {0.5: 0.1}, cache=None,
         trained=trained)
    assert len(trained) == 3 and str(trained[2]).startswith(str(tmp_path / "run" / "c3"))


def test_tripwire_is_not_cached(dataset_root, tmp_path, monkeypatch):
    mined = _stub_mining(monkeypatch, n_gt=1, n_fp=0)
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    cache = tmp_path / "run" / "classifier_cache"
    for c in ("c1", "c2"):
        res, _, _ = _run(_cfg(), ds, tmp_path / "run" / c, {}, cache=cache)
        assert res["decision"]["tripwire"] is True and res["adapt_lambda"] is None
    assert len(mined) == 2 and not list(cache.rglob(adapt_round.CACHE_META))


@pytest.mark.parametrize("bad", [[], [1.5], [0.5, 0.5], ["0.5"], [True]])
def test_lambdas_are_validated(bad):
    from agentic_configuration_search.config import ConfigError
    with pytest.raises(ConfigError, match="crop_classifier.lambdas"):
        _cfg(lambdas=bad)


# ------------------------------------------------------------------ runner

def _blend_infer(root):
    """Class 2 is confused as class 1 unless a classifier is loaded with λ ≤ 0.5
    (higher λ is made to hurt) — so the λ grid matters end to end."""
    from agentic_configuration_search.core.coco import Split
    splits = {n: Split.load(root / n) for n in ("train", "valid")}
    by_path = {str(s.image_path(e)): e for s in splits.values() for e in s.images}

    def fake(ctx, client, config, queries, ontology, image_paths,
             alias_to_name=None, **kw):
        lam = gateway.classifier_blend()
        fixed = lam is not None and lam <= 0.5
        out = []
        for p in image_paths:
            dets = []
            for a in by_path[str(p)].annotations:
                cid = a.category_id
                if cid == 2 and not fixed:
                    cid = 1
                dets.append({"box": list(a.bbox_xyxy), "score": 0.9,
                             "category_id": cid, "sources": ["full"]})
            out.append(dets)
        return out
    return fake


def _round_cfg(root, **cc):
    return config_from_dict({
        "planner": {"kind": "deterministic"}, "dataset": str(root),
        "crop_classifier": {"min_samples": 4, "min_background": 0, "margin": 0.01,
                            "lambdas": [0.25, 0.5, 1.0], **cc}})


def test_runner_records_lambda_and_reuses_training(root, tmp_path,  # noqa: F811
                                                   monkeypatch):
    _human_candidates_only(monkeypatch)
    monkeypatch.setattr(gateway, "infer_config", _blend_infer(root))
    trained = []
    monkeypatch.setattr(adapt_round, "train_crop_classifier", _train_fn(trained))
    loaded = []
    hooks = {"weights_ctl": lambda w: None, "adapt_ctl": lambda p: loaded.append(p),
             "repo_root": tmp_path, "base_weights": tmp_path / "b"}
    cands = tmp_path / "candidates.yaml"
    cands.write_text("- id: cls\n  axes: {crop_classifier: true}\n  rationale: x\n")
    run = tmp_path / "run"
    result = run_rounds(root, out_dir=run, planner="deterministic", rounds=1,
                        cfg=_round_cfg(root), candidates_file=cands,
                        detector_hooks=hooks, **_fakes())
    assert result.decisions[1]["adopted"] is True
    assert result.champion["adapt_lambda"] == 0.25         # best, smaller of the tie
    assert len(trained) == 1 and str(trained[0]).startswith(str(run / "classifier_cache"))
    meta = json.loads((run / "champion.json").read_text())
    assert meta["adapt_lambda"] == 0.25
    assert gateway.classifier_blend() == 0.25              # the round ends in the champion state
    # λ changes did not reload the artifact — loads are path changes only
    assert all(a is not None for a in loaded[:1]) and len(set(map(str, loaded))) <= 2
    hist = [json.loads(ln) for ln in (run / "history.jsonl").read_text().splitlines()]
    row = next(c for c in hist[1]["candidates"] if c["id"] == "r1_human_cls")
    assert row["classifier_lambda"] == 0.25
    assert hist[1]["champion"]["classifier_lambda"] == 0.25


def test_resume_restores_the_champion_lambda(root, tmp_path, monkeypatch):  # noqa: F811
    _human_candidates_only(monkeypatch)
    monkeypatch.setattr(gateway, "infer_config", _blend_infer(root))
    monkeypatch.setattr(adapt_round, "train_crop_classifier", _train_fn([]))
    loaded = []
    hooks = {"weights_ctl": lambda w: None, "adapt_ctl": lambda p: loaded.append(p),
             "repo_root": tmp_path, "base_weights": tmp_path / "b"}
    cands = tmp_path / "candidates.yaml"
    cands.write_text("- id: cls\n  axes: {crop_classifier: true}\n  rationale: x\n")
    run = tmp_path / "run"
    first = run_rounds(root, out_dir=run, planner="deterministic", rounds=1,
                       cfg=_round_cfg(root), candidates_file=cands,
                       detector_hooks=hooks, **_fakes())
    gateway.set_classifier_blend(None)                     # a new process starts blend-off
    loaded.clear()
    cands.write_text("")
    run_rounds(root, out_dir=run, planner="deterministic", rounds=2,
               cfg=_round_cfg(root), candidates_file=cands, detector_hooks=hooks,
               resume=True, **_fakes())
    assert str(loaded[0]) == first.champion["adapt_root"]
    assert gateway.classifier_blend() == 0.25


def _choose(lambdas, scores, *, base=0.25, retune=True, margin=0.01):
    """Loop._choose_lambda on a stand-in loop — the screen score per blend λ."""
    from types import SimpleNamespace

    from agentic_configuration_search.loop.runner import Loop
    cfg = config_from_dict({"dataset": "x", "crop_classifier": {
        "lambdas": lambdas, "margin": margin, "retune_lambda": retune}})
    seen = []

    def screen(materials, axes):
        lam = gateway.classifier_blend()
        seen.append(lam)
        return {"fit_dev_micro_f_beta": scores[lam], "lam": lam}

    fake = SimpleNamespace(cfg=cfg, _adapt_path="/a", _adapt_lambda=base, _screen=screen)
    gateway.set_classifier_blend(base)
    entry = {"id": "c"}
    setting = Loop._choose_lambda(fake, None, {}, entry, {"fit_dev_micro_f_beta":
                                                          scores[base], "lam": base})
    return setting, entry, seen


def test_ordinary_candidate_rechooses_lambda_only_past_margin():
    setting, entry, seen = _choose([0.25, 0.5, 1.0], {0.25: 0.80, 0.5: 0.805, 1.0: 0.85})
    assert seen == [0.5, 1.0] and gateway.classifier_blend() == 0.25   # restored after
    assert entry["_adapt_lambda"] == 1.0 and setting["lam"] == 1.0
    assert [g["lam"] for g in entry["classifier_lambda_grid"]] == [0.25, 0.5, 1.0]
    # within margin — the champion λ stays
    setting, entry, _ = _choose([0.25, 0.5, 1.0], {0.25: 0.80, 0.5: 0.805, 1.0: 0.808})
    assert entry["_adapt_lambda"] == 0.25 and setting["lam"] == 0.25


def test_no_rechoice_without_a_champion_classifier_or_with_retune_off():
    _, entry, seen = _choose([0.25, 1.0], {0.25: 0.8, 1.0: 0.9}, retune=False)
    assert seen == [] and "_adapt_lambda" not in entry
    from types import SimpleNamespace

    from agentic_configuration_search.loop.runner import Loop
    fake = SimpleNamespace(cfg=_cfg(), _adapt_path=None, _adapt_lambda=None,
                           _screen=lambda *a: pytest.fail("screened without a classifier"))
    s0 = {"fit_dev_micro_f_beta": 0.5}
    assert Loop._choose_lambda(fake, None, {}, {}, s0) is s0


def test_serve_champion_restores_a_leftover_arm_state():
    """An arm that cleared its margin but was dropped before holdout leaves the
    detector in its own state — the next attempt starts from the champion state."""
    from agentic_configuration_search.loop.runner import Loop, _RoundAcc
    fake, calls = _state_fake("/champ", 1.0)
    champ = {"weights": None, "adapt_root": "/champ", "adapt_lambda": 0.5}
    acc = _RoundAcc(served_w="/arm.json", served_a="/champ", served_l=1.0)
    Loop._serve_champion(fake, champ, acc)
    assert calls == [("w", None), ("a", "/champ", 0.5)]
    assert (acc.served_w, acc.served_a, acc.served_l) == (None, "/champ", 0.5)
    calls.clear()
    Loop._serve_champion(fake, champ, acc)                 # already champion — nothing to do
    assert calls == []


def _state_fake(path, lam):
    """A Loop stand-in whose _adapt_ctl records the call and the loaded state."""
    from types import SimpleNamespace
    calls = []
    fake = SimpleNamespace(detector_hooks={"weights_ctl": lambda w: calls.append(("w", w))},
                           _adapt_path=path, _adapt_lambda=lam)

    def adapt_ctl(p, lam):
        calls.append(("a", p, lam))
        fake._adapt_path, fake._adapt_lambda = p, lam
    fake._adapt_ctl = adapt_ctl
    return fake, calls


def test_serve_champion_checks_the_loaded_state_not_the_round_record():
    """The round record says champion, but another artifact is actually loaded —
    it is still put back to the champion's."""
    from agentic_configuration_search.loop.runner import Loop, _RoundAcc
    fake, calls = _state_fake("/other", 0.25)
    champ = {"weights": None, "adapt_root": "/champ", "adapt_lambda": 0.5}
    acc = _RoundAcc(served_w=None, served_a="/champ", served_l=0.5)
    Loop._serve_champion(fake, champ, acc)
    assert calls == [("a", "/champ", 0.5)]
    assert (fake._adapt_path, fake._adapt_lambda) == ("/champ", 0.5)
