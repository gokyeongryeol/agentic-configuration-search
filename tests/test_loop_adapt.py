"""crop classifier axis e2e — a stub where client-side reclassification fixes class_confusion.

Scenario: the stub detector **misclassifies** class-2 GT as class 1 (boxes are exact).
Once the classifier artifact is loaded (FakeAdapt.root) labels are corrected -> fit_dev Δ passes
-> holdout gate accepts -> the champion carries adapt_root. Training is injected via train_fn.
"""

import json
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
    size = lambda i: 20 + (i % 3) * 10  # noqa: E731
    ds = tmp_path / "ds"
    make_split(ds / "train", [
        {"name": f"tr{i:02d}.png", "w": 100, "h": 80,
         "anns": [(1 if i % 2 else 2, 10, 10, size(i), size(i))]}
        for i in range(12)])
    make_split(ds / "valid", [        # the loop's holdout
        {"name": f"t{i}.png", "w": 100, "h": 80,
         "anns": [(1 if i < 4 else 2, 10, 10, size(i), size(i))]}
        for i in range(8)])
    make_split(ds / "test", [{"name": "v.png", "w": 100, "h": 80,   # report-only — the loop never reads it
                              "anns": [(1, 10, 10, 20, 20)]}])
    return ds


class FakeAdapt:
    """Recording fake for adapt_ctl — when root is loaded, the stub corrects labels."""

    def __init__(self):
        self.root = None
        self.calls: list = []

    def ctl(self, root):
        self.root = root
        self.calls.append(root)


def _fake_infer(root, server: FakeAdapt, confuse: bool = True):
    splits = {n: Split.load(root / n) for n in ("train", "valid")}
    by_path = {str(s.image_path(e)): e for s in splits.values() for e in s.images}

    def fake(ctx, client, config, queries, ontology, image_paths,
             alias_to_name=None, **kw):
        out = []
        for p in image_paths:
            dets = []
            for a in by_path[str(p)].annotations:
                cid = a.category_id
                if confuse and cid == 2 and server.root is None:
                    cid = 1                      # class_confusion — fixed by reclassification
                dets.append({"box": list(a.bbox_xyxy), "score": 0.9,
                             "category_id": cid, "sources": ["full"]})
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
        refine_alias_fn=lambda name, cn, pool, others: {"alias": None,
                                                        "reason": "none"},
        refine_negatives_fn=lambda name, cn, pool, mk: {"keep": [],
                                                        "rejected": []},
        client=object())


def _fake_train(tmp_path):
    def train_fn(repo_root, *, manifest, ontology, out_dir, device, log_path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "classifier.pt").write_bytes(b"fake")
        return out_dir
    return train_fn


def _human_candidates_only(monkeypatch):
    """These scenarios measure the classifier path on the human-injected candidate.
    When the deterministic planner proposes the same axes combination
    (crop_classifier=True), combo dedup keeps the planner's (earlier) candidate and
    folds human_*. With untrainable lora blocked up front (DatasetFacts), the
    deterministic planner fills that slot with exactly this combination on the
    small fixture, so planner proposals are emptied here."""
    from agentic_configuration_search.planner import deterministic
    monkeypatch.setattr(deterministic, "propose",
                        lambda history, diagnosis, **kw: {
                            "planner": "deterministic", "candidates": [],
                            "stop": False})


def test_classifier_candidate_adopted_end_to_end(root, tmp_path, monkeypatch):
    _human_candidates_only(monkeypatch)
    server = FakeAdapt()
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, server))
    import agentic_configuration_search.loop.adapt_round as ar
    monkeypatch.setattr(ar, "train_crop_classifier", _fake_train(tmp_path))

    cfg = config_from_dict({
        "planner": {"kind": "deterministic"},
        "dataset": str(root),
        # misclassified FPs sit on GT (IoU 1.0), so no background arises — relax the tripwire
        "crop_classifier": {"min_samples": 4, "min_background": 0,
                            "margin": 0.01},
    })
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text(
        "- id: fixcls\n  axes: {crop_classifier: true}\n  rationale: fix misclassification\n")
    result = run_rounds(
        root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
        cfg=cfg, candidates_file=cand_file,
        detector_hooks={"weights_ctl": lambda w: None, "adapt_ctl": server.ctl,
                   "repo_root": tmp_path, "base_weights": tmp_path / "b-large"},
        **_fakes())

    base = result.decisions[0]["micro_f_beta"]
    assert base == pytest.approx(0.5)            # all 4 cat2 images misclassified
    r1 = result.decisions[1]
    assert r1["adopted"] is True and r1["id"] == "r1_human_fixcls"
    assert result.champion["axes"]["crop_classifier"] is True
    assert result.champion["eval"]["micro_f_beta"] == pytest.approx(1.0)
    # champion carries the artifact, and the detector stays in that state
    assert result.champion["adapt_root"] and "artifact" in result.champion["adapt_root"]
    assert server.root is not None
    meta = json.loads((tmp_path / "run" / "champion.json").read_text())
    assert meta["adapt_root"] == result.champion["adapt_root"]


def test_classifier_rejected_restores_state(root, tmp_path, monkeypatch):
    """If the artifact fixes nothing (stub: confuse=False, already perfect), fit_dev
    Δ=0 -> arm rejected -> current state (None) restored."""
    server = FakeAdapt()
    monkeypatch.setattr(gateway, "infer_config",
                        _fake_infer(root, server, confuse=False))
    import agentic_configuration_search.loop.adapt_round as ar
    monkeypatch.setattr(ar, "train_crop_classifier", _fake_train(tmp_path))

    cfg = config_from_dict({
        "planner": {"kind": "deterministic"},
        "dataset": str(root),
        "crop_classifier": {"min_samples": 4, "min_background": 0},
    })
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text(
        "- id: fixcls\n  axes: {crop_classifier: true}\n  rationale: try correction\n")
    result = run_rounds(
        root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
        cfg=cfg, candidates_file=cand_file,
        detector_hooks={"weights_ctl": lambda w: None, "adapt_ctl": server.ctl,
                   "repo_root": tmp_path, "base_weights": tmp_path / "b-large"},
        **_fakes())
    assert result.decisions[1]["adopted"] is False
    assert "r1_human_fixcls" in result.decisions[1]["skipped"]
    assert server.root is None                   # current state restored
    assert result.champion["adapt_root"] is None


def test_classifier_without_adapt_ctl_skips(root, tmp_path, monkeypatch):
    server = FakeAdapt()
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, server))
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text(
        "- id: fixcls\n  axes: {crop_classifier: true}\n  rationale: correction\n")
    result = run_rounds(root, out_dir=tmp_path / "run", planner="deterministic",
                        rounds=1, candidates_file=cand_file, **_fakes())
    assert "r1_human_fixcls" in result.decisions[1]["skipped"]


def _ONTO_FOR(root):
    from agentic_configuration_search.core.data import load_dataset
    return load_dataset(root).ontology


def test_arm_refits_rescorer_instead_of_dropping(root, tmp_path, monkeypatch):
    """Upstream arms keep the rescorer and refit it on the new distribution — measuring without
    it would make the gate's Δ a compound "+upstream, −rescorer", so the upstream axis would fight
    handicapped by the rescorer's gain. If the refit is identity, only the label is corrected."""
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop import rescore
    from agentic_configuration_search.loop.runner import Loop

    loop = Loop(config_from_dict({"dataset": str(root)}), root, tmp_path / "run")
    setting, materials = {"per_class": {}, "configs": []}, object()

    class _Active:
        active = True

    calls = []
    monkeypatch.setattr(rescore, "fit_for_setting",
                        lambda *a, **k: (_Active(), {}, {"p": 1}))
    monkeypatch.setattr(rescore, "rescored_screen",
                        lambda *a, **k: calls.append(a) or {"rescored": True})

    entry = {"id": "c1", "axes": {"lora": True, "rescorer": True}}
    out = loop._arm_rescorer(entry, setting, materials, tmp_path / "m")
    assert out == {"rescored": True} and calls, "rescoring must be applied on top of the arm setting"
    assert entry["axes"]["rescorer"] is True          # label kept
    assert entry["rescorer_path"].endswith("rescorer.json")

    # if the refit is identity (α=0 for every class), axes are corrected
    class _Identity:
        active = False

    monkeypatch.setattr(rescore, "fit_for_setting",
                        lambda *a, **k: (_Identity(), {}, {}))
    entry2 = {"id": "c2", "axes": {"lora": True, "rescorer": True}}
    assert loop._arm_rescorer(entry2, setting, materials, tmp_path / "m") is setting
    assert entry2["axes"]["rescorer"] is False
    assert "rescorer_path" not in entry2

    # no rescorer on the champion -> do nothing
    entry3 = {"id": "c3", "axes": {"lora": True, "rescorer": False}}
    assert loop._arm_rescorer(entry3, setting, materials, tmp_path / "m") is setting


def test_resume_restores_detector_state(root, tmp_path, monkeypatch):
    """resume must reapply the champion's detector state — a fresh process starts with the base
    model, so without reapplying, candidates are scored by a different detector than the
    champion and Δ is contaminated."""
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop.runner import Loop

    seen = {"w": [], "a": []}
    loop = Loop(config_from_dict({"dataset": str(root)}), root, tmp_path / "run",
                detector_hooks={"weights_ctl": lambda w: seen["w"].append(w),
                           "adapt_ctl": lambda p: seen["a"].append(p),
                           "repo_root": tmp_path, "base_weights": tmp_path / "b"})

    loop._restore_detector_state({"weights": "/w/merged.safetensors",
                                "adapt_root": "/a/artifact", "adapt_lambda": 0.25})
    assert [str(x) for x in seen["w"]] == ["/w/merged.safetensors"]
    assert [str(x) for x in seen["a"]] == ["/a/artifact"]
    assert gateway.classifier_blend() == 0.25          # the blend λ comes back too

    # a champion from before the blend (artifact without λ) cannot be continued
    with pytest.raises(RuntimeError, match="adapt_lambda"):
        Loop(config_from_dict({"dataset": str(root)}), root, tmp_path / "r3",
             detector_hooks={"weights_ctl": lambda w: None, "adapt_ctl": lambda p: None,
                             "repo_root": tmp_path, "base_weights": tmp_path / "b"}
             )._restore_detector_state({"weights": None, "adapt_root": "/a/artifact"})

    # champion without detector state -> reapply nothing (avoid needless swaps)
    seen["w"].clear(); seen["a"].clear()
    loop._restore_detector_state({"weights": None, "adapt_root": None})
    assert seen["w"] == [] and seen["a"] == []

    # a run without detector state controls (no detector_hooks) skips silently
    bare = Loop(config_from_dict({"dataset": str(root)}), root, tmp_path / "r2")
    bare._restore_detector_state({"weights": "/w", "adapt_root": "/a"})


def test_classifier_trains_on_fit_train_only(root, tmp_path, monkeypatch):
    """The arm is scored by its fit_dev Δ — if fit_dev crops are in the training
    material, scoring becomes an in-sample comparison on crops the classifier has
    already seen, and the margin filters nothing. Training uses fit_train only."""
    from agentic_configuration_search.core.data import load_dataset
    import agentic_configuration_search.loop.adapt_round as ar

    server = FakeAdapt()
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, server))
    seen: dict = {}
    real_mine = ar.mine_classifier_crops

    def spy_mine(split, ontology, dets_by_image, thresholds, out_dir):
        seen["images"] = {e.file_name for e in split.images}
        seen["n_dets"] = len(dets_by_image)
        return real_mine(split, ontology, dets_by_image, thresholds, out_dir)

    def train_fn(repo_root, *, manifest, **kw):
        seen["manifest"] = [json.loads(l) for l in
                            Path(manifest).read_text().splitlines()]
        return _fake_train(tmp_path)(repo_root, manifest=manifest, **kw)

    monkeypatch.setattr(ar, "mine_classifier_crops", spy_mine)
    monkeypatch.setattr(ar, "train_crop_classifier", train_fn)
    cfg = config_from_dict({
        "planner": {"kind": "deterministic"},
        "dataset": str(root),
        "crop_classifier": {"min_samples": 1, "min_background": 0, "margin": 0.01},
    })
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text("- id: fixcls\n  axes: {crop_classifier: true}\n")
    run_rounds(root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
               cfg=cfg, candidates_file=cand_file,
               detector_hooks={"weights_ctl": lambda w: None, "adapt_ctl": server.ctl,
                               "repo_root": tmp_path, "base_weights": tmp_path / "b"},
               **_fakes())

    ds = load_dataset(root, seed=0, fit_train_ratio=0.8)
    fit_train = {e.file_name for e in ds.fit_train.images}
    fit_dev_anns = {a.id for e in ds.fit_dev.images for a in e.annotations}
    assert seen["images"] == fit_train
    assert seen["n_dets"] == len(fit_train)          # the inference pass is fit_train only too
    gt_ids = {int(Path(r["crop_path"]).stem.split("_")[1])
              for r in seen["manifest"] if r["source"] == "train_gt"}
    assert gt_ids and not (gt_ids & fit_dev_anns)


def test_classifier_tripwire_records_empty_axis(root, tmp_path, monkeypatch):
    """The classifier tripwire (too few training crops) cannot be decided up front —
    the background FP count is known only after inference. When it trips, the axis is
    recorded as empty material (empty_axis), so the axis status counts it as no
    material (the board's run column) — not as a failure."""
    import agentic_configuration_search.loop.adapt_round as ar

    _human_candidates_only(monkeypatch)
    server = FakeAdapt()
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, server))
    monkeypatch.setattr(ar, "train_crop_classifier", _fake_train(tmp_path))
    cfg = config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(root),
                            "crop_classifier": {"min_samples": 10_000}})
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text("- id: fixcls\n  axes: {crop_classifier: true}\n")
    run_rounds(root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
               cfg=cfg, candidates_file=cand_file,
               detector_hooks={"weights_ctl": lambda w: None, "adapt_ctl": server.ctl,
                               "repo_root": tmp_path, "base_weights": tmp_path / "b"},
               **_fakes())
    hist = [json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rec = {c["id"]: c for c in hist[1]["candidates"]}["r1_human_fixcls"]
    assert rec["skipped"] == "classifier_gate"
    assert rec["empty_axis"] == "crop_classifier"

    from agentic_configuration_search.planner.axis_status import axis_status
    s = axis_status(hist, 2).axes["crop_classifier"]
    assert s.no_material == [True] and s.measured == 0     # the tripwire is not a failure


def test_classifier_margin_rejection_is_not_dead(root, tmp_path, monkeypatch):
    """A rejection after training that falls short of the fit_dev margin is not a dead
    axis — the material was not short; it simply had no effect."""
    import agentic_configuration_search.loop.adapt_round as ar

    server = FakeAdapt()
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, server, confuse=False))
    monkeypatch.setattr(ar, "train_crop_classifier", _fake_train(tmp_path))
    cfg = config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(root),
                            "crop_classifier": {"min_samples": 1, "min_background": 0}})
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text("- id: fixcls\n  axes: {crop_classifier: true}\n")
    run_rounds(root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
               cfg=cfg, candidates_file=cand_file,
               detector_hooks={"weights_ctl": lambda w: None, "adapt_ctl": server.ctl,
                               "repo_root": tmp_path, "base_weights": tmp_path / "b"},
               **_fakes())
    hist = [json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rec = {c["id"]: c for c in hist[1]["candidates"]}["r1_human_fixcls"]
    assert rec["skipped"] == "classifier_gate" and "empty_axis" not in rec


# ------------------------------- detector-state candidates mine their materials on the champion state

def _two_arm_loop(root, tmp_path, monkeypatch, state):
    from agentic_configuration_search.fitting.materials import Materials
    from agentic_configuration_search.loop.runner import Loop
    loop = Loop(config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(root),
                                  "lora": {"min_images": 1}}),
                root, tmp_path / "run",
                detector_hooks={"weights_ctl": lambda w: state.__setitem__("w", w),
                                "adapt_ctl": lambda p: state.__setitem__("a", p),
                                "repo_root": tmp_path, "base_weights": tmp_path / "b"})
    mined = []

    def materialize(self, axes, mat_dir, wall):
        mined.append((state.get("w"), state.get("a"), gateway.classifier_blend()))
        return Materials()
    monkeypatch.setattr(Loop, "_materialize_timed", materialize)
    monkeypatch.setattr(Loop, "_guard_material", lambda self, *a, **k: False)
    monkeypatch.setattr(Loop, "_arm_rescorer", lambda self, e, s, m, d: s)
    monkeypatch.setattr(Loop, "_fit_dev_cmp", lambda self, s: None)
    monkeypatch.setattr(Loop, "_arm_skip", lambda self, row: False)
    monkeypatch.setattr(Loop, "_gate", lambda self, ch, ev: {})
    monkeypatch.setattr(Loop, "_evaluate_candidate", lambda self, e, s, m: {})
    loop._round_no = 1
    return loop, mined


def _two_candidates(loop, tmp_path, axis, prefix):
    from agentic_configuration_search.loop.runner import _RoundAcc
    champion = {"weights": None, "adapt_root": None, "adapt_lambda": None, "axes": {}}
    acc = _RoundAcc(served_w=None, served_a=None, served_l=None)
    loop._attempt(tmp_path / "rd", champion,
                  {"candidates": [{"id": f"{prefix}1", "axes": {axis: True}},
                                  {"id": f"{prefix}2", "axes": {axis: True}}],
                   "stop": False},
                  acc)
    assert len(acc.evaluated) == 2, "both candidates were not evaluated — scenario invalid"


def test_second_lora_candidate_mines_on_champion_weights(root, tmp_path, monkeypatch):
    """An adopted lora candidate is measured on holdout with its arm weights loaded.
    The next lora candidate's materials (FP-mining train pass) must be mined on the
    champion detector — run_tail_candidate restores the baseline only after the
    materials are built."""
    from agentic_configuration_search.loop import tail_round
    state = {"w": None, "a": None}
    loop, mined = _two_arm_loop(root, tmp_path, monkeypatch, state)

    def fake_tail(cfg, ds, axes, materials, out_dir, *, weights_ctl, **kw):
        arm = tmp_path / f"{Path(out_dir).name}.json"
        weights_ctl(arm)                                 # ends with the adopted arm loaded
        return {"weights": arm, "setting": {"fit_dev_micro_f_beta": 0.5},
                "grid": [{"f_beta": 0.5}], "decision": {"adopted": True}}
    monkeypatch.setattr(tail_round, "run_tail_candidate", fake_tail)
    _two_candidates(loop, tmp_path, "lora", "l")
    assert [m[0] for m in mined] == [None, None], "mined on the previous arm's weights"


def test_second_classifier_candidate_mines_without_previous_arm(root, tmp_path,
                                                                 monkeypatch):
    """Same for classifier candidates — mining the next candidate's materials with
    the adopted arm's artifact and λ loaded picks FPs from reclassified detections."""
    from agentic_configuration_search.loop import adapt_round
    state = {"w": None, "a": None}
    loop, mined = _two_arm_loop(root, tmp_path, monkeypatch, state)

    def fake_clf(cfg, ds, axes, materials, out_dir, *, adapt_ctl, restore_root,
                 restore_lambda, **kw):
        adapt_ctl(restore_root, restore_lambda)
        art = Path(out_dir) / "artifact"
        adapt_ctl(art, 1.0)                              # ends with the adopted arm loaded
        return {"adapt_root": art, "adapt_lambda": 1.0,
                "setting": {"fit_dev_micro_f_beta": 0.5}, "decision": {"adopted": True}}
    monkeypatch.setattr(adapt_round, "run_classifier_candidate", fake_clf)
    _two_candidates(loop, tmp_path, "crop_classifier", "k")
    assert [m[1:] for m in mined] == [(None, None), (None, None)], \
        "mined on the previous arm's artifact"


# ------------------------------------------------- same weights are not swapped in again

def test_same_weights_are_not_reloaded(root, tmp_path):
    """Like the artifact, weights already loaded are not swapped in again — the
    pass cache flush goes with it. At first (state unknown) and after a failed
    swap they are swapped in again."""
    from agentic_configuration_search.loop.runner import Loop
    calls = []

    def ctl(w):
        calls.append(w)
        if w == "boom":
            raise RuntimeError("swap failed")
    loop = Loop(config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(root)}),
                root, tmp_path / "run",
                detector_hooks={"weights_ctl": ctl, "adapt_ctl": lambda p: None,
                                "repo_root": tmp_path, "base_weights": tmp_path / "b"})
    loop._weights_ctl(None)
    loop._weights_ctl(None)
    assert calls == [None], "only the first — the same state is not swapped in again"
    loop._weights_ctl(tmp_path / "arm")
    loop._weights_ctl(str(tmp_path / "arm"))              # str and Path mixed are the same
    assert calls == [None, tmp_path / "arm"]
    with pytest.raises(RuntimeError):
        loop._weights_ctl("boom")
    loop._weights_ctl(tmp_path / "arm")                   # after a failure the state is unknown
    assert calls[-1] == tmp_path / "arm" and len(calls) == 4


def test_lora_candidate_baseline_keeps_the_pass_cache(root, tmp_path, monkeypatch):
    """Once _serve_champion has the champion weights loaded, run_tail_candidate's
    baseline restore (weights_ctl(restore_weights)) does not touch the detector
    or empty the pass cache."""
    from agentic_configuration_search.loop import tail_round
    from agentic_configuration_search.loop.runner import _RoundAcc
    state = {"w": None, "a": None}
    loop, _ = _two_arm_loop(root, tmp_path, monkeypatch, state)
    loop._weights_ctl(None)                               # knows the champion state (base)
    flushed = []
    monkeypatch.setattr(gateway, "invalidate_pass_cache", lambda: flushed.append(1))

    def fake_tail(cfg, ds, axes, materials, out_dir, *, weights_ctl, restore_weights, **kw):
        weights_ctl(restore_weights)
        return {"weights": None, "setting": {"fit_dev_micro_f_beta": 0.5},
                "grid": [], "decision": {"adopted": False, "reason": "x"}}
    monkeypatch.setattr(tail_round, "run_tail_candidate", fake_tail)
    champion = {"weights": None, "adapt_root": None, "adapt_lambda": None, "axes": {}}
    loop._attempt(tmp_path / "rd", champion,
                  {"candidates": [{"id": "l1", "axes": {"lora": True}}], "stop": False},
                  _RoundAcc(served_w=None, served_a=None, served_l=None))
    assert flushed == []


# ------------------------------------- round-end and resume restores, pinned at the round level

def _state_loop(root, tmp_path, calls):
    from agentic_configuration_search.loop.runner import Loop
    return Loop(config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(root)}),
                root, tmp_path / "run",
                detector_hooks={"weights_ctl": lambda w: calls.append(("w", w)),
                                "adapt_ctl": lambda p: calls.append(("a", p)),
                                "repo_root": tmp_path, "base_weights": tmp_path / "b"})


def test_round_end_restores_the_loaded_artifact_not_the_round_record(root, tmp_path,
                                                                     monkeypatch):
    """The round record (acc.served_a) still says champion, but another artifact
    is loaded when the round ends — run_round puts the champion artifact back, so
    nothing but the champion state is left on the detector after the last round."""
    from agentic_configuration_search.loop.runner import Loop
    calls = []
    loop = _state_loop(root, tmp_path, calls)
    champ_art, other = tmp_path / "champ", tmp_path / "other"
    champion = {"round": 0, "id": "baseline", "axes": {"crop_classifier": True},
                "weights": None, "adapt_root": str(champ_art), "adapt_lambda": 0.5,
                "setting": {"fit_dev_micro_f_beta": 0.5, "configs": [], "per_class": {}},
                "materials_dir": str(tmp_path / "m"), "eval": {"micro_f_beta": 0.5}}
    loop._adapt_ctl(champ_art, 0.5)

    def attempt(self, rd, champion, proposal, acc):
        acc.n_runnable += 1                               # a candidate ran and was rejected
        self._adapt_ctl(other, 0.25)                      # a swap the round record missed
    monkeypatch.setattr(Loop, "_attempt", attempt)
    monkeypatch.setattr(Loop, "_diagnosis", lambda self, *a, **k: {})
    monkeypatch.setattr(Loop, "_crop_notes", lambda self, *a, **k: None)
    monkeypatch.setattr(Loop, "_propose",
                        lambda self, *a, **k: {"candidates": [], "stop": False})
    monkeypatch.setattr(Loop, "_reask", lambda self, *a, **k: None)
    calls.clear()
    loop.run_round(1, champion, [])
    assert (Path(loop._adapt_path), loop._adapt_lambda) == (champ_art, 0.5)
    assert calls[-1] == ("a", str(champ_art))
    assert gateway.classifier_blend() == 0.5


def test_resume_restore_marks_the_champion_weights_as_loaded(root, tmp_path):
    """The resume weight restore also goes through _weights_ctl — a following
    request for the same weights (the lora baseline) does not swap again or empty
    the pass cache."""
    calls = []
    loop = _state_loop(root, tmp_path, calls)
    arm = tmp_path / "arm.json"
    loop._restore_detector_state({"weights": str(arm), "adapt_root": None})
    loop._weights_ctl(arm)
    assert calls == [("w", arm)]


# ------------------------------- lora winners re-choose the classifier λ too

def _lora_on_classifier(root, tmp_path, monkeypatch, f, *, cfg_extra=None,
                        arm_skip=False):
    """Run one lora candidate through _attempt on a classifier champion (λ=0.5) —
    with the winning arm loaded, screen returns f[λ] per blend λ. Returns
    (loop, acc, seen, the λs measured on the winning arm)."""
    from agentic_configuration_search.fitting.materials import Materials
    from agentic_configuration_search.loop import tail_round
    from agentic_configuration_search.loop.runner import Loop, _RoundAcc
    state = {"w": None}
    loop = Loop(config_from_dict({"planner": {"kind": "deterministic"}, "dataset": str(root),
                                  "lora": {"min_images": 1},
                                  "crop_classifier": {"margin": 0.01, **(cfg_extra or {})}}),
                root, tmp_path / "run",
                detector_hooks={"weights_ctl": lambda w: state.__setitem__("w", w),
                                "adapt_ctl": lambda p: None,
                                "repo_root": tmp_path, "base_weights": tmp_path / "b"})
    loop._adapt_ctl(tmp_path / "artifact", 0.5)
    arm = tmp_path / "arm.json"
    screened_on = []

    def screen(materials, axes, split=None):
        lam = gateway.classifier_blend()
        screened_on.append((state["w"], lam))
        return {"fit_dev_micro_f_beta": f[lam], "configs": [], "per_class": {}, "lam": lam}
    loop._screen = screen
    seen = {}

    def arm_rescorer(self, entry, setting, materials, mat_dir):
        seen["refit"], seen["refit_setting"] = gateway.classifier_blend(), setting["lam"]
        return setting

    def evaluate(self, setting, materials):
        seen["holdout"] = gateway.classifier_blend()
        return {}
    monkeypatch.setattr(Loop, "_materialize_timed", lambda self, *a, **k: Materials())
    monkeypatch.setattr(Loop, "_guard_material", lambda self, *a, **k: False)
    monkeypatch.setattr(Loop, "_arm_rescorer", arm_rescorer)
    monkeypatch.setattr(Loop, "_fit_dev_cmp", lambda self, s: None)
    monkeypatch.setattr(Loop, "_arm_skip", lambda self, row: arm_skip)
    monkeypatch.setattr(Loop, "_evaluate", evaluate)
    monkeypatch.setattr(Loop, "_gate", lambda self, ch, ev: {})

    def fake_tail(cfg, ds, axes, materials, out_dir, *, weights_ctl, screen_fn, **kw):
        weights_ctl(arm)                                 # ends with the winning arm loaded
        return {"weights": arm, "setting": screen_fn(materials, axes),
                "grid": [{"f_beta": 0.6}], "decision": {"adopted": True}}
    monkeypatch.setattr(tail_round, "run_tail_candidate", fake_tail)
    champion = {"weights": None, "adapt_root": str(tmp_path / "artifact"),
                "adapt_lambda": 0.5, "axes": {"crop_classifier": True}}
    loop._champion, loop._round_no = champion, 1
    acc = _RoundAcc(served_w=None, served_a=str(tmp_path / "artifact"), served_l=0.5)
    loop._attempt(tmp_path / "rd", champion,
                  {"candidates": [{"id": "l1", "axes": {"lora": True,
                                                         "crop_classifier": True}}],
                   "stop": False}, acc)
    return loop, acc, seen, [lam for w, lam in screened_on if w == arm]


def test_lora_winner_reselects_classifier_lambda(root, tmp_path, monkeypatch):
    """lora changes the detections — on the winning arm the classifier λ is
    re-chosen by the same rule as ordinary candidates (margin over the champion
    λ), the rescorer is refit on that λ and holdout is measured with it. The
    history row's setting and fit_dev are that final setting too."""
    _, acc, seen, on_arm = _lora_on_classifier(
        root, tmp_path, monkeypatch, {0.0: 0.6, 0.25: 0.6, 0.5: 0.6, 0.75: 0.7, 1.0: 0.6})
    assert set(on_arm) == {0.0, 0.25, 0.5, 0.75, 1.0}       # the λ grid on the winning arm
    (ev,) = acc.evaluated
    assert ev["_adapt_lambda"] == 0.75
    assert seen == {"refit": 0.75, "refit_setting": 0.75, "holdout": 0.75}
    assert gateway.classifier_blend() == 0.5             # back to the detector state's λ
    (row,) = acc.screened
    assert row["classifier_lambda"] == 0.75 and len(row["classifier_lambda_grid"]) == 5
    assert row["setting"]["lam"] == 0.75 and row["fit_dev_micro_f_beta"] == 0.7


def test_lora_winner_keeps_champion_lambda_within_margin(root, tmp_path, monkeypatch):
    _, acc, seen, _ = _lora_on_classifier(
        root, tmp_path, monkeypatch, {0.0: 0.6, 0.25: 0.605, 0.5: 0.6, 0.75: 0.6, 1.0: 0.6})
    (ev,) = acc.evaluated
    assert ev["_adapt_lambda"] == 0.5 and seen["holdout"] == 0.5
    assert gateway.classifier_blend() == 0.5


def test_lora_without_retune_keeps_champion_lambda(root, tmp_path, monkeypatch):
    _, acc, seen, on_arm = _lora_on_classifier(
        root, tmp_path, monkeypatch, {0.5: 0.6, 0.75: 0.9},
        cfg_extra={"retune_lambda": False})
    assert on_arm == [0.5] and "_adapt_lambda" not in acc.evaluated[0]
    assert seen["holdout"] == 0.5


def test_lora_arm_skipped_after_retune_restores_the_blend(root, tmp_path, monkeypatch):
    """If λ changed and the pre-holdout fit_dev verdict then drops the arm, the
    blend still returns to the detector state's λ."""
    _, acc, _, _ = _lora_on_classifier(
        root, tmp_path, monkeypatch, {0.0: 0.6, 0.25: 0.6, 0.5: 0.6, 0.75: 0.7, 1.0: 0.6},
        arm_skip=True)
    assert not acc.evaluated and gateway.classifier_blend() == 0.5
