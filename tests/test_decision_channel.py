"""Guard — decision channel.

Contract: the adoption decision really comes from the holdout (valid) — perturbing
holdout GT **must change** the decision (positive control for the purity guard).
The planner is pinned to deterministic to rule out LLM nondeterminism.

Inference is stubbed to return fixed detections regardless of the candidate
setting, so holdout GT is the only variable that can move the decision.
"""

import json

import numpy as np
import pytest


def _stub_infer(ctx, client, config, queries, ontology, image_paths,
                alias_to_name=None, **kw):
    return [[{"box": [10.0, 10.0, 30.0, 30.0], "score": 0.9, "category_id": 1,
              "sources": ["full"]}] for _ in image_paths]


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


def test_decision_changes_when_holdout_changes(dataset_root, tmp_path, monkeypatch):
    from agentic_detector_config.infer import gateway
    from agentic_detector_config.loop.runner import run_rounds

    monkeypatch.setattr(gateway, "infer_config", _stub_infer)

    dec_a = run_rounds(dataset_root, out_dir=tmp_path / "a",
                       planner="deterministic", rounds=1, **_fakes()).decisions

    coco_p = dataset_root / "valid" / "_annotations.coco.json"
    coco = json.loads(coco_p.read_text())
    for a in coco["annotations"]:
        a["category_id"] = 1 if a["category_id"] == 2 else 2
    coco_p.write_text(json.dumps(coco))

    dec_b = run_rounds(dataset_root, out_dir=tmp_path / "b",
                       planner="deterministic", rounds=1, **_fakes()).decisions

    assert dec_a != dec_b, \
        "decision unchanged under perturbed holdout GT — the decision does not come from holdout"
    # concretely: the holdout micro F1 in the baseline adoption record must differ
    assert dec_a[0]["micro_f_beta"] != pytest.approx(dec_b[0]["micro_f_beta"])


# --- Guard: score channel of the adoption decision ----------------------------
# The test above pins "the decision comes from the holdout". The one below pins that,
# on that holdout, **champion and candidate are scored on the same score channel**.

from tests.test_tail import FakeDetector, _fake_infer, _fakes, root  # noqa: E402,F401


class _DummyRescorer:
    active = True

    def save(self, path):
        pass


def test_arm_candidate_with_rescorer_is_gated_on_calibrated_channel(
        root, tmp_path, monkeypatch):
    """If an arm-path candidate (lora, crop classifier) refit the rescorer, the holdout
    evaluation the gate sees must also be on the **rescored channel**.

    Breakage is silent. With the champion scored on the rescored channel and the
    candidate on raw scores, the paired Δ becomes "candidate − rescorer layer" rather
    than "the candidate's increment" — the arm axis competes handicapped by the
    rescorer's gain (a compound Δ). Conversely, if such a candidate is adopted, raw-score
    records land in the ledger and next round's rescored candidates get a free gain
    equal to the channel gap.
    """
    from agentic_detector_config.config import config_from_dict
    from agentic_detector_config.infer import gateway
    from agentic_detector_config.loop import rescore, runner as runner_mod, tail_round
    from agentic_detector_config.loop.runner import run_rounds

    server = FakeDetector()
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root, server))

    def fake_train(cfg, *, out_dir, **kw):
        p = __import__("pathlib").Path(out_dir) / "tuned_tail.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
        return p

    monkeypatch.setattr(tail_round, "train_tail", fake_train)

    events: list[str] = []
    real_eval = runner_mod.evaluate_on_holdout

    def fake_fit(cfg, client, ds, setting, materials, mat_dir):
        events.append("fit")                      # rescorer refit on top of the arm (non-identity)
        return _DummyRescorer(), {"alphas": {"1": 0.5}}, {}

    def fake_screen(cfg, ds, setting, rescorer, dev_preds):
        return setting                            # only the channel is under test — setting unchanged

    def spy_rescored(cfg, client, ds, setting, materials, rescorer):
        events.append("rescored")
        return real_eval(cfg, client, ds.holdout, ds.ontology, setting, materials)

    def spy_raw(cfg, client, split, ontology, setting, materials):
        events.append("raw")
        return real_eval(cfg, client, split, ontology, setting, materials)

    monkeypatch.setattr(rescore, "fit_for_setting", fake_fit)
    monkeypatch.setattr(rescore, "rescored_screen", fake_screen)
    monkeypatch.setattr(rescore, "rescored_evaluate", spy_rescored)
    monkeypatch.setattr(runner_mod, "evaluate_on_holdout", spy_raw)

    cfg = config_from_dict({
        "dataset": str(root),
        "lora": {"lambdas": [0.0, 1.0], "margin": 0.01, "min_images": 4},
    })
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text("- id: adapt\n  axes: {lora: true, rescorer: true}\n"
                         "  rationale: adapt+rescore\n")
    run_rounds(root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
               cfg=cfg, candidates_file=cand_file,
               detector_hooks={"weights_ctl": server.ctl, "repo_root": tmp_path,
                          "base_weights": tmp_path / "yoloe.pt"},
               **_fakes())

    assert "fit" in events, "no rescorer re-fit on top of the arm — scenario invalid"
    after = events[events.index("fit") + 1:]
    assert after and after[0] == "rescored", (
        "arm candidate was evaluated on the raw-score channel with the rescorer on — it reaches the "
        f"gate on a different channel than the champion (observed: {events})")


# --- guard: baseline state of detector-state candidates ------------------------
# lora (tail arms) and the crop classifier both swap **detector state** (weights,
# artifact). When both arrive in one round, the winning lora arm can still be
# loaded while the classifier is trained, has its τ estimated and is evaluated on
# holdout. If that classifier wins, the champion is served without the lora arm
# (on the current weights), so the recorded τ and holdout F do not match the
# serving state.

class _FakeDetectorState:
    """Fake that records weights and adapt_root together."""

    def __init__(self):
        self.weights = None
        self.adapt = None

    def ctl(self, weights):
        self.weights = weights

    def adapt_ctl(self, root):
        self.adapt = root


def _detector_state_round(root, tmp_path, monkeypatch, candidates: str, fixed_by):
    """One round with the given candidates. The base detector misclassifies class 2
    as class 1; `fixed_by(state)` decides whether the loaded state corrects it."""
    from pathlib import Path

    from agentic_detector_config.config import config_from_dict
    from agentic_detector_config.core.coco import Split
    from agentic_detector_config.infer import gateway
    from agentic_detector_config.loop import adapt_round, tail_round
    from agentic_detector_config.loop.runner import run_rounds

    state = _FakeDetectorState()
    splits = {n: Split.load(root / n) for n in ("train", "valid")}
    by_path = {str(s.image_path(e)): e for s in splits.values() for e in s.images}

    def fake_infer(ctx, client, config, queries, ontology, image_paths,
                   alias_to_name=None, **kw):
        fixed = fixed_by(state)
        return [[{"box": list(a.bbox_xyxy), "score": 0.9,
                  "category_id": a.category_id if fixed or a.category_id == 1 else 1,
                  "sources": ["full"]}
                 for a in by_path[str(p)].annotations] for p in image_paths]

    def fake_train_tail(cfg, *, out_dir, **kw):
        p = Path(out_dir) / "tuned_tail.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
        return p

    def fake_train_cls(repo_root, *, manifest, ontology, out_dir, device, log_path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "classifier.pt").write_bytes(b"fake")
        return out_dir

    weights_at_classifier: list = []
    real_cls = adapt_round.run_classifier_candidate

    def spy_cls(*a, **kw):
        weights_at_classifier.append(state.weights)
        return real_cls(*a, **kw)

    monkeypatch.setattr(gateway, "infer_config", fake_infer)
    monkeypatch.setattr(tail_round, "train_tail", fake_train_tail)
    monkeypatch.setattr(adapt_round, "train_crop_classifier", fake_train_cls)
    monkeypatch.setattr(adapt_round, "run_classifier_candidate", spy_cls)

    cfg = config_from_dict({
        "dataset": str(root),
        "lora": {"lambdas": [0.0, 1.0], "margin": 0.01, "min_images": 4},
        "crop_classifier": {"min_samples": 4, "min_background": 0, "margin": 0.01},
    })
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text(candidates)
    result = run_rounds(
        root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
        cfg=cfg, candidates_file=cand_file,
        detector_hooks={"weights_ctl": state.ctl, "adapt_ctl": state.adapt_ctl,
                        "repo_root": tmp_path, "base_weights": tmp_path / "yoloe.pt"},
        **_fakes())
    return state, result, weights_at_classifier


def _assert_state_is_champion(state, champion):
    from pathlib import Path
    norm = lambda p: None if p is None else str(Path(p))  # noqa: E731
    assert norm(state.weights) == norm(champion.get("weights")), (
        f"detector weights {state.weights} != champion {champion.get('weights')}")
    assert norm(state.adapt) == norm(champion.get("adapt_root")), (
        f"detector artifact {state.adapt} != champion {champion.get('adapt_root')}")


_LORA_AND_CLS = ("- id: adapt\n  axes: {lora: true}\n  rationale: adapt\n"
                 "- id: fixcls\n  axes: {crop_classifier: true}\n  rationale: fix\n")


def _either_fixes(s):
    return (s.weights is not None and "arm_lam" in str(s.weights)) or s.adapt is not None


def test_classifier_candidate_runs_on_champion_weights(root, tmp_path, monkeypatch):
    """A classifier candidate must be trained and evaluated on the **champion's
    weights** — the winning arm of an earlier lora candidate must not stay loaded."""
    _, result, seen = _detector_state_round(root, tmp_path, monkeypatch,
                                            _LORA_AND_CLS, _either_fixes)
    assert "human_adapt" in result.decisions[1]["evaluated"], \
        "lora candidate did not pass its arm gate — scenario invalid"
    assert seen, "classifier candidate did not run — scenario invalid"
    assert all(w is None for w in seen), (
        "classifier candidate ran on a lora arm — if adopted, its τ and holdout F "
        f"do not match the serving state (champion weights) (observed: {seen})")


def test_detector_ends_round_in_champion_state(root, tmp_path, monkeypatch):
    """At the end of the round the detector must hold the **final champion's**
    weights and artifact — the next round's diagnosis runs on top of it."""
    state, result, _ = _detector_state_round(root, tmp_path, monkeypatch,
                                             _LORA_AND_CLS, _either_fixes)
    _assert_state_is_champion(state, result.champion)


@pytest.mark.parametrize("axis", ["crop_classifier", "lora"])
def test_detector_keeps_winner_state_when_later_candidate_restores(
        axis, root, tmp_path, monkeypatch):
    """Two detector-state candidates on the same axis where **the first one wins**:
    the later one fails its arm gate and restores the current state itself, yet
    the detector must end in the winner's state."""
    def fixed_by(s):
        loaded = s.adapt if axis == "crop_classifier" else s.weights
        return "human_good" in str(loaded)

    cands = (f"- id: good\n  axes: {{{axis}: true}}\n  rationale: fix\n"
             f"- id: bad\n  axes: {{{axis}: true, alias: vlm}}\n"
             f"  rationale: no effect\n")
    state, result, _ = _detector_state_round(root, tmp_path, monkeypatch, cands, fixed_by)
    dec = result.decisions[1]
    assert dec["id"] == "human_good", f"scenario invalid — good did not win: {dec}"
    assert "human_bad" in dec["skipped"], f"scenario invalid — bad was not rejected: {dec}"
    _assert_state_is_champion(state, result.champion)
