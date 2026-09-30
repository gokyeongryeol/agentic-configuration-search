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
