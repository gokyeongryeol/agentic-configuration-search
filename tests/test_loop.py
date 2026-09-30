"""loop/runner — e2e dry-run (all stubs): acceptance, stopping, resume, human candidate injection.

Stub scenario: the text pass finds only class 1 (class 2 fully missed), while the semantic
pass (exemplar candidates) finds everything -> the baseline (text-only) diagnosis reports missed,
the deterministic planner proposes exemplar=base, and the gate accepts it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from agentic_detector_config.core.coco import Split
from agentic_detector_config.infer import gateway
from agentic_detector_config.loop.runner import run_rounds
from tests.conftest import make_split


@pytest.fixture
def big_root(tmp_path: Path) -> Path:
    """train 12 + holdout (valid) 8 (class 1: 4, class 2: 4) — with the default n=2 fixture
    the bootstrap CI lower bound sticks to 0 and no gate scenario can be built."""
    ds = tmp_path / "big"
    # Mix GT sizes — with uniform sizes every FN is classified missed_small (bottom quartile =
    # its own area) and the planner only proposes tiling (see the caveat in test_diagnose)
    size = lambda i: 20 + (i % 3) * 10  # noqa: E731
    make_split(ds / "train", [
        {"name": f"tr{i:02d}.png", "w": 100, "h": 80,
         "anns": [(1 if i % 2 else 2, 10, 10, size(i), size(i))]}
        for i in range(12)])
    make_split(ds / "valid", [        # the loop's holdout
        {"name": f"t{i}.png", "w": 100, "h": 80,
         "anns": [(1 if i < 4 else 2, 10, 10, size(i), size(i))]}
        for i in range(8)])
    make_split(ds / "test", [         # report-only — the loop never reads it
        {"name": "v.png", "w": 100, "h": 80, "anns": [(1, 10, 10, 20, 20)]}])
    return ds


def _gt_lookup(root):
    splits = {name: Split.load(root / name) for name in ("train", "valid")}
    by_path = {}
    for s in splits.values():
        for e in s.images:
            by_path[str(s.image_path(e))] = e
    return by_path


def _fake_infer(by_path):
    def fake(ctx, client, config, queries, ontology, image_paths,
             alias_to_name=None, **kw):
        semantic = bool(queries) and any("prompt_image" in q for q in queries)
        out = []
        for p in image_paths:
            entry = by_path[str(p)]
            dets = [{"box": list(a.bbox_xyxy), "score": 0.9,
                     "category_id": a.category_id, "sources": ["full"]}
                    for a in entry.annotations
                    if semantic or a.category_id == 1]   # text misses class 2
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


def test_loop_adopts_then_stops(big_root, tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    out = tmp_path / "run"
    result = run_rounds(big_root, out_dir=out, planner="deterministic",
                        rounds=4, **_fakes())

    # round 0: baseline (text-only) — class 2 fully missed (TP 4, FN 4 -> micro 2/3)
    assert result.decisions[0]["kind"] == "baseline"
    assert result.decisions[0]["micro_f_beta"] == pytest.approx(2 / 3)
    # round 1: missed diagnosis -> exemplar=base proposed -> gate accepts (micro 1.0)
    r1 = result.decisions[1]
    assert r1["adopted"] is True
    assert result.champion["axes"]["exemplar"] == "base"
    assert result.champion["eval"]["micro_f_beta"] == pytest.approx(1.0)
    # afterwards: errors are gone, so it stops at some round (planner_stop or saturation)
    assert result.stopped in ("planner_stop", "saturation")

    # artifacts — round directories and the ledger
    assert (out / "history.jsonl").exists()
    assert (out / "rounds" / "round_001" / "decision.json").exists()
    assert (out / "rounds" / "round_001" / "diagnosis_train.json").exists()
    assert (out / "rounds" / "round_001" / "diagnosis_test.json").exists()
    d_test = json.loads((out / "rounds" / "round_001" /
                         "diagnosis_test.json").read_text())
    assert d_test["anonymized"] is True and "fn" not in d_test
    assert (out / "champion.json").exists()


def test_loop_resume_continues_from_history(big_root, tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    out = tmp_path / "run"
    first = run_rounds(big_root, out_dir=out, planner="deterministic",
                       rounds=1, **_fakes())
    n_first = len(first.decisions)
    assert first.decisions[-1]["adopted"] is True

    resumed = run_rounds(big_root, out_dir=out, planner="deterministic",
                         rounds=3, resume=True, **_fakes())
    # baseline is not rerun — resumes from round 2
    assert [d["round"] for d in resumed.decisions][:n_first] == \
        [d["round"] for d in first.decisions]
    assert resumed.champion["axes"]["exemplar"] == "base"   # champion restored


def test_human_candidates_are_injected(big_root, tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text(
        "- id: my_idea\n  axes: {exemplar: fn_arm}\n  rationale: human-injected\n")
    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
                     rounds=1, candidates_file=cand_file, **_fakes())
    screened = out.decisions[1]["screened"]
    assert any(cid.startswith("human_") for cid in screened)


def test_rescorer_without_embeddings_skips_as_identity(big_root, tmp_path,
                                                        monkeypatch):
    """If the stub provides no embedding the discriminator has 0 rows — recorded as identity only."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text(
        "- id: try_rescorer\n  axes: {rescorer: true}\n  rationale: try rescoring\n")
    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
                     rounds=1, candidates_file=cand_file, **_fakes())
    assert "human_try_rescorer" in out.decisions[1]["skipped"]


def test_lora_without_deps_skips_and_records(big_root, tmp_path, monkeypatch):
    """In a run without detector_hooks, lora candidates are skipped — recorded in tried to prevent re-proposal."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    cand_file = tmp_path / "candidates.yaml"
    cand_file.write_text(
        "- id: try_lora\n  axes: {lora: true}\n  rationale: try adaptation\n")
    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
                     rounds=1, candidates_file=cand_file, **_fakes())
    assert "human_try_lora" in out.decisions[1]["skipped"]


def test_quantity_axes_override_mine_quota(big_root, tmp_path, monkeypatch):
    """The exemplar_k quantity axis overrides the MineCfg quota — None means the config default."""
    import numpy as np

    from agentic_detector_config.config import config_from_dict
    from agentic_detector_config.core.data import load_dataset
    from agentic_detector_config.loop.materialize import materialize

    cfg = config_from_dict({"dataset": str(big_root)})
    ds = load_dataset(big_root, seed=0, fit_train_ratio=0.8)
    enc = lambda cid, paths: np.array(  # noqa: E731
        [[float(i), 0.0] for i in range(len(paths))])

    m1 = materialize(cfg, ds, {"exemplar": "base", "exemplar_k": 2},
                     tmp_path / "k2", encode_fn=enc, seed=0)
    assert all(len(bs) == 2 for bs in m1.exemplars.values())
    m2 = materialize(cfg, ds, {"exemplar": "base"},          # None -> default 5
                     tmp_path / "kd", encode_fn=enc, seed=0)
    assert all(len(bs) == cfg.mine.exemplars_per_class
               for bs in m2.exemplars.values())


def test_runner_filters_retried_combos_from_any_planner(big_root, tmp_path,
                                                        monkeypatch):
    """Shared combo filter in runner — even if the LLM re-proposes the same combo (outside
    its prompt window, or ignoring it), it is not re-screened."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))

    def stubborn_llm(messages, **kw):   # LLM that insists on the same combo every round
        return {"candidates": [{"id": "again", "axes": {"alias": "vlm"},
                                "rationale": "repeat the same proposal"}], "stop": False}

    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
                     llm_chat=stubborn_llm, rounds=3, **_fakes())
    r1 = out.decisions[1]
    assert "again" in r1["screened"]                    # first attempt is screened
    for d in out.decisions[2:]:
        assert "again" not in d.get("screened", []), \
            "an already-tried combination was re-screened"


def test_dataset_notes_reach_llm_prompt(big_root, tmp_path, monkeypatch):
    """Wiring: config.dataset_notes -> runner -> LLMPlanner prompt."""
    from agentic_detector_config.config import config_from_dict

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    # Non-ASCII (Korean) notes on purpose: must reach the prompt unescaped.
    cfg = config_from_dict({"dataset": str(big_root), "rounds": 1,
                            "planner": {"kind": "llm"},
                            "dataset_notes": "축구 중계 화면, 선수는 소형"})
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    run_rounds(big_root, out_dir=tmp_path / "run", cfg=cfg,
               llm_chat=chat, rounds=1, **_fakes())
    assert "축구 중계 화면, 선수는 소형" in seen["prompt"]


def test_dedup_dropped_recorded_as_combo_tried(big_root, tmp_path, monkeypatch):
    """Re-proposals dropped by the combo filter must not be silent — they stay in history as
    skipped:"combo_tried" and are fed back into the next round's prompt."""
    import json as _json

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))

    def stubborn_llm(messages, **kw):
        return {"candidates": [{"id": "again", "axes": {"alias": "vlm"},
                                "rationale": "repeat the same proposal"}], "stop": False}

    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
                     llm_chat=stubborn_llm, rounds=2, **_fakes())
    assert "again" in out.decisions[2].get("skipped", [])
    hist = [_json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    r2 = {c["id"]: c for c in hist[2]["candidates"]}
    assert r2["again"]["skipped"] == "combo_tried"


def test_stagnant_counts_adopted_below_epsilon():
    """Saturation rule: stagnant if nothing was accepted, or if accepted with Δ CI lower bound ≤ ε —
    with the default ε=0.0 an acceptance (lo > margin=0.0) never counts as stagnation (same as
    before); only when ε > margin does a marginal acceptance count as stagnation."""
    from agentic_detector_config.loop.runner import _stagnant

    assert _stagnant({"adopted": False}, 0.0) is True
    win = lambda lo: {"adopted": True, "id": "w",              # noqa: E731
                      "evaluated": {"w": {"gate": {"lo": lo}}}}
    assert _stagnant(win(0.02), 0.0) is False    # clear acceptance
    assert _stagnant(win(0.005), 0.01) is True   # gain below ε — stagnant
    assert _stagnant(win(0.02), 0.01) is False
    # a baseline-style acceptance (no evaluated) is not stagnation
    assert _stagnant({"adopted": True, "id": "baseline"}, 0.01) is False


def test_crop_notes_reach_planner_prompt(big_root, tmp_path, monkeypatch):
    """Runner wiring: VLM descriptions are merged into diagnosis["train"]["crop_notes"] and
    only appear in the planner prompt — diagnostic signal only."""
    from agentic_detector_config.loop import runner as runner_mod

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    # Non-ASCII (Korean) note on purpose: the prompt JSON must use ensure_ascii=False.
    monkeypatch.setattr(runner_mod.describe, "crop_notes",
                        lambda split, diag, chat, out: [
                            {"crop": 0, "kind": "background", "category_id": 1,
                             "image": "x.png", "box": [1, 2, 3, 4],
                             "note": "반사광으로 보인다"}])
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
               llm_chat=chat, rounds=1, **_fakes())
    assert "crop_notes" in seen["prompt"] and "반사광" in seen["prompt"]


def test_resume_honors_saturation_and_writes_summary(big_root, tmp_path,
                                                     monkeypatch):
    """resume restores the saturation counter from the history tail — an already saturated
    run does not run more rounds on resume (regression guard against restarts resetting the
    counter to 0 and burning extra GPU rounds on a saturated run). For analysis, history
    round timestamps (ts, dur_s) and run_summary.json are also written."""
    import json as _json

    from agentic_detector_config.config import config_from_dict

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    cfg = config_from_dict({"dataset": str(big_root), "rounds": 2,
                            "planner": {"kind": "llm"},
                            "saturation": {"rounds": 2}})   # threshold pinned for the test
    proposals = [{"candidates": [{"id": "c1", "axes": {"alias": "vlm"},
                                  "rationale": "x"}], "stop": False},
                 {"candidates": [{"id": "c2", "axes": {"text_negatives": "fp_crop"},
                                  "rationale": "x"}], "stop": False}]
    calls = []

    def chat(messages, **kw):
        calls.append(1)
        return proposals[min(len(calls) - 1, 1)]

    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", cfg=cfg,
                     llm_chat=chat, rounds=2, **_fakes())
    assert out.stopped == "saturation"          # perfect-detection stub -> Δ=0 -> 2 consecutive rejections
    hist = [_json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert all("ts" in e and "dur_s" in e for e in hist)
    summary = _json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["stopped"] == "saturation"
    assert summary["champion"]["id"] == "baseline"
    n_calls = len(calls)

    out2 = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", cfg=cfg,
                      llm_chat=chat, rounds=6, resume=True, **_fakes())
    assert out2.stopped == "saturation"
    assert len(out2.decisions) == len(out.decisions)   # 0 rounds added
    assert len(calls) == n_calls                       # 0 extra LLM calls


def test_server_state_routing_is_increment_based():
    """lora/crop_classifier routing is based on axes **newly turned on** relative to the champion —
    (1) turning both on at once is rejected as compound (unattributable), (2) a single increment of
    the other axis on top of a champion that already holds one is allowed via that arm (sequential
    composition), (3) pure inheritance uses normal wiring (no needless rerun of the LoRA grid)."""
    from agentic_detector_config.loop.runner import _route_detector_state

    both = {"lora": True, "crop_classifier": True}
    assert _route_detector_state(both, {}) == "compound"
    assert _route_detector_state(both, {"lora": True}) == "classifier"
    assert _route_detector_state(both, {"crop_classifier": True}) == "lora"
    assert _route_detector_state({"lora": True, "alias": "vlm"},
                               {"lora": True}) == "wired"
    assert _route_detector_state({"crop_classifier": True, "alias": "vlm"},
                               {"crop_classifier": True}) == "wired"
    assert _route_detector_state({"lora": True}, {}) == "lora"
    assert _route_detector_state({"crop_classifier": True}, {}) == "classifier"
    assert _route_detector_state({"alias": "vlm"}, {}) == "wired"


def test_winner_axes_drop_stale_rescorer_label():
    """Champion label consistency: if the rescorer was not applied when evaluating the winner (the
    winner has no rescorer_path), axes.rescorer=True is a false label — it must be corrected to False
    so a rescorer refit can be re-proposed as a new increment in later rounds (guards against the
    calibration layer silently dropping when a server-state axis is accepted on a rescorer champion)."""
    from agentic_detector_config.loop.runner import _winner_axes

    ax, dropped = _winner_axes({"axes": {"rescorer": True, "lora": True},
                                "rescorer_path": None})
    assert ax["rescorer"] is False and ax["lora"] is True and dropped
    ax, dropped = _winner_axes({"axes": {"rescorer": True},
                                "rescorer_path": "settings/c/rescorer.json"})
    assert ax["rescorer"] is True and not dropped
    ax, dropped = _winner_axes({"axes": {"lora": True, "rescorer": False}})
    assert ax["rescorer"] is False and not dropped


def test_measured_axes_follow_the_refit_outcome():
    """axes must be the **actually measured state**. Upstream arms also refit the rescorer and
    measure with it on, so the label stays True; it is corrected to False only when the refit
    ends as identity — history, dedup and champion treat these axes as ground truth."""
    from agentic_detector_config.loop.runner import _measured_axes

    ax = {"lora": True, "rescorer": True, "crop_classifier": True, "alias": "vlm"}
    assert _measured_axes(ax, rescorer=True) == ax          # refit succeeded = unchanged
    off = _measured_axes(ax, rescorer=False)
    assert off["rescorer"] is False
    assert off["crop_classifier"] is True and off["alias"] == "vlm"  # rest unchanged


def test_empty_material_candidate_is_skipped():
    """If an enabled axis has empty material, the candidate equals the baseline — it is rejected
    before spending screen/holdout budget, and that fact is fed back to the planner via history."""
    from agentic_detector_config.fitting.materials import Materials
    from agentic_detector_config.loop.runner import _empty_material

    empty = Materials()
    assert _empty_material({"alias": "vlm"}, empty, {}) == "alias"
    assert _empty_material({"text_negatives": "fp_crop"}, empty, {}) == "text_negatives"
    assert _empty_material({"visual_negatives": True}, empty, {}) == "visual_negatives"
    assert _empty_material({"exemplar": "base"}, empty, {}) == "exemplar"
    # manual aliases are always applied regardless of axes.alias — empty generation means no increment
    assert _empty_material({"alias": "vlm"}, Materials(aliases={"a": "b"}),
                           {"a": "b"}) == "alias"
    filled = Materials(aliases={"cat": "furry pet"},
                       text_negatives={"cat": ["dog"]})
    assert _empty_material({"alias": "vlm"}, filled, {}) is None
    assert _empty_material({"text_negatives": "gt_crop"}, filled, {}) is None
    assert _empty_material({"rescorer": True, "tiling": True}, empty, {}) is None


def test_missing_llm_hooks_skip_dependent_axes():
    """Without an LLM hook, axes that depend on it are filtered out up front as unwireable.

    alias needs refine_alias_fn and text_negatives needs refine_negatives_fn (safety-rule
    check) to build material. Without them materialize raises in _require, but by then mining
    has already started and round budget is spent — same discipline as filtering out LoRA
    candidates up front when detector_hooks is not injected."""
    from agentic_detector_config.loop.runner import _missing_hook

    cap = {"caption_fn": object(), "vlm_caption_fn": object()}   # no LLM refine
    full = {**cap, "refine_alias_fn": object(), "refine_negatives_fn": object()}
    # captioners present, no LLM refine
    assert _missing_hook({"alias": "vlm"}, cap) == "refine_alias_fn"
    assert _missing_hook({"text_negatives": "gt_crop"}, cap) == "refine_negatives_fn"
    # with no hooks at all, it fails on the first required hook — for alias, the VLM captioner
    assert _missing_hook({"alias": "vlm"}, {}) == "vlm_caption_fn"
    assert _missing_hook({"text_negatives": "gt_crop"}, {}) == "caption_fn"
    assert _missing_hook({"alias": "vlm"}, full) is None
    # axes needing no hook always pass
    assert _missing_hook({"rescorer": True, "tiling": True}, {}) is None


def test_invalid_axes_dropped_recorded_in_history(big_root, tmp_path, monkeypatch):
    """Candidates failing axis validation must not be dropped silently either — they stay as
    skipped:"invalid_axes" and, like combo_tried, are fed back into the next round's prompt.

    Without feedback the planner cannot tell "evaluated and rejected" from "discarded outright"
    and keeps re-proposing the same candidate, wasting rounds."""
    import json as _json

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))

    def llm(messages, **kw):
        return {"candidates": [
            {"id": "bad_fn_k", "axes": {"rescorer": True, "fn_k": 2},
             "rationale": "fn_k is exemplar=fn_arm only — caught by validation"},
            {"id": "ok_tiling", "axes": {"tiling": True}, "rationale": "valid"}],
            "stop": False}

    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
                     llm_chat=llm, rounds=1, **_fakes())

    assert "bad_fn_k" in out.decisions[1].get("skipped", [])
    hist = [_json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rec = {c["id"]: c for c in hist[1]["candidates"]}
    assert rec["bad_fn_k"]["skipped"] == "invalid_axes"
    assert "fn_k" in rec["bad_fn_k"].get("note", "")
    assert "ok_tiling" in rec and rec["ok_tiling"].get("skipped") is None


def test_invalid_axes_note_reaches_next_prompt(big_root, tmp_path, monkeypatch):
    """The drop reason must reach the next round's prompt to break the repetition."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    prompts: list = []

    def llm(messages, **kw):
        prompts.append(messages[-1]["content"])
        return {"candidates": [
            {"id": "bad_fn_k", "axes": {"rescorer": True, "fn_k": 2},
             "rationale": "invalid"},
            {"id": "ok_tiling", "axes": {"tiling": True}, "rationale": "valid"}],
            "stop": False}

    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
               llm_chat=llm, rounds=2, **_fakes())

    # llm_chat is also used for crop descriptions, not just the planner — pick out planner calls only
    planner_prompts = [p for p in prompts if "Round history" in p]
    assert len(planner_prompts) >= 2
    assert "invalid_axes" in planner_prompts[1]
    assert "fn_k" in planner_prompts[1], "the reason (note) must be carried too, or the repetition does not stop"
