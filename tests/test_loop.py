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
    # Mix GT sizes (r≈0.22–0.45 relative to the image — no missed_small at the
    # default cut 0.04).
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
    run does not run more rounds on resume. For analysis, history round timestamps
    (ts, dur_s) and run_summary.json are also written."""
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
    so a rescorer refit can be re-proposed as a new increment in later rounds."""
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
    assert _empty_material({"rescorer": True, "tiling": 2}, empty, {}) is None


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
    assert _missing_hook({"rescorer": True, "tiling": 2}, {}) is None


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
            {"id": "ok_tiling", "axes": {"tiling": 2}, "rationale": "valid"}],
            "stop": False}

    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
                     llm_chat=llm, rounds=1, **_fakes())

    assert "bad_fn_k" in out.decisions[1].get("skipped", [])
    hist = [_json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rec = {c["id"]: c for c in hist[1]["candidates"]}
    assert rec["bad_fn_k"]["skipped"] == "invalid_axes"
    assert "fn_k" in rec["bad_fn_k"].get("note", "")
    # a valid candidate passes sanitization — it may still be rejected later (tiling_unused etc.)
    assert "ok_tiling" in rec and rec["ok_tiling"].get("skipped") != "invalid_axes"


def test_invalid_axes_note_reaches_next_prompt(big_root, tmp_path, monkeypatch):
    """The drop reason must reach the next round's prompt to break the repetition."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    prompts: list = []

    def llm(messages, **kw):
        prompts.append(messages[-1]["content"])
        return {"candidates": [
            {"id": "bad_fn_k", "axes": {"rescorer": True, "fn_k": 2},
             "rationale": "invalid"},
            {"id": "ok_tiling", "axes": {"tiling": 2}, "rationale": "valid"}],
            "stop": False}

    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
               llm_chat=llm, rounds=2, **_fakes())

    # llm_chat is also used for crop descriptions, not just the planner — pick out planner calls only
    planner_prompts = [p for p in prompts if "Round history" in p]
    assert len(planner_prompts) >= 2
    assert "invalid_axes" in planner_prompts[1]
    assert "fn_k" in planner_prompts[1], "the reason (note) must be carried too, or the repetition does not stop"


def test_single_class_gt_crop_never_reaches_materialize(tmp_path, monkeypatch):
    """Single-class dataset: gt_crop from the LLM or a human is recorded as
    invalid_axes at sanitization, and is absent from the prompt's value list."""
    import json as _json

    from tests import conftest

    monkeypatch.setattr(conftest, "CATS", [
        {"id": 0, "name": "stuff", "supercategory": "none"},
        {"id": 1, "name": "cat a", "supercategory": "stuff"}])
    root = tmp_path / "one"
    size = lambda i: 20 + (i % 3) * 10  # noqa: E731
    make_split(root / "train", [
        {"name": f"tr{i:02d}.png", "w": 100, "h": 80,
         "anns": [(1, 10, 10, size(i), size(i))]} for i in range(12)])
    make_split(root / "valid", [
        {"name": f"t{i}.png", "w": 100, "h": 80,
         "anns": [(1, 10, 10, size(i), size(i))]} for i in range(8)])
    make_split(root / "test", [
        {"name": "v.png", "w": 100, "h": 80, "anns": [(1, 10, 10, 20, 20)]}])
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(root)))
    human = tmp_path / "candidates.yaml"
    human.write_text("- {id: hg, axes: {text_negatives: gt_crop}}\n")
    prompts: list = []

    def llm(messages, **kw):
        prompts.append(messages[-1]["content"])
        return {"candidates": [
            {"id": "g", "axes": {"text_negatives": "gt_crop"}, "rationale": "x"},
            {"id": "t", "axes": {"tiling": 2}, "rationale": "y"}], "stop": False}

    run_rounds(root, out_dir=tmp_path / "run", planner="llm", llm_chat=llm,
               rounds=1, candidates_file=human, **_fakes())

    planner_prompts = [p for p in prompts if "Round history" in p]
    assert planner_prompts and "gt_crop" not in planner_prompts[0]
    hist = [_json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rec = {c["id"]: c for c in hist[1]["candidates"]}
    assert rec["g"]["skipped"] == "invalid_axes"
    assert rec["human_hg"]["skipped"] == "invalid_axes"
    assert rec["t"].get("skipped") != "invalid_axes"     # passes sanitization


def test_tiling_variants_follow_axis_grid():
    """screen compares only the grid the candidate chose against tiling off — it no
    longer runs every grid from config (1 + (1+g²) passes per candidate)."""
    from agentic_detector_config.config import config_from_dict
    from agentic_detector_config.loop.runner import _tiling_variants

    s = config_from_dict({"dataset": "x"}).search
    assert _tiling_variants(s, {"tiling": False}) == ()
    assert _tiling_variants(s, {"tiling": 3}) == (
        {"grid": 3, "overlap": 0.5, "merge": "nmm"},)


def test_history_records_whether_tiling_was_used(big_root, tmp_path, monkeypatch):
    """A tiling candidate records whether screen actually assigned a tile config
    (tiling_used) — the signal the planner reads to raise or drop the zoom."""
    import json as _json

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))

    def llm(messages, **kw):
        return {"candidates": [{"id": "t3", "axes": {"tiling": 3},
                                "rationale": "small objects"}], "stop": False}

    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
               llm_chat=llm, rounds=1, **_fakes())
    hist = [_json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rec = {c["id"]: c for c in hist[1]["candidates"]}
    assert isinstance(rec["t3"]["tiling_used"], bool)


def test_diagnosis_uses_configured_small_object_cut(big_root, tmp_path, monkeypatch):
    """The runner passes the config cut to the diagnosis. Fixture GTs are r≈0.22–0.45
    relative to the image: missed at the default cut (0.04), missed_small at a large cut."""
    import json as _json

    from agentic_detector_config.config import config_from_dict

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))

    def kinds(out, cut):
        cfg = config_from_dict({"dataset": str(big_root),
                                "diagnose": {"small_object_ratio": cut}})
        run_rounds(big_root, out_dir=out, planner="deterministic", rounds=1,
                   cfg=cfg, **_fakes())
        d = _json.loads((out / "rounds" / "round_001" / "diagnosis_train.json")
                        .read_text())
        return {r["kind"] for r in d["fn"]}

    assert "missed_small" not in kinds(tmp_path / "a", 0.04)
    assert "missed_small" in kinds(tmp_path / "b", 0.9)


def test_untrainable_lora_never_reaches_tail_round(big_root, tmp_path, monkeypatch):
    """If fit_train has fewer than lora.min_images images, a lora candidate is recorded
    as invalid_axes at sanitization and never reaches the tail round (training wiring)."""
    import json as _json

    from agentic_detector_config.config import config_from_dict
    from agentic_detector_config.loop import tail_round

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    monkeypatch.setattr(tail_round, "run_tail_candidate", lambda *a, **k: (
        _ for _ in ()).throw(AssertionError("reached the tail round")))

    def llm(messages, **kw):
        return {"candidates": [{"id": "l", "axes": {"lora": True}, "rationale": "x"},
                               {"id": "t", "axes": {"tiling": 2}, "rationale": "y"}],
                "stop": False}

    cfg = config_from_dict({"dataset": str(big_root), "planner": {"kind": "llm"},
                            "lora": {"min_images": 50}})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=llm,
               rounds=1, cfg=cfg, **_fakes())
    hist = [_json.loads(l) for l in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    rec = {c["id"]: c for c in hist[1]["candidates"]}
    assert rec["l"]["skipped"] == "invalid_axes" and "min_images" in rec["l"]["note"]


# --- text-only tiling: measure whether tiles are actually used, align the label ---

def _tile_helps_infer(by_path):
    """Like _fake_infer, but the text pass finds class 2 only on a tile config."""
    def fake(ctx, client, config, queries, ontology, image_paths,
             alias_to_name=None, **kw):
        semantic = bool(queries) and any("prompt_image" in q for q in queries)
        tiled = config.get("tiling") is not None
        return [[{"box": list(a.bbox_xyxy), "score": 0.9,
                  "category_id": a.category_id, "sources": ["full"]}
                 for a in by_path[str(p)].annotations
                 if semantic or tiled or a.category_id == 1]
                for p in image_paths]
    return fake


def _llm_tiling(messages, **kw):
    return {"candidates": [{"id": "t2", "axes": {"tiling": 2},
                            "rationale": "small objects"}], "stop": False}


def _hist(run_dir):
    return [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines()]


def test_text_only_tiling_that_does_not_help_is_not_evaluated(big_root, tmp_path,
                                                              monkeypatch):
    """If screen picks an untiled config the candidate equals the champion —
    reject it as tiling_unused without holdout evaluation, but keep it as tried."""
    from agentic_detector_config.planner.axes import validate_axes
    from agentic_detector_config.planner.deterministic import tried_combos

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
                     llm_chat=_llm_tiling, rounds=1, **_fakes())
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["t2"]
    assert rec["skipped"] == "tiling_unused" and rec["tiling_used"] is False
    assert "t2" not in (out.decisions[1].get("evaluated") or {})
    key = tuple(sorted(validate_axes({"tiling": 2}).items()))
    assert key in tried_combos(_hist(tmp_path / "run"))


def test_text_only_tiling_that_helps_keeps_its_label(big_root, tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _tile_helps_infer(_gt_lookup(big_root)))
    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
                     llm_chat=_llm_tiling, rounds=1, **_fakes())
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["t2"]
    assert rec.get("skipped") is None and rec["tiling_used"] is True
    assert rec["axes"]["tiling"] == 2 and "requested_axes" not in rec
    assert "t2" in out.decisions[1]["evaluated"]


def test_untiled_setting_is_relabelled_with_request_kept():
    """A candidate that requested tiling but differs from the champion through
    other axes is evaluated, with its label corrected to the measured state (no
    tiling) and the request kept in requested_axes — so the champion never
    carries tiling it does not run, and the same request stays blocked."""
    from agentic_detector_config.loop.runner import _relabel_untiled, _tiling_record
    from agentic_detector_config.planner.axes import validate_axes
    from agentic_detector_config.planner.deterministic import tried_combos
    from agentic_detector_config.planner.llm import _tried_lines

    setting = {"configs": [{"config_id": 0, "tiling": None},
                           {"config_id": 1, "tiling": {"grid": 2}}],
               "per_class": {1: {"config_id": 0}}}
    entry = {"id": "x", "axes": validate_axes({"tiling": 2, "alias": "vlm"}),
             "setting": setting}
    _relabel_untiled(entry)
    assert entry["axes"]["tiling"] is False and entry["axes"]["alias"] == "vlm"
    assert entry["requested_axes"]["tiling"] == 2
    assert _tiling_record(entry) == {"tiling_used": False}

    hist = [{"round": 1, "champion": {"axes": validate_axes({})},
             "candidates": [{"id": "x", "axes": entry["axes"],
                             "requested_axes": entry["requested_axes"]}]}]
    assert tuple(sorted(entry["requested_axes"].items())) in tried_combos(hist)
    assert any("tiling=2" in l for l in _tried_lines(hist, validate_axes({})))


# --- forwards per image ---------------------------------------------------

def _setting(groups, tilings):
    """groups: {cid: (config_id, modality)}, tilings: {config_id: grid g | None}."""
    return {"configs": [{"config_id": i,
                         "tiling": ({"grid": g, "overlap": 0.5, "merge": "nmm"}
                                    if g else None)} for i, g in tilings.items()],
            "per_class": {cid: {"config_id": c, "modality": m}
                          for cid, (c, m) in groups.items()}}


def test_setting_forwards_counts_passes_and_tiles():
    """Forwards per image = 1 + g² per (config, modality) pass the setting runs.
    A grid has g² tiles regardless of image size, so the setting alone decides it."""
    from agentic_detector_config.loop.runner import setting_forwards

    assert setting_forwards(_setting({1: (0, "text"), 2: (0, "text")}, {0: None})) == 1
    assert setting_forwards(_setting({1: (0, "semantic"), 2: (0, "text")}, {0: None})) == 2
    assert setting_forwards(_setting({1: (1, "text")}, {0: None, 1: 3})) == 10
    assert setting_forwards(_setting({1: (1, "semantic"), 2: (1, "text")},
                                     {0: None, 1: 2})) == 10


def test_forwards_recorded_on_champion_and_candidates(big_root, tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
               rounds=1, **_fakes())
    champ = json.loads((tmp_path / "run" / "champion.json").read_text())
    assert champ["forwards_per_image"] >= 1
    hist = _hist(tmp_path / "run")
    assert all("forwards_per_image" in h["champion"] for h in hist)
    screened = [c for c in hist[1]["candidates"] if not c.get("skipped")]
    assert screened and all(c["forwards"] >= 1 for c in screened)
