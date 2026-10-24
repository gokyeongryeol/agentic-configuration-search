"""Planner prompt structure — budget card, champion, axis ledger, allowed moves,
recent rounds in detail with older rounds summarized, diagnosis digest, and the
response format (expected gain, cost, variant note).

Why: the prompt of a validation run (soccer R7) carried 8 rounds of history
JSON and every tried combination over the whole history (mostly ones turning
off an old champion's axes), but not how many times the same value lost, what
it cost, or how many rounds were left. The planner kept proposing fn_arm
variants.
"""

import json

from agentic_configuration_search.planner.axes import validate_axes
from agentic_configuration_search.planner.llm import LLMPlanner
from tests.test_loop import big_root  # noqa: F401 — fixture reuse


def _row(cid, *, ub=None, delta=None, skipped=None, wall=None, champ=None, **axes):
    row = {"id": cid, "axes": validate_axes({**(champ or {}), **axes})}
    if ub is not None:
        row["fit_dev_cmp"] = {"delta": delta if delta is not None else -0.02,
                              "ub": ub, "n": 22}
    if skipped:
        row["skipped"] = skipped
    if wall is not None:
        row["wall_min"] = {"total": wall}
    return row


def _round(r, champ_round, cands, *, champ=None, adopted=None, evaluated=None,
           f=0.9):
    return {"round": r,
            "champion": {"axes": validate_axes(champ or {}), "round": champ_round,
                         "id": f"c{champ_round}", "micro_f_beta": f,
                         "fit_dev_micro_f_beta": f + 0.02, "forwards_per_image": 5},
            "candidates": cands,
            "decision": {"adopted": adopted is not None, "id": adopted,
                         "evaluated": evaluated or {}}}


def _history():
    base = {"exemplar": "base"}
    win = {**_row("w", exemplar="base", wall=6.0), "adopted": True}
    ev = {"w": {"micro_f_beta": 0.9, "gate": {"delta": 0.2, "lo": 0.15}}}
    ev3 = {"vlm": {"micro_f_beta": 0.89, "gate": {"delta": -0.01, "lo": -0.03}}}
    return [
        _round(0, 0, [], f=0.7),
        _round(1, 1, [win], champ=base, adopted="w", evaluated=ev),
        _round(2, 1, [{**_row("t3", tiling=3, champ=base, skipped="axis_unused",
                              wall=20.0), "unused_axes": ["tiling"],
                       "tiling_used": False},
                      _row("fn", exemplar="fn_arm", fn_k=4, champ=base, ub=0.004,
                           delta=0.003, wall=25.0)], champ=base),
        _round(3, 1, [_row("vlm", alias="vlm", champ=base, ub=-0.004, delta=-0.012,
                           wall=8.0)], champ=base, evaluated=ev3),
    ]


def _prompt(history, diagnosis=None, **kw):
    seen = {}

    def chat(messages, **k):
        seen["m"] = messages
        return {"candidates": [], "stop": True}
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        LLMPlanner(chat=chat, artifact_dir=Path(td), **kw).propose(
            history, diagnosis or {})
    return seen["m"][-1]["content"]


def test_prompt_sections_in_order():
    p = _prompt(_history(), budget={"round": 4, "rounds": 8})
    heads = ["## Budget", "## Current champion", "## Axis ledger",
             "## Allowed moves", "## Round history", "## Diagnosis digest",
             "## Response"]
    pos = [p.index(h) for h in heads]
    assert pos == sorted(pos)


def test_budget_card_shows_rounds_left_counters_and_costs():
    p = _prompt(_history(), budget={"round": 4, "rounds": 8, "stagnant": 2,
                                    "saturation_rounds": 5, "empty": 1,
                                    "empty_rounds": 2})
    card = p.split("## Budget", 1)[1].split("## Current champion", 1)[0]
    assert "Round 4 of 8: 5 rounds left including this one." in card
    assert "without adoption: 2; the run stops at 5" in card
    assert "(a round where a candidate was dropped for losing on fit_dev does not count): 1; the run stops at 2" in card
    assert "- exemplar: median" in card                  # observed cost
    assert "lora: fine-tunes the detector tail" in card   # unobserved axes get a static note
    assert "- tiling=3: median 20 min" in card and "1+g^2" not in card


def test_budget_card_static_tiling_cost_uses_champion_forwards():
    hist = _history()[:2]                                 # before tiling is observed
    card = _prompt(hist).split("## Budget", 1)[1].split("## Current champion", 1)[0]
    assert "tiling=g: every detector pass costs 1+g^2 forwards per image " \
           "(champion: 5)" in card


def test_budget_card_states_unmeasured_costs_without_judging_them():
    """The cost note for an unobserved axis states facts only — a value judgment
    like "the slowest axis" makes the planner put that axis off (in a validation
    run two datasets put LoRA off to R3–R4). No duration estimate either — it
    differs per dataset."""
    hist = _history()[:2]                                 # before lora and classifier are observed
    card = _prompt(hist).split("## Budget", 1)[1].split("## Current champion", 1)[0]
    assert "slowest" not in card and " min" not in card.split("Not measured yet", 1)[1]
    assert ("lora: fine-tunes the detector tail before screening; crop_classifier: "
            "trains a crop classifier before screening") in card


def test_budget_card_skips_unknown_fields_and_disabled_axes():
    p = _prompt(_history(), disabled_axes=("lora",))
    card = p.split("## Budget", 1)[1].split("## Current champion", 1)[0]
    assert "Round" not in card and "lora" not in card


def test_champion_block_states_scores_and_cost():
    p = _prompt(_history())
    block = p.split("## Current champion", 1)[1].split("## Axis ledger", 1)[0]
    assert '"exemplar": "base"' in block
    assert "held-out micro F 0.9000" in block and "fit_dev micro F 0.9200" in block
    assert "adopted in round 1 (c1)" in block and "5 detector passes" in block


def test_ledger_table_lists_values_with_status_and_prescreen():
    p = _prompt(_history(), prescreen={"on": True, "n_images": 20, "of": 80})
    ledger = p.split("## Axis ledger", 1)[1].split("## Allowed moves", 1)[0]
    assert "Rounds after round 1" in ledger
    vlm = next(l for l in ledger.splitlines() if l.startswith("alias=vlm |"))
    assert "clearly-lost (prescreened)" in vlm
    assert "-0.0120 (-0.0040)" in vlm and "-0.0100 (-0.0300)" in vlm
    fn = next(l for l in ledger.splitlines() if l.startswith("exemplar=fn_arm |"))
    assert "near-miss" in fn and "| 25" in fn
    t3 = next(l for l in ledger.splitlines() if l.startswith("tiling=3 |"))
    assert "folded" in t3
    # row order: clearly-lost values first
    assert ledger.index("alias=vlm |") < ledger.index("exemplar=fn_arm |")
    assert "{exemplar=fn_arm, fn_k=4}" in ledger and "{alias=vlm}" in ledger

    allowed = p.split("## Allowed moves", 1)[1].split("## Round history", 1)[0]
    assert "Prescreened values" in allowed and "alias=vlm (round 3 vlm" in allowed
    assert "20 of 80 fit_dev images" in allowed and "lost_value_prescreen" in allowed
    assert "grid >= 3 is folded" in allowed


def test_allowed_moves_when_prescreen_is_off():
    p = _prompt(_history(), prescreen={"on": False})
    allowed = p.split("## Allowed moves", 1)[1].split("## Round history", 1)[0]
    assert "not prescreened on this run" in allowed and "alias=vlm" in allowed


def test_recent_rounds_in_detail_older_rounds_one_line():
    from agentic_configuration_search.planner.llm import MAX_HISTORY

    hist = _history()
    extra = [_round(r, 1, [_row(f"x{r}", text_negatives="background",
                                text_negative_k=None, champ={"exemplar": "base"},
                                skipped="no_material")],
                    champ={"exemplar": "base"}) for r in range(4, 4 + MAX_HISTORY)]
    p = _prompt(hist + extra)
    section = p.split("## Round history", 1)[1].split("## Diagnosis digest", 1)[0]
    detail = json.loads(next(l for l in section.splitlines() if l.startswith("[")))
    assert [r["round"] for r in detail] == [e["round"] for e in extra]
    assert "Earlier rounds, one line each:" in section
    assert "R0: baseline, held-out micro F 0.7000" in section
    assert "R1: adopted w (exemplar=base) -> held-out micro F 0.9000, gate delta " \
           "+0.2000" in section
    assert "R3: no adoption; 1 candidates, 1 evaluated (best gate delta -0.0100)" \
        in section


def test_detail_rows_carry_fit_dev_cmp_and_expected_gain():
    from agentic_configuration_search.planner.llm import _compact_history

    hist = _history()
    hist[3]["candidates"][0]["plan"] = {"expected_gain": 0.02, "cost_min": 8}
    hist[3]["candidates"][0]["lost_values"] = ["alias=vlm"]
    row = _compact_history(hist)[-1]["candidates"][0]
    assert row["fit_dev_cmp"] == {"delta": -0.012, "ub": -0.004}
    assert row["expected_gain"] == 0.02 and row["lost_values"] == ["alias=vlm"]
    assert "skipped" not in row                      # empty values are left out


def test_digest_section_replaces_summary_json():
    diag = {"train": {"n_images": 3, "summary": {"fn": {"missed": 4}},
                      "per_class": {"a": {"gt": 5}}},
            "test": {"n_images": 2, "summary": {"fn": {"missed": 1}}}}
    p = _prompt(_history(), diag)
    digest = p.split("## Diagnosis digest", 1)[1].split("Error detail.", 1)[0]
    assert "- missed: 5 (train 4, held-out 1) -> exemplar, lora, alias" in digest
    detail = p.split("what the errors look like.\n", 1)[1].split("\n\n## Response")[0]
    assert '"summary"' not in detail and '"per_class"' in detail


def test_response_section_asks_for_gain_cost_and_differs():
    p = _prompt(_history(), max_candidates=2)
    resp = p.split("## Response", 1)[1]
    for key in ("expected_gain", "cost_min", "differs", "rationale"):
        assert f'"{key}"' in resp
    assert "Propose up to 2 candidates" in resp


# --- response parsing ---------------------------------------------------------------------

def _propose(tmp_path, response, history=None):
    out = LLMPlanner(chat=lambda m, **k: response, artifact_dir=tmp_path).propose(
        history or _history(), {})
    stored = json.loads((tmp_path / "proposal.json").read_text())
    return out, stored


def test_plan_fields_are_kept(tmp_path):
    resp = {"candidates": [{"id": "a", "axes": {"tiling": 2}, "expected_gain": 0.012,
                            "cost_min": 15, "differs": "", "rationale": "r"},
                           {"id": "b", "axes": {"alias": "vlm", "exemplar": "fn_arm"},
                            "expected_gain": 0.004, "cost_min": 9,
                            "differs": "fn_arm exemplars fix the misses vlm alone "
                                       "could not", "rationale": "r"}],
            "stop": False}
    out, stored = _propose(tmp_path, resp)
    plans = {c["id"]: c.get("plan") for c in out["candidates"]}
    assert plans["a"] == {"expected_gain": 0.012, "cost_min": 15.0}
    assert plans["b"]["differs"].startswith("fn_arm exemplars")
    assert stored["fallback"] is None


def test_missing_or_malformed_plan_fields_do_not_drop_or_fall_back(tmp_path):
    """A missing or malformed field neither drops the candidate nor falls back to
    deterministic."""
    resp = {"candidates": [{"id": "a", "axes": {"tiling": 2}, "rationale": "r"},
                           {"id": "b", "axes": {"alias": "vlm"}, "rationale": "r",
                            "expected_gain": "a lot", "cost_min": -3,
                            "differs": 7}],
            "stop": False}
    out, stored = _propose(tmp_path, resp)
    assert [c["id"] for c in out["candidates"]] == ["a", "b"]
    assert all("plan" not in c for c in out["candidates"])
    assert stored["fallback"] is None and out["planner"] == "llm"


def test_plan_differs_is_bounded():
    from agentic_configuration_search.planner.axes import PLAN_DIFFERS_MAX, sanitize_candidates

    out = sanitize_candidates([{"axes": {"tiling": 2}, "differs": "x" * 900,
                                "expected_gain": float("nan")}], max_candidates=1)
    assert len(out[0]["plan"]["differs"]) == PLAN_DIFFERS_MAX
    assert "expected_gain" not in out[0]["plan"]


def test_schema_orders_plan_fields_before_rationale_and_bounds_them(tmp_path):
    from agentic_configuration_search.planner import llm as L

    seen = {}

    def chat(messages, **kw):
        seen.update(kw)
        return {"candidates": [], "stop": True}
    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(_history(), {})
    item = seen["schema"]["schema"]["properties"]["candidates"]["items"]
    order = list(item["properties"])
    assert order.index("expected_gain") < order.index("rationale")
    assert order.index("differs") < order.index("rationale")
    assert {"expected_gain", "cost_min", "differs"} <= set(item["required"])
    assert item["properties"]["differs"]["maxLength"] <= 300


def test_reask_message_names_lost_value_prescreen():
    from agentic_configuration_search.planner.llm import REASK
    assert "lost_value_prescreen" in REASK


def test_reask_outcome_line_carries_the_prescreen_numbers():
    from agentic_configuration_search.planner.llm import _outcome_lines

    row = {**_row("k2", ub=-0.03, delta=-0.05, text_negatives="gt_crop",
                  text_negative_k=2, skipped="lost_value_prescreen"),
           "note": "prescreen: " + "x" * 200}
    line = _outcome_lines([row], validate_axes({}))[0]
    assert "skipped=lost_value_prescreen" in line
    assert "fit_dev delta -0.0500 (ub -0.0300)" in line


def test_runner_feeds_budget_card_into_the_prompt(big_root, tmp_path,  # noqa: F811
                                                 monkeypatch):
    """The runner feeds rounds left and stop counters (the tail of history) into
    the prompt, and the response's expected gain and cost stay on the candidate
    row as plan."""
    from tests.test_loop import _fake_infer, _fakes, _gt_lookup
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.infer import gateway
    from agentic_configuration_search.loop.runner import run_rounds

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    prompts = []

    def chat(messages, **kw):
        if (kw.get("schema") or {}).get("name") != "proposal":
            return {}
        prompts.append(messages[-1]["content"])
        return {"candidates": [{"id": f"t{len(prompts)}", "axes": {"tiling": 2},
                                "rationale": "x", "expected_gain": 0.01,
                                "cost_min": 3, "differs": ""}], "stop": False}
    cfg = config_from_dict({"dataset": str(big_root),
                            "planner": {"kind": "llm", "max_reasks": 0},
                            "saturation": {"rounds": 4, "empty_rounds": 3}})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=2, cfg=cfg, **_fakes())
    assert "Round 1 of 2: 2 rounds left including this one." in prompts[0]
    assert "without adoption: 0; the run stops at 4" in prompts[0]
    assert "Round 2 of 2: 1 round left including this one." in prompts[1]
    assert "without adoption: 1; the run stops at 4" in prompts[1]
    hist = [json.loads(line) for line in
            (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    row = {c["id"]: c for c in hist[1]["candidates"]}["r1_t1"]
    assert row["plan"] == {"expected_gain": 0.01, "cost_min": 3.0}


# --- untried axes ----------------------------------------------------------------
# Ported from zero-agent: with one candidate per round the planner proposed four
# exemplar_k variants before reaching crop_classifier, an axis that had been
# worth +0.06 in earlier runs. With few candidates the order is the budget.

def _untried(p):
    allowed = p.split("## Allowed moves", 1)[1].split("## Round history", 1)[0]
    return next((ln for ln in allowed.splitlines()
                 if ln.startswith("- Untried axes under the current champion")), None)


def test_untried_axes_line_lists_enabled_axes_not_tried_under_the_champion():
    line = _untried(_prompt(_history()))
    assert line is not None
    names = line.split(":", 1)[1].split(".", 1)[0]
    # axes tried under the current champion (exemplar=base) are left out —
    # tiling (t3), exemplar (fn_arm variant), alias (vlm). Other axes that can
    # still be turned on are listed
    for axis in ("text_negatives", "visual_negatives", "rescorer", "lora",
                 "crop_classifier"):
        assert axis in names, axis
    for axis in ("tiling", "exemplar", "alias", "exemplar_k", "fn_k"):
        assert axis not in names.replace("visual_negatives", "").replace(
            "text_negatives", ""), axis
    assert "prefer one of them over another variant of an already-tried value" in line
    assert "unless the diagnosis gives a concrete reason" in line


def test_untried_axes_skip_disabled_and_blocked_and_champion_on_axes():
    from agentic_configuration_search.planner.axes import DatasetFacts

    hist = _history()
    for e in hist[1:]:                                   # the champion turned rescorer on
        e["champion"]["axes"] = validate_axes({"exemplar": "base", "rescorer": True})
    line = _untried(_prompt(hist, disabled_axes=("crop_classifier",),
                            disabled_values={"text_negatives": ["fp_crop", "gt_crop",
                                                                "background"]},
                            facts=DatasetFacts(n_classes=2, n_fit_train=3,
                                               lora_min_images=20)))
    names = line.split(":", 1)[1].split(".", 1)[0]
    assert "crop_classifier" not in names                # disabled by the operator
    assert "text_negatives" not in names                 # every on-value disabled
    assert "lora" not in names                           # blocked by the data
    assert "rescorer" not in names                       # boolean the champion already turned on
    assert "visual_negatives" in names


def test_untried_axes_marks_axes_never_tried_in_the_run():
    """An axis tried under an older champion is untried under the current one but
    tried in the run — axes never proposed are marked separately."""
    hist = _history()
    hist[1]["candidates"].append(_row("tn", text_negatives="fp_crop", ub=-0.01))
    line = _untried(_prompt(hist))
    names, never = line.split("never tried in this run:", 1)
    assert "text_negatives" in names.split(":", 1)[1]
    never = never.split(")", 1)[0]
    assert "text_negatives" not in never and "crop_classifier" in never


def test_hygiene_drops_do_not_count_as_tried():
    """A row dropped without measuring it (combo dedup and the like) did not try the axis."""
    hist = _history()
    hist[3]["candidates"].append(_row("cc", crop_classifier=True,
                                      champ={"exemplar": "base"},
                                      skipped="combo_tried"))
    names = _untried(_prompt(hist)).split(":", 1)[1].split(".", 1)[0]
    assert "crop_classifier" in names


def test_no_untried_line_when_everything_was_tried():
    hist = _history()
    for i, axes in enumerate([{"text_negatives": "fp_crop"},
                              {"visual_negatives": True}, {"rescorer": True},
                              {"lora": True}, {"crop_classifier": True}]):
        hist[3]["candidates"].append(_row(f"x{i}", champ={"exemplar": "base"}, **axes))
    assert _untried(_prompt(hist)) is None


def test_visual_negatives_is_not_untried_without_an_exemplar_champion():
    """Visual negatives are a layer on exemplars, so with no exemplar in the
    champion they cannot be turned on alone — not suggested as untried."""
    from agentic_configuration_search.planner.axes import axis_values
    from agentic_configuration_search.planner.llm import _untried_axis_lines

    hist = [_round(0, 0, [], f=0.7), _round(1, 0, [], f=0.7)]
    line = _untried_axis_lines(hist, axis_values())[0]
    assert "visual_negatives" not in line and "exemplar" in line
