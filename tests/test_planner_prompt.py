"""Planner prompt structure — budget card, champion, axis board, recent rounds in
detail with older rounds summarized, diagnosis digest, and the response format
(expected gain, cost, variant note).
"""

import json
from pathlib import Path

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
                      _row("fn", exemplar="fn_arm", exemplar_fn_k=4, champ=base, ub=0.004,
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
    with tempfile.TemporaryDirectory() as td:
        LLMPlanner(chat=chat, artifact_dir=Path(td), **kw).propose(
            history, diagnosis or {})
    return seen["m"][-1]["content"]


def _board(p):
    return p.split("## Axis board", 1)[1].split("## Round history", 1)[0]


def _brow(p, axis):
    """One axis row of the board — column name → value."""
    cols = ["axis", "status", "propose", "run", "champion", "min", "note"]
    line = next(l for l in _board(p).splitlines() if l.startswith(f"{axis} | "))
    return dict(zip(cols, line.split(" | ")))


def test_prompt_sections_in_order():
    p = _prompt(_history(), budget={"round": 4, "rounds": 8})
    heads = ["Hard constraints", "Rules:", "## Budget", "## Current champion",
             "## Axis board", "## Round history", "## Diagnosis digest", "## Response"]
    pos = [p.index(h) for h in heads]
    assert pos == sorted(pos)


def test_budget_card_shows_rounds_left_and_counters_only():
    p = _prompt(_history(), budget={"round": 4, "rounds": 8, "stagnant": 2,
                                    "saturation_rounds": 5, "empty": 1,
                                    "empty_rounds": 2})
    card = p.split("## Budget", 1)[1].split("## Current champion", 1)[0]
    assert "Round 4 of 8: 5 rounds left including this one." in card
    assert "without adoption: 2; the run stops at 5" in card
    assert "(a round where a candidate was dropped for losing on fit_dev does not count): 1; the run stops at 2" in card
    assert " min" not in card                              # cost is the board's min column


def test_budget_card_skips_unknown_fields():
    card = _prompt(_history()).split("## Budget", 1)[1].split("## Current champion", 1)[0]
    assert "Round" not in card and "(no budget information)" in card


def test_board_costs_measured_and_static():
    """Observed cost in the min column (median, n); axes not observed yet get facts
    in the note — no value judgment."""
    p = _prompt(_history())
    assert _brow(p, "exemplar")["min"] == "16 (n=2, max 25)"          # median of 6 and 25 min
    assert _brow(p, "tiling")["min"] == "g3 20 (n=1, max 20)"
    assert "1+g^2 passes per image (champion: 5)" in _brow(p, "tiling")["note"]
    assert _brow(p, "lora")["min"] == "-"
    assert "fine-tunes the detector tail before screening" in _brow(p, "lora")["note"]
    assert "slowest" not in _board(p)


def test_board_skips_disabled_axes():
    p = _prompt(_history(), disabled_axes=("lora",))
    assert not any(l.startswith("lora | ") for l in _board(p).splitlines())
    assert "Disabled on this dataset (never turn these on" in _board(p)


def test_champion_block_states_scores_and_cost():
    p = _prompt(_history())
    block = p.split("## Current champion", 1)[1].split("## Axis board", 1)[0]
    assert '"exemplar": "base"' in block
    assert "held-out micro F 0.9000" in block and "fit_dev micro F 0.9200" in block
    assert "adopted in round 1 (c1)" in block and "5 detector passes" in block


def test_board_champion_column_and_prescreen():
    p = _prompt(_history(), prescreen={"on": True, "n_images": 20, "of": 80})
    alias = _brow(p, "alias")["champion"]
    assert "vlm clearly-lost (prescreened)" in alias
    assert "fd -0.0120 (ub -0.0040)" in alias and "ho -0.0100 (lo -0.0300)" in alias
    ex = _brow(p, "exemplar")
    assert ex["champion"].startswith("fn_arm near-miss")
    assert ex["status"] == "ON" and "fn_arm" in ex["propose"]
    assert "exemplar_base_k 1|2|4|8" in ex["propose"]
    tiling = _brow(p, "tiling")
    assert "3 folded" in tiling["champion"] and "grid >= 3 folded" in tiling["note"]
    assert tiling["propose"] == "2"                       # grids at or above the folded one are left out
    board = _board(p)
    assert "{exemplar=fn_arm, exemplar_fn_k=4}" in board and "{alias=vlm}" in board
    assert "Prescreened values" in board and "alias=vlm (round 3 vlm" in board
    assert "20 of 80 fit_dev images" in board and "lost_value_prescreen" in board


def test_board_when_prescreen_is_off():
    p = _prompt(_history(), prescreen={"on": False})
    assert "not prescreened on this run" in _board(p) and "alias=vlm" in _board(p)


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
    # exemplar=base, which the champion already uses, is left out — only what can be proposed now
    assert "- missed: 5 (train 4, held_out 1) -> exemplar=fn_arm, lora, alias=vlm" in digest
    detail = p.split("what the errors look like.\n", 1)[1].split("\n\n## Response")[0]
    assert '"summary"' not in detail and "train | a | 5 |" in detail


def test_response_section_asks_for_gain_cost_and_differs():
    p = _prompt(_history(), max_candidates=2)
    resp = p.split("## Response", 1)[1]
    for key in ("expected_gain", "cost_min", "differs", "rationale"):
        assert f'"{key}"' in resp
    assert "Propose up to 2 candidates" in resp


# --- response parsing ---------------------------------------------------------------

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
    """A missing or malformed field neither drops the candidate nor hands it to deterministic."""
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
    """The runner passes the rounds left and the stop counters (history tail) into the
    prompt, and the response's expected gain and cost stay in the candidate row's plan."""
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


# --- untried axes ---------------------------------------------------------------------
# With one candidate per round the order is the budget — untried axes must not be
# pushed back while variants repeat. The basis is the axis status's tried
# (planner.axis_status) — an axis never proposed in the whole run.

def _never(p):
    """Axes whose run column on the board is "never tried in this run"."""
    return [l.split(" | ")[0] for l in _board(p).splitlines()
            if " | " in l and "never tried in this run" in l.split(" | ")[3]]


def test_board_marks_axes_never_tried_in_the_run():
    p = _prompt(_history())
    never = _never(p)
    for axis in ("text_negatives", "visual_negatives", "rescorer", "lora",
                 "crop_classifier"):
        assert axis in never, axis
    for axis in ("tiling", "exemplar", "alias"):
        assert axis not in never, axis
    assert ("prefer one of them over another variant of an already-tried value"
            in " ".join(p.split()))


def test_board_status_for_disabled_blocked_and_champion_on_axes():
    from agentic_configuration_search.planner.axes import DatasetFacts

    hist = _history()
    for e in hist[1:]:                                   # the champion turned the rescorer on
        e["champion"]["axes"] = validate_axes({"exemplar": "base", "rescorer": True})
    p = _prompt(hist, disabled_axes=("crop_classifier",),
                facts=DatasetFacts(n_classes=2, n_fit_train=3, lora_min_images=20))
    assert not any(l.startswith("crop_classifier | ") for l in _board(p).splitlines())
    lora = _brow(p, "lora")                               # an axis the data blocks
    assert lora["status"] == "BLOCKED" and "too few fit_train images" in lora["note"]
    assert _brow(p, "rescorer")["status"] == "ON"         # an axis the champion has on
    assert _brow(p, "text_negatives")["status"] == "OPEN"


def test_axes_tried_under_an_old_champion_are_not_never_tried():
    """An axis tried under an old champion was tried in the whole run too."""
    hist = _history()
    hist[1]["candidates"].append(_row("tn", text_negatives="fp_crop", ub=-0.01,
                                      skipped="fit_dev_lost"))
    never = _never(_prompt(hist))
    assert "text_negatives" not in never and "crop_classifier" in never


def test_hygiene_drops_do_not_count_as_tried():
    """A row dropped without being measured (combo dedup etc.) did not try the axis."""
    hist = _history()
    hist[3]["candidates"].append(_row("cc", crop_classifier=True,
                                      champ={"exemplar": "base"},
                                      skipped="combo_tried"))
    assert "crop_classifier" in _never(_prompt(hist))


def test_visual_negatives_needs_an_exemplar_champion():
    """Visual negatives are a layer on exemplars — the note says so when the champion has none."""
    hist = [_round(0, 0, [], f=0.7), _round(1, 0, [], f=0.7)]
    vn = _brow(_prompt(hist), "visual_negatives")
    assert "needs exemplar" in vn["note"]


# --- prompt terms ------------------------------------------------------------------

def test_prompt_names_the_held_out_split_held_out():
    """The stored key "test" is legacy — the prompt carries it only as held_out."""
    diag = {"train": {"n_images": 3, "summary": {"fn": {"missed": 4}}},
            "test": {"n_images": 2, "summary": {"fn": {"missed": 1}},
                     "per_class": {"a": {"gt": 1}}}}
    p = _prompt(_history(), diag)
    assert "held_out 2 images" in p and "held_out | a | 1 |" in p
    assert '"test"' not in p and "test |" not in p
    assert "test_micro_f_beta" not in p and 'named "test"' not in p


def test_digest_picks_the_negative_source_for_the_error_kind():
    from agentic_configuration_search.planner.axes import DatasetFacts, axis_values
    from agentic_configuration_search.planner.llm import _digest_lines

    diag = {"train": {"summary": {"fp": {"class_confusion": 5}}}}
    two = _digest_lines(diag, values=axis_values(DatasetFacts(n_classes=2)),
                        champion={"exemplar": "base"})[0]
    assert "visual_negatives=fp_crop|gt_crop" in two and "text_negatives=gt_crop" in two
    # with a single class gt_crop is blocked — that source is not carried
    one = _digest_lines(diag, values=axis_values(DatasetFacts(n_classes=1)))[0]
    assert "negatives" not in one


def test_board_costs_fold_quantity_keys_into_their_axis():
    from agentic_configuration_search.planner.board import axis_costs

    hist = _history()
    hist[3]["candidates"].append(_row("k", exemplar_base_k=4, champ={"exemplar": "base"},
                                      wall=9.0))
    labels = set(axis_costs(hist))
    assert "exemplar" in labels and not any(l.endswith("_k") for l in labels)


# --- budget ladder -----------------------------------------------------------------

def _sample_diag(n=12):
    rows = [{"kind": "missed", "category": "a", "image": f"i{i}.jpg", "box": [0, 0, 9, 9],
             "size_ratio": 0.2, "note": "a long vision note " * 8} for i in range(n)]
    return {"train": {"n_images": 9, "summary": {"fn": {"missed": n}}, "fn": rows,
                      "per_class": {"a": {"gt": 20}}}}


def test_budget_ladder_trims_samples_before_the_last_round():
    """The last round's detail is worth more than the diagnosis samples — the window
    drops to 0 only after the samples are halved and removed."""
    from agentic_configuration_search.planner.llm import (BUDGET_LADDER, MAX_HISTORY,
                                                          _est_tokens)

    assert BUDGET_LADDER[0] == (MAX_HISTORY, "full")
    assert BUDGET_LADDER[-3:] == ((1, "half"), (1, "none"), (0, "none"))
    hist, diag = _history(), _sample_diag()
    planner = LLMPlanner(chat=lambda m, **k: None, artifact_dir=Path("/nonexistent"))
    build, _ = planner._builder(hist, diag)
    size = {step: _est_tokens(build(*step)) for step in BUDGET_LADDER}
    assert size[(1, "half")] < size[(1, "full")] and size[(1, "none")] < size[(1, "half")]
    # a budget only window 1 with half the samples fits — the last round stays
    planner.token_budget = size[(1, "half")]
    msgs, window, samples = planner._fit(build)
    assert (window, samples) == (1, "half") and '"round": 3' in msgs[-1]["content"]
    full = build(1, "full")[-1]["content"].count("a long vision note")
    assert 0 < msgs[-1]["content"].count("a long vision note") < full
    # if nothing fits, the last step
    planner.token_budget = 1
    assert planner._fit(build)[1] == 0


def test_budget_ladder_also_trims_leftover_crop_notes():
    """Crop notes without a matching sample are samples too — halved and removed with them."""
    from agentic_configuration_search.planner.llm import _halve_samples, _strip_samples
    diag = {"train": {"fn_sample": [1, 2, 3], "crop_notes": [1, 2, 3, 4], "fn_total": 9}}
    half = _halve_samples(diag)["train"]
    assert half["fn_sample"] == [1, 2] and half["crop_notes"] == [1, 2]
    none = _strip_samples(diag)["train"]
    assert none == {"fn_total": 9}


def test_proposal_records_the_budget_step(tmp_path):
    from agentic_configuration_search.planner.llm import BUDGET_LADDER
    LLMPlanner(chat=lambda m, **k: {"candidates": [], "stop": True},
               artifact_dir=tmp_path).propose(_history(), {})
    stored = json.loads((tmp_path / "proposal.json").read_text())
    assert (stored["budget"]["window"], stored["budget"]["samples"]) == BUDGET_LADDER[0]


# --- reasons, not notes -----------------------------------------------------------

def test_prompt_carries_english_reasons_not_notes():
    """A rejection row's note is a free sentence for people — the prompt carries only
    the English reason built from the skip code and structured fields."""
    from agentic_configuration_search.planner.llm import _outcome_lines, _reason

    hist = _history()
    hist[3]["candidates"] += [
        {"id": "bad", "axes": {"exemplar": "base", "exemplar_fn_k": 2},
         "skipped": "invalid_axes", "note": "NOTE-A exemplar_fn_k is fn_arm only",
         "invalid": "exemplar_fn_k applies only with exemplar=fn_arm"},
        {**_row("cls", champ={"exemplar": "base"}, crop_classifier=True,
                skipped="classifier_gate"),
         "note": "NOTE-B fit_dev ΔF(β) -0.0010 ≤ margin 0.005",
         "arm_gate": {"delta": -0.001, "margin": 0.005, "lam": 0.5}},
        {**_row("err", champ={"exemplar": "base"}, rescorer=True, skipped="error"),
         "note": "ValueError: NOTE-C discriminator fit failed"},
        {**_row("pre", champ={"exemplar": "base"}, tiling=2, skipped="lost_value_prescreen"),
         "note": "prescreen: NOTE-D clearly lost", "prescreen": {"n_images": 20, "of": 80}}]
    p = _prompt(hist)
    assert "NOTE-" not in p
    rows = {c["id"]: c for c in hist[3]["candidates"]}
    assert "exemplar_fn_k" in _reason(rows["bad"])
    assert _reason(rows["cls"]) == ("crop classifier arm gate failed: best fit_dev delta "
                                    "-0.0010 <= margin 0.005 (best lambda 0.5)")
    assert _reason(rows["err"]) == "it failed with a runtime error (ValueError)"
    assert "prescreen of 20/80" in _reason(rows["pre"])
    lines = _outcome_lines(list(rows.values()), {"exemplar": "base"})
    assert "NOTE-" not in "\n".join(lines)


# --- diagnosis tables and conditional explanations ---------------------------------

def test_diagnosis_renders_tables_not_json():
    from agentic_configuration_search.planner.llm import _compact_diagnosis, _diagnosis_text
    diag = {"train": {"n_images": 9, "per_class": {"a": {"gt": 20, "miss_rate": 0.25,
                                                         "fn": {"missed": 5}}},
                      "fn": [{"kind": "missed", "category": "a", "image": "x.jpg",
                              "box": [0, 0, 30, 10], "size_ratio": 0.12}]}}
    text = _diagnosis_text(_compact_diagnosis(diag))
    assert "train | a | 20 | 0.250 | missed 5 | -" in text
    assert "FN | missed | a | size 0.12 | 30x10 | -" in text
    assert "{" not in text and "x.jpg" not in text


def test_field_explanations_follow_the_rows_present():
    """A field is explained only when a history row has it — a glossary of absent fields is noise."""
    from agentic_configuration_search.planner.llm import _glossary
    assert _glossary([]) == ""
    g = _glossary([{"candidates": [{"fit_dev_cmp": {"delta": 0.1}, "same_as": "champion"}]}])
    assert g.startswith("Fields in these rows:") and "fit_dev_cmp" in g and "same_as" in g
    assert "neg_eta_used" not in g and "modality" not in g
    # not fooled by a field name that happens to appear in a value string — judged by key
    assert "modality" not in _glossary([{"candidates": [{"material": "modality"}]}])


def test_tiling_rule_only_when_tiling_is_enabled():
    assert "is a grid g" in _prompt(_history())
    assert "is a grid g" not in _prompt(_history(), disabled_axes=("tiling",))


def test_diagnosis_table_cells_and_leftover_notes():
    """Unmatched crop notes tell FP from FN by their fields (the score for an FP) and
    are not counted in the sample count. A | or line break in a cell does not break
    the columns. Without a class name the id is carried."""
    from agentic_configuration_search.planner.llm import _diagnosis_text, _strip_samples
    compact = {"train": {
        "n_images": 4, "fp_total": 5, "fn_total": 3,
        "per_class": {"a": {"gt": 9, "miss_rate": 0.0041, "fn": {"missed": 1}}},
        "fp_sample": [{"kind": "background", "category": None, "category_id": 7,
                       "score": 0.5, "wh": [3, 4], "note": "pole | sign\nsecond line"}],
        "crop_notes": [{"kind": "background", "category": "a", "score": 0.77,
                        "wh": [4, 4], "note": "left over"}]}}
    text = _diagnosis_text(compact)
    assert "train 4 images, 5 FP, 3 FN" in text and "1 of 5 FP" in text
    assert "train | a | 9 | 0.004 |" in text
    assert "FP | background | id 7 | score 0.500 | 3x4 | pole / sign second line" in text
    assert "FP-note | background | a | score 0.770 | 4x4 | left over" in text
    none = _diagnosis_text(_strip_samples(compact))
    assert "-note" not in none and "Error samples" not in none
    assert "train 4 images, 5 FP, 3 FN" in none               # totals stay without samples


def test_every_skip_code_has_an_english_reason():
    """Every skip code the ledger knows has an English reason — so a new code does not leak as str(code)."""
    from agentic_configuration_search.planner.ledger import _BLOCKED, _OUTCOME
    from agentic_configuration_search.planner.llm import _REASON
    arm = {"lora_gate", "classifier_gate"}
    assert set(_BLOCKED) | set(_OUTCOME) <= set(_REASON) | arm


def test_reason_details_follow_structured_fields():
    from agentic_configuration_search.planner.axes import sanitize_candidates
    from agentic_configuration_search.planner.llm import _reason
    dropped: list = []
    sanitize_candidates([{"id": "u", "axes": {"foo": 1}},
                         {"id": "v", "axes": {"exemplar": "bogus"}}],
                        max_candidates=3, dropped=dropped)
    assert [_reason(d) for d in dropped] == [
        "the axes break the value lists or Hard constraints (unknown axis foo)",
        "the axes break the value lists or Hard constraints (exemplar=\"bogus\" is not "
        "a valid value)"]
    passed = {"skipped": "fit_dev_lost", "prescreen": {"passed": True, "n_images": 20,
                                                      "of": 80}}
    assert "prescreen" not in _reason(passed)   # once passed, the full fit_dev decided
    assert _reason({"skipped": "lora_gate", "arm_gate": {"tripwire": True}}) == \
        "tail fine-tune not trained: too few fit_train images"
    assert _reason({"skipped": "error", "note": "a message without an exception type"}) == \
        "it failed with a runtime error"


# --- axis board details ----------------------------------------------------------------

def test_board_states_how_to_turn_an_axis_off():
    """The off value differs per axis — "none" for exemplar, false for the rest, null for quantities."""
    board = _board(_prompt(_history(), disabled_axes=("visual_negatives",)))
    assert 'set exemplar to "none", any other axis to false' in board
    assert "quantity key set to null" in board


def test_never_tried_skips_axes_that_cannot_be_proposed_alone():
    """A lora the data blocks and visual_negatives under a champion without exemplars are not recommended as untried."""
    from agentic_configuration_search.planner.axes import DatasetFacts
    hist = [_round(0, 0, [], f=0.7), _round(1, 0, [], f=0.7)]
    p = _prompt(hist, facts=DatasetFacts(n_classes=2, n_fit_train=3, lora_min_images=20))
    assert _brow(p, "lora")["run"] == "not tried"
    assert _brow(p, "visual_negatives")["run"] == "not tried"
    assert _brow(p, "lora")["note"] == "too few fit_train images to train"   # no cost note
    assert "never tried in this run" in _brow(p, "alias")["run"]


def test_board_keep_on_probe_and_off_rows():
    from agentic_configuration_search.planner.axes import DatasetFacts
    hist = _history()
    for e in hist[1:]:
        e["champion"]["axes"] = validate_axes({"exemplar": "base", "lora": True})
    p = _prompt(hist)
    assert _brow(p, "lora")["status"] == "ON" and _brow(p, "lora")["propose"] == "keep on"
    one = _prompt(_history(), facts=DatasetFacts(n_classes=1))
    assert "gt_crop" not in _brow(one, "text_negatives")["propose"]       # a single class


def test_board_champion_column_lists_hidden_values_by_name():
    """Numbers only for the first two, the rest by name and status — a value that lost narrowly comes before one that clearly lost."""
    hist = _history()
    for v in ("fp_crop", "gt_crop"):
        hist[3]["candidates"].append(_row(v, champ={"exemplar": "base"}, text_negatives=v,
                                          ub=-0.02, skipped="fit_dev_lost"))
    hist[3]["candidates"].append(_row("bg", champ={"exemplar": "base"},
                                      text_negatives="background", ub=0.01, delta=-0.001))
    cell = _brow(_prompt(hist), "text_negatives")["champion"]
    assert cell.startswith("background near-miss") and "also " in cell


def test_prompt_sections_agree_on_off_values_and_costs():
    """The off value is stated once, in the legend; Rules and Response point to the board."""
    p = _prompt(_history())
    rules = p.split("Rules:", 1)[1].split("## Budget", 1)[0]
    assert "set it to false" not in rules and "off value (Axis board legend)" in " ".join(
        rules.split())
    assert "from the min column of the Axis board" in " ".join(p.split())
    assert "from the budget" not in p


def test_board_k_variants_when_only_the_champion_value_is_left():
    from agentic_configuration_search.planner.axes import axis_values
    from agentic_configuration_search.planner.axis_status import axis_status
    from agentic_configuration_search.planner.board import board_lines
    hist = _history()
    for e in hist[1:]:
        e["champion"]["axes"] = validate_axes({"text_negatives": "fp_crop"})
    values = {**axis_values(), "text_negatives": (False, "fp_crop")}
    row = next(l for l in board_lines(hist, values, axis_status(hist, 2))
               if l.startswith("text_negatives | ")).split(" | ")
    assert row[2] == "k variants: text_negative_k 1|2|4"


def test_champion_block_pluralizes_detector_passes():
    hist = _history()
    hist[-1]["champion"]["forwards_per_image"] = 1
    assert "1 detector pass per image" in _prompt(hist)


def test_board_writes_values_as_json():
    """Boolean values as true, like the response JSON — the same notation as the champion line."""
    hist = [_round(0, 0, [], f=0.7), _round(1, 0, [], f=0.7)]
    assert _brow(_prompt(hist), "lora")["propose"] == "true"


def test_glossary_covers_carryover_and_lost_value_rows():
    from agentic_configuration_search.planner.llm import _glossary
    g = _glossary([{"candidates": [{"gate_passed_but_lost": True, "lost_values": ["a"]}]}])
    assert "gate_passed_but_lost" in g and "lost_values" in g


def test_note_legend_only_when_note_rows_exist():
    from agentic_configuration_search.planner.llm import _diagnosis_text
    base = {"train": {"fp_total": 1, "fp_sample": [{"kind": "background", "score": 0.5}]}}
    assert "-note rows" not in _diagnosis_text(base)
    base["train"]["crop_notes"] = [{"kind": "background", "score": 0.4, "note": "x"}]
    assert "-note rows" in _diagnosis_text(base)


def test_over_budget_prompt_is_not_sent_and_says_why(tmp_path):
    """If even the last step is over budget, nothing goes to the server and it falls back with the reason."""
    calls = []
    planner = LLMPlanner(chat=lambda m, **k: calls.append(m) or {"candidates": []},
                         artifact_dir=tmp_path, token_budget=10)
    planner.propose(_history(), {"train": {"summary": {"fn": {"missed": 5}}}})
    stored = json.loads((tmp_path / "proposal.json").read_text())
    assert not calls and stored["fallback"] == "deterministic"
    assert stored["error"].startswith("PromptOverBudget")


def test_budget_follows_the_server_context():
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop.runner import _token_budget
    from agentic_configuration_search.planner.llm import CONTEXT_MARGIN, OUTPUT_MAX_TOKENS
    cfg = config_from_dict({"dataset": "x"})
    assert _token_budget(cfg) == 32768 - OUTPUT_MAX_TOKENS - CONTEXT_MARGIN
    assert _token_budget(cfg, 16384) == 16384 - OUTPUT_MAX_TOKENS - CONTEXT_MARGIN


def test_run_summary_collects_planner_fallbacks(tmp_path):
    from agentic_configuration_search.loop.runner import _planner_fallbacks
    for r, rec in ((1, {"fallback": None}), (2, {"fallback": "deterministic",
                                                 "error": "PromptOverBudget: x"})):
        d = tmp_path / "rounds" / f"round_{r:03d}"
        d.mkdir(parents=True)
        (d / "proposal.json").write_text(json.dumps(rec))
    assert _planner_fallbacks(tmp_path) == [{"round": 2, "error": "PromptOverBudget: x"}]


def test_llm_client_reads_the_server_context():
    from agentic_configuration_search.clients.llm import LLMClient

    class _Resp:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    class _Session:
        def get(self, url, timeout):
            assert url.endswith("/v1/models")
            return _Resp({"data": [{"id": "other", "max_model_len": 8192},
                                   {"id": "m", "max_model_len": 32768}]})
    assert LLMClient("http://x", "m", session=_Session()).max_model_len() == 32768


def test_missed_small_block_only_when_tiling_can_be_proposed():
    """When tiling cannot be proposed (disabled, every grid folded) the first-grid hint is not carried."""
    diag = {"train": {"summary": {"fn": {"missed_small": 3}},
                      "fn": [{"kind": "missed_small", "size_ratio": 0.02, "category": "a"}]}}
    assert "Missed-small sizes" in _prompt(_history(), diag)
    assert "Missed-small sizes" not in _prompt(_history(), diag, disabled_axes=("tiling",))
    hist = _history()
    hist[3]["candidates"].append({**_row("t2", champ={"exemplar": "base"}, tiling=2,
                                         skipped="axis_unused"),
                                  "unused_axes": ["tiling"], "tiling_used": False})
    assert "Missed-small sizes" not in _prompt(hist, diag)   # grid >= 2 folded


def test_loop_wires_the_server_context_into_the_planner(tmp_path, monkeypatch):
    """The runner sets the planner budget from the context the server reports (not the config value)."""
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop.runner import Loop
    from agentic_configuration_search.planner.llm import CONTEXT_MARGIN, OUTPUT_MAX_TOKENS
    loop = Loop.__new__(Loop)
    loop.cfg = config_from_dict({"dataset": "x", "planner": {"base_url": "http://x"}})
    loop.llm_chat = lambda m, **k: {}
    monkeypatch.setattr(Loop, "_server_context", lambda self: 16384)
    monkeypatch.setattr(Loop, "_facts", lambda self: None)
    monkeypatch.setattr(Loop, "_prescreen_info", lambda self: {})
    loop.disabled_axes, loop.disabled_values = (), {}
    planner = loop._llm_planner(tmp_path)
    assert planner.token_budget == 16384 - OUTPUT_MAX_TOKENS - CONTEXT_MARGIN


def test_board_champion_column_uses_json_values_and_skips_empty_sources():
    """Booleans as true in the champion column; values whose material came out empty leave propose."""
    hist = _history()
    hist[3]["candidates"] += [
        _row("l", champ={"exemplar": "base"}, lora=True, ub=-0.01, skipped="fit_dev_lost"),
        {**_row("tn", champ={"exemplar": "base"}, text_negatives="gt_crop",
                skipped="no_material"), "empty_axis": "text_negatives"}]
    p = _prompt(hist)
    assert _brow(p, "lora")["champion"].startswith("true ")
    assert "gt_crop" not in _brow(p, "text_negatives")["propose"]


def test_missed_small_block_lists_only_grids_above_the_champion():
    from agentic_configuration_search.planner.llm import _small_block
    diag = {"train": {"fn": [{"kind": "missed_small", "size_ratio": 0.02}]}}
    text = _small_block(diag, 0.5, grids_open=[4, 5])
    assert "grid 4:" in text and "grid 5:" in text and "grid 2:" not in text


def test_blocked_axis_says_why_when_every_value_is_empty():
    rows = [{**_row("vlm", alias="vlm", skipped="no_material"), "empty_axis": "alias"}]
    hist = [_round(0, 0, [], f=0.7), _round(1, 0, rows, f=0.7)]
    row = _brow(_prompt(hist), "alias")
    assert row["status"] == "BLOCKED" and "mined empty" in row["note"]


def test_blocked_detector_state_axis_says_not_trained():
    rows = [{**_row("c", crop_classifier=True, skipped="classifier_gate"),
             "empty_axis": "crop_classifier", "arm_gate": {"tripwire": True}}]
    hist = [_round(0, 0, [], f=0.7), _round(1, 0, rows, f=0.7)]
    row = _brow(_prompt(hist), "crop_classifier")
    assert row["status"] == "BLOCKED" and "not trained under this champion" in row["note"]


def test_board_explains_what_each_axis_does():
    """The board does not carry axis names alone — one line per axis says what it
    does, so the planner does not pick base vs fn_arm from the record only. Axes
    that are off are not explained."""
    from agentic_configuration_search.planner.axes import axis_values
    from agentic_configuration_search.planner.board import _axis_doc_line

    line = _axis_doc_line(axis_values())
    for token in ("alias", "vlm", "exemplar", "fn_arm", "rescorer", "crop_classifier",
                  "lora"):
        assert token in line, token
    assert "rescorer" not in _axis_doc_line(axis_values(), disabled=("rescorer",))


def test_prompt_defines_error_kinds_and_drops_redundant_lines():
    from agentic_configuration_search.core.metrics import FN_KINDS, FP_KINDS
    from agentic_configuration_search.planner import llm

    for kind in FP_KINDS + FN_KINDS:
        assert f"{kind} =" in llm.ERROR_KINDS, kind
    assert "{error_kinds}" in llm.PROMPT
    # the tried-combinations line (_combos_line) states the no-exact-repeat rule —
    # the constraints do not carry it twice
    from agentic_configuration_search.planner.axis_status import axis_status
    assert not any("exact combination" in x
                   for x in llm._constraint_lines((), axis_status([], None)))
    # how the prompt is assembled is not a rule
    assert "shown while" not in llm.TILING_RULE
