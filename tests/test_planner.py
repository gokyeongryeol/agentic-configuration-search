"""planner — axes contract, deterministic cause -> remedy, LLM hygiene and fallback."""

import pytest

from agentic_detector_config.planner import deterministic
from agentic_detector_config.planner.axes import BASELINE, sanitize_candidates, validate_axes
from agentic_detector_config.planner.llm import LLMPlanner


# ------------------------------------------------------------------- axes

def test_validate_axes_fills_baseline_and_rejects_unknown():
    full = validate_axes({"alias": "vlm"})
    assert full["alias"] == "vlm" and full["exemplar"] == "none"
    with pytest.raises(ValueError, match="unknown key"):
        validate_axes({"exemplars": True})
    with pytest.raises(ValueError, match="axes.exemplar"):
        validate_axes({"exemplar": "all"})


def test_visual_negative_requires_exemplar():
    with pytest.raises(ValueError, match="visual_negatives"):
        validate_axes({"visual_negatives": True})
    validate_axes({"visual_negatives": True, "exemplar": "base"})   # OK


def test_sanitize_drops_invalid_and_dedups():
    raw = [{"axes": {"visual_negatives": True}, "rationale": "invalid — no exemplar"},
           {"axes": {"alias": "vlm"}, "rationale": "a"},
           {"axes": {"alias": "vlm"}, "rationale": "duplicate"},
           "garbage",
           {"axes": {"rescorer": True}, "rationale": "b"}]
    out = sanitize_candidates(raw, max_candidates=5)
    assert [bool(c["axes"]["alias"] or c["axes"]["rescorer"])
            for c in out] == [True, True]
    assert len(out) == 2


# ---------------------------------------------------------- deterministic

def _diag(fn=None, fp=None, split="train"):
    return {split: {"summary": {"fn": fn or {}, "fp": fp or {}}}}


def test_remedy_maps_dominant_error_to_axis():
    """missed dominates -> promoting exemplar is the first candidate."""
    out = deterministic.propose([], _diag(fn={"missed": 10, "below_threshold": 2}))
    assert out["stop"] is False
    first = out["candidates"][0]
    assert first["axes"]["exemplar"] == "base"
    # a below_threshold remedy follows
    assert any(c["axes"]["rescorer"] for c in out["candidates"])


def test_single_increment_from_champion():
    """A candidate moves exactly one axis away from the champion — a single increment with clean attribution."""
    history = [{"champion": {"axes": {**BASELINE, "exemplar": "base"}},
                "candidates": []}]
    out = deterministic.propose(history, _diag(fn={"missed": 5}))
    first = out["candidates"][0]
    assert first["axes"]["exemplar"] == "fn_arm"            # base -> fn_arm promotion
    diff = {k for k in BASELINE if first["axes"][k] !=
            history[0]["champion"]["axes"][k]}
    assert diff == {"exemplar"}


def test_tried_candidates_not_reproposed():
    history = [{"champion": {"axes": BASELINE},
                "candidates": [{"axes": {**BASELINE, "exemplar": "base"}}]}]
    out = deterministic.propose(history, _diag(fn={"missed": 5}))
    assert all(c["axes"]["exemplar"] != "base" for c in out["candidates"])


def test_fn_arm_requires_base_first():
    """If the champion is text-only but base was already tried, do not skip ahead to fn_arm."""
    history = [{"champion": {"axes": BASELINE},
                "candidates": [{"axes": {**BASELINE, "exemplar": "base"}}]}]
    out = deterministic.propose(history, _diag(fn={"missed": 5}))
    assert all(c["axes"]["exemplar"] != "fn_arm" for c in out["candidates"])


def test_stop_when_mapping_exhausted():
    out = deterministic.propose([], _diag(fn={"duplicate": 3}))
    assert out["stop"] is True and out["candidates"] == []


def test_test_split_diagnosis_also_counts():
    out = deterministic.propose([], _diag(fn={"below_threshold": 4}, split="test"))
    assert any(c["axes"]["rescorer"] for c in out["candidates"])


# ------------------------------------------------------------------- llm

def test_llm_invalid_response_falls_back_to_deterministic(tmp_path):
    planner = LLMPlanner(chat=lambda m, **k: {"candidates": ["mess"], "stop": False},
                         artifact_dir=tmp_path)
    out = planner.propose([], _diag(fn={"missed": 5}))
    assert out["candidates"], "the fallback should have filled the candidates"
    assert out["candidates"][0]["axes"]["exemplar"] == "base"
    import json
    assert json.loads((tmp_path / "proposal.json").read_text())["fallback"] == \
        "deterministic"


def test_llm_error_falls_back(tmp_path):
    def boom(messages, **kw):
        raise RuntimeError("down")

    planner = LLMPlanner(chat=boom, artifact_dir=tmp_path)
    out = planner.propose([], _diag(fn={"missed": 5}))
    assert out["candidates"] and out["planner"] == "llm"


def test_llm_valid_candidates_pass_through(tmp_path):
    planner = LLMPlanner(
        chat=lambda m, **k: {"candidates": [
            {"id": "my", "axes": {"exemplar": "base", "visual_negatives": True},
             "rationale": "suppress background FPs"}], "stop": False},
        artifact_dir=tmp_path)
    out = planner.propose([], {})
    assert out["candidates"][0]["id"] == "my"
    assert out["candidates"][0]["axes"]["visual_negatives"] is True


def test_llm_prompt_is_compacted(tmp_path):
    """Train diagnosis details can run to thousands of entries — including them verbatim overflows
    the context. The prompt carries only aggregates + a small sample."""
    import json

    big_detail = [{"kind": "background", "image": f"i{i}.png",
                   "box": [1, 2, 3, 4], "category_id": 1, "score": 0.5}
                  for i in range(2000)]
    diagnosis = {"train": {"n_images": 100,
                             "summary": {"fn": {}, "fp": {"background": 2000}},
                             "per_class": {}, "fp": big_detail, "fn": []}}
    seen = {}

    def chat(messages, **kw):
        seen["chars"] = sum(len(m["content"]) for m in messages)
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose([], diagnosis)
    assert seen["chars"] < 30_000, f"prompt is still large: {seen['chars']}"


def test_prompt_omits_image_identifiers_and_merges_crop_notes(tmp_path):
    """Samples carry no image identifiers — descriptions are folded into the samples, so there is
    nothing to join on, and a bare name without coordinates cannot point at anything. No filename leaks either."""
    import json

    from agentic_detector_config.planner.llm import LLMPlanner

    name = "cam07/2024-01-02T03_04_05_jpg.rf.deadbeefcafe.jpg"
    fp = [{"kind": "background", "image": name, "box": [1, 2, 16, 46],
           "category_id": 1, "category": "ball", "score": 0.9},
          {"kind": "class_confusion", "image": name, "box": [5, 6, 7, 8],
           "category_id": 2, "category": "player", "score": 0.8}]
    diagnosis = {"train": {
        "n_images": 2, "summary": {}, "per_class": {}, "fp": fp, "fn": [],
        "crop_notes": [{"crop": 0, "kind": "background", "image": name,
                        "box": [1, 2, 16, 46], "score": 0.9,
                        "category_id": 1, "note": "blurry fence post"}]}}
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = "\n".join(m["content"] for m in messages)
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose([], diagnosis)
    prompt = seen["prompt"]
    assert name not in prompt and "deadbeefcafe" not in prompt   # no filename leak
    assert '"image"' not in prompt                               # no identifier at all

    diag = json.loads(prompt.split("what the errors look like.\n", 1)[1]
                      .split("\n\nAxis leverage this round", 1)[0])["train"]
    assert all("image" not in r for r in diag["fp_sample"])
    # descriptions are folded into the same detection's sample, so no separate crop_notes remain
    bg = next(r for r in diag["fp_sample"] if r["kind"] == "background")
    assert bg["note"] == "blurry fence post" and bg["wh"] == [15, 44]
    assert "crop_notes" not in diag


def test_samples_carry_class_name_not_id():
    """Samples must use the same vocabulary (class names) as per_class — with ids only, the prompt
    has no mapping and individual errors cannot be attributed to a class."""
    from agentic_detector_config.planner.llm import _compact_diagnosis

    diag = {"train": {
        "n_images": 1, "summary": {"fp": {"background": 1, "duplicate": 0}},
        "per_class": {"soccer ball": {"gt": 3, "miss_rate": 0.1}},
        "fp": [{"kind": "background", "image": "a.png", "box": [0, 0, 15, 44],
                "category_id": 1, "category": "soccer ball", "score": 0.7}],
        "fn": []}}
    c = _compact_diagnosis(diag)["train"]
    row = c["fp_sample"][0]
    assert row["category"] == "soccer ball" and "category_id" not in row
    assert row["category"] in c["per_class"]          # same vocabulary as per_class
    assert row["wh"] == [15, 44]
    # causes with zero count are omitted from the prompt
    assert c["summary"]["fp"] == {"background": 1}


def test_leverage_block_is_computed_per_round():
    """Replaces the static "cause -> axis" table: each axis gets the error count it addresses this
    round, listed in descending order. Axis names come from the canonical REMEDY."""
    from agentic_detector_config.planner.axes import AXES
    from agentic_detector_config.planner.llm import _leverage_lines

    diag = {"train": {"summary": {"fn": {"below_threshold": 80, "missed": 2},
                                  "fp": {"class_confusion": 40}}},
            "test": {"summary": {"fn": {"below_threshold": 20}, "fp": {}}}}
    lines = _leverage_lines(diag)
    axes = [l.split(":")[0][2:] for l in lines]
    assert axes == sorted(set(axes), key=axes.index)          # no duplicates
    assert all(a in AXES for a in axes)                       # no phantom axes
    # sorted by count descending — axes covering several causes come first
    totals = [int(l.rsplit("(total ", 1)[1].rstrip(")")) for l in lines]
    assert totals == sorted(totals, reverse=True)
    rescorer = next(l for l in lines if l.startswith("- rescorer:"))
    assert "below_threshold 100" in rescorer      # train 80 + holdout ("test" key, legacy name) 20 summed
    # axes dedicated to unobserved causes are omitted (tiling <- missed_small)
    assert "tiling" not in axes
    assert _leverage_lines({}) == []


def test_dead_axes_survive_the_history_window(tmp_path):
    """Axes with empty material are collected over the full history — once they slide out of the
    history window the model proposes the same axis again (re-proposals did repeat in practice)."""
    from agentic_detector_config.planner.llm import MAX_HISTORY, LLMPlanner, _dead_axis_lines

    old = [{"round": 1, "champion": {"axes": BASELINE}, "candidates": [
        {"id": "c1", "axes": {**BASELINE, "alias": "vlm"},
         "skipped": "no_material", "empty_axis": "alias"}]}]
    filler = [{"round": i, "champion": {"axes": BASELINE}, "candidates": []}
              for i in range(2, MAX_HISTORY + 4)]
    history = old + filler
    assert _dead_axis_lines(history) == ["- alias (values tried: vlm)"]

    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(history, {})
    assert "came out empty on this dataset" in seen["prompt"]
    assert "- alias (values tried: vlm)" in seen["prompt"]


def test_history_window_zero_means_no_history():
    """history[-0:] is the whole list — if the prompt grows back the moment the budget guard shrinks
    the window to 0, the last line of defense becomes the worst case."""
    from agentic_detector_config.planner.llm import _compact_history

    history = [{"round": i, "champion": {"axes": BASELINE}, "candidates": []}
               for i in range(1, 5)]
    assert _compact_history(history, 0) == []
    assert len(_compact_history(history, 1)) == 1


def test_quantity_axes_require_their_feature():
    """Quantity axes are meaningful only when their feature axis is on."""
    validate_axes({"exemplar": "base", "exemplar_k": 2})           # OK
    validate_axes({"exemplar": "fn_arm", "fn_k": 4})               # OK
    validate_axes({"text_negatives": "fp_crop", "text_negative_k": 4})  # OK
    with pytest.raises(ValueError, match="exemplar_k"):
        validate_axes({"exemplar_k": 2})                # quantity with exemplar=none
    with pytest.raises(ValueError, match="fn_k"):
        validate_axes({"exemplar": "base", "fn_k": 4})  # not fn_arm
    with pytest.raises(ValueError, match="visual_negative_k"):
        validate_axes({"visual_negative_k": 2})
    with pytest.raises(ValueError, match="axes.exemplar_k"):
        validate_axes({"exemplar": "base", "exemplar_k": 3})  # value off the grid


def test_quantity_axes_default_none_is_baseline():
    full = validate_axes({"exemplar": "base"})
    assert full["exemplar_k"] is None and full["fn_k"] is None


def test_same_axis_reproposed_after_champion_change():
    """Core contract of combo-level dedup: an axis rejected under an old champion may be retried
    once the champion changes (a new combo) — e.g. lora rejected in the text-only era is
    proposed again after exemplar is accepted."""
    old_try = {**BASELINE, "lora": True}                 # lora tried on text-only
    new_champ = {**BASELINE, "exemplar": "base"}
    history = [{"champion": {"axes": BASELINE}, "candidates": [{"axes": old_try}]},
               {"champion": {"axes": new_champ}, "candidates": []}]
    out = deterministic.propose(history, _diag(fn={"missed": 5}))
    lora_cands = [c for c in out["candidates"] if c["axes"]["lora"]]
    assert lora_cands, "champion changed but lora is not re-proposed"
    assert lora_cands[0]["axes"]["exemplar"] == "base"   # new combo on top of the new champion


def test_exact_combo_never_reproposed():
    """The exact same combo is blocked permanently while the champion is unchanged."""
    tried = {**BASELINE, "exemplar": "base"}
    history = [{"champion": {"axes": BASELINE}, "candidates": [{"axes": tried}]}]
    out = deterministic.propose(history, _diag(fn={"missed": 5}))
    assert all(c["axes"] != tried for c in out["candidates"])


def test_class_confusion_proposes_crop_classifier():
    out = deterministic.propose([], _diag(fp={"class_confusion": 9}))
    axes_list = [c["axes"] for c in out["candidates"]]
    assert any(a["crop_classifier"] for a in axes_list)


# ------------------------------------------------------- llm dataset notes

def test_llm_prompt_includes_dataset_notes(tmp_path):
    """dataset_notes are placed verbatim into the LLM prompt — the only free-text channel through which
    the planner can use dataset traits known to the operator."""
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path,
               notes="night-time CCTV; objects are tiny").propose([], {})
    assert "night-time CCTV; objects are tiny" in seen["prompt"]
    assert "Dataset notes" in seen["prompt"]


def test_llm_prompt_without_notes_is_byte_identical(tmp_path):
    """Without notes, the prompt must be byte-identical to the pre-notes prompt — a running
    experiment's prompt must not change because of a code upgrade."""
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose([], {})
    assert "Dataset notes" not in seen["prompt"]
    # junction between the hint block and the closing sentence — no empty block in between
    assert "picking the largest number.\n- (no errors observed)\n\nPropose up to" \
        in seen["prompt"]


def test_llm_notes_recorded_in_replay_artifact(tmp_path):
    """The prompt with notes is kept as-is in proposal.json (replay contract)."""
    import json

    LLMPlanner(chat=lambda m, **k: {"candidates": [], "stop": True},
               artifact_dir=tmp_path, notes="drone footage").propose([], {})
    stored = json.loads((tmp_path / "proposal.json").read_text())
    assert any("drone footage" in m["content"] for m in stored["prompt"])


def test_compact_history_carries_rejection_magnitude():
    """The *degree* of a rejection stays in the prompt — _compact_history carries each candidate's
    holdout score (test_micro_f_beta, legacy name), gate Δ / CI lower bound (decision.evaluated),
    the fit_dev score and the head of the skip reason (note). The LLM must tell a near miss from
    a large regression to decide whether to retry or give up."""
    from agentic_detector_config.planner.llm import MAX_NOTE, _compact_history

    entry = {
        "round": 3,
        "champion": {"axes": dict(BASELINE), "micro_f_beta": 0.7},
        "candidates": [
            {"id": "near", "axes": {"alias": "vlm"}, "adopted": False,
             "fit_dev_micro_f_beta": 0.72},
            {"id": "arm", "axes": {"lora": True}, "skipped": "lora_gate",
             "note": "fit_dev ΔF(β) +0.0020 ≤ margin 0.01" + " long tail" * 40},
        ],
        "decision": {"adopted": False, "id": None, "evaluated": {
            "near": {"micro_f_beta": 0.705,
                     "gate": {"delta": 0.005, "lo": -0.001, "hi": 0.011,
                              "adopted": False}}}},
    }
    near, arm = _compact_history([entry])[0]["candidates"]
    assert near["test_micro_f_beta"] == 0.705
    assert near["gate_delta"] == 0.005 and near["gate_ci_lo"] == -0.001
    assert near["fit_dev_micro_f_beta"] == 0.72
    assert arm["note"].startswith("fit_dev ΔF(β) +0.0020")
    assert len(arm["note"]) <= MAX_NOTE          # only the head of the note (token budget)
    assert "test_micro_f_beta" not in arm        # absent for candidates that never reached the holdout gate


def test_compact_payloads_round_floats_to_4dp():
    """Floats in the prompt are rounded to 4 digits — spurious precision (18-char floats) wastes
    tokens. Why 4: gate Δ/CI decisions at the margin (0.005) scale are decided at the 4th digit
    (with 3 digits, 0.0049 and 0.0051 become equal)."""
    from agentic_detector_config.planner.llm import _compact_diagnosis, _compact_history

    entry = {"round": 1,
             "champion": {"axes": dict(BASELINE),
                          "micro_f_beta": 0.9685230024213075},
             "candidates": [{"id": "c", "axes": {"alias": "vlm"},
                             "adopted": False,
                             "fit_dev_micro_f_beta": 0.7123456789}],
             "decision": {"adopted": False, "id": None, "evaluated": {
                 "c": {"micro_f_beta": 0.7049999999,
                       "gate": {"delta": 0.0049123, "lo": -0.00123456,
                                "hi": 0.011, "adopted": False}}}}}
    h = _compact_history([entry])[0]
    assert h["champion_micro_f_beta"] == 0.9685
    row = h["candidates"][0]
    assert row["fit_dev_micro_f_beta"] == 0.7123
    assert row["test_micro_f_beta"] == 0.705
    assert row["gate_delta"] == 0.0049 and row["gate_ci_lo"] == -0.0012

    diag = {"train": {"n_images": 10, "summary": {"fp": {"background": 1}},
                      "per_class": {"cat": {"miss_rate": 0.3333333333}},
                      "fp": [{"kind": "background", "score": 0.51234567,
                              "box": [1.23456789, 2, 3, 4]}], "fn": []}}
    c = _compact_diagnosis(diag)["train"]
    assert c["per_class"]["cat"]["miss_rate"] == 0.3333
    assert c["fp_sample"][0]["score"] == 0.5123
    # box coordinates are replaced by integer size (wh) — the planner has no use for coordinates
    assert c["fp_sample"][0]["wh"] == [2, 2] and "box" not in c["fp_sample"][0]


def test_detail_sample_is_representative():
    """Representative sampling rules: round-robin over kind (cause coverage) + round-robin over
    classes within a kind (avoid skew) + significance order (FP score desc / FN area desc) +
    IMAGE_CAP per image + KIND_CAP per kind. The first N by filename carries no information criterion."""
    from agentic_detector_config.diagnose.sample import IMAGE_CAP, KIND_CAP, MAX_DETAIL

    fp = ([{"kind": "background", "image": f"a{i}.png", "category_id": 1,
            "score": 0.1 + i * 0.01, "box": [0, 0, 10, 10]} for i in range(10)]
          + [{"kind": "background", "image": "hot.png", "category_id": 2,
              "score": 0.95 - i * 0.01, "box": [0, 0, 10, 10]} for i in range(5)]
          + [{"kind": "class_confusion", "image": f"c{i}.png", "category_id": 1,
              "score": 0.5, "box": [0, 0, 10, 10]} for i in range(3)])
    fn = [{"kind": "missed", "image": f"m{i}.png", "category_id": 1,
           "box": [0, 0, 10 * (i + 1), 10]} for i in range(6)]
    from agentic_detector_config.diagnose.sample import pick_detail

    d = {"fp_sample": pick_detail(fp), "fn_sample": pick_detail(fn),
         "fp_total": len(fp)}
    fps, fns = d["fp_sample"], d["fn_sample"]
    bg = [r for r in fps if r["kind"] == "background"]
    assert len(bg) <= KIND_CAP
    # selection sees the original filename (it is dropped only when rendering the prompt)
    assert bg[0]["score"] == 0.95 and bg[0]["image"] == "hot.png"   # top significance
    assert {r["category_id"] for r in bg} == {1, 2}                 # class round-robin
    assert sum(1 for r in bg if r["image"] == "hot.png") <= IMAGE_CAP
    assert any(r["kind"] == "class_confusion" for r in fps)         # kind coverage
    assert fns[0]["image"] == "m5.png"                              # top FN area
    assert d["fp_total"] == 18 and len(fps) <= MAX_DETAIL


def test_no_rebuild_nudge_after_identity_refit():
    """Downstream layers are no longer detached (the arm refits them) — the champion loses the
    rescorer only when the refit ends as identity, and re-proposing on the same distribution is
    identity again. Leaving a rebuild nudge would just burn candidate slots."""
    champ = validate_axes({**BASELINE, "crop_classifier": True})
    history = [{"round": 2, "champion": {"axes": champ}, "candidates": [],
                "decision": {"adopted": True, "rescorer_identity": True}}]
    out = deterministic.propose(history, _diag(fn={"missed": 3}))
    assert all(not c["id"].startswith("refit:") for c in out["candidates"])
    # only remedies pointed to by the diagnosis
    assert out["candidates"][0]["id"].startswith("missed:")


def test_prompt_has_no_rebuild_hint(tmp_path):
    """With no drops, the prompt has no rebuild-nudge block either."""
    champ = validate_axes({**BASELINE, "crop_classifier": True})
    history = [{"round": 2, "champion": {"axes": champ, "micro_f_beta": 0.5},
                "candidates": [],
                "decision": {"adopted": True, "rescorer_identity": True}}]
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(history, {})
    assert "lost its rescorer layer" not in seen["prompt"]
    assert "lost its crop-classifier layer" not in seen["prompt"]


def test_text_negative_source_axis():
    """text_negatives takes the mining source as its value — with FP-crop captions alone, most
    phrases hit the safety rules (own parts, same kind, category name) and material came out empty."""
    from agentic_detector_config.planner.axes import AXES
    assert AXES["text_negatives"] == (False, "fp_crop", "gt_crop", "background")
    for src in ("fp_crop", "gt_crop", "background"):
        full = validate_axes({"text_negatives": src})
        assert full["text_negatives"] == src
    validate_axes({"text_negatives": "gt_crop", "text_negative_k": 2})
    with pytest.raises(ValueError, match="text_negative_k"):
        validate_axes({"text_negative_k": 2})            # quantity without a source
    with pytest.raises(ValueError, match="text_negatives"):
        validate_axes({"text_negatives": True})          # the old boolean is invalid


def test_alias_captioner_axis():
    """alias takes the crop->text converter as its value — currently only VLM."""
    from agentic_detector_config.planner.axes import AXES
    assert AXES["alias"] == (False, "vlm")
    assert validate_axes({"alias": "vlm"})["alias"] == "vlm"
    with pytest.raises(ValueError, match="alias"):
        validate_axes({"alias": "pcm"})              # removed captioner
    with pytest.raises(ValueError, match="alias"):
        validate_axes({"alias": True})               # the old boolean is invalid


def test_prompt_lists_tried_combinations(tmp_path):
    """Already-tried combos are packed compactly into the prompt — if the planner forgets combos
    outside the MAX_HISTORY window and re-proposes them, the runner filter silently drops them and
    only burns candidate budget. They are written as increments over the champion to stay short."""
    from agentic_detector_config.planner.llm import LLMPlanner, _tried_lines

    champ = validate_axes({**BASELINE, "rescorer": True})
    history = [
        {"round": 1, "champion": {"axes": dict(BASELINE), "micro_f_beta": 0.5},
         "candidates": [{"id": "a", "axes": validate_axes({"rescorer": True})},
                        {"id": "b", "axes": validate_axes({"tiling": True})}],
         "decision": {"adopted": True, "id": "a"}},
        {"round": 2, "champion": {"axes": champ, "micro_f_beta": 0.6},
         "candidates": [{"id": "c", "axes": validate_axes(
             {"rescorer": True, "alias": "vlm"})}],
         "decision": {"adopted": False}},
    ]
    lines = _tried_lines(history, champ)
    joined = " ".join(lines)
    assert "tiling=True" in joined and "alias=vlm" in joined
    assert "baseline" in joined or "rescorer" in joined
    assert all(len(l) < 200 for l in lines)

    seen = {}

    def chat(messages, **kw):
        seen["p"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(history, {})
    assert "Already tried" in seen["p"] and "alias=vlm" in seen["p"]


def test_sanitize_records_invalid_axes_instead_of_dropping_silently():
    """Candidates failing validation must not be dropped silently — they must land in the dropped
    sink with a reason.

    A silent drop breaks two things at once: round budget shrinks unnoticed (only 2 of 3
    proposals evaluated), and the planner cannot learn from its mistake, so it **re-proposes the
    same candidate next round** (e.g. fn_k attached without fn_arm, lost for several rounds in a row).
    combo_tried already solves the same problem via skipped records (see the no-silent-drop
    comment in runner.py) — validation failures were the only exception.
    """
    raw = [{"id": "bad_fn_k", "axes": {"rescorer": True, "fn_k": 2},
            "rationale": "invalid: fn_k is exemplar=fn_arm only"},
           {"id": "good", "axes": {"alias": "vlm"}, "rationale": "valid"}]
    dropped: list = []
    out = sanitize_candidates(raw, max_candidates=5, dropped=dropped)

    assert [c["id"] for c in out] == ["good"]
    assert len(dropped) == 1
    rec = dropped[0]
    assert rec["id"] == "bad_fn_k"
    assert rec["skipped"] == "invalid_axes"
    assert "fn_k" in rec["note"]           # the ValueError message must be carried for the planner to learn
    assert rec["axes"] == raw[0]["axes"]   # original axes kept — to trace what was wrong


def test_sanitize_records_duplicate_and_overflow_drops():
    """Drops for duplicates or exceeding the quota follow the same principle — kept with distinct reasons."""
    raw = [{"id": "a", "axes": {"alias": "vlm"}},
           {"id": "a_dup", "axes": {"alias": "vlm"}},
           {"id": "b", "axes": {"rescorer": True}},
           {"id": "c", "axes": {"tiling": True}}]
    dropped: list = []
    out = sanitize_candidates(raw, max_candidates=2, dropped=dropped)

    assert [c["id"] for c in out] == ["a", "b"]
    reasons = {r["id"]: r["skipped"] for r in dropped}
    assert reasons == {"a_dup": "duplicate_axes", "c": "max_candidates"}


def test_sanitize_without_sink_keeps_old_signature():
    """Existing call sites that do not pass dropped keep working."""
    out = sanitize_candidates([{"id": "x", "axes": {"visual_negatives": True}},
                               {"id": "y", "axes": {"alias": "vlm"}}],
                              max_candidates=5)
    assert [c["id"] for c in out] == ["y"]


def test_llm_proposal_carries_dropped_and_replays_it(tmp_path):
    """The LLM planner ships dropped candidates in the proposal and stores them in the artifact, so
    **replay restores them identically** — if resume lost the drop records, only resumed runs would
    silently leak budget again (replay guard)."""
    import json as _json

    def chat(messages, **kw):
        return {"candidates": [{"id": "bad", "axes": {"rescorer": True, "fn_k": 2},
                                "rationale": "invalid"},
                               {"id": "ok", "axes": {"alias": "vlm"},
                                "rationale": "valid"}],
                "stop": False}

    planner = LLMPlanner(chat=chat, artifact_dir=tmp_path)
    proposal = planner.propose(history=[], diagnosis={})

    assert [c["id"] for c in proposal["candidates"]] == ["ok"]
    assert [(d["id"], d["skipped"]) for d in proposal["dropped"]] \
        == [("bad", "invalid_axes")]

    stored = _json.loads((tmp_path / "proposal.json").read_text())
    assert [d["id"] for d in stored["dropped"]] == ["bad"]

    def boom(messages, **kw):            # replay does not call the LLM
        raise AssertionError("LLM was called during replay")

    replayed = LLMPlanner(chat=boom, artifact_dir=tmp_path).propose([], {})
    assert [d["id"] for d in replayed["dropped"]] == ["bad"]
