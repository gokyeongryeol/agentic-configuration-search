"""planner — axes contract, deterministic cause -> remedy, LLM hygiene and fallback."""

import json

import pytest

from agentic_configuration_search.planner import deterministic
from agentic_configuration_search.planner.axes import BASELINE, sanitize_candidates, validate_axes
from agentic_configuration_search.planner.llm import LLMPlanner


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

    from agentic_configuration_search.planner.llm import LLMPlanner

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
                      .split("\n\n## Response", 1)[0])["train"]
    assert all("image" not in r for r in diag["fp_sample"])
    # descriptions are folded into the same detection's sample, so no separate crop_notes remain
    bg = next(r for r in diag["fp_sample"] if r["kind"] == "background")
    assert bg["note"] == "blurry fence post" and bg["wh"] == [15, 44]
    assert "crop_notes" not in diag


def test_samples_carry_class_name_not_id():
    """Samples must use the same vocabulary (class names) as per_class — with ids only, the prompt
    has no mapping and individual errors cannot be attributed to a class."""
    from agentic_configuration_search.planner.llm import _compact_diagnosis

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


def test_digest_maps_error_kinds_to_axes_per_round():
    """Replaces the static "cause -> axis" table: one line per cause with this
    round's count (train and holdout separately) and the axes that target it,
    most first. Axis names come from the canonical REMEDY, and counts are not
    repeated per axis."""
    from agentic_configuration_search.planner.axes import AXES
    from agentic_configuration_search.planner.llm import _digest_lines

    diag = {"train": {"summary": {"fn": {"below_threshold": 80, "missed": 2},
                                  "fp": {"class_confusion": 40}}},
            "test": {"summary": {"fn": {"below_threshold": 20}, "fp": {}}}}
    lines = _digest_lines(diag)
    kinds = [l[2:].split(":")[0] for l in lines]
    assert kinds == ["below_threshold", "class_confusion", "missed"]   # most first
    bt = lines[0]
    assert "below_threshold: 100 (train 80, held-out 20)" in bt    # sum and split
    axes = [a.strip() for a in bt.split("->")[1].split(",")]
    assert axes and all(a in AXES for a in axes)                  # no phantom axes
    assert "rescorer" in axes
    # axes dedicated to unobserved causes are omitted (tiling <- missed_small)
    assert not any("tiling" in l for l in lines)
    assert _digest_lines({}) == []
    # disabled axes are left out — they cannot be proposed
    assert "rescorer" not in _digest_lines(diag, disabled=("rescorer",))[0]


def test_digest_counts_background_for_crop_classifier_and_unaddressable_kinds():
    from agentic_configuration_search.planner.llm import _digest_lines

    lines = _digest_lines({"train": {"summary": {"fp": {"background": 7,
                                                        "duplicate": 3}}}})
    bg = next(l for l in lines if l.startswith("- background:"))
    assert "background: 7 (train 7)" in bg and "crop_classifier" in bg
    dup = next(l for l in lines if l.startswith("- duplicate:"))
    assert "no axis" in dup


def test_dead_axes_survive_the_history_window(tmp_path):
    """Axes with empty material are collected over the full history — once they slide out of the
    history window the model proposes the same axis again (re-proposals did repeat in practice)."""
    from agentic_configuration_search.planner.llm import MAX_HISTORY, LLMPlanner, _dead_axis_lines

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
    assert "came out empty, or too small to train on, on this" in seen["prompt"]
    assert "- alias (values tried: vlm)" in seen["prompt"]


def test_history_window_zero_means_no_history():
    """history[-0:] is the whole list — if the prompt grows back the moment the budget guard shrinks
    the window to 0, the last line of defense becomes the worst case."""
    from agentic_configuration_search.planner.llm import _compact_history

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


def test_background_proposes_crop_classifier():
    """The client drops detections reclassified as background (0) — the classifier
    is a background-FP filter, not only a class fixer. On a single-class dataset
    class_confusion is zero by definition, so without the background remedy the
    diagnosis could not reach this axis."""
    assert ("crop_classifier", True) in deterministic.REMEDY["background"]
    out = deterministic.propose([], _diag(fp={"background": 9}))
    assert any(c["axes"]["crop_classifier"] for c in out["candidates"])


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
    # junction between the diagnosis digest and the response section — no empty
    # block in between
    assert "and the samples.\n- (no errors observed)\n\nError detail." \
        in seen["prompt"]
    assert "\n{}\n\n## Response\nPropose up to" in seen["prompt"]


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
    from agentic_configuration_search.planner.llm import MAX_NOTE, _compact_history

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
    from agentic_configuration_search.planner.llm import _compact_diagnosis, _compact_history

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
    from agentic_configuration_search.diagnose.sample import IMAGE_CAP, KIND_CAP, MAX_DETAIL

    fp = ([{"kind": "background", "image": f"a{i}.png", "category_id": 1,
            "score": 0.1 + i * 0.01, "box": [0, 0, 10, 10]} for i in range(10)]
          + [{"kind": "background", "image": "hot.png", "category_id": 2,
              "score": 0.95 - i * 0.01, "box": [0, 0, 10, 10]} for i in range(5)]
          + [{"kind": "class_confusion", "image": f"c{i}.png", "category_id": 1,
              "score": 0.5, "box": [0, 0, 10, 10]} for i in range(3)])
    fn = [{"kind": "missed", "image": f"m{i}.png", "category_id": 1,
           "box": [0, 0, 10 * (i + 1), 10]} for i in range(6)]
    from agentic_configuration_search.diagnose.sample import pick_detail

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
    assert all(not c["rationale"].startswith("refit") for c in out["candidates"])
    # only remedies pointed to by the diagnosis
    assert out["candidates"][0]["rationale"].endswith("for missed errors")


def test_deterministic_ids_are_folder_safe():
    """Candidate ids become material and arm folder names — cand_<i> like the LLM planner's,
    so the folder name is the id as is and an id unique within the round (runner._round_ids)
    is a unique folder."""
    import re
    from agentic_configuration_search.loop.runner import _slug
    champ = validate_axes(dict(BASELINE))
    history = [{"round": 0, "champion": {"axes": champ}, "candidates": [],
                "decision": {"adopted": True}}]
    out = deterministic.propose(history, _diag(fn={"missed": 3, "missed_small": 2},
                                               fp={"background": 4,
                                                   "class_confusion": 2}))
    ids = [c["id"] for c in out["candidates"]]
    assert ids and ids == [f"cand_{i}" for i in range(len(ids))]   # same format as the LLM planner
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", i) and _slug(i) == i for i in ids)


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
    from agentic_configuration_search.planner.axes import AXES
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
    from agentic_configuration_search.planner.axes import AXES
    assert AXES["alias"] == (False, "vlm")
    assert validate_axes({"alias": "vlm"})["alias"] == "vlm"
    with pytest.raises(ValueError, match="alias"):
        validate_axes({"alias": "pcm"})              # removed captioner
    with pytest.raises(ValueError, match="alias"):
        validate_axes({"alias": True})               # the old boolean is invalid


def test_prompt_lists_tried_combinations(tmp_path):
    """Tried combos go into the prompt as increments over the champion — if the
    planner forgets one and re-proposes it, the runner filter drops it and only
    burns candidate budget. All combos under the current champion are carried;
    earlier ones only if they do not turn off a champion axis (those are rarely
    proposed again and are noise — the filter still blocks them all)."""
    from agentic_configuration_search.planner.ledger import tried_under_champion
    from agentic_configuration_search.planner.llm import LLMPlanner

    champ = validate_axes({**BASELINE, "rescorer": True})
    history = [
        {"round": 0, "champion": {"axes": dict(BASELINE), "round": 0,
                                  "micro_f_beta": 0.4}, "candidates": []},
        {"round": 1, "champion": {"axes": champ, "round": 1, "micro_f_beta": 0.5},
         "candidates": [{"id": "a", "axes": champ, "adopted": True},
                        {"id": "b", "axes": validate_axes({"tiling": 2})},
                        {"id": "t", "axes": validate_axes({"text_negatives":
                                                           "fp_crop"})}],
         "decision": {"adopted": True, "id": "a"}},
        {"round": 2, "champion": {"axes": champ, "round": 1, "micro_f_beta": 0.5},
         "candidates": [{"id": "c", "axes": validate_axes(
             {"rescorer": True, "alias": "vlm"})}],
         "decision": {"adopted": False}},
    ]
    lines = tried_under_champion(history)
    assert lines == ["alias=vlm"]           # the two old combos turn rescorer off
    assert all(len(l) < 200 for l in lines)

    seen = {}

    def chat(messages, **kw):
        seen["p"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(history, {})
    assert "Combos already tried" in seen["p"] and "{alias=vlm}" in seen["p"]


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
           {"id": "c", "axes": {"tiling": 2}}]
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


# ------------------------------------------------- single class — gt_crop

from agentic_configuration_search.planner.axes import DatasetFacts  # noqa: E402

ONE_CLASS = DatasetFacts(n_classes=1)

def test_gt_crop_invalid_for_single_class():
    """gt_crop is sourced from **other classes'** GT crops — on a single class the
    pool is empty by definition and the candidate only uses up a slot."""
    axes = {"text_negatives": "gt_crop"}
    with pytest.raises(ValueError, match="gt_crop"):
        validate_axes(axes, facts=ONE_CLASS)
    assert validate_axes(axes, facts=DatasetFacts(n_classes=2))["text_negatives"] == "gt_crop"
    assert validate_axes(axes)["text_negatives"] == "gt_crop"   # unspecified = no constraint
    # the other sources stay valid on a single class
    for src in ("fp_crop", "background"):
        validate_axes({"text_negatives": src}, facts=ONE_CLASS)


def test_sanitize_records_gt_crop_drop_for_single_class():
    dropped: list = []
    out = sanitize_candidates(
        [{"id": "g", "axes": {"text_negatives": "gt_crop"}},
         {"id": "f", "axes": {"text_negatives": "fp_crop"}}],
        max_candidates=3, dropped=dropped, facts=ONE_CLASS)
    assert [c["id"] for c in out] == ["f"]
    assert dropped[0]["id"] == "g" and dropped[0]["skipped"] == "invalid_axes"
    assert "gt_crop" in dropped[0]["note"]


def test_deterministic_skips_gt_crop_for_single_class():
    diag = _diag(fp={"class_confusion": 9})
    multi = deterministic.propose([], diag, max_candidates=10)
    assert any(c["axes"]["text_negatives"] == "gt_crop" for c in multi["candidates"])
    single = deterministic.propose([], diag, max_candidates=10, facts=ONE_CLASS)
    assert single["candidates"]
    assert all(c["axes"]["text_negatives"] != "gt_crop"
               for c in single["candidates"])


def test_llm_prompt_hides_gt_crop_for_single_class(tmp_path):
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [
            {"id": "g", "axes": {"text_negatives": "gt_crop"}, "rationale": "x"},
            {"id": "t", "axes": {"tiling": 2}, "rationale": "y"}], "stop": False}

    out = LLMPlanner(chat=chat, artifact_dir=tmp_path / "one",
                     facts=ONE_CLASS).propose([], {})
    assert "gt_crop" not in seen["prompt"]
    assert [c["id"] for c in out["candidates"]] == ["t"]
    assert out["dropped"][0]["skipped"] == "invalid_axes"

    LLMPlanner(chat=chat, artifact_dir=tmp_path / "two",
               facts=DatasetFacts(n_classes=2)).propose([], {})
    assert "gt_crop" in seen["prompt"]     # the multi-class prompt is unchanged


# ------------------------------------------------- tiling = grid g axis

def test_tiling_axis_is_grid_value():
    """The tiling value is grid g (g×g tiles, zoom (g-1)(1-o)+1, 1+g² passes) — the
    planner picks the zoom directly and raises it when it falls short. The code sets the range."""
    from agentic_configuration_search.planner.axes import AXES, TILING_GRIDS
    assert AXES["tiling"] == (False, *TILING_GRIDS)
    for g in TILING_GRIDS:
        assert validate_axes({"tiling": g})["tiling"] == g
    for bad in (True, 1, 7, 2.0, "2"):
        with pytest.raises(ValueError, match="tiling"):
            validate_axes({"tiling": bad})


def test_deterministic_tiling_steps_up_from_champion():
    """The missed_small remedy only moves to grids larger than the champion's —
    going down would undo an already-adopted zoom, so it is not a remedy."""
    diag = _diag(fn={"missed_small": 9})
    base = deterministic.propose([], diag, max_candidates=10)
    assert [c["axes"]["tiling"] for c in base["candidates"]][:1] == [2]

    champ = validate_axes({"tiling": 3})
    history = [{"round": 1, "champion": {"axes": champ}, "candidates": []}]
    out = deterministic.propose(history, diag, max_candidates=10)
    grids = [c["axes"]["tiling"] for c in out["candidates"]]
    assert grids and grids[0] == 4
    assert all(g > 3 for g in grids)


def test_llm_prompt_explains_tiling_grid(tmp_path):
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose([], {})
    assert "tiling: one of [False, 2, 3, 4, 5, 6]" in seen["prompt"]
    assert "1 + g^2" in seen["prompt"]          # cost
    assert "tiling_used" in seen["prompt"]      # how to read the shortfall signal


def test_compact_history_carries_tiling_used():
    from agentic_configuration_search.planner.llm import _compact_history

    champ = validate_axes({})
    hist = [{"round": 1, "champion": {"axes": champ, "micro_f_beta": 0.5},
             "candidates": [{"id": "t", "axes": validate_axes({"tiling": 3}),
                             "adopted": False, "tiling_used": False}],
             "decision": {"adopted": False}}]
    row = _compact_history(hist)[0]["candidates"][0]
    assert row["tiling_used"] is False


# ------------------------------------------------- lora tripwire, decided up front
# If fit_train has fewer than lora.min_images images, a lora candidate is rejected by
# lora_gate without training — an outcome the data already decides, so it is
# blocked before it uses a candidate slot.

SMALL = DatasetFacts(n_classes=3, n_fit_train=8, lora_min_images=20)


def test_lora_invalid_when_fit_train_below_min_images():
    with pytest.raises(ValueError, match="lora.*8.*20"):
        validate_axes({"lora": True}, facts=SMALL)
    ok = DatasetFacts(n_classes=3, n_fit_train=20, lora_min_images=20)
    assert validate_axes({"lora": True}, facts=ok)["lora"] is True
    assert validate_axes({"lora": True})["lora"] is True       # no facts = no constraint


def test_axis_values_hide_untrainable_lora():
    from agentic_configuration_search.planner.axes import axis_values
    assert axis_values(SMALL)["lora"] == (False,)
    assert axis_values(DatasetFacts())["lora"] == (False, True)


def test_deterministic_skips_untrainable_lora():
    diag = _diag(fn={"missed": 9})
    assert any(c["axes"]["lora"] for c in
               deterministic.propose([], diag, max_candidates=10)["candidates"])
    out = deterministic.propose([], diag, max_candidates=10, facts=SMALL)
    assert out["candidates"] and not any(c["axes"]["lora"] for c in out["candidates"])


def test_llm_prompt_and_response_drop_untrainable_lora(tmp_path):
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[-1]["content"]
        return {"candidates": [
            {"id": "l", "axes": {"lora": True}, "rationale": "x"},
            {"id": "t", "axes": {"tiling": 2}, "rationale": "y"}], "stop": False}

    out = LLMPlanner(chat=chat, artifact_dir=tmp_path, facts=SMALL).propose([], {})
    assert "- lora: one of [False]" in seen["prompt"]
    assert [c["id"] for c in out["candidates"]] == ["t"]
    assert out["dropped"][0]["id"] == "l"
    assert out["dropped"][0]["skipped"] == "invalid_axes"


# ------------------------- carry-over of gate passers that lost to the winner

def _carry_history(lost_axes, *, winner_axes, base_axes=None):
    base = validate_axes(base_axes or {})
    win = validate_axes(winner_axes)
    return [{"round": 0, "champion": {"axes": base}, "candidates": []},
            {"round": 1, "champion": {"axes": win},
             "candidates": [{"id": "w", "axes": win, "adopted": True},
                            {"id": "l", "axes": validate_axes(lost_axes),
                             "adopted": False, "gate_passed_but_lost": True},
                            {"id": "x", "axes": validate_axes({"alias": "vlm"}),
                             "adopted": False}]}]


def test_carryover_puts_lost_axis_on_new_champion():
    """A candidate that passed the gate but lost to the winner comes back next
    round as 'new champion + the axes it changed' — a single increment over the
    new champion, so attribution stays clean."""
    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    out = deterministic.carryover(hist)
    assert [c["id"] for c in out] == ["carry_l"]
    assert out[0]["axes"] == validate_axes({"exemplar": "base", "rescorer": True})
    assert out[0]["source"] == "carryover"


def test_carryover_skips_invalid_identical_or_disabled():
    from agentic_configuration_search.planner.axes import DatasetFacts
    # nothing to carry if it equals the new champion
    assert deterministic.carryover(
        _carry_history({"exemplar": "base"}, winner_axes={"exemplar": "base"})) == []
    # combinations the data blocks (facts) are not carried
    small = DatasetFacts(n_classes=2, n_fit_train=5, lora_min_images=20)
    assert deterministic.carryover(
        _carry_history({"lora": True}, winner_axes={"exemplar": "base"}),
        facts=small) == []
    # nor are ones that turn on a disabled axis
    assert deterministic.carryover(
        _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"}),
        disabled_axes=("rescorer",)) == []
    assert deterministic.carryover([]) == []


def test_carryover_uses_requested_axes_not_relabel():
    """A candidate whose label was corrected for an unused axis is carried with
    its requested increment — reading the corrected label (exemplar=none) as the
    increment would turn the new champion's exemplar off."""
    champ0 = validate_axes({"exemplar": "base"})
    win = validate_axes({"exemplar": "base", "tiling": 2})
    requested = validate_axes({"exemplar": "base", "text_negatives": "fp_crop"})
    measured = validate_axes({"text_negatives": "fp_crop"})
    hist = [{"round": 1, "champion": {"axes": champ0}, "candidates": []},
            {"round": 2, "champion": {"axes": win},
             "candidates": [{"id": "w", "axes": win, "adopted": True},
                            {"id": "l", "axes": measured, "requested_axes": requested,
                             "unused_axes": ["exemplar"], "adopted": False,
                             "gate_passed_but_lost": True}]}]
    out = deterministic.carryover(hist)
    assert [c["axes"] for c in out] == [
        validate_axes({"exemplar": "base", "tiling": 2, "text_negatives": "fp_crop"})]


def test_carryover_resets_dependent_axes_on_new_champion():
    """Carried increments are applied with merge_onto too — if the new champion
    turned on an axis tied to exemplar (visual_negatives) and the losing
    candidate turned exemplar off, an overwrite would be a contract-violating
    combination and the gate passer would vanish without a record."""
    prev = validate_axes({"exemplar": "base"})
    win = validate_axes({"exemplar": "base", "visual_negatives": True})
    lost = validate_axes({"alias": "vlm"})          # turned exemplar off and alias on
    hist = [{"round": 1, "champion": {"axes": prev}, "candidates": []},
            {"round": 2, "champion": {"axes": win},
             "candidates": [{"id": "w", "axes": win, "adopted": True},
                            {"id": "l", "axes": lost, "adopted": False,
                             "gate_passed_but_lost": True}]}]
    out = deterministic.carryover(hist)
    assert [c["axes"] for c in out] == [validate_axes({"alias": "vlm"})]


def test_llm_prompt_lists_carried_candidates_only_when_present(tmp_path):
    seen = {}

    def chat(messages, **kw):
        seen["p"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    LLMPlanner(chat=chat, artifact_dir=tmp_path / "a").propose(hist, {})
    assert "Re-queued automatically" in seen["p"]
    assert "rescorer=True" in seen["p"].split("Re-queued automatically", 1)[1][:400]
    LLMPlanner(chat=chat, artifact_dir=tmp_path / "b").propose(hist[:1], {})
    assert "Re-queued automatically" not in seen["p"]


# ------------------------------------- candidate axes are relative to the champion

def test_sanitize_merges_candidate_over_champion():
    """The planner lists only the axes it changes — omitted axes keep the
    champion's value. They used to be filled from BASELINE, silently turning the
    champion's axes (rescorer, lora, ...) off."""
    champ = validate_axes({"rescorer": True, "lora": True})
    out = sanitize_candidates([{"id": "t", "axes": {"tiling": 2}},
                               {"id": "off", "axes": {"rescorer": False}}],
                              max_candidates=3, base=champ)
    assert out[0]["axes"] == validate_axes({"rescorer": True, "lora": True, "tiling": 2})
    assert out[1]["axes"]["rescorer"] is False and out[1]["axes"]["lora"] is True


def test_llm_candidates_keep_champion_axes(tmp_path):
    champ = validate_axes({"rescorer": True, "crop_classifier": True})
    hist = [{"round": 0, "champion": {"axes": champ}, "candidates": []}]

    def chat(messages, **kw):
        return {"candidates": [{"id": "t", "axes": {"tiling": 2}, "rationale": "x"}],
                "stop": False}

    out = LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(hist, {})
    assert out["candidates"][0]["axes"] == validate_axes(
        {"rescorer": True, "crop_classifier": True, "tiling": 2})


def test_compact_history_delta_is_against_pre_round_champion():
    """History axes_delta is against the champion at the **start** of the round —
    comparing with the champion after it mixes in the axes adopted that round as
    if they were turned off."""
    from agentic_configuration_search.planner.llm import _compact_history

    base = validate_axes({})
    after = validate_axes({"rescorer": True})
    hist = [{"round": 0, "champion": {"axes": base}, "candidates": []},
            {"round": 1, "champion": {"axes": after},
             "candidates": [{"id": "r", "axes": after, "adopted": True},
                            {"id": "e", "axes": validate_axes({"exemplar": "base"}),
                             "adopted": False}]}]
    rows = {c["id"]: c for c in _compact_history(hist)[1]["candidates"]}
    assert rows["e"]["axes_delta"] == {"exemplar": "base"}
    assert rows["r"]["axes_delta"] == {"rescorer": True}


def test_merge_resets_dependents_of_a_changed_feature_axis():
    """Turning a feature axis off or changing it resets the dependent and count
    axes the candidate does not name — if the champion's stayed, the validation
    rules would drop the whole candidate."""
    champ = validate_axes({"exemplar": "fn_arm", "exemplar_k": 4, "fn_k": 2,
                           "visual_negatives": True, "visual_negative_k": 2,
                           "text_negatives": "fp_crop", "text_negative_k": 2,
                           "rescorer": True})
    def one(ax):
        out = sanitize_candidates([{"id": "c", "axes": ax}], max_candidates=1, base=champ)
        assert out, f"{ax} was dropped"
        return out[0]["axes"]
    off = one({"exemplar": "none"})
    assert off["exemplar_k"] is None and off["fn_k"] is None
    assert off["visual_negatives"] is False and off["visual_negative_k"] is None
    assert off["rescorer"] is True and off["text_negatives"] == "fp_crop"
    base = one({"exemplar": "base"})
    assert base["fn_k"] is None and base["exemplar_k"] == 4 and base["visual_negatives"]
    assert one({"visual_negatives": False})["visual_negative_k"] is None
    assert one({"text_negatives": False})["text_negative_k"] is None
    # an explicit value is used as given
    assert one({"exemplar": "base", "exemplar_k": 8})["exemplar_k"] == 8


def test_candidate_identical_to_champion_is_dropped():
    champ = validate_axes({"rescorer": True})
    dropped: list = []
    out = sanitize_candidates([{"id": "same", "axes": {}},
                               {"id": "same2", "axes": {"rescorer": True}}],
                              max_candidates=3, dropped=dropped, base=champ)
    assert out == [] and {d["skipped"] for d in dropped} == {"no_change"}


def test_legacy_invalid_base_falls_back_to_absolute_axes():
    """A champion that does not validate under the current contract (an old run's
    tiling: True, say) as base does not kill the round — candidate axes are read
    as-is, without the increment."""
    legacy = {**BASELINE, "tiling": True}
    dropped: list = []
    out = sanitize_candidates([{"id": "c", "axes": {"tiling": 2}}],
                              max_candidates=3, dropped=dropped, base=legacy)
    assert [c["axes"]["tiling"] for c in out] == [2] and dropped == []


def test_dropped_rows_record_merged_axes():
    """A candidate dropped after merging onto base is recorded with the merged
    axes — left as the increment, tried_combos would fill it from BASELINE and
    block a combination that was never tried."""
    champ = validate_axes({"exemplar": "base"})
    dropped: list = []
    sanitize_candidates([{"id": "a", "axes": {"tiling": 2}},
                         {"id": "b", "axes": {"tiling": 2}}],
                        max_candidates=3, dropped=dropped, base=champ)
    assert [d["skipped"] for d in dropped] == ["duplicate_axes"]
    assert dropped[0]["axes"] == validate_axes({"exemplar": "base", "tiling": 2})


def test_planner_bounds_its_output_by_schema_and_cap(tmp_path):
    """The schema bounds the longest response (rationale and id length, number of
    candidates) and the output cap sits above it — at a 1024-token cap a
    rationale that kept running truncated the response and the planner silently
    fell back to deterministic."""
    from agentic_configuration_search.planner import llm as L

    seen = {}

    def chat(messages, **kw):
        seen.update(kw)
        return {"candidates": [], "stop": True}

    from agentic_configuration_search.config import PLANNER_MAX_CANDIDATES as K

    LLMPlanner(chat=chat, artifact_dir=tmp_path, max_candidates=K).propose(
        [{"round": 0, "champion": {"axes": validate_axes({})}, "candidates": []}], {})
    assert seen["max_tokens"] == L.OUTPUT_MAX_TOKENS
    cands = seen["schema"]["schema"]["properties"]["candidates"]
    assert cands["maxItems"] == K
    props = cands["items"]["properties"]
    assert props["rationale"]["maxLength"] == L.RATIONALE_MAX_CHARS
    assert props["id"]["maxLength"] <= 80
    # the longest response the schema allows fits under the cap — at a
    # conservative 2.0 chars/token, with
    # 600 characters per candidate for axes (12 enumerated keys) and 100 for
    # the two number fields (expected_gain, cost_min) and key names. differs is
    # the longest string field, so it is counted too
    assert props["differs"]["maxLength"] <= 300
    # counted with the most candidates config allows (planner.max_candidates ≤ K)
    worst = K * (L.RATIONALE_MAX_CHARS + props["id"]["maxLength"]
                 + props["differs"]["maxLength"] + 600 + 100)
    assert worst / 2.0 < L.OUTPUT_MAX_TOKENS


def test_token_budget_is_language_aware_and_fits_context():
    """The input budget is in estimated tokens — a character budget plus the
    output share can overflow the server context on CJK-heavy prompts. ASCII
    counts at 1.85 chars/token, other characters generously at 1.5 tokens each."""
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop.runner import _token_budget
    from agentic_configuration_search.planner import llm as L

    est = lambda text: L._est_tokens([{"role": "user", "content": text}])  # noqa: E731
    assert est("a" * 1850) == 1000
    assert est("가" * 100) == 150
    cfg = config_from_dict({"dataset": "x"})
    ctx = cfg.planner.launch.max_model_len
    assert _token_budget(cfg) == ctx - L.OUTPUT_MAX_TOKENS - L.CONTEXT_MARGIN
    assert _token_budget(cfg) == L.DEFAULT_TOKEN_BUDGET
    # a long, mostly-ASCII prompt (about 21,000 characters) fits without shrinking history
    assert est("a" * 21_003) < _token_budget(cfg)


# ------------------------------- candidates equal to carried ones take no slot

def _scripted_chat(*responses):
    """Returns the next response on each planner call."""
    calls = []

    def chat(messages, **kw):
        calls.append(messages)
        return responses[len(calls) - 1]
    return chat, calls


def test_candidates_repeating_carried_ones_are_dropped_and_slots_refilled(tmp_path):
    """When the planner repeats carried candidates, they are dropped as
    duplicates and the freed slots are asked for once more — even with the
    prompt saying not to spend slots on them, the planner can fill most of its
    slots with carried candidates."""
    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    chat, calls = _scripted_chat(
        {"candidates": [{"id": "dup", "axes": {"rescorer": True}, "rationale": "r"},
                        {"id": "new", "axes": {"alias": "vlm"}, "rationale": "r"}],
         "stop": False},
        {"candidates": [{"id": "more", "axes": {"tiling": 2}, "rationale": "r"},
                        {"id": "dup2", "axes": {"rescorer": True}, "rationale": "r"}],
         "stop": False})
    out = LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(hist, {})
    assert [c["id"] for c in out["candidates"]] == ["new", "more"]
    assert {d["id"]: d["skipped"] for d in out["dropped"]} == {
        "dup": "carried_duplicate", "dup2": "carried_duplicate"}
    assert len(calls) == 2                               # asked again only once
    follow = calls[1][-1]["content"]
    assert "rescorer=True" in follow and "up to 2" in follow
    assert "Kept from your answer: alias=vlm" in follow     # so kept ones are not repeated
    assert calls[1][-2]["role"] == "assistant"           # continues the conversation
    stored = json.loads((tmp_path / "proposal.json").read_text())
    assert stored["followup"]["response"]["candidates"][0]["id"] == "more"


def test_no_refill_without_carried_duplicates(tmp_path):
    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    chat, calls = _scripted_chat(
        {"candidates": [{"id": "new", "axes": {"alias": "vlm"}, "rationale": "r"}],
         "stop": False})
    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(hist, {})
    assert len(calls) == 1


def test_only_carried_duplicates_do_not_trigger_deterministic_fallback(tmp_path):
    """If every candidate equals a carried one and the follow-up fails too, the
    proposal is empty — the runner runs the carried candidates anyway, so the
    deterministic fallback does not fill it."""
    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})

    def chat(messages, **kw):
        if len(messages) > 2:
            raise RuntimeError("follow-up failed")
        return {"candidates": [{"id": "dup", "axes": {"rescorer": True},
                                "rationale": "r"}], "stop": False}

    out = LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(hist, {})
    assert out["candidates"] == [] and out["stop"] is False
    stored = json.loads((tmp_path / "proposal.json").read_text())
    assert stored["fallback"] is None and "follow-up failed" in stored["followup"]["error"]


def test_refill_ids_never_collide_with_dropped_or_carried_ids(tmp_path):
    """Ids of refilled candidates collide with neither kept nor dropped nor
    carried candidates — two rows with one id in a round would make the next
    round's history attach the evaluation score to the dropped row, and the
    planner would read the dropped combination as evaluated at that score."""
    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    chat, _ = _scripted_chat(
        {"candidates": [{"id": "a", "axes": {"alias": "vlm"}, "rationale": "r"},
                        {"id": "a_2", "axes": {"text_negatives": "fp_crop"}, "rationale": "r"},
                        {"id": "c2", "axes": {"rescorer": True}, "rationale": "r"}],
         "stop": False},
        {"candidates": [{"id": "c2", "axes": {"tiling": 2}, "rationale": "r"}],
         "stop": False})
    out = LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(hist, {})
    ids = [c["id"] for c in out["candidates"]] + [d["id"] for d in out["dropped"]]
    assert len(ids) == len(set(ids)), ids
    assert "carry_l" not in [c["id"] for c in out["candidates"]]


def test_refill_renames_until_free():
    from agentic_configuration_search.planner.llm import _free_id
    assert _free_id("a", {"b"}) == "a"
    assert _free_id("a", {"a", "a_2"}) == "a_3"


def test_stop_with_only_carried_duplicates_keeps_running(tmp_path):
    """stop=True with only carried candidates is not a stop — proposing them
    means looking at the carry-overs further, and before this change those
    candidates stayed, so it was not a stop either."""
    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    chat, calls = _scripted_chat(
        {"candidates": [{"id": "dup", "axes": {"rescorer": True}, "rationale": "r"}],
         "stop": True})
    out = LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(hist, {})
    assert out["candidates"] == [] and out["stop"] is False
    assert len(calls) == 1                               # no follow-up on stop


def test_prompt_explains_carried_duplicate():
    from agentic_configuration_search.planner.llm import PROMPT
    assert "carried_duplicate" in PROMPT


def test_refill_shrinks_history_to_fit_instead_of_skipping(tmp_path):
    """The follow-up conversation is longer by the first response and the
    follow-up message — if the first prompt nearly filled the budget, shrink
    the history window to fit instead of skipping the follow-up."""
    from agentic_configuration_search.planner.llm import MAX_HISTORY, _est_tokens

    filler = [{"round": i, "champion": {"axes": validate_axes({})},   # the carry's previous champion
               "candidates": [{"id": f"f{i}_{j}", "axes": validate_axes(
                   {"exemplar": "base", "exemplar_k": 2}),
                   "skipped": "no_material", "note": "x" * 80} for j in range(4)]}
              for i in range(1, 9)]
    carry = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    hist = [carry[0]] + filler + [{**carry[1], "round": 9}]
    first = {"candidates": [{"id": "dup", "axes": {"rescorer": True}, "rationale": "r"}],
             "stop": False}
    seen = []

    def probe(messages, **kw):
        seen.append(messages)
        return first
    LLMPlanner(chat=probe, artifact_dir=tmp_path / "p").propose(hist, {})
    budget = _est_tokens(seen[0]) + 20          # the first prompt fits without shrinking

    chat, calls = _scripted_chat(
        first, {"candidates": [{"id": "n", "axes": {"tiling": 2}, "rationale": "r"}],
                "stop": False})
    out = LLMPlanner(chat=chat, artifact_dir=tmp_path / "a",
                     token_budget=budget).propose(hist, {})
    assert len(calls) == 2, "the follow-up was skipped"
    assert _est_tokens(calls[1]) <= budget
    assert [c["id"] for c in out["candidates"]] == ["n"]
    stored = json.loads((tmp_path / "a" / "proposal.json").read_text())
    assert stored["followup"]["window"] < MAX_HISTORY


def test_carried_duplicate_note_does_not_claim_evaluation():
    """A carry-over can still be filtered in the runner — the note must not
    claim it will be evaluated."""
    from agentic_configuration_search.planner.llm import PROMPT, _drop_carried
    dropped: list = []
    _drop_carried([{"id": "x", "axes": validate_axes({})}],
                  {tuple(sorted(validate_axes({}).items()))}, dropped)
    assert "evaluated" not in dropped[0]["note"] and "carry_" in dropped[0]["note"]
    assert "evaluated anyway" not in PROMPT


def test_tried_combos_name_the_champion_itself_first(tmp_path):
    """The current champion's own combination is not a candidate — it is named as
    the first item of the tried-combination list (an empty increment is dropped
    as no_change). Carried even without history."""
    from agentic_configuration_search.planner.axes import validate_axes
    from agentic_configuration_search.planner.llm import LLMPlanner

    champ = validate_axes({"exemplar": "fn_arm", "exemplar_k": 8})
    history = [
        {"round": 0, "champion": {"axes": champ, "round": 0, "micro_f_beta": 0.5},
         "candidates": [], "decision": {"adopted": True, "id": "baseline"}},
        {"round": 1, "champion": {"axes": champ, "round": 0, "micro_f_beta": 0.5},
         "candidates": [{"id": "k4", "axes": validate_axes(
             {"exemplar": "fn_arm", "exemplar_k": 4})}],
         "decision": {"adopted": False}},
    ]
    seen = {}

    def chat(messages, **kw):
        seen["p"] = messages[-1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(history, {})
    line = next(l for l in seen["p"].splitlines()
                if l.startswith("Combos already tried"))
    items = line.split("): ", 1)[1]
    assert items.startswith("{} = the current champion itself")
    assert "no_change" in items and items.index("{exemplar_k=4}") > 0


def test_refill_respects_disabled_axes(tmp_path):
    """Refilled candidates go through the same per-run bans as the first
    response — a disabled axis cannot slip in through the follow-up."""
    hist = _carry_history({"rescorer": True}, winner_axes={"exemplar": "base"})
    chat, calls = _scripted_chat(
        {"candidates": [{"id": "dup", "axes": {"rescorer": True}, "rationale": "r"}],
         "stop": False},
        {"candidates": [{"id": "t", "axes": {"tiling": 2}, "rationale": "r"},
                        {"id": "v", "axes": {"alias": "vlm"}, "rationale": "r"}],
         "stop": False})
    out = LLMPlanner(chat=chat, artifact_dir=tmp_path,
                     disabled_axes=("tiling",)).propose(hist, {})
    assert len(calls) == 2
    assert [c["id"] for c in out["candidates"]] == ["v"]
    assert {d["id"]: d["skipped"] for d in out["dropped"]}["t"] == "disabled_axes"


def test_a_transient_error_gets_one_retry():
    """skipped=error (OOM, worker failure) is not a measurement — the same
    combination is allowed once more and blocked after the second failure (so a
    deterministic failure does not repeat every round)."""
    from agentic_configuration_search.planner.axes import validate_axes
    from agentic_configuration_search.planner.deterministic import tried_combos
    axes = validate_axes({"lora": True})
    key = tuple(sorted(axes.items()))
    once = [{"candidates": [{"id": "a", "axes": axes, "requested_axes": axes,
                             "skipped": "error"}]}]
    assert key not in tried_combos(once)
    twice = once + [{"candidates": [{"id": "b", "axes": axes, "skipped": "error"}]}]
    assert key in tried_combos(twice)
    measured = [{"candidates": [{"id": "a", "axes": axes}]}]
    assert key in tried_combos(measured)
