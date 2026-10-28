"""search.disabled_axes — **blocks proposals** on axes that cannot be wired in a given
setup (e.g. lora, crop_classifier), rather than failing at run time. Pins all three
planner surfaces (sanitize, deterministic, LLM prompt) and Loop validation."""

import pytest

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.planner import deterministic
from agentic_configuration_search.planner.axes import BASELINE, sanitize_candidates, uses_disabled_axes
from agentic_configuration_search.planner.llm import LLMPlanner

DISABLED = ("lora", "crop_classifier")


def _diag(kind_counts: dict):
    return {"train": {"summary": {"fn": kind_counts, "fp": {}}}, "test": {}}


def test_uses_disabled_axes_only_flags_non_baseline():
    assert not uses_disabled_axes(dict(BASELINE), DISABLED)
    assert uses_disabled_axes({**BASELINE, "lora": True}, DISABLED)
    assert not uses_disabled_axes({**BASELINE, "alias": True}, DISABLED)


def test_sanitize_drops_disabled_axis_candidates():
    raw = [{"id": "a", "axes": {"lora": True}, "rationale": ""},
           {"id": "b", "axes": {"alias": "vlm"}, "rationale": ""}]
    out = sanitize_candidates(raw, max_candidates=5, disabled_axes=DISABLED)
    assert [c["id"] for c in out] == ["b"]


def test_deterministic_skips_disabled_remedies():
    # remedies for missed go exemplar→alias→lora — a disabled lora drops out of proposals
    prop = deterministic.propose([], _diag({"missed": 5}), max_candidates=10,
                                 disabled_axes=DISABLED)
    axes_on = {k for c in prop["candidates"] for k, v in c["axes"].items()
               if v != BASELINE[k]}
    assert "lora" not in axes_on and "crop_classifier" not in axes_on
    assert prop["candidates"]                      # other remedies survive


def test_deterministic_skips_disabled_refit():
    history = [{"round": 1,
                "champion": {"axes": dict(BASELINE)},
                "decision": {"adopted": True, "classifier_dropped": True},
                "candidates": []}]
    prop = deterministic.propose(history, _diag({}), disabled_axes=DISABLED)
    assert all(not c["id"].startswith("refit:crop_classifier")
               for c in prop["candidates"])


def test_llm_prompt_omits_disabled_axes(tmp_path):
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[1]["content"]
        return {"candidates": [{"id": "x", "axes": {"lora": True},
                                "rationale": "r"},
                               {"id": "y", "axes": {"alias": "vlm"},
                                "rationale": "r"}], "stop": False}

    planner = LLMPlanner(chat=chat, artifact_dir=tmp_path,
                         disabled_axes=DISABLED)
    prop = planner.propose([], _diag({"missed": 1}))
    assert "- lora:" not in seen["prompt"]
    assert "- crop_classifier:" not in seen["prompt"]
    assert "never turn these on" in seen["prompt"] and "lora" in seen["prompt"]
    # LLM candidates that turn on a disabled axis are dropped by sanitize
    assert [c["id"] for c in prop["candidates"]] == ["y"]


def test_llm_prompt_unchanged_when_no_disabled(tmp_path):
    prompts = {}

    def chat_for(tag):
        def chat(messages, **kw):
            prompts[tag] = messages[1]["content"]
            return {"candidates": [], "stop": True}
        return chat

    LLMPlanner(chat=chat_for("default"), artifact_dir=tmp_path / "a").propose(
        [], _diag({}))
    LLMPlanner(chat=chat_for("empty"), artifact_dir=tmp_path / "b",
               disabled_axes=()).propose([], _diag({}))
    assert prompts["default"] == prompts["empty"]  # empty disabled list = byte-identical


def test_loop_rejects_unknown_disabled_axis(dataset_root, tmp_path):
    from agentic_configuration_search.loop.runner import Loop
    cfg = config_from_dict({"dataset": str(dataset_root),
                            "search": {"disabled_axes": ["warp_drive"]}})
    with pytest.raises(ValueError, match="warp_drive"):
        Loop(cfg, dataset_root, tmp_path / "run")


# ------------------------------------------------- disabled_values (per value)

def test_uses_disabled_values_flags_only_listed_values():
    dv = {"exemplar": ("fn_arm",)}
    assert uses_disabled_axes({**BASELINE, "exemplar": "fn_arm"}, (), dv)
    assert not uses_disabled_axes({**BASELINE, "exemplar": "base"}, (), dv)
    assert not uses_disabled_axes(dict(BASELINE), (), dv)


def test_sanitize_drops_disabled_value_candidates():
    raw = [{"id": "a", "axes": {"exemplar": "fn_arm"}, "rationale": ""},
           {"id": "b", "axes": {"exemplar": "base"}, "rationale": ""}]
    out = sanitize_candidates(raw, max_candidates=5,
                              disabled_values={"exemplar": ("fn_arm",)})
    assert [c["id"] for c in out] == ["b"]


def test_deterministic_skips_disabled_value_remedies():
    # exemplar step-up for missed: with a base champion, fn_arm is the next
    # increment, but disabling that value blocks the proposal (other values such as base survive).
    history = [{"round": 1,
                "champion": {"axes": {**BASELINE, "exemplar": "base"}},
                "decision": {"adopted": True}, "candidates": []}]
    prop = deterministic.propose(history, _diag({"missed": 5}),
                                 max_candidates=10,
                                 disabled_values={"exemplar": ("fn_arm",)})
    assert all(c["axes"]["exemplar"] != "fn_arm" for c in prop["candidates"])


def test_llm_prompt_omits_disabled_values(tmp_path):
    seen = {}

    def chat(messages, **kw):
        seen["prompt"] = messages[1]["content"]
        return {"candidates": [], "stop": True}

    LLMPlanner(chat=chat, artifact_dir=tmp_path,
               disabled_values={"exemplar": ("fn_arm",)}).propose([], _diag({}))
    row = next(l for l in seen["prompt"].splitlines() if l.startswith("exemplar | "))
    assert "fn_arm" not in row.split(" | ")[2] and "base" in row.split(" | ")[2]
    assert "never turn these on" in seen["prompt"] and "exemplar=fn_arm" in seen["prompt"]


def test_loop_rejects_unknown_disabled_value(dataset_root, tmp_path):
    from agentic_configuration_search.loop.runner import Loop
    cfg = config_from_dict({"dataset": str(dataset_root),
                            "search": {"disabled_values": {"exemplar": ["warp"]}}})
    with pytest.raises(ValueError, match="warp"):
        Loop(cfg, dataset_root, tmp_path / "run")


# ------------------------------------------- champion inheritance (base)

def test_disabled_axis_value_inherited_from_champion_is_kept():
    """Only turning an axis on is blocked — blocking a value the champion already
    has would reject every candidate once the scope is narrowed on resume."""
    champ = {**BASELINE, "tiling": 2}
    assert not uses_disabled_axes({**champ, "alias": "vlm"}, ("tiling",), base=champ)
    assert uses_disabled_axes({**champ, "tiling": 3}, ("tiling",), base=champ)
    assert uses_disabled_axes({**BASELINE, "tiling": 2}, ("tiling",))
    dv = {"exemplar": ("fn_arm",)}
    fn_champ = {**BASELINE, "exemplar": "fn_arm"}
    assert not uses_disabled_axes({**fn_champ, "alias": "vlm"}, (), dv, base=fn_champ)


def test_llm_and_human_candidates_inherit_champion_disabled_axis(tmp_path):
    champ = {**BASELINE, "tiling": 2}
    history = [{"round": 1, "champion": {"axes": champ},
                "decision": {"adopted": True}, "candidates": []}]

    def chat(messages, **kw):
        return {"candidates": [{"id": "keep", "axes": {"tiling": 2, "alias": "vlm"},
                                "rationale": "r"},
                               {"id": "raise", "axes": {"tiling": 3},
                                "rationale": "r"}], "stop": False}

    prop = LLMPlanner(chat=chat, artifact_dir=tmp_path,
                      disabled_axes=("tiling",)).propose(history, _diag({}))
    assert [c["id"] for c in prop["candidates"]] == ["keep"]
    assert [c["id"] for c in prop["dropped"]] == ["raise"]

    from agentic_configuration_search.loop.runner import _load_candidates_file
    f = tmp_path / "candidates.yaml"
    f.write_text("- {id: k, axes: {tiling: 2, alias: vlm}}\n"
                 "- {id: r, axes: {tiling: 3}}\n")
    kept, dropped = _load_candidates_file(f, ("tiling",), base=champ)
    assert [c["id"] for c in kept] == ["human_k"]
    assert [(c["id"], c["skipped"]) for c in dropped] == [("human_r", "disabled_axes")]
