"""config schema — unknown-key errors, defaults, validation."""

import pytest

from agentic_configuration_search.config import ConfigError, config_from_dict, load_config

BASE = {"dataset": "demo"}


def test_defaults():
    cfg = config_from_dict(BASE)
    assert cfg.planner.kind == "deterministic"
    assert cfg.split.fit_train_ratio == 0.8
    assert cfg.gate.margin == 0.0
    assert cfg.saturation.rounds == 5
    assert cfg.screen.top_k == 2


def test_dataset_required():
    with pytest.raises(ConfigError, match="dataset"):
        config_from_dict({})


def test_unknown_top_key_errors():
    with pytest.raises(ConfigError, match="render"):
        config_from_dict({**BASE, "render": {}})


def test_unknown_section_key_errors():
    with pytest.raises(ConfigError, match="kine"):
        config_from_dict({**BASE, "planner": {"kine": "llm"}})


def test_section_must_be_mapping():
    with pytest.raises(ConfigError, match="planner"):
        config_from_dict({**BASE, "planner": "llm"})


def test_bad_planner_kind():
    with pytest.raises(ConfigError, match="planner.kind"):
        config_from_dict({**BASE, "planner": {"kind": "random"}})


def test_bad_ratio():
    with pytest.raises(ConfigError, match="fit_train_ratio"):
        config_from_dict({**BASE, "split": {"fit_train_ratio": 1.0}})


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("dataset: demo\nplanner:\n  kind: llm\nscreen:\n  top_k: 3\n")
    cfg = load_config(p)
    assert cfg.planner.kind == "llm"
    assert cfg.screen.top_k == 3
    # unspecified sections take defaults
    assert cfg.gate.bootstrap == 2000


def test_empty_yaml_errors(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ConfigError):
        load_config(p)


def test_beta_default_and_validation():
    assert config_from_dict(BASE).beta == 1.0
    assert config_from_dict({**BASE, "beta": 2.0}).beta == 2.0
    with pytest.raises(ConfigError, match="beta"):
        config_from_dict({**BASE, "beta": 0})


def test_default_yaml_matches_dataclass_defaults():
    """config/default.yaml is reference documentation — if it drifts from the code
    defaults it lies, so drift is blocked here."""
    from pathlib import Path

    ref = load_config(Path(__file__).parent.parent / "config" / "default.yaml")
    default = config_from_dict({"dataset": ref.dataset})
    assert ref == default


def test_dataset_notes_default_and_roundtrip():
    assert config_from_dict(BASE).dataset_notes == ""
    # Non-ASCII (Korean) sample on purpose: user notes must round-trip unescaped.
    cfg = config_from_dict({**BASE, "dataset_notes": "야간 CCTV, 소형 객체 위주"})
    assert cfg.dataset_notes == "야간 CCTV, 소형 객체 위주"


def test_dataset_notes_must_be_string():
    with pytest.raises(ConfigError, match="dataset_notes"):
        config_from_dict({**BASE, "dataset_notes": ["list", "forbidden"]})


def test_tiling_grid_comes_from_axis_not_config():
    """The tile grid (grid g) is a planner-chosen axis value — config keeps only the
    overlap and merge shared by every grid. The old tiling_configs list is an unknown key."""
    s = config_from_dict(BASE).search
    assert s.tiling_overlap == 0.5 and s.tiling_merge == "nmm"
    with pytest.raises(ConfigError, match="tiling_configs"):
        config_from_dict({**BASE, "search": {"tiling_configs": [
            {"grid": 2, "overlap": 0.5, "merge": "nmm"}]}})


@pytest.mark.parametrize("key,value", [
    ("tiling_overlap", 1.0), ("tiling_overlap", -0.1), ("tiling_overlap", "0.5"),
    ("tiling_merge", "union"),
])
def test_bad_tiling_params_error(key, value):
    with pytest.raises(ConfigError, match=f"search.{key}"):
        config_from_dict({**BASE, "search": {key: value}})


def test_small_object_ratio_default_and_validation():
    assert config_from_dict(BASE).diagnose.small_object_ratio == 0.04
    for bad in (0, -0.1, 1.0, 1.5):
        with pytest.raises(ConfigError, match="diagnose.small_object_ratio"):
            config_from_dict({**BASE, "diagnose": {"small_object_ratio": bad}})


def test_kcenter_outlier_mad_validation_and_old_keys():
    """The outlier trim is distribution-based (median + k·MAD) — the fraction keys are unknown keys."""
    for key in ("kcenter_outlier_mad", "fn_kcenter_outlier_mad"):
        with pytest.raises(ConfigError, match=f"mine.{key}"):
            config_from_dict({**BASE, "mine": {key: -1.0}})
    for old in ("kcenter_trim", "fn_kcenter_trim"):
        with pytest.raises(ConfigError, match=old):
            config_from_dict({**BASE, "mine": {old: 0.2}})
