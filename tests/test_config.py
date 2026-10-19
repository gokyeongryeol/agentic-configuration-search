"""config schema — unknown-key errors, defaults, validation."""

import pytest

from agentic_detector_config.config import ConfigError, config_from_dict, load_config

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


def test_tiling_configs_default_is_resolution_relative():
    """Default tile grids use the grid form — absolute px slices grew the tile count
    quadratically at high resolution and made a round several times longer."""
    for t in config_from_dict(BASE).search.tiling_configs:
        assert "grid" in t and "slice" not in t


@pytest.mark.parametrize("bad", [
    {"overlap": 0.25},                                    # no grid
    {"grid": 2, "slice": [320, 320], "overlap": 0.25},    # slice not supported
    {"grid": 1, "overlap": 0.25},                         # 1 = no tiles
    {"grid": 2.5, "overlap": 0.25},
    {"grid": 2, "overlap": 1.0},
    {"grid": 2, "overlap": -0.1},
    {"slice": [416, 416], "overlap": 0.25, "merge": "nmm"},  # old form
    {"grid": 2, "overlap": 0.25, "merge": "union"},
    {"grid": 2, "overlap": 0.25, "zoom": 2},              # typo / unknown key
])
def test_bad_tiling_config_errors(bad):
    with pytest.raises(ConfigError, match="search.tiling_configs"):
        config_from_dict({"dataset": "x", "search": {"tiling_configs": [bad]}})
