"""diagnose/report — trajectory, report, A/B, retrospective hit rate. No GPU needed."""

from pathlib import Path

import pytest

from agentic_configuration_search.diagnose.report import (ab_compare, render_report,
                                                     retrospective_hit_rate, trajectory)
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from agentic_configuration_search.planner import deterministic
from tests.conftest import make_split
from tests.test_loop import _fake_infer, _fakes, _gt_lookup


@pytest.fixture
def finished_run(tmp_path, monkeypatch):
    """A run completed with the same scenario as test_loop — baseline 2/3 -> exemplar accepted, 1.0."""
    size = lambda i: 20 + (i % 3) * 10  # noqa: E731
    ds = tmp_path / "ds"
    make_split(ds / "train", [
        {"name": f"tr{i:02d}.png", "w": 100, "h": 80,
         "anns": [(1 if i % 2 else 2, 10, 10, size(i), size(i))]}
        for i in range(12)])
    make_split(ds / "valid", [        # the loop's holdout
        {"name": f"t{i}.png", "w": 100, "h": 80,
         "anns": [(1 if i < 4 else 2, 10, 10, size(i), size(i))]}
        for i in range(8)])
    make_split(ds / "test", [{"name": "v.png", "w": 100, "h": 80,   # report-only — the loop never reads it
                              "anns": [(1, 10, 10, 20, 20)]}])
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(ds)))
    out = tmp_path / "run"
    run_rounds(ds, out_dir=out, planner="deterministic", rounds=3, **_fakes())
    return out


def test_trajectory_and_report(finished_run):
    t = trajectory(finished_run)
    assert t["baseline_micro_f_beta"] == pytest.approx(2 / 3)
    assert t["final_micro_f_beta"] == pytest.approx(1.0)
    assert t["gain"] > 0.3 and t["n_adopted"] == 1

    md = render_report(finished_run)
    assert "setting search report" in md
    assert "1.0000" in md and "baseline 0.6667" in md
    assert "exemplar" in md                         # axis turned on in the champion
    assert (finished_run / "report.md").exists()
    assert "residual error" in md                        # residual-error summary of the last round's diagnosis


def test_ab_compare_self_is_tie(finished_run):
    out = ab_compare(finished_run, finished_run, label_a="det", label_b="llm")
    assert out["winner"] == "tie"
    assert out["det"]["final"] == out["llm"]["final"] == pytest.approx(1.0)
    assert out["rounds"][0]["det"] == pytest.approx(2 / 3)


def test_retrospective_hit_rate_with_deterministic(finished_run):
    """Replaying the stored diagnoses, the deterministic planner must re-propose the actually
    accepted candidate — retrospective hit rate 1.0 (retrospective hit-rate harness contract)."""
    out = retrospective_hit_rate(finished_run, deterministic.propose)
    assert out["total"] == 1
    assert out["hit_rate"] == pytest.approx(1.0)
    assert out["detail"][0]["adopted_axes"]["exemplar"] == "base"


def test_report_missing_run_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="history"):
        trajectory(tmp_path / "nope")
