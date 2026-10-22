"""loop/runner — the final champion's diagnosis at the end of the run (rounds/final).

A round's diagnosis is of the champion at the **start** of the round, so when
the last round adopts, the final champion's residual errors are recorded
nowhere. run_rounds diagnoses the final champion once at the end — if the last
round adopted nothing it only copies that round's diagnosis (no inference); if
it adopted, it is one train inference. Resume is idempotent, and the next round
continuing the run reuses this diagnosis.
"""

import json
import logging

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from tests.test_loop import _fake_infer, _fakes, _gt_lookup
from tests.test_loop import big_root  # noqa: F401 — fixture reuse


def _counting(big_root, monkeypatch):  # noqa: F811
    fake = _fake_infer(_gt_lookup(big_root))
    seen = []

    def counted(*a, **kw):
        seen.append([str(p) for p in a[5]])
        return fake(*a, **kw)
    monkeypatch.setattr(gateway, "infer_config", counted)
    return seen


def _train_calls(seen, big_root):  # noqa: F811
    train = str(big_root / "train")
    return [c for c in seen if c and all(p.startswith(train) for p in c)
            and len(c) == 12]


def test_final_diagnosis_after_adopting_last_round(big_root,  # noqa: F811
        tmp_path, monkeypatch):
    """Round 1 adopts exemplar and the run ends — the final diagnosis must be the
    adopted champion's (class 2 misses disappear)."""
    _counting(big_root, monkeypatch)
    out = tmp_path / "run"
    res = run_rounds(big_root, out_dir=out, planner="deterministic", rounds=1,
                     **_fakes())
    assert res.decisions[-1]["adopted"] is True
    final = out / "rounds" / "final"
    d_train = json.loads((final / "diagnosis_train.json").read_text())
    d_test = json.loads((final / "diagnosis_test.json").read_text())
    r1 = json.loads((out / "rounds" / "round_001" / "diagnosis_train.json").read_text())
    assert sum(r1["summary"]["fn"].values()) > 0          # the champion before adoption
    assert sum(d_train["summary"]["fn"].values()) == 0     # the final champion
    assert d_test["anonymized"] is True and "fn" not in d_test
    key = json.loads((final / "diagnosis_key.json").read_text())
    assert key["champion"]["round"] == 1
    residual = json.loads((final / "residual.json").read_text())
    assert residual["train"]["n_images"] == 12
    assert residual["holdout"]["n_images"] == 8
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["final_diagnosis"]["reused"] is False
    assert summary["final_diagnosis"]["champion_round"] == 1


def test_final_diagnosis_reuses_non_adopting_last_round(big_root,  # noqa: F811
        tmp_path, monkeypatch):
    """If the last round adopted nothing, its diagnosis is copied — no inference."""
    seen = _counting(big_root, monkeypatch)
    out = tmp_path / "run"
    res = run_rounds(big_root, out_dir=out, planner="deterministic", rounds=3,
                     **_fakes())
    assert res.decisions[-1]["adopted"] is False
    last = res.decisions[-1]["round"]
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["final_diagnosis"]["reused"] is True
    rd = out / "rounds" / f"round_{last:03d}"
    final = out / "rounds" / "final"
    assert (final / "diagnosis_train.json").read_text() == \
        (rd / "diagnosis_train.json").read_text()
    n_train = len(_train_calls(seen, big_root))

    # resume (an already finished run): the final diagnosis is not recomputed either
    seen.clear()
    run_rounds(big_root, out_dir=out, planner="deterministic", rounds=3,
               resume=True, **_fakes())
    assert _train_calls(seen, big_root) == []
    assert n_train >= 1


def test_next_round_reuses_final_diagnosis(big_root,  # noqa: F811
        tmp_path, monkeypatch, caplog):
    """Continuing a run that ended at rounds=1 with rounds=2 makes round 2 reuse the final diagnosis."""
    _counting(big_root, monkeypatch)
    out = tmp_path / "run"
    run_rounds(big_root, out_dir=out, planner="deterministic", rounds=1, **_fakes())
    caplog.set_level(logging.INFO, logger="agentic_configuration_search.loop.runner")
    run_rounds(big_root, out_dir=out, planner="deterministic", rounds=2,
               resume=True, **_fakes())
    assert any("diagnosis reused: final" in r.getMessage() for r in caplog.records)
    assert json.loads((out / "rounds" / "round_002" / "diagnosis_key.json")
                      .read_text())["champion"]["round"] == 1


def test_final_diagnosis_failure_does_not_kill_run(big_root,  # noqa: F811
        tmp_path, monkeypatch):
    """The final diagnosis is report-only — on failure the run still ends with its results."""
    from agentic_configuration_search.diagnose import residual
    _counting(big_root, monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(residual, "residual", boom)
    out = tmp_path / "run"
    run_rounds(big_root, out_dir=out, planner="deterministic", rounds=1, **_fakes())
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["final_diagnosis"]["error"].startswith("RuntimeError")
    assert summary["champion"]["id"]
