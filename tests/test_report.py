"""diagnose/report — trajectory, candidate attempts, A/B, retrospective hit rate, and
the ledger edge cases of the run report (diagnose.story). No GPU needed."""

import json as _json
from pathlib import Path

import pytest

from agentic_configuration_search.diagnose.report import (ab_compare, retrospective_hit_rate,
                                                          trajectory)
from agentic_configuration_search.diagnose.story import render_report
from agentic_configuration_search.core.coco import Ontology
from agentic_configuration_search.diagnose import residual as _residual
from agentic_configuration_search.diagnose.errors import diagnose as _diagnose
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from agentic_configuration_search.planner import deterministic
from tests.conftest import CATS, make_split
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

    assert (finished_run / "report.md").exists()     # the run leaves a report at its end


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


# --- synthetic run directories — the per-section report contract ------------------

_BASE = {"exemplar": "none", "alias": False, "text_negatives": False,
         "visual_negatives": False, "tiling": False, "rescorer": False,
         "lora": False, "crop_classifier": False, "exemplar_k": None,
         "fn_k": None, "visual_negative_k": None, "text_negative_k": None}
_EX = {**_BASE, "exemplar": "base", "exemplar_k": 4}
_EXT = {**_EX, "tiling": 2}


def _wall(m, s, h=0.0):
    return {"materialize": m, "screen": s, **({"holdout": h} if h else {}),
            "total": m + s + h}


def _cand(cid, axes, **kw):
    return {"id": cid, "axes": axes, "adopted": False,
            "fit_dev_micro_f_beta": 0.5, **kw}


def _gate(delta, lo, adopted=False):
    return {"micro_f_beta": 0.5 + delta,
            "gate": {"delta": delta, "lo": lo, "hi": None, "adopted": adopted}}


def _history_rows(*, timing=True):
    """R0 baseline → R1 exemplar adopted (tiling=2 rejected at the gate,
    text_negatives passed but lost, alias same_setting) → R2 tiling=2 re-adopted
    (the carry rejected, alias same_setting again, 1 re-ask) → R3 an empty round
    (rescorer no_material, 1 re-ask)."""
    def ch(axes, rnd, cid, f):
        return {"axes": axes, "round": rnd, "id": cid, "micro_f_beta": f}
    t = (lambda **kw: kw) if timing else (lambda **kw: {})
    rows = [
        {"round": 0, "champion": ch(_BASE, 0, "baseline", 0.5), "candidates": [],
         "decision": {"round": 0, "kind": "baseline", "adopted": True,
                      "id": "baseline", "micro_f_beta": 0.5},
         **t(dur_s=60.0)},
        {"round": 1, "champion": ch(_EX, 1, "ex", 0.6),
         "candidates": [
             _cand("til2", {**_BASE, "tiling": 2}, **t(wall_min=_wall(1, 2, 1))),
             _cand("ex", _EX, adopted=True, **t(wall_min=_wall(3, 4, 1))),
             _cand("neg", {**_BASE, "text_negatives": "fp_crop"},
                   gate_passed_but_lost=True, **t(wall_min=_wall(2, 2, 1))),
             _cand("al", {**_BASE, "alias": "vlm"}, skipped="same_setting",
                   **t(wall_min=_wall(1, 1)))],
         "decision": {"round": 1, "adopted": True, "id": "ex",
                      "screened": ["til2", "ex", "neg"],
                      "evaluated": {"til2": _gate(-0.01, -0.03),
                                    "ex": _gate(0.10, 0.05, True),
                                    "neg": _gate(0.03, 0.01, True)},
                      "skipped": ["al"], "micro_f_beta": 0.6},
         **t(dur_s=1500.0, pass_cache={"hits": 10, "misses": 30,
                                       "invalidations": 0, "hit_rate": 0.25,
                                       "entries": 30})},
        {"round": 2, "champion": ch(_EXT, 2, "til2b", 0.65),
         "candidates": [
             _cand("carry_neg", {**_EX, "text_negatives": "fp_crop"},
                   source="carryover", **t(wall_min=_wall(1, 1, 1))),
             _cand("til2b", _EXT, adopted=True, reask=1,
                   **t(wall_min=_wall(0, 8, 2))),
             _cand("al2", {**_EX, "alias": "vlm"}, skipped="same_setting",
                   **t(wall_min=_wall(1, 1)))],
         "decision": {"round": 2, "adopted": True, "id": "til2b", "reasks": 1,
                      "screened": ["carry_neg", "til2b"],
                      "evaluated": {"carry_neg": _gate(-0.005, -0.02),
                                    "til2b": _gate(0.05, 0.02, True)},
                      "skipped": ["al2"], "micro_f_beta": 0.65},
         **t(dur_s=1200.0, pass_cache={"hits": 20, "misses": 20,
                                       "invalidations": 1, "hit_rate": 0.5,
                                       "entries": 50})},
        {"round": 3, "champion": ch(_EXT, 2, "til2b", 0.65),
         "candidates": [_cand("res", {**_EXT, "rescorer": True},
                              skipped="no_material", reask=1,
                              **t(wall_min=_wall(2, 0)))],
         "decision": {"round": 3, "adopted": False, "kind": "no_candidates",
                      "skipped": ["res"], "reasks": 1},
         **t(dur_s=600.0, pass_cache={"hits": 0, "misses": 0, "invalidations": 0,
                                      "hit_rate": 0.0, "entries": 50})},
    ]
    return rows


ONTO = Ontology.from_categories(CATS)


def _img(file, gts, dets=(), w=100, h=100):
    return {"file": file, "width": w, "height": h,
            "detections": [{"box": b, "score": 0.9, "category_id": c}
                           for b, c in dets],
            "gts": [{"box": b, "category_id": c} for b, c in gts]}


def _train_images():
    """cat a: finds most large GT and misses the small border GT. cat b: misses all (large ones)."""
    imgs = []
    for i in range(6):
        imgs.append(_img(f"a{i}.png", [([40, 40, 70, 70], 1), ([0, 10, 3, 13], 1)],
                         dets=[([40, 40, 70, 70], 1)]))
    imgs.append(_img("b0.png", [([10, 10, 60, 60], 2)]))
    imgs.append(_img("bad.png", [([20, 20, 21, 40], 1), ([50, 50, 70, 70], 2),
                                 ([50, 50, 70, 70], 2)],
                     dets=[([20, 20, 21, 40], 1), ([50, 50, 70, 70], 2)]))
    return imgs


def _holdout_images():
    return [_img("h0.png", [([40, 40, 70, 70], 1), ([0, 0, 2, 2], 1)],
                 dets=[([40, 40, 70, 70], 1)]),
            _img("h1.png", [([10, 10, 60, 60], 2)])]


def _write_diag(d: Path, *, notes=True, with_residual=True):
    d.mkdir(parents=True, exist_ok=True)
    tau = {1: 0.5, 2: 0.5}
    train, hold = _train_images(), _holdout_images()
    d_train = _diagnose(train, None, tau, ONTO, iou_thr=0.5, anonymize=False)
    d_test = _diagnose(hold, None, tau, ONTO, iou_thr=0.5, anonymize=True)
    (d / "diagnosis_train.json").write_text(_json.dumps(d_train))
    (d / "diagnosis_test.json").write_text(_json.dumps(d_test))
    if with_residual:
        res = _residual.residual([{k: v for k, v in r.items() if k != "detections"}
                                  for r in train], d_train, hold, None, tau, ONTO,
                                 iou_thr=0.5, small_ratio=0.04)
        (d / "residual.json").write_text(_json.dumps(res))
    if notes:
        fn = {r["image"]: r for r in d_train["fn"]}
        rows = [
            {"crop": 0, "kind": "missed_small", "category": "cat a",
             "category_id": 1, "image": "a0.png", "box": fn["a0.png"]["box"],
             "note": "A tiny blurry plate partially cropped at the frame edge."},
            {"crop": 1, "kind": "missed_small", "category": "cat a",
             "category_id": 1, "image": "a1.png", "box": fn["a1.png"]["box"],
             "note": "Tiny, blurry object near the border."},
            {"crop": 2, "kind": "missed", "category": "cat b", "category_id": 2,
             "image": "b0.png", "box": fn["b0.png"]["box"],
             "note": "A dark object in shadow, unusual viewpoint."},
        ]
        (d / "crop_notes.json").write_text(_json.dumps({"notes": rows}))


def _synthetic_run(tmp_path, *, timing=True, final=True, notes=True, upto=None):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    rows = _history_rows(timing=timing)[:upto]
    (run / "history.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n")
    (run / "champion.json").write_text(_json.dumps(
        {"round": 2, "id": "til2b", "axes": _EXT, "micro_f_beta": 0.65}))
    summary = {"stopped": "empty_rounds", "last_round": 3}
    if timing:
        summary["final_diagnosis"] = {"dir": "rounds/final", "reused": True,
                                      "champion_round": 2, "after_round": 3,
                                      "dur_s": 0.4}
    (run / "run_summary.json").write_text(_json.dumps(summary))
    for r in range(len(rows)):
        _write_diag(run / "rounds" / f"round_{r:03d}", notes=notes,
                    with_residual=False)
    if final:
        _write_diag(run / "rounds" / "final", notes=notes)
    return run


def test_report_falls_back_without_final_diagnosis(tmp_path):
    """Without rounds/final the last round's diagnosis stands in — the sections do not break."""
    run = _synthetic_run(tmp_path, timing=False, final=False, upto=3)
    md = render_report(run)
    assert "## 2. How did it improve?" in md and "## 3. What problems are left?" in md
    assert "could not be generated" not in md


def test_report_survives_history_only_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = _history_rows(timing=False)
    (run / "history.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n")
    md = render_report(run)
    assert "## 2. How did it improve?" in md
    assert "There is no diagnosis record" in md


def test_report_section_error_is_contained(tmp_path, monkeypatch):
    """One broken section does not stop the rest of the report."""
    from agentic_configuration_search.diagnose import story
    run = _synthetic_run(tmp_path)

    def boom(*a, **k):
        raise KeyError("x")
    monkeypatch.setattr(story, "_section_rounds", boom)
    md = render_report(run)
    assert "## 2. How did it improve?" in md and "could not be generated" in md
    assert "## 4. What to do next" in md
