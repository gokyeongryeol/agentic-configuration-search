"""Guards against wasted time — the tiling fold gate, the subset prescreen,
per-candidate wall clock, the same_setting material summary, pass cache counts.

Why: with the champion stalled, a planner can keep raising tiling g one step at
a time even though every grid is rejected as tiling_used=false, and the screens
at 1+g² cost take most of the remaining time.
"""

import json
from pathlib import Path

import pytest

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from agentic_configuration_search.planner.axes import BASELINE, validate_axes
from tests.test_loop import _fake_infer, _fakes, _gt_lookup, _hist, _tile_helps_infer
from tests.test_loop import big_root  # noqa: F401 — fixture reuse


def _row(cid, tiling, *, used=None, fit_dev=None, skipped=None, **extra):
    axes = validate_axes({"tiling": tiling, **extra})
    row = {"id": cid, "axes": axes}
    if used is not None:
        row["tiling_used"] = used
    if fit_dev is not None:
        row["fit_dev_micro_f_beta"] = fit_dev
    if skipped:
        row["skipped"] = skipped
        if used is False:
            row["unused_axes"] = ["tiling"]
    return row


def _entry(r, champ_round, cands, *, champ_axes=None, champ_fit_dev=0.5):
    return {"round": r, "champion": {"axes": validate_axes(champ_axes or {}),
                                     "round": champ_round, "id": f"c{champ_round}",
                                     "micro_f_beta": 0.5,
                                     "fit_dev_micro_f_beta": champ_fit_dev},
            "candidates": cands, "decision": {"adopted": False}}


# --- T1: tiling fold gate -----------------------------------------------------

def test_tiling_gate_leaves_the_first_grid_to_the_planner():
    """The first grid is not blocked — the planner picks it from the missed_small sizes."""
    from agentic_configuration_search.planner.gates import tiling_gate

    gate = tiling_gate([_entry(0, 0, [])])
    for g in (2, 3, 4, 6):
        assert gate.block(validate_axes({"tiling": g})) is None, g
    assert gate.block(validate_axes({})) is None        # candidates without tiling are unaffected


def test_tiling_gate_folds_after_unused_grid():
    """If grid g went unused, grids ≥ g are blocked until the champion changes."""
    from agentic_configuration_search.planner.gates import tiling_gate

    hist = [_entry(0, 0, []),
            _entry(1, 0, [_row("t2", 2, used=False, skipped="axis_unused")])]
    gate = tiling_gate(hist)
    for g in (2, 3, 4, 6):
        why = gate.block(validate_axes({"tiling": g}))
        assert why and "tiling_used=false" in why, g
    assert gate.folded_at == 2


def test_tiling_gate_does_not_force_steps_after_a_used_grid():
    """Raising past a used grid is only recommended by the prompt, not blocked —
    even without a gain, only an unused grid folds."""
    from agentic_configuration_search.planner.gates import tiling_gate

    hurt = [_entry(0, 0, []),
            _entry(1, 0, [_row("t2", 2, used=True, fit_dev=0.26)])]
    gate = tiling_gate(hurt)
    assert gate.folded_at is None
    for g in (2, 3, 5):
        assert gate.block(validate_axes({"tiling": g})) is None, g


def test_tiling_gate_folds_only_at_and_above_the_unused_grid():
    from agentic_configuration_search.planner.gates import tiling_gate

    hist = [_entry(0, 0, []),
            _entry(1, 0, [_row("t4", 4, used=False, skipped="axis_unused")])]
    gate = tiling_gate(hist)
    assert gate.block(validate_axes({"tiling": 3})) is None      # cheaper grids stay open
    assert gate.block(validate_axes({"tiling": 4}))
    assert gate.block(validate_axes({"tiling": 5}))


def test_tiling_gate_resets_when_champion_changes():
    """A fold is evidence under that champion — a new champion starts over."""
    from agentic_configuration_search.planner.gates import tiling_gate

    hist = [_entry(0, 0, []),
            _entry(1, 0, [_row("t2", 2, used=False, skipped="axis_unused")]),
            _entry(2, 2, [], champ_axes={"exemplar": "base"})]   # adopted in round 2
    gate = tiling_gate(hist)
    assert gate.block(validate_axes({"tiling": 2, "exemplar": "base"})) is None


def test_tiling_gate_keeps_inherited_lower_and_off_grids_open():
    """With the champion on grid g, inheriting g, lowering, turning off and larger
    grids all stay open (when no grid is folded)."""
    from agentic_configuration_search.planner.gates import tiling_gate

    hist = [_entry(3, 3, [], champ_axes={"tiling": 3})]
    gate = tiling_gate(hist)
    for axes in ({"tiling": 4}, {"tiling": 5}, {"tiling": 3, "alias": "vlm"},
                 {"tiling": 2}, {}):
        assert gate.block(validate_axes(axes)) is None, axes


def test_tiling_gate_reads_requested_grid():
    """For a relabelled candidate, requested_axes is the request."""
    from agentic_configuration_search.planner.gates import tiling_gate

    relabelled = {"id": "t2", "axes": validate_axes({"alias": "vlm"}),
                  "requested_axes": validate_axes({"tiling": 2, "alias": "vlm"}),
                  "tiling_used": False, "unused_axes": ["tiling"],
                  "fit_dev_micro_f_beta": 0.7}
    hist = [_entry(0, 0, []), _entry(1, 0, [relabelled])]
    assert tiling_gate(hist).folded_at == 2


def test_deterministic_planner_respects_tiling_gate():
    from agentic_configuration_search.planner import deterministic

    diag = {"train": {"summary": {"fn": {"missed_small": 10}}}}
    hist = [_entry(0, 0, []),
            _entry(1, 0, [_row("t2", 2, used=False, skipped="axis_unused")])]
    out = deterministic.propose(hist, diag)
    assert not any(c["axes"]["tiling"] for c in out["candidates"])


def _llm_by_round(plan):
    calls = {"n": 0}

    def chat(messages, **kw):
        # crop descriptions share this chat — count only planner calls (proposal schema)
        if (kw.get("schema") or {}).get("name") != "proposal":
            return {}
        calls["n"] += 1
        return {"candidates": plan.get(calls["n"], []), "stop": False}
    return chat


def test_blocked_tiling_is_dropped_before_materialize(big_root, tmp_path,  # noqa: F811
                                                      monkeypatch):
    """After grid 2 went unused, grid 3 is recorded as tiling_blocked without materials or screen."""
    from agentic_configuration_search.loop import runner

    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    chat = _llm_by_round({1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}],
                          2: [{"id": "t3", "axes": {"tiling": 3}, "rationale": "y"}]})
    seen = []
    orig = runner.materialize
    monkeypatch.setattr(runner, "materialize",
                        lambda cfg, ds, axes, *a, **k: (seen.append(axes.get("tiling")),
                                                        orig(cfg, ds, axes, *a, **k))[1])
    # this checks the fold across rounds — re-asking within the round is off
    # (that path is in tests/test_empty_round.py)
    from agentic_configuration_search.config import config_from_dict
    cfg = config_from_dict({"dataset": str(big_root), "rounds": 2,
                            "planner": {"kind": "llm", "max_reasks": 0}})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=2, cfg=cfg, **_fakes())
    h = _hist(tmp_path / "run")
    assert h[1]["candidates"][0]["skipped"] == "axis_unused", h[1]["candidates"]
    rec = {c["id"]: c for c in h[2]["candidates"]}["r2_t3"]
    assert rec["skipped"] == "tiling_blocked" and "grid" in rec["note"]
    assert 3 not in seen


def test_prompt_states_tiling_cost_and_fold_rule():
    from agentic_configuration_search.planner.axis_status import AxisStatus
    from agentic_configuration_search.planner.llm import TILING_RULE, _constraint_lines

    flat = " ".join(TILING_RULE.split())
    assert "1 + g^2" in flat and "used and helped" in flat
    assert "folds that grid and larger ones" in flat
    assert "first grid from the missed-small sizes" in flat
    assert any("tiling_blocked" in l for l in _constraint_lines((), AxisStatus(2)))


def test_small_block_reports_missed_small_input_px_and_grid_magnification():
    """Input-px quantiles of all train missed_small (the fn list, not samples) and
    the median after each grid's magnification — evidence for the first grid.
    No missed_small, no block."""
    from agentic_configuration_search.planner.llm import _small_block

    fn = [{"kind": "missed_small", "size_ratio": r} for r in (0.01, 0.02, 0.03, 0.04)]
    fn.append({"kind": "missed", "size_ratio": 0.5})
    text = _small_block({"train": {"fn": fn}}, overlap=0.5)
    assert "4 objects" in text and "median 19 px" in text
    assert "grid 2: 1.5x -> median 29 px, cost 5 passes" in text
    assert "grid 3: 2x -> median 38 px, cost 10 passes" in text
    assert _small_block({"train": {"fn": fn[-1:]}}, overlap=0.5) == ""
    assert _small_block({}, overlap=0.5) == ""


# --- T2: subset prescreen for tiling grid ≥ min_grid ---------------------------

def test_prescreen_config_defaults_and_validation():
    from agentic_configuration_search.config import ConfigError, config_from_dict

    s = config_from_dict({"dataset": "x"}).screen
    assert (s.prescreen_fraction, s.prescreen_min_images, s.prescreen_min_grid,
            s.prescreen_max_share) == (0.25, 20, 3, 0.5)
    for bad in ({"prescreen_fraction": -0.1}, {"prescreen_fraction": 1.5},
                {"prescreen_fraction": 0}, {"prescreen_fraction": "0.25"},
                {"prescreen_min_images": 0}, {"prescreen_min_grid": 1},
                {"prescreen_max_share": 0.0}, {"prescreen_max_share": 1.5}):
        with pytest.raises(ConfigError):
            config_from_dict({"dataset": "x", "screen": bad})


def test_prescreen_subset_is_deterministic_fit_dev_only(dataset_root):
    """The subset is a subset of fit_dev (split roles), deterministic by filename hash."""
    from agentic_configuration_search.core.data import load_dataset, prescreen_subset

    ds = load_dataset(dataset_root)
    dev = ds.fit_dev
    names = [e.file_name for e in dev.images]
    a = prescreen_subset(dev, 0.5, 1, max_share=1.0)
    b = prescreen_subset(load_dataset(dataset_root).fit_dev, 0.5, 1, max_share=1.0)
    assert a is not None and len(a.images) == 1
    assert [e.file_name for e in a.images] == [e.file_name for e in b.images]
    assert set(e.file_name for e in a.images) <= set(names)
    # None is off (0 is refused) · equal to the whole split is pointless, so None
    assert prescreen_subset(dev, None, 1, max_share=1.0) is None
    with pytest.raises(ValueError, match="None"):
        prescreen_subset(dev, 0.0, 1, max_share=1.0)
    assert prescreen_subset(dev, 0.5, len(names), max_share=1.0) is None


def _names_split(n):
    from types import SimpleNamespace

    from agentic_configuration_search.core.data import Split
    sp = Split.__new__(Split)
    sp.root = sp.raw = sp.ontology = None
    sp.images = [SimpleNamespace(file_name=f"{i:03d}.jpg") for i in range(n)]
    return sp


@pytest.mark.parametrize("n_all,expect", [
    (22, None),     # 20 of 22 — a passing candidate re-screens all 22, nothing saved
    (40, None),     # 20/40 = 50% — the boundary skips
    (41, 20),       # 20/41 < 50%
    (200, 50),      # ⌈0.25·200⌉ = 50
])
def test_prescreen_subset_skipped_when_it_is_most_of_fit_dev(n_all, expect):
    """When the subset is max_share of fit_dev or more, the prescreen saves no
    time — a passing candidate is screened in full again. None (prescreen
    skipped)."""
    from agentic_configuration_search.core.data import prescreen_subset

    sub = prescreen_subset(_names_split(n_all), 0.25, 20, max_share=0.5)
    assert (None if sub is None else len(sub.images)) == expect


def _recording(fake, log):
    def wrapped(ctx, client, config, queries, ontology, image_paths, **kw):
        log.append((bool(config.get("tiling")), len(image_paths)))
        return fake(ctx, client, config, queries, ontology, image_paths, **kw)
    return wrapped


def _prescreen_cfg(root, fraction=0.5, max_share=1.0):
    from agentic_configuration_search.config import config_from_dict
    return config_from_dict({"dataset": str(root), "planner": {"kind": "llm"},
                             "screen": {"prescreen_fraction": fraction,
                                        "prescreen_min_images": 1,
                                        "prescreen_min_grid": 2,
                                        "prescreen_max_share": max_share}})


def test_prescreen_skips_unused_tiling_without_full_screen(big_root, tmp_path,  # noqa: F811
                                                           monkeypatch):
    from agentic_configuration_search.core.data import load_dataset

    calls = []
    monkeypatch.setattr(gateway, "infer_config",
                        _recording(_fake_infer(_gt_lookup(big_root)), calls))
    chat = _llm_by_round({1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}]})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=1, cfg=_prescreen_cfg(big_root), **_fakes())
    n_dev = len(load_dataset(big_root).fit_dev.images)
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["r1_t2"]
    assert rec["skipped"] == "axis_unused" and rec["tiling_used"] is False
    assert rec["note"].startswith("prescreen")
    assert rec["prescreen"]["n_images"] < rec["prescreen"]["of"] == n_dev
    # tile configs ran on the subset only — never on the full fit_dev
    assert calls and all(n < n_dev for tiled, n in calls if tiled)


def test_prescreen_skipped_when_subset_is_most_of_fit_dev(big_root, tmp_path,  # noqa: F811
                                                        monkeypatch):
    """When the subset is half of fit_dev or more, the full screen runs directly
    without a prescreen — tile configs never run at the subset size."""
    from agentic_configuration_search.core.data import load_dataset

    calls = []
    monkeypatch.setattr(gateway, "infer_config",
                        _recording(_fake_infer(_gt_lookup(big_root)), calls))
    chat = _llm_by_round({1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}]})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=1, cfg=_prescreen_cfg(big_root, 0.5, max_share=0.5),
               **_fakes())
    n_dev = len(load_dataset(big_root).fit_dev.images)
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["r1_t2"]
    assert rec["tiling_used"] is False and "prescreen" not in rec
    assert not str(rec.get("note", "")).startswith("prescreen")
    assert calls and all(n == n_dev for tiled, n in calls if tiled)


def test_prescreen_pass_keeps_full_screen_numbers(big_root, tmp_path,  # noqa: F811
                                                  monkeypatch):
    """If tiles are used on the subset, the full fit_dev is screened as before — same numbers."""
    monkeypatch.setattr(gateway, "infer_config", _tile_helps_infer(_gt_lookup(big_root)))
    plan = {1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}]}
    recs = []
    for name, frac in (("on", 0.5), ("off", None)):
        run_rounds(big_root, out_dir=tmp_path / name, planner="llm",
                   llm_chat=_llm_by_round(dict(plan)), rounds=1,
                   cfg=_prescreen_cfg(big_root, frac), **_fakes())
        recs.append({c["id"]: c for c in _hist(tmp_path / name)[1]["candidates"]}["r1_t2"])
    on, off = recs
    assert on.get("skipped") is None and on["tiling_used"] is True
    assert on["fit_dev_micro_f_beta"] == off["fit_dev_micro_f_beta"]
    assert on["prescreen"]["passed"] is True and "prescreen" not in off


def test_prescreen_only_for_tiling_only_increments(big_root, tmp_path,  # noqa: F811
                                                  monkeypatch):
    """A candidate that also changes other axes is not the champion even if tiles
    go unused — it is not rejected by the prescreen and gets the full screen."""
    calls = []
    monkeypatch.setattr(gateway, "infer_config",
                        _recording(_fake_infer(_gt_lookup(big_root)), calls))
    chat = _llm_by_round({1: [{"id": "mix", "axes": {"tiling": 2,
                                                     "text_negatives": "gt_crop"},
                               "rationale": "x"}]})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=1, cfg=_prescreen_cfg(big_root), **_fakes())
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["r1_mix"]
    assert "prescreen" not in rec
    assert not str(rec.get("note", "")).startswith("prescreen")


def test_unused_tiling_row_records_the_loss(big_root, tmp_path,  # noqa: F811
                                            monkeypatch):
    """A row whose full screen did not use tiles records the best tiled and untiled
    F (how far short) — even with the prescreen off."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    chat = _llm_by_round({1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"}]})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=1, cfg=_prescreen_cfg(big_root, None), **_fakes())
    rec = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}["r1_t2"]
    assert rec["tiling_used"] is False and "prescreen" not in rec
    tf = rec["tiling_f"]
    assert set(tf) == {"tiled", "untiled", "gap_ub"} and tf["tiled"] <= tf["untiled"]


# --- T4: per-candidate wall clock ----------------------------------------------

def test_candidate_rows_carry_wall_minutes(big_root, tmp_path,  # noqa: F811
                                           monkeypatch):
    """An evaluated candidate records materialize, screen and holdout; a rejected
    one the first two — in minutes, with a total."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    chat = _llm_by_round({1: [{"id": "t2", "axes": {"tiling": 2}, "rationale": "x"},
                              {"id": "ex", "axes": {"exemplar": "base"},
                               "rationale": "y"}]})
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm", llm_chat=chat,
               rounds=1, cfg=_prescreen_cfg(big_root, None), **_fakes())
    rows = {c["id"]: c for c in _hist(tmp_path / "run")[1]["candidates"]}
    ev = rows["r1_ex"]["wall_min"]
    assert set(ev) == {"materialize", "screen", "holdout", "total"}
    assert all(v >= 0 for v in ev.values())
    assert abs(ev["total"] - (ev["materialize"] + ev["screen"] + ev["holdout"])) < 0.02
    assert rows["r1_t2"]["skipped"] == "axis_unused"
    assert set(rows["r1_t2"]["wall_min"]) == {"materialize", "screen", "total"}


def test_compact_history_and_prompt_show_cost():
    from agentic_configuration_search.planner.board import _cost_cell, axis_costs
    from agentic_configuration_search.planner.llm import _compact_history

    cands = [{**_row("t2", 2, used=False, skipped="axis_unused"),
              "wall_min": {"materialize": 1.0, "screen": 50.0, "total": 51.0},
              "tiling_f": {"tiled": 0.4, "untiled": 0.5}},
             {**_row("t4", 4, used=False, skipped="axis_unused"),
              "wall_min": {"materialize": 1.0, "screen": 143.0, "total": 144.0}},
             {"id": "ex", "axes": validate_axes({"exemplar": "base"}),
              "wall_min": {"materialize": 2.0, "screen": 8.0, "holdout": 2.0,
                           "total": 12.0}}]
    hist = [_entry(0, 0, []), _entry(1, 0, cands)]
    row = _compact_history(hist)[1]["candidates"][0]
    assert row["wall_min"] == 51.0
    assert row["tiling_f"] == {"tiled": 0.4, "untiled": 0.5}
    costs = axis_costs(hist)
    assert _cost_cell("tiling", costs) == \
        "g2 51 (n=1, max 51), g4 144 (n=1, max 144)"            # per grid
    assert _cost_cell("exemplar", costs) == "12 (n=1, max 12)"
    assert axis_costs([_entry(0, 0, [])]) == {}                 # nothing before anything is recorded


# --- T5: material summary on same_setting rows ---------------------------------

def test_same_setting_row_summarizes_served_material(big_root, tmp_path,  # noqa: F811
                                                     monkeypatch):
    """A row rejected as the same setting carries what was the same (the served text-negative phrases)."""
    from tests.test_loop import _text_beats_semantic_infer
    from agentic_configuration_search.planner.llm import _compact_history

    monkeypatch.setattr(gateway, "infer_config",
                        _text_beats_semantic_infer(_gt_lookup(big_root)))
    human = tmp_path / "c.yaml"
    human.write_text("- {id: a, axes: {text_negatives: gt_crop}}\n"
                     "- {id: b, axes: {text_negatives: gt_crop, exemplar: base}}\n")
    run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
               rounds=1, candidates_file=human, **_fakes())
    hist = _hist(tmp_path / "run")
    rec = {c["id"]: c for c in hist[1]["candidates"]}["r1_human_b"]
    assert rec["skipped"] == "same_setting"
    assert rec["material"].startswith("text_negatives ")
    assert "served material: text_negatives" in rec["note"]
    row = {r["id"]: r for r in _compact_history(hist)[1]["candidates"]}["r1_human_b"]
    assert row["material"] == rec["material"]


def test_material_summary_is_bounded_and_modality_aware():
    from agentic_configuration_search.fitting.materials import Materials
    from agentic_configuration_search.loop.runner import MATERIAL_SUMMARY_MAX, _material_summary

    m = Materials(text_negatives={"tower": ["building"], "pole": []},
                  aliases={"tower": "cell tower"})
    text = {"per_class": {"1": {"config_id": 0, "modality": "text"}}}
    assert _material_summary(text, m) == \
        "text_negatives tower=[building] | aliases tower=cell tower"
    sem = {"per_class": {"1": {"config_id": 0, "modality": "semantic"}}}
    assert _material_summary(sem, m) == "exemplars 0"
    long = Materials(text_negatives={f"c{i}": ["x" * 20] for i in range(20)})
    assert len(_material_summary(text, long)) == MATERIAL_SUMMARY_MAX


# --- pass cache counts -----------------------------------------------------------

def test_pass_cache_counts_hits_misses_invalidations(tmp_path):
    from tests.test_gateway import CFG_OFF, ONTO, TILE_CFG, ContentClient, _ctx, \
        _noise_images

    assert gateway.pass_cache_stats() is None                 # outside the block
    paths = _noise_images(tmp_path, 3)
    client = ContentClient()
    with gateway.pass_cache():
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
        assert gateway.pass_cache_stats() == {"hits": 0, "misses": 3,
                                              "invalidations": 0, "entries": 3}
        gateway.infer_config(_ctx(), client, TILE_CFG, [], ONTO, paths)
        st = gateway.pass_cache_stats()
        assert (st["hits"], st["misses"]) == (3, 3 + 12)       # full pieces reused, tiles new
        gateway.invalidate_pass_cache()
        st = gateway.pass_cache_stats()
        assert (st["invalidations"], st["entries"]) == (1, 0)
    assert gateway.pass_cache_stats() is None


def test_history_rows_carry_round_pass_cache_counts(big_root, tmp_path,  # noqa: F811
                                                    monkeypatch):
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(_gt_lookup(big_root)))
    run_rounds(big_root, out_dir=tmp_path / "run", planner="deterministic",
               rounds=1, **_fakes())
    for e in _hist(tmp_path / "run"):
        pc = e["pass_cache"]
        assert {"hits", "misses", "invalidations", "hit_rate", "entries"} <= set(pc)


# --- folded-axis hints -----------------------------------------------------------

def test_board_folds_tiling_only():
    """"Folding" is for tiling grids only — other axes left unused repeatedly are handled by the axis status (OFF)."""
    from agentic_configuration_search.planner.axes import axis_values
    from agentic_configuration_search.planner.axis_status import axis_status
    from agentic_configuration_search.planner.board import board_lines

    ex = {"id": "e1", "axes": validate_axes({"exemplar": "base"}),
          "skipped": "axis_unused", "unused_axes": ["exemplar"]}
    t2 = _row("t2", 2, used=False, skipped="axis_unused")
    hist = [_entry(0, 0, []), _entry(1, 0, [ex, t2])]
    rows = {l.split(" | ")[0]: l.split(" | ")
            for l in board_lines(hist, axis_values(), axis_status(hist, 2)) if " | " in l}
    assert "grid >= 2 folded" in rows["tiling"][6] and rows["tiling"][2] == "-"
    assert "folded" not in rows["exemplar"][6]
    # a champion change starts over
    moved = hist + [_entry(2, 2, [], champ_axes={"alias": "vlm"})]
    rows = {l.split(" | ")[0]: l.split(" | ")
            for l in board_lines(moved, axis_values(), axis_status(moved, 2)) if " | " in l}
    assert "folded" not in rows["tiling"][6]


def test_prescreen_is_only_for_grid_increases(monkeypatch):
    """A grid below the champion's is not prescreened — it is cheaper, and unused
    tiles on the subset do not make it the same setting as the champion (which
    uses tiles). The full path measures it."""
    from types import SimpleNamespace

    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop import runner
    from agentic_configuration_search.loop.runner import Loop

    monkeypatch.setattr(runner, "prescreen_subset",
                        lambda *a, **kw: SimpleNamespace(images=[0] * 40))

    class Screened(Exception):
        pass

    def screen(*a, **kw):
        raise Screened

    champ = validate_axes({"tiling": 4})
    fake = SimpleNamespace(
        cfg=config_from_dict({"dataset": "x", "screen": {"prescreen_min_grid": 2,
                                                         "lost_value_prescreen": False}}),
        _champion={"axes": champ}, _screen=screen,
        ds=SimpleNamespace(fit_dev=SimpleNamespace(images=[0] * 400)))
    lower = validate_axes({"tiling": 3})
    assert Loop._prescreen(fake, {"axes": lower}, lower, None, {}, []) is False
    higher = validate_axes({"tiling": 5})
    with pytest.raises(Screened):
        Loop._prescreen(fake, {"axes": higher}, higher, None, {}, [])
