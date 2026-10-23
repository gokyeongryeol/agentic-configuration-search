"""Variants of clearly-lost values are measured on a fit_dev subset first
(lost-value prescreen).

Why: in validation runs (soccer, aquarium) exact combination dedup blocked only
the same combination, so variants of values that lost badly (alias=vlm,
exemplar=fn_arm) went through the full screen and holdout only to confirm the
same result. Variants are not blocked — instead, a candidate containing a value
that clearly lost **on fit_dev** under the current champion goes to the full
screen only if the bootstrap upper bound of (candidate − champion) micro F(β) on
the subset is at or above 0.

The verdict is a pure function reading only history rows (`fit_dev_cmp`,
planner.ledger). Holdout results are not used for it.
"""

import json

import numpy as np
import pytest

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from agentic_configuration_search.planner.axes import validate_axes
from tests.test_loop import _fake_infer, _fakes, _gt_lookup, _hist
from tests.test_loop import big_root  # noqa: F401 — fixture reuse
from tests.test_time_efficiency import _llm_by_round


# --- ledger (planner.ledger) — a pure function of history ---------------------------------

def _cand(cid, *, ub=None, delta=None, skipped=None, champ=None, wall=None, **axes):
    full = validate_axes({**(champ or {}), **axes})
    row = {"id": cid, "axes": full}
    if ub is not None:
        row["fit_dev_cmp"] = {"delta": delta if delta is not None else min(ub, 0) - 0.01,
                              "ub": ub, "n": 20}
    if skipped:
        row["skipped"] = skipped
    if wall is not None:
        row["wall_min"] = {"total": wall}
    return row


def _round(r, champ_round, cands, *, champ=None, evaluated=None):
    return {"round": r,
            "champion": {"axes": validate_axes(champ or {}), "round": champ_round,
                         "id": f"c{champ_round}", "micro_f_beta": 0.7,
                         "fit_dev_micro_f_beta": 0.72},
            "candidates": cands,
            "decision": {"adopted": False, "evaluated": evaluated or {}}}


def _ledger(hist):
    from agentic_configuration_search.planner.ledger import axis_ledger
    return {f"{k[0]}={k[1]}": v for k, v in axis_ledger(hist).items()}


def test_value_keys_fold_quantity_axes_into_their_feature():
    from agentic_configuration_search.planner.ledger import value_keys

    champ = validate_axes({"exemplar": "base"})
    fn = validate_axes({"exemplar": "fn_arm", "fn_k": 4})
    assert value_keys(fn, champ) == (("exemplar", "fn_arm"),)
    k_only = validate_axes({"exemplar": "base", "exemplar_k": 2})
    assert value_keys(k_only, champ) == (("exemplar", "base"),)   # a variant of the champion's value
    combo = validate_axes({"exemplar": "fn_arm", "alias": "vlm"})
    assert value_keys(combo, champ) == (("alias", "vlm"), ("exemplar", "fn_arm"))


def test_single_increment_clear_loss_marks_value_lost():
    hist = [_round(0, 0, []), _round(1, 0, [_cand("a", ub=-0.01, alias="vlm")])]
    led = _ledger(hist)
    assert led["alias=vlm"].clearly_lost and led["alias=vlm"].prescreened
    assert led["alias=vlm"].status == "clearly-lost (prescreened)"
    assert led["alias=vlm"].evidence["id"] == "a"


def test_near_miss_is_never_lost_even_after_a_clear_loss():
    """A near miss (ub ≥ 0) releases the value from the prescreen even if it came
    from a combination — it showed a variant can get close."""
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("a", ub=-0.01, alias="vlm"),
                          _cand("b", ub=0.002, alias="vlm", exemplar="base")])]
    led = _ledger(hist)
    assert led["alias=vlm"].near_miss and not led["alias=vlm"].clearly_lost
    assert led["alias=vlm"].status == "near-miss"


def test_fit_dev_identical_row_is_not_a_near_miss():
    """A row whose per-image fit_dev rows all equal the champion's
    (fit_dev_identical — delta=0, ub=0) is neither a near miss nor a loss — it
    does not release a clearly-lost value."""
    same = _cand("b", ub=0.0, delta=0.0, skipped="fit_dev_identical", alias="vlm")
    same["fit_dev_cmp"]["identical"] = True
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("a", ub=-0.01, alias="vlm")]),
            _round(2, 0, [same])]
    led = _ledger(hist)
    rec = led["alias=vlm"]
    assert not rec.near_miss and rec.clearly_lost and rec.status != "near-miss"
    from agentic_configuration_search.planner.ledger import lost_values
    assert ("alias", "vlm") in lost_values(hist)


def test_fit_dev_identical_alone_is_not_a_clear_loss():
    same = _cand("b", ub=-0.0, delta=0.0, skipped="fit_dev_identical", alias="vlm")
    same["fit_dev_cmp"]["identical"] = True
    same["fit_dev_cmp"]["ub"] = -1e-9                 # an identical row is ignored whatever its ub
    rec = _ledger([_round(0, 0, []), _round(1, 0, [same])])["alias=vlm"]
    assert not rec.near_miss and not rec.single_loss and rec.evidence is None


def test_combo_loss_does_not_mark_its_values_lost():
    """A combination's loss cannot be attributed to one value."""
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("a", ub=-0.05, alias="vlm", exemplar="base")])]
    led = _ledger(hist)
    assert not led["alias=vlm"].clearly_lost
    assert not led["exemplar=base"].clearly_lost


def test_holdout_loss_alone_is_not_a_clear_loss():
    """A candidate that won on fit_dev and lost at the holdout gate did not
    clearly lose — the holdout Δ is only shown (soccer's fn_arm variants were
    this case)."""
    ev = {"a": {"micro_f_beta": 0.68, "gate": {"delta": -0.03, "lo": -0.05}}}
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("a", ub=0.01, delta=0.004, exemplar="base")],
                   evaluated=ev)]
    rec = _ledger(hist)["exemplar=base"]
    assert not rec.clearly_lost and rec.best_holdout == {"delta": -0.03, "lo": -0.05}


def test_ledger_resets_when_champion_changes():
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("a", ub=-0.01, alias="vlm")]),
            _round(2, 2, [], champ={"exemplar": "base"})]    # adopted in round 2
    assert _ledger(hist) == {}


def test_values_that_changed_a_champion_are_never_prescreened():
    """A value that ever changed the champion is not prescreened."""
    adopted = {**_cand("w", exemplar="base"), "adopted": True}
    hist = [_round(0, 0, []),
            _round(1, 1, [adopted], champ={"exemplar": "base"}),
            _round(2, 1, [_cand("off", ub=-0.2, exemplar="none",
                                champ={"exemplar": "base"})],
                   champ={"exemplar": "base"}),
            _round(3, 3, [], champ={"alias": "vlm"}),
            _round(4, 3, [_cand("back", ub=-0.1, exemplar="base",
                                champ={"alias": "vlm"})], champ={"alias": "vlm"})]
    rec = _ledger(hist)["exemplar=base"]
    assert rec.clearly_lost and rec.protected and not rec.prescreened


def test_server_state_and_rescorer_values_are_not_prescreened():
    """Axes whose expensive part (training, refit) comes before the fit_dev
    comparison are not prescreened."""
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("r", ub=-0.01, rescorer=True),
                          _cand("l", ub=-0.01, lora=True),
                          _cand("c", ub=-0.01, crop_classifier=True)])]
    led = _ledger(hist)
    assert all(led[k].clearly_lost and not led[k].prescreened
               for k in ("rescorer=True", "lora=True", "crop_classifier=True"))


def test_ledger_counts_outcomes_minutes_and_merges_arm_rows():
    """The lora and classifier arms leave a screen row and a gate rejection row
    with the same id — counted once."""
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("l", lora=True, wall=18.0),
                          _cand("l", lora=True, skipped="lora_gate", wall=18.0),
                          _cand("u", tiling=2, skipped="axis_unused", wall=3.0),
                          _cand("u2", tiling=2, exemplar_k=None, skipped="axis_unused",
                                wall=2.0, alias="vlm")])]
    led = _ledger(hist)
    assert led["lora=True"].tries == 1 and led["lora=True"].status == "arm_gate"
    assert led["lora=True"].minutes == pytest.approx(18.0)
    assert led["tiling=2"].tries == 2 and led["tiling=2"].single == 1
    assert led["tiling=2"].minutes == pytest.approx(5.0)


def test_status_prefers_the_furthest_outcome():
    """Without comparison numbers, a value's status is the furthest outcome
    (holdout > screen > rejection)."""
    ev = {"e": {"micro_f_beta": 0.7, "gate": {"delta": 0.001, "lo": -0.01}}}
    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("s", alias="vlm"), _cand("e", alias="vlm",
                                                          exemplar_k=None),
                          _cand("u", alias="vlm", tiling=2, skipped="axis_unused")],
                   evaluated=ev)]
    assert _ledger(hist)["alias=vlm"].status == "evaluated"


def test_lost_values_is_a_pure_function_of_history():
    from agentic_configuration_search.planner.ledger import lost_values

    hist = [_round(0, 0, []), _round(1, 0, [_cand("a", ub=-0.01, alias="vlm")])]
    again = json.loads(json.dumps(hist))
    assert lost_values(hist).keys() == lost_values(again).keys() == {("alias", "vlm")}


def test_variant_of_lists_lost_values_a_candidate_contains():
    from agentic_configuration_search.planner.ledger import lost_values, variant_of

    hist = [_round(0, 0, []),
            _round(1, 0, [_cand("a", ub=-0.01, text_negatives="gt_crop")])]
    lost = lost_values(hist)
    champ = validate_axes({})
    k = validate_axes({"text_negatives": "gt_crop", "text_negative_k": 2})
    combo = validate_axes({"text_negatives": "gt_crop", "tiling": 2})
    other = validate_axes({"text_negatives": "fp_crop"})
    assert variant_of(k, champ, lost) == ["text_negatives=gt_crop"]
    assert variant_of(combo, champ, lost) == ["text_negatives=gt_crop"]
    assert variant_of(other, champ, lost) == []


def test_tried_under_champion_drops_old_combos_that_turn_champion_axes_off():
    from agentic_configuration_search.planner.ledger import tried_under_champion

    base = {"exemplar": "base"}
    hist = [_round(0, 0, []),
            _round(1, 1, [_cand("old_alias", alias="vlm"),
                          {**_cand("w", exemplar="base"), "adopted": True}],
                   champ=base),
            _round(2, 1, [_cand("fn", exemplar="fn_arm", fn_k=4, champ=base)],
                   champ=base)]
    lines = tried_under_champion(hist)
    assert "exemplar=fn_arm, fn_k=4" in lines          # under the current champion
    # the old combo alias=vlm (exemplar=none) turns the champion's exemplar off — left out
    assert not any("alias=vlm" in x for x in lines)


# --- screen's fit_dev comparison -----------------------------------------------------------

def test_count_rows_micro_matches_records_micro():
    from agentic_configuration_search.core import metrics
    from agentic_configuration_search.eval.screen import count_rows, micro_f_rows

    recs = [{"gts": [{"box": [0, 0, 10, 10], "category_id": 1}],
             "detections": [{"box": [0, 0, 10, 10], "score": 0.9, "category_id": 1},
                            {"box": [50, 50, 60, 60], "score": 0.9, "category_id": 2}]},
            {"gts": [{"box": [0, 0, 10, 10], "category_id": 2}], "detections": []}]
    rows = count_rows(recs, [1, 2], 0.5)
    assert rows == [[[1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]
    assert micro_f_rows(rows) == pytest.approx(
        metrics.records_micro_f_beta(recs, [1, 2], iou_thr=0.5))


def test_fit_dev_gap_identical_is_near_miss_and_uniform_loss_is_clear():
    from agentic_configuration_search.eval.screen import fit_dev_gap

    champ = [[[2, 0, 0]] for _ in range(12)]
    worse = [[[1, 1, 1]] for _ in range(12)]
    kw = dict(beta=1.0, n_boot=200, seed=0, alpha=0.05)
    same = fit_dev_gap(champ, champ, **kw)
    assert same["delta"] == 0.0 and same["ub"] >= 0 and same["n"] == 12
    lost = fit_dev_gap(champ, worse, **kw)
    assert lost["delta"] < 0 and lost["ub"] < 0


def test_fit_dev_gap_small_loss_on_few_images_is_within_noise():
    """One worse image with the rest equal puts the upper bound at 0 — with a small
    sample it is noise."""
    from agentic_configuration_search.eval.screen import fit_dev_gap

    champ = [[[2, 0, 0]] for _ in range(5)]
    cand = [[[1, 1, 1]]] + champ[1:]
    out = fit_dev_gap(champ, cand, beta=1.0, n_boot=500, seed=0, alpha=0.05)
    assert out["delta"] < 0 and out["ub"] >= 0


def test_config_knob_defaults_on():
    from agentic_configuration_search.config import config_from_dict
    assert config_from_dict({"dataset": "x"}).screen.lost_value_prescreen is True


# --- loop integration -------------------------------------------------------------------

def _neg_hurts_infer(by_path):
    """Every pass finds the GT as is (the baseline is perfect on every image), but
    when the text query carries suppression phrases (the 'look' of the _fakes
    captions) it adds two false positives per image — the suppression phrases
    lose on every fit_dev image (a clear loss)."""
    def fake(ctx, client, config, queries, ontology, image_paths, **kw):
        semantic = bool(queries) and any("prompt_image" in q for q in queries)
        neg = not semantic and "look" in json.dumps(queries, default=str)
        out = []
        for p in image_paths:
            dets = [{"box": list(a.bbox_xyxy), "score": 0.9,
                     "category_id": a.category_id, "sources": ["full"]}
                    for a in by_path[str(p)].annotations]
            if neg:
                dets += [{"box": [60.0, 50.0, 70.0, 60.0], "score": 0.95,
                          "category_id": 1, "sources": ["full"]},
                         {"box": [80.0, 60.0, 90.0, 70.0], "score": 0.95,
                          "category_id": 2, "sources": ["full"]}]
            out.append(dets)
        return out
    return fake


def _recording(fake, log):
    def wrapped(ctx, client, config, queries, ontology, image_paths, **kw):
        neg = "look" in json.dumps(queries, default=str)
        log.append((neg, len(image_paths)))
        return fake(ctx, client, config, queries, ontology, image_paths, **kw)
    return wrapped


PLAN = {1: [{"id": "neg", "axes": {"text_negatives": "gt_crop"}, "rationale": "x"}],
        2: [{"id": "neg_k2", "axes": {"text_negatives": "gt_crop",
                                      "text_negative_k": 2}, "rationale": "y"}]}


def _cfg(root, **screen):
    from agentic_configuration_search.config import config_from_dict
    return config_from_dict({
        "dataset": str(root), "rounds": 2,
        "planner": {"kind": "llm", "max_reasks": 0},
        "gate": {"bootstrap": 200},
        "saturation": {"rounds": 5, "empty_rounds": 0},
        "screen": {"prescreen_fraction": 0.5, "prescreen_min_images": 1,
                   "prescreen_max_share": 1.0, **screen}})


def _run(root, out, monkeypatch, calls=None, rounds=2, resume=False, by_path=None,
         **screen):
    fake = _neg_hurts_infer(by_path or _gt_lookup(root))
    monkeypatch.setattr(gateway, "infer_config",
                        _recording(fake, calls) if calls is not None else fake)
    run_rounds(root, out_dir=out, planner="llm", llm_chat=_llm_by_round(dict(PLAN)),
               rounds=rounds, cfg=_cfg(root, **screen), resume=resume, **_fakes())
    return _hist(out)


def test_clear_loss_is_recorded_then_variant_is_prescreened(big_root, tmp_path,  # noqa: F811
                                                            monkeypatch):
    from agentic_configuration_search.core.data import load_dataset

    from agentic_configuration_search.loop.runner import Loop

    calls: list = []
    real = Loop.run_round

    def marked(self, r, *a, **k):
        calls.append(("round", r))
        return real(self, r, *a, **k)
    monkeypatch.setattr(Loop, "run_round", marked)
    h = _run(big_root, tmp_path / "run", monkeypatch, calls)
    r1 = {c["id"]: c for c in h[1]["candidates"]}["r1_neg"]
    assert r1["fit_dev_cmp"]["delta"] < 0 and r1["fit_dev_cmp"]["ub"] < 0
    r2 = {c["id"]: c for c in h[2]["candidates"]}["r2_neg_k2"]
    assert r2["skipped"] == "lost_value_prescreen"
    assert r2["lost_values"] == ["text_negatives=gt_crop"]
    assert "text_negatives=gt_crop" in r2["note"] and "upper bound" in r2["note"]
    assert r2["fit_dev_cmp"]["ub"] < 0
    n_dev = len(load_dataset(big_root).fit_dev.images)
    assert r2["prescreen"]["n_images"] < r2["prescreen"]["of"] == n_dev
    assert "neg_k2" not in (h[2]["decision"].get("evaluated") or {})
    # round 2's suppression-phrase passes ran on the subset only — they reached
    # neither the full fit_dev nor holdout (round 1 ran the full fit_dev)
    start = calls.index(("round", 2))
    r1 = [n for neg, n in calls[:start] if neg is True]
    r2 = [n for neg, n in calls[start + 1:] if neg is True]
    assert n_dev in r1
    assert r2 and all(n < n_dev for n in r2)


def test_prescreen_switch_off_screens_variant_fully(big_root, tmp_path,  # noqa: F811
                                                   monkeypatch):
    h = _run(big_root, tmp_path / "run", monkeypatch, lost_value_prescreen=False)
    r2 = {c["id"]: c for c in h[2]["candidates"]}["r2_neg_k2"]
    assert r2.get("skipped") != "lost_value_prescreen" and "lost_values" not in r2
    assert "prescreen" not in r2


def test_prescreen_without_subset_uses_full_fit_dev(big_root, tmp_path,  # noqa: F811
                                                    monkeypatch):
    """Without a subset (fit_dev is small) it is measured on the full fit_dev —
    that still saves holdout."""
    h = _run(big_root, tmp_path / "run", monkeypatch, prescreen_fraction=0.0)
    r2 = {c["id"]: c for c in h[2]["candidates"]}["r2_neg_k2"]
    assert r2["skipped"] == "lost_value_prescreen"
    assert r2["prescreen"]["n_images"] == r2["prescreen"]["of"]


def _round_marks(monkeypatch, calls):
    from agentic_configuration_search.loop.runner import Loop

    real = Loop.run_round

    def marked(self, r, *a, **k):
        calls.append(("round", r))
        return real(self, r, *a, **k)
    monkeypatch.setattr(Loop, "run_round", marked)


def test_lost_value_skips_subset_when_it_is_most_of_fit_dev(big_root, tmp_path,  # noqa: F811
                                                            monkeypatch):
    """If the subset is screen.prescreen_max_share of fit_dev or more, the subset is
    skipped as with the tiling prescreen — the variant goes to the full fit_dev
    screen and the fit_dev comparison applies there (a lost value is rejected
    without the rescorer refit or holdout)."""
    from agentic_configuration_search.core.data import load_dataset

    calls: list = []
    _round_marks(monkeypatch, calls)
    h = _run(big_root, tmp_path / "run", monkeypatch, calls, prescreen_max_share=0.5)
    n_dev = len(load_dataset(big_root).fit_dev.images)
    r2 = {c["id"]: c for c in h[2]["candidates"]}["r2_neg_k2"]
    assert r2["skipped"] == "lost_value_prescreen" and r2["fit_dev_cmp"]["ub"] < 0
    assert r2["prescreen"]["n_images"] == r2["prescreen"]["of"] == n_dev
    assert "neg_k2" not in (h[2]["decision"].get("evaluated") or {})
    start = calls.index(("round", 2))
    r2_calls = [n for neg, n in calls[start + 1:] if neg is True]
    assert r2_calls and all(n == n_dev for n in r2_calls)   # no subset-sized passes


def test_full_fit_dev_prescreen_pass_is_not_screened_twice(big_root, tmp_path,  # noqa: F811
                                                           monkeypatch):
    """A variant measured on the full fit_dev without a subset that passes reuses
    that screen — the same fit_dev is not screened twice."""
    from agentic_configuration_search.loop import runner as runner_mod

    marks: list = []
    _round_marks(monkeypatch, marks)
    real_cmp = runner_mod.Loop._fit_dev_cmp

    def cmp(self, setting, *a, **k):
        if k.get("raw"):                       # let only the prescreen comparison pass
            return {"delta": -0.001, "ub": 0.01, "n": 1}
        return real_cmp(self, setting, *a, **k)
    monkeypatch.setattr(runner_mod.Loop, "_fit_dev_cmp", cmp)
    real_screen = runner_mod.screen_candidate

    def screen(cfg, client, split, *a, **k):
        marks.append(("screen", len(split.images)))
        return real_screen(cfg, client, split, *a, **k)
    monkeypatch.setattr(runner_mod, "screen_candidate", screen)
    h = _run(big_root, tmp_path / "run", monkeypatch, prescreen_max_share=0.5)
    r2 = {c["id"]: c for c in h[2]["candidates"]}["r2_neg_k2"]
    assert r2.get("skipped") != "lost_value_prescreen"
    assert r2["prescreen"]["passed"] is True
    assert r2["prescreen"]["n_images"] == r2["prescreen"]["of"]
    start = marks.index(("round", 2))
    assert [m for m in marks[start + 1:] if m[0] == "screen"] == \
        [("screen", r2["prescreen"]["of"])]


def test_human_candidates_are_not_prescreened(big_root, tmp_path, monkeypatch):  # noqa: F811
    from agentic_configuration_search.planner.ledger import lost_values

    fake = _neg_hurts_infer(_gt_lookup(big_root))
    monkeypatch.setattr(gateway, "infer_config", fake)
    plan = {1: PLAN[1]}
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
               llm_chat=_llm_by_round(plan), rounds=1, cfg=_cfg(big_root), **_fakes())
    assert lost_values(_hist(tmp_path / "run"))
    cand = tmp_path / "c.yaml"
    cand.write_text("- id: k2\n  axes: {text_negatives: gt_crop, text_negative_k: 2}\n"
                    "  rationale: human\n")
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
               llm_chat=_llm_by_round({}), rounds=2, cfg=_cfg(big_root),
               candidates_file=cand, resume=True, **_fakes())
    row = {c["id"]: c for c in _hist(tmp_path / "run")[2]["candidates"]}["r2_human_k2"]
    assert row.get("skipped") != "lost_value_prescreen" and "lost_values" not in row


def test_prescreen_decision_does_not_depend_on_holdout(big_root, tmp_path,  # noqa: F811
                                                       monkeypatch):
    """Perturbing the holdout GT leaves the fit_dev comparison and the prescreen
    verdict unchanged — screens are not filtered with values from holdout."""
    by_path = _gt_lookup(big_root)            # detections follow the unperturbed GT
    a = _run(big_root, tmp_path / "a", monkeypatch, by_path=by_path)
    p = big_root / "valid" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    for ann in coco["annotations"]:
        ann["category_id"] = 1 if ann["category_id"] == 2 else 2
    p.write_text(json.dumps(coco))
    b = _run(big_root, tmp_path / "b", monkeypatch, by_path=by_path)
    assert a[0]["decision"]["micro_f_beta"] != b[0]["decision"]["micro_f_beta"]

    def view(h):
        return [(c["id"], c.get("skipped"), c.get("fit_dev_cmp"), c.get("lost_values"))
                for e in h[1:] for c in e["candidates"]]
    assert view(a) == view(b)


def test_resume_reaches_the_same_prescreen_decision(big_root, tmp_path,  # noqa: F811
                                                    monkeypatch):
    """The ledger reads only history and the comparison baseline only the fit_dev
    rows in champion.json — a run split in two reaches the same verdict as one
    run in one go."""
    whole = _run(big_root, tmp_path / "whole", monkeypatch)
    _run(big_root, tmp_path / "split", monkeypatch, rounds=1)
    champ = json.loads((tmp_path / "split" / "champion.json").read_text())
    assert champ["setting"]["fit_dev_counts"]                  # the baseline is saved
    monkeypatch.setattr(gateway, "infer_config", _neg_hurts_infer(_gt_lookup(big_root)))
    plan = {1: PLAN[2]}                                        # resume's first call = round 2
    run_rounds(big_root, out_dir=tmp_path / "split", planner="llm",
               llm_chat=_llm_by_round(plan), rounds=2, cfg=_cfg(big_root),
               resume=True, **_fakes())
    split = _hist(tmp_path / "split")

    def r2(h):
        c = {c["id"]: c for c in h[2]["candidates"]}["r2_neg_k2"]
        return c["skipped"], c["fit_dev_cmp"], c["prescreen"]["n_images"]
    assert r2(whole) == r2(split)


def test_screen_records_fit_dev_counts_and_rescored_keeps_raw(monkeypatch):
    """A setting with the rescorer keeps both the rescored-channel and the
    pre-rescorer-channel rows — the prescreen of a candidate inheriting the
    rescorer runs before the refit, so it compares pre-rescorer channels."""
    from types import SimpleNamespace

    from agentic_configuration_search.loop import rescore

    raw_rows = [[[1, 0, 0]]]

    def fake_score(cfg, ds, setting, rescorer, preds, split):
        recs = [{"gts": [{"box": [0, 0, 10, 10], "category_id": 1}],
                 "detections": []}]
        return {"per_class": setting["per_class"], "micro_f_beta": 0.0,
                "records": recs}
    monkeypatch.setattr(rescore, "rescored_split_score", fake_score)
    cfg = SimpleNamespace(detector=SimpleNamespace(iou_threshold=0.5))
    ds = SimpleNamespace(fit_dev=None,
                         ontology=SimpleNamespace(classes=[SimpleNamespace(id=1)]))
    setting = {"configs": [], "per_class": {1: {}}, "fit_dev_counts": raw_rows}
    out = rescore.rescored_screen(cfg, ds, setting, None, {})
    assert out["fit_dev_counts"] == [[[0, 0, 1]]]
    assert out["fit_dev_counts_raw"] == raw_rows
    again = rescore.rescored_screen(cfg, ds, out, None, {})      # applied twice, the original is kept
    assert again["fit_dev_counts_raw"] == raw_rows


def test_loop_row_fields_reach_the_ledger_through_reask_rows():
    """The same-round rows a re-ask sees carry the fit_dev comparison too
    (_ledger_record)."""
    from agentic_configuration_search.loop.runner import _ledger_record

    c = {"fit_dev_cmp": {"delta": -0.1, "ub": -0.05, "n": 3},
         "lost_values": ["alias=vlm"], "plan": {"expected_gain": 0.01}}
    assert _ledger_record(c) == c
    assert _ledger_record({}) == {}
    assert np.isfinite(c["fit_dev_cmp"]["ub"])


def test_lost_value_prescreen_records_tiling_used(big_root, tmp_path,  # noqa: F811
                                                  monkeypatch):
    """Even if a candidate requesting tiling is rejected by the lost-value
    prescreen, that grid reached the (subset) screen — tiling_used is recorded so
    the fold gate's near-miss exception is spent."""
    from agentic_configuration_search.planner.gates import _screened

    plan = {1: PLAN[1],
            2: [{"id": "neg_t2", "axes": {"text_negatives": "gt_crop",
                                          "text_negative_k": 2, "tiling": 2},
                 "rationale": "y"}]}
    monkeypatch.setattr(gateway, "infer_config",
                        _neg_hurts_infer(_gt_lookup(big_root)))
    run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
               llm_chat=_llm_by_round(plan), rounds=2, cfg=_cfg(big_root),
               **_fakes())
    r2 = {c["id"]: c for c in _hist(tmp_path / "run")[2]["candidates"]}["r2_neg_t2"]
    assert r2["skipped"] == "lost_value_prescreen"
    assert isinstance(r2["tiling_used"], bool) and _screened(r2)
