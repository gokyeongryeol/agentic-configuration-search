"""A candidate that clearly lost on fit_dev or equals the champion is not sent to
the holdout gate.

Why: in validation runs, screen sent the top top_k (2) by fit_dev to holdout
regardless of how they fared against the champion, so candidates whose
`fit_dev_cmp` was a clear loss (ub < 0) or exactly the champion (delta = ub = 0)
reused holdout across three datasets.

The verdict reads only fit_dev (per-image (tp, fp, fn) counted at screen
thresholds) — holdout plays no part.
"""

import json

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import Loop, _holdout_pick, run_rounds
from tests.test_loop import _gt_lookup, _hist, _fakes
from tests.test_loop import big_root  # noqa: F401 — fixture reuse
from tests.test_tail import root  # noqa: F401 — fixture reuse
from tests.test_lost_value_prescreen import _neg_hurts_infer
from tests.test_time_efficiency import _llm_by_round


# --- pick rule (pure function) ---------------------------------------------------------

def _e(cid, fd, cmp=None):
    out = {"id": cid, "fit_dev_micro_f_beta": fd}
    if cmp is not None:
        out["fit_dev_cmp"] = cmp
    return out


def test_pick_skips_clear_losers_and_identical_then_refills_top_k():
    """Clear losers (ub < 0) and candidates equal to the champion drop out with a
    reason, and the next candidate fills the freed holdout slot — top_k counts
    only sendable candidates."""
    same = _e("same", 0.80, {"delta": 0.0, "ub": 0.0, "n": 9, "identical": True})
    lost = _e("lost", 0.79, {"delta": -0.03, "ub": -0.01, "n": 9})
    near = _e("near", 0.70, {"delta": -0.01, "ub": 0.02, "n": 9})
    nocmp = _e("nocmp", 0.60)                    # without comparison rows nothing is filtered
    tail = _e("tail", 0.50, {"delta": 0.01, "ub": 0.03, "n": 9})
    top = _holdout_pick([tail, lost, nocmp, same, near], top_k=2, skip=True)
    assert [c["id"] for c in top] == ["near", "nocmp"]
    assert same["skipped"] == "fit_dev_identical" and "note" in same
    assert lost["skipped"] == "fit_dev_lost" and "-0.0100" in lost["note"]
    assert not any(c.get("skipped") for c in (near, nocmp, tail))


def test_pick_rounded_zero_delta_is_not_identity():
    """Even when delta and ub round to 0, rows that are not the same (no identical
    mark) are not filtered — identity is only an exact per-image row match."""
    c = _e("c", 0.8, {"delta": 0.0, "ub": 0.0, "n": 9})
    assert _holdout_pick([c], top_k=2, skip=True) == [c]
    assert not c.get("skipped")


def test_pick_switch_off_keeps_old_top_k():
    lost = _e("lost", 0.9, {"delta": -0.03, "ub": -0.01, "n": 9})
    other = _e("other", 0.5)
    assert _holdout_pick([other, lost], top_k=1, skip=False) == [lost]
    assert not lost.get("skipped")


def test_fit_dev_cmp_marks_exact_identity_only():
    """Identity is an exact match of every per-image fit_dev (tp, fp, fn) row."""
    from types import SimpleNamespace

    rows = [[[2, 0, 0], [1, 1, 0]] for _ in range(6)]
    fake = SimpleNamespace(
        _champion={"setting": {"fit_dev_counts": rows}},
        cfg=config_from_dict({"dataset": "x"}), ds=None)
    same = Loop._fit_dev_cmp(fake, {"fit_dev_counts": json.loads(json.dumps(rows))})
    assert same["identical"] is True and same["delta"] == 0.0
    other = [list(map(list, r)) for r in rows]
    other[0][1] = [1, 0, 1]
    diff = Loop._fit_dev_cmp(fake, {"fit_dev_counts": other})
    assert "identical" not in diff


def test_config_switch_default_on():
    from agentic_configuration_search.config import ScreenCfg
    assert ScreenCfg().skip_fit_dev_losers is True
    assert config_from_dict({"dataset": "x"}).screen.skip_fit_dev_losers is True
    off = config_from_dict({"dataset": "x",
                            "screen": {"skip_fit_dev_losers": False}})
    assert off.screen.skip_fit_dev_losers is False


# --- loop integration -------------------------------------------------------------------

PLAN = {1: [{"id": "neg", "axes": {"text_negatives": "gt_crop"}, "rationale": "x"}]}


def _cfg(root, **screen):
    return config_from_dict({
        "dataset": str(root), "rounds": 1,
        "planner": {"kind": "llm", "max_reasks": 0},
        "gate": {"bootstrap": 200},
        "saturation": {"rounds": 5, "empty_rounds": 0},
        "screen": {"lost_value_prescreen": False, **screen}})


def _run(root, out, monkeypatch, plan=PLAN, **screen):
    monkeypatch.setattr(gateway, "infer_config", _neg_hurts_infer(_gt_lookup(root)))
    evaluated: list = []
    real = Loop._evaluate_candidate

    def spy(self, entry, *a, **k):
        evaluated.append(entry["id"])
        return real(self, entry, *a, **k)
    monkeypatch.setattr(Loop, "_evaluate_candidate", spy)
    run_rounds(root, out_dir=out, planner="llm", llm_chat=_llm_by_round(dict(plan)),
               rounds=1, cfg=_cfg(root, **screen), **_fakes())
    return _hist(out), evaluated


def test_clear_fit_dev_loser_never_reaches_holdout(big_root, tmp_path,  # noqa: F811
                                                   monkeypatch):
    h, evaluated = _run(big_root, tmp_path / "run", monkeypatch)
    row = {c["id"]: c for c in h[1]["candidates"]}["r1_neg"]
    assert row["skipped"] == "fit_dev_lost"
    assert row["fit_dev_cmp"]["ub"] < 0 and "holdout" in row["note"]
    # the screen record stays — the ledger counts it as a clear loss
    assert row["fit_dev_micro_f_beta"] is not None and row.get("setting_fp")
    assert "r1_neg" not in evaluated
    d = h[1]["decision"]
    assert "r1_neg" in d["skipped"] and not d.get("evaluated")
    from agentic_configuration_search.planner.ledger import lost_values
    assert ("text_negatives", "gt_crop") in lost_values(h)


def test_switch_off_sends_the_loser_to_holdout(big_root, tmp_path,  # noqa: F811
                                               monkeypatch):
    h, evaluated = _run(big_root, tmp_path / "run", monkeypatch,
                        skip_fit_dev_losers=False)
    row = {c["id"]: c for c in h[1]["candidates"]}["r1_neg"]
    assert not row.get("skipped") and evaluated == ["r1_neg"]


def test_identical_candidate_never_reaches_holdout(big_root, tmp_path,  # noqa: F811
                                                   monkeypatch):
    """If every per-image fit_dev row equals the champion's (even with a different
    serving-state fingerprint), it is not sent to holdout — fit_dev_identical."""
    real = Loop._fit_dev_cmp

    def ident(self, setting, *a, **k):
        champ = self._champion["setting"]
        return real(self, {**setting, "fit_dev_counts": champ["fit_dev_counts"]},
                    *a, **k)
    monkeypatch.setattr(Loop, "_fit_dev_cmp", ident)
    h, evaluated = _run(big_root, tmp_path / "run", monkeypatch)
    row = {c["id"]: c for c in h[1]["candidates"]}["r1_neg"]
    assert row["skipped"] == "fit_dev_identical"
    assert row["fit_dev_cmp"]["identical"] is True and evaluated == []


def test_holdout_skip_does_not_depend_on_holdout(big_root, tmp_path,  # noqa: F811
                                                 monkeypatch):
    """Perturbing the holdout GT leaves the skip verdict unchanged — it reads only
    fit_dev."""
    a, _ = _run(big_root, tmp_path / "a", monkeypatch)
    p = big_root / "valid" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    for ann in coco["annotations"]:
        ann["category_id"] = 1 if ann["category_id"] == 2 else 2
    p.write_text(json.dumps(coco))
    b, _ = _run(big_root, tmp_path / "b", monkeypatch)

    def view(h):
        return [(c["id"], c.get("skipped"), c.get("fit_dev_cmp"))
                for c in h[1]["candidates"]]
    assert view(a) == view(b)


def test_lora_arm_that_lost_on_fit_dev_skips_holdout(root, tmp_path,  # noqa: F811
                                                    monkeypatch):
    """The lora arm follows the same rule — even past the arm's fit_dev margin
    gate, if the (winning arm − champion) fit_dev comparison is a clear loss it
    is not sent to holdout, and the champion's weights stay served."""
    from agentic_configuration_search.loop import tail_round
    from tests.test_tail import FakeDetector, _fake_infer as tail_infer
    from tests.test_tail import _fakes as tail_fakes

    server = FakeDetector()
    monkeypatch.setattr(gateway, "infer_config", tail_infer(root, server))

    def fake_train(cfg, *, out_dir, **kw):
        p = tmp_path / "tuned" / "tuned_tail.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
        return p
    monkeypatch.setattr(tail_round, "train_tail", fake_train)
    monkeypatch.setattr(Loop, "_fit_dev_cmp",
                        lambda self, setting, **k: {"delta": -0.2, "ub": -0.1, "n": 4})
    evaluated: list = []
    real = Loop._evaluate_candidate
    monkeypatch.setattr(Loop, "_evaluate_candidate",
                        lambda self, entry, *a, **k: (evaluated.append(entry["id"]),
                                                      real(self, entry, *a, **k))[1])
    cfg = config_from_dict({
        "planner": {"kind": "deterministic"},
        "dataset": str(root),
        "lora": {"lambdas": [0.0, 1.0], "margin": 0.01, "min_images": 4}})
    cand = tmp_path / "c.yaml"
    cand.write_text("- id: adapt\n  axes: {lora: true}\n  rationale: x\n")
    result = run_rounds(
        root, out_dir=tmp_path / "run", planner="deterministic", rounds=1,
        cfg=cfg, candidates_file=cand,
        detector_hooks={"weights_ctl": server.ctl, "repo_root": tmp_path,
                        "base_weights": tmp_path / "yoloe.pt"},
        **tail_fakes())
    # the deterministic planner may propose the same lora combination — look by axes
    rows = [c for c in _hist(tmp_path / "run")[1]["candidates"]
            if c["axes"].get("lora") and c.get("skipped") != "combo_tried"]
    assert rows and any(c.get("skipped") == "fit_dev_lost" for c in rows)
    assert not evaluated
    assert result.champion["weights"] is None and server.weights is None


# --- empty rounds — a round dropped by the fit_dev verdict is not empty ----------------

def test_fit_dev_skipped_round_is_not_empty_but_is_stagnant():
    """The skip only saves holdout time; it does not change when the run stops —
    a round that missed holdout only because of the fit_dev verdict
    (fit_dev_lost, fit_dev_identical) would have measured something on holdout
    without the skip. It is not counted as empty, but as stagnant since nothing
    was adopted. The in-run check (_empty_round) and the resume restore
    (_trailing_empty) agree."""
    from agentic_configuration_search.loop.runner import (_empty_round, _stagnant,
                                                          _trailing_empty)

    base = {"round": 0, "decision": {"kind": "baseline", "adopted": True}}
    skip = {"round": 1, "decision": {"adopted": False, "evaluated": {},
                                     "fit_dev_skipped": ["r1_neg"]}}
    empty = {"round": 2, "decision": {"adopted": False, "evaluated": {}}}
    assert not _empty_round(skip["decision"]) and _stagnant(skip["decision"], 0.0)
    assert _trailing_empty([base, skip]) == 0
    assert _trailing_empty([base, empty, skip]) == 0
    assert _trailing_empty([base, skip, empty]) == 1


def test_fit_dev_skip_rounds_do_not_trigger_empty_round_stop(big_root, tmp_path,  # noqa: F811
                                                             monkeypatch):
    plan = {1: PLAN[1],
            2: [{"id": "neg_k2", "axes": {"text_negatives": "gt_crop",
                                          "text_negative_k": 2}, "rationale": "y"}]}
    monkeypatch.setattr(gateway, "infer_config", _neg_hurts_infer(_gt_lookup(big_root)))
    cfg = config_from_dict({
        "dataset": str(big_root), "rounds": 2,
        "planner": {"kind": "llm", "max_reasks": 0},
        "gate": {"bootstrap": 200},
        "saturation": {"rounds": 5, "empty_rounds": 1},
        "screen": {"lost_value_prescreen": False}})
    out = run_rounds(big_root, out_dir=tmp_path / "run", planner="llm",
                     llm_chat=_llm_by_round(plan), rounds=2, cfg=cfg, **_fakes())
    h = _hist(tmp_path / "run")
    d1 = h[1]["decision"]
    assert not d1.get("evaluated") and d1["fit_dev_skipped"] == ["r1_neg"]
    assert out.stopped == "rounds" and [d["round"] for d in out.decisions] == [0, 1, 2]
