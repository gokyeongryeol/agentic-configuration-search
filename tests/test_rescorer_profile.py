"""Per-stage time of the rescorer refit and the champion discriminator comparison
scores — decisions unchanged.

Weighing a scheme that trains the discriminator only when the champion changes
and re-picks only α per candidate needs a measure of whether it reaches the same
fit_dev verdicts as refitting. It is computed from the fit_dev predictions the
refit already has, so no inference is added.
"""

import json

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.loop.runner import run_rounds
from tests.test_loop_rescorer import _cfg, _fake_infer, _fakes
from tests.test_loop_rescorer import root  # noqa: F401 — fixture reuse
from tests.test_rescorer import _fit, _mk_preds, _mk_train, _rs


def test_fit_threads_do_not_change_the_model(tmp_path):
    """The GBDT thread count changes only the time — the same α, the same P̂."""
    base = dict(min_samples=50, min_samples_per_class=1, ensemble_members=2)
    one, info1 = _fit(tmp_path / "a", _rs(**base, fit_threads=1))
    many, info4 = _fit(tmp_path / "b", _rs(**base, fit_threads=4))
    assert one.alphas == many.alphas
    assert info1["auc_dev"] == info4["auc_dev"]
    assert set(info1["fit_s"]) == {"discriminator", "alpha"}


def test_refit_alphas_keeps_the_discriminator(tmp_path):
    """The discriminator stays and only α is re-picked — the same fit_dev gives the same α."""
    from agentic_configuration_search.fitting.rescorer import refit_alphas

    rs = _rs(min_samples=50, min_samples_per_class=1, ensemble_members=2)
    rescorer, _ = _fit(tmp_path, rs)
    dev = _mk_train(tmp_path / "dv2", "d", 12)
    again = refit_alphas(rs, rescorer.discriminator,
                         dev_preds=_mk_preds(dev, "d", seed=1), dev_split=dev,
                         dev_image_paths=[dev.image_path(e) for e in dev.images])
    assert again.discriminator is rescorer.discriminator
    assert again.alphas == rescorer.alphas


def test_history_records_refit_minutes(root, tmp_path, monkeypatch):  # noqa: F811
    """A candidate row that fit the rescorer keeps minutes per stage (rescorer_min).
    Round 1's champion has no rescorer, so there are no comparison scores (rescorer_cmp)."""
    monkeypatch.setattr(gateway, "infer_config", _fake_infer(root))
    out = tmp_path / "run"
    run_rounds(root, out_dir=out, planner="deterministic", rounds=1,
               cfg=_cfg(root), **_fakes())
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    refit = [c for c in rows[1]["candidates"] if "rescorer_min" in c]
    assert refit, rows[1]["candidates"]
    for c in refit:
        assert set(c["rescorer_min"]) == {"infer_train", "infer_dev",
                                          "discriminator", "alpha"}
        assert "rescorer_cmp" not in c


def _setting_and_ds(tmp_path, rs):
    from pathlib import Path
    from types import SimpleNamespace

    from agentic_configuration_search.config import config_from_dict

    dev = _mk_train(tmp_path / "dv2", "d", 12)
    train = _mk_train(tmp_path / "tr2", "t", 24)
    cfg = config_from_dict({"dataset": "x", "rescorer": rs})
    ds = SimpleNamespace(train=train, fit_dev=dev, ontology=train.ontology,
                         root=Path("/data/demo"))
    setting = {"configs": [], "per_class": {1: {}, 2: {}}}
    return cfg, ds, setting, _mk_preds(dev, "d", seed=1)


def test_champion_scores_on_the_candidate_dev_preds(tmp_path):
    """Two scores under the champion's discriminator — the champion's α as is, and α
    re-picked. With a rescorer fit on the same fit_dev as the champion, both equal
    the refit score."""
    from agentic_configuration_search.loop import rescore

    rs = dict(min_samples=50, min_samples_per_class=1, ensemble_members=2)
    rescorer, _ = _fit(tmp_path, _rs(**rs))
    cfg, ds, setting, preds = _setting_and_ds(tmp_path, rs)
    own = rescore.rescored_split_score(cfg, ds, setting, rescorer, preds,
                                       ds.fit_dev)["micro_f_beta"]
    got = rescore.champion_scores(cfg, ds, setting, rescorer, preds, ds.fit_dev)
    assert set(got) == {"champion_rescorer", "champion_disc_alpha"}
    assert got["champion_rescorer"] == own and got["champion_disc_alpha"] == own


def test_rescorer_record_is_artifact_only():
    """The records are in history rows only — not in the planner prompt's history."""
    from agentic_configuration_search.loop.runner import _rescorer_record
    from agentic_configuration_search.planner.llm import _compact_history

    c = {"_rescorer_min": {"infer_train": 1.0}, "_rescorer_cmp": {"refit": 0.5}}
    assert _rescorer_record(c) == {"rescorer_min": {"infer_train": 1.0},
                                   "rescorer_cmp": {"refit": 0.5}}
    assert _rescorer_record({}) == {}
    row = {"id": "c", "axes": {"rescorer": True}, "fit_dev_micro_f_beta": 0.5,
           "rescorer_min": {"infer_train": 1.0}, "rescorer_cmp": {"refit": 0.5}}
    hist = [{"round": 1, "champion": {"axes": {}}, "candidates": [row],
             "decision": {"adopted": False}}]
    text = json.dumps(_compact_history(hist), default=str)
    assert "rescorer_min" not in text and "rescorer_cmp" not in text


def test_loop_compares_against_the_saved_champion_rescorer(tmp_path):
    """The runner reads the artifact at champion.rescorer_path and puts the comparison
    scores on the candidate — with no champion rescorer it puts nothing."""
    from types import SimpleNamespace

    from agentic_configuration_search.loop import rescore
    from agentic_configuration_search.loop.runner import Loop

    rs = dict(min_samples=50, min_samples_per_class=1, ensemble_members=2)
    rescorer, _ = _fit(tmp_path, _rs(**rs))
    rescorer.save(tmp_path / "rescorer.json")
    cfg, ds, setting, preds = _setting_and_ds(tmp_path, rs)
    refit = rescore.rescored_screen(cfg, ds, setting, rescorer, preds)
    me = SimpleNamespace(cfg=cfg, ds=ds,
                         _champion={"rescorer_path": str(tmp_path / "rescorer.json")})
    entry = {}
    Loop._rescorer_compare(me, entry, setting, preds, refit)
    assert entry["_rescorer_cmp"]["refit"] == refit["fit_dev_micro_f_beta"]
    assert set(entry["_rescorer_cmp"]) == {"refit", "champion_rescorer",
                                           "champion_disc_alpha"}
    me._champion = {}
    entry = {}
    Loop._rescorer_compare(me, entry, setting, preds, refit)
    assert entry == {}


def test_compare_failure_is_recorded_not_raised(tmp_path):
    """The comparison is record-only — when it fails the candidate goes on and an error is kept."""
    from pathlib import Path
    from types import SimpleNamespace

    from agentic_configuration_search.loop.runner import Loop

    me = SimpleNamespace(cfg=None, ds=SimpleNamespace(root=Path("/data/demo"),
                                                      ontology=None),
                         _champion={"rescorer_path": str(tmp_path / "missing.json")})
    entry = {"id": "c"}
    Loop._rescorer_compare(me, entry, {}, {}, {"fit_dev_micro_f_beta": 0.5})
    assert entry["_rescorer_cmp"]["refit"] == 0.5 and "error" in entry["_rescorer_cmp"]


def test_fit_threads_apply_from_the_first_fit():
    """threadpool_limits changes only the OpenMP loaded at that time — if sklearn were
    first imported inside the block, the first fit would run on one thread. Checked in
    a new process (one thread, like the entry point)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "from threadpoolctl import threadpool_limits\n"
        "import agentic_configuration_search.fitting.rescorer\n"
        "with threadpool_limits(limits=4, user_api='openmp'):\n"
        "    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads\n"
        "    print(_openmp_effective_n_threads())\n")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": src}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "4", out.stderr


def test_record_only_time_is_not_charged_to_the_candidate():
    """The comparison time is kept out of the cost the planner sees (wall_min) — it is record-only."""
    import time

    from agentic_configuration_search.loop.runner import _close_wall, _wall_min

    wall = {"materialize": 60.0, "_record_only": 30.0}
    _close_wall(wall, time.monotonic() - 120.0, ())
    assert abs(wall["screen"] - 30.0) < 1.0
    got = _wall_min(wall)
    assert "_record_only" not in got
    assert abs(got["total"] - 1.5) < 0.02


def test_compare_time_goes_to_rescorer_min(tmp_path):
    from types import SimpleNamespace

    from agentic_configuration_search.loop import rescore
    from agentic_configuration_search.loop.runner import Loop

    rs = dict(min_samples=50, min_samples_per_class=1, ensemble_members=2)
    rescorer, _ = _fit(tmp_path, _rs(**rs))
    rescorer.save(tmp_path / "rescorer.json")
    cfg, ds, setting, preds = _setting_and_ds(tmp_path, rs)
    refit = rescore.rescored_screen(cfg, ds, setting, rescorer, preds)
    me = SimpleNamespace(cfg=cfg, ds=ds,
                         _champion={"rescorer_path": str(tmp_path / "rescorer.json")})
    entry = {"_wall": {}, "_rescorer_min": {"infer_train": 1.0}}
    Loop._rescorer_compare(me, entry, setting, preds, refit)
    assert entry["_wall"]["_record_only"] > 0
    assert entry["_rescorer_min"]["cmp"] == round(entry["_wall"]["_record_only"] / 60, 2)


def test_arm_rows_keep_their_records_in_history():
    """lora and classifier arms put the row (not the entry) into screened, with the
    records under their output keys — the function building history rows must read
    those keys too."""
    from agentic_configuration_search.loop.runner import (_classifier_lambda_record,
                                                          _rescorer_record)

    row = {"rescorer_min": {"infer_train": 1.0}, "rescorer_cmp": {"refit": 0.5}}
    assert _rescorer_record(row) == row
    lora_row = {"lora": {"adopted": True}, "classifier_lambda": 0.75,
                "classifier_lambda_grid": [{"lam": 0.75}]}
    assert _classifier_lambda_record(lora_row) == {
        "classifier_lambda": 0.75, "classifier_lambda_grid": [{"lam": 0.75}]}
