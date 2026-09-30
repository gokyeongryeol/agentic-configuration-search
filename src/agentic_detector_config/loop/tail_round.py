"""Round wiring for the lora axis — YOLOE tail fine-tune + λ-interpolation arms.

Contract: train on fit_train, score λ arms on fit_dev (screen_fn is re-run —
adapted weights are upstream of fixing the setting), margin gate. If adopted,
the in-process client is left on the winning arm and `weights` carries the arm
descriptor path; otherwise restore_weights is restored. The λ=0 arm is not
scored — the baseline measured in the restored state is that value.

Grid entries are filled with preset="tail", rank=0 — tail has no preset/rank
grid (its boundary is the single TAIL_MODULES in workers/yoloe_tail.py), but
this keeps the lora decision record format in history.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..fitting.tail import build_tail_arms, select_candidate, train_tail

log = logging.getLogger(__name__)


def run_tail_candidate(cfg, ds, base_axes: dict, materials, out_dir: Path, *,
                       screen_fn, weights_ctl, base_weights: Path,
                       restore_weights=None, seed: int) -> dict:
    """Score the λ grid on fit_dev — {"decision", "weights", "grid", "setting"}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(ds.fit_train.images) < cfg.lora.min_images:
        dec = {"adopted": False,
               "reason": f"fit_train {len(ds.fit_train.images)} images < "
                         f"min_images {cfg.lora.min_images} — training skipped"}
        _write(out_dir, dec, [])
        return {"decision": dec, "weights": None, "grid": [], "setting": None}

    weights_ctl(restore_weights)          # baseline = current serving state
    base_setting = screen_fn(materials, base_axes)
    base_f1 = base_setting["fit_dev_micro_f_beta"]
    grid = [{"source": "grid", "preset": "tail", "rank": 0, "lam": 0.0,
             "f_beta": base_f1, "delta": 0.0}]

    tuned = train_tail(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
                       base_weights=base_weights, out_dir=out_dir / "tail",
                       seed=seed, log_path=out_dir / "tail" / "train.log")
    arms = build_tail_arms(tuned, cfg.lora.lambdas, out_dir / "arms")
    arm_settings: dict[float, dict] = {}
    for lam in sorted(arms):
        weights_ctl(arms[lam])
        setting = screen_fn(materials, base_axes)
        f1 = setting["fit_dev_micro_f_beta"]
        grid.append({"source": "grid", "preset": "tail", "rank": 0,
                     "lam": lam, "f_beta": f1, "delta": f1 - base_f1})
        arm_settings[lam] = setting
        log.info("tail arm λ=%g: fit_dev micro %.4f (Δ%+.4f)",
                 lam, f1, f1 - base_f1)

    best, dec = select_candidate(grid, cfg.lora.margin)
    _write(out_dir, dec, grid)
    if dec.get("adopted"):
        lam = best["lam"]
        weights_ctl(arms[lam])            # switch to the adopted arm — caller runs holdout eval
        return {"decision": dec, "weights": arms[lam], "grid": grid,
                "setting": arm_settings[lam]}
    weights_ctl(restore_weights)          # not adopted — restore current
    return {"decision": dec, "weights": None, "grid": grid,
            "setting": base_setting}


def _write(out_dir: Path, dec: dict, grid: list) -> None:
    (out_dir / "tail_decision.json").write_text(
        json.dumps({"decision": dec, "grid": grid}, indent=2,
                   ensure_ascii=False, default=str))
