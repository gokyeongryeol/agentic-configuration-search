"""Round wiring for the lora axis — YOLOE tail fine-tune + λ-interpolation arms.

Contract: train on fit_train, score λ arms on fit_dev (screen_fn is re-run —
adapted weights are upstream of fixing the setting), margin gate. If adopted,
the in-process client is left on the winning arm and `weights` carries the arm
descriptor path; otherwise restore_weights is restored. The λ=0 arm is not
scored — the baseline measured in the restored state is that value.

Grid entries are filled with preset="tail", rank=0 — tail has no preset/rank
grid (its boundary is the single TAIL_MODULES in workers/yoloe_tail.py), but
this keeps the lora decision record format in history.

Training reuse: the training inputs (fit_train GT, base weights, seed, training
hyperparameters — fitting.tail.train_key) do not depend on the champion, so
they are the same throughout a run. When the runner passes cache_dir
(`<run>/lora_cache/`), a tuned subset with the same key is reused and later
lora candidates (carried ones included) skip training, only rebuilding the λ
arm descriptors and re-scoring them (_train_or_reuse). The crop classifier is
not reused — its training crops are mined from the current setting's FPs and
change with the champion (loop.adapt_round).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from ..fitting.tail import (build_tail_arms, key_digest, select_candidate,
                            train_key, train_tail)

log = logging.getLogger(__name__)

# completion mark of a run-level training cache entry — written only after
# training finished and the tuned subset exists
CACHE_META = "key.json"


def run_tail_candidate(cfg, ds, base_axes: dict, materials, out_dir: Path, *,
                       screen_fn, weights_ctl, base_weights: Path,
                       restore_weights=None, seed: int,
                       cache_dir: Path | None = None,
                       origin: dict | None = None) -> dict:
    """Score the λ grid on fit_dev — {"decision", "weights", "grid", "setting"}.

    With cache_dir, training is reused per run (_train_or_reuse) — if a tuned
    subset with the same key exists, the arms are rebuilt and re-scored without
    training, and the decision records `delta_reused` ({"tail": origin of the
    candidate that trained it first}). With None, every call trains in the
    candidate directory."""
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

    tuned, source = _train_or_reuse(cfg, ds, base_weights=base_weights, seed=seed,
                                    out_dir=out_dir / "tail", cache_dir=cache_dir,
                                    origin=origin)
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
    if source is not None:
        dec["delta_reused"] = {"tail": source}
    _write(out_dir, dec, grid)
    if dec.get("adopted"):
        lam = best["lam"]
        weights_ctl(arms[lam])            # switch to the adopted arm — caller runs holdout eval
        return {"decision": dec, "weights": arms[lam], "grid": grid,
                "setting": arm_settings[lam]}
    weights_ctl(restore_weights)          # not adopted — restore current
    return {"decision": dec, "weights": None, "grid": grid,
            "setting": base_setting}


def _train_or_reuse(cfg, ds, *, base_weights, seed, out_dir: Path,
                    cache_dir: Path | None,
                    origin: dict | None) -> tuple[Path, dict | None]:
    """The tuned subset path and the reuse origin (None if freshly trained).

    The training inputs (fitting.tail.train_key) do not depend on the champion
    within a run — training always starts from the base weights on fit_train
    GT. So if `cache_dir/<digest>/` holds a subset with the same key, nothing is
    trained. A cache entry writes CACHE_META (key, origin, subset path) only
    after training finished — a directory without the mark is the remains of an
    interrupted training and is removed and retrained. The cache lives in the
    run directory, so it is reused after resume too; arm descriptors point at
    the cached subset, and nothing in the run deletes it."""
    if cache_dir is None:
        return train_tail(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
                          base_weights=base_weights, out_dir=out_dir, seed=seed,
                          log_path=out_dir / "train.log"), None
    key = train_key(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
                    base_weights=base_weights, seed=seed)
    entry = Path(cache_dir) / key_digest(key)[:16]
    meta = entry / CACHE_META
    if meta.exists():
        rec = json.loads(meta.read_text())
        tuned = Path(rec.get("tuned") or "")
        if rec.get("key") == key and tuned.is_file():
            log.info("tail: reusing the tuned subset of the same training inputs — "
                     "%s (first trained: %s)", tuned, rec.get("origin"))
            return tuned, dict(rec.get("origin") or {})
    if entry.exists():
        shutil.rmtree(entry)             # remains of an interrupted training (or a key collision)
    tuned = train_tail(cfg, fit_train=ds.fit_train, ontology=ds.ontology,
                       base_weights=base_weights, out_dir=entry, seed=seed,
                       log_path=entry / "train.log")
    # the mark goes last — if the process dies mid-training there is no mark and
    # the next call retrains
    entry.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"key": key, "origin": origin or {},
                                "tuned": os.path.abspath(str(tuned))},
                               indent=2, ensure_ascii=False))
    return Path(tuned), None


def _write(out_dir: Path, dec: dict, grid: list) -> None:
    (out_dir / "tail_decision.json").write_text(
        json.dumps({"decision": dec, "grid": grid}, indent=2,
                   ensure_ascii=False, default=str))
