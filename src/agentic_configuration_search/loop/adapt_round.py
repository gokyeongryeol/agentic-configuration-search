"""Round wiring for the crop classifier axis — trained on fit_train, λ grid scored on fit_dev.

Like lora, this is a "detector-state candidate": it loads an
artifact into the in-process detector client, so the runner handles it after
all other candidates are evaluated, and restores the restore state (the current
champion's artifact and blend λ, or None) when not adopted.

**Blend λ grid**: the client gives per-box class probabilities and the loop
blends the score as s^(1-λ)·p(c*)^λ (infer.gateway.blend_classifier). The
artifact is loaded once and fit_dev is re-screened for each λ in
crop_classifier.lambdas; the best ΔF(β) > margin wins (ties go to the smaller λ
— the one that moved less). λ=0 is the relabel-only arm (scores unchanged), and
no classifier (current state) stays on the grid as lam=None — the baseline
measured in the restore state. Mixing the two would make "loaded but scores
unchanged" and "not loaded" indistinguishable. Changing only λ leaves the
detector response (the probabilities) the same, so the artifact is not
reloaded and the pass cache serves the re-inference.

**Training is reused within the run** — with cache_dir
(`<run>/classifier_cache/`), the artifact trained first is reused by later
proposals without mining or training, and only the λ grid is re-scored on the
new champion. The background crops were mined on the champion setting of the
first training, so the material drifts when the champion changes; instead of
retraining every time (tens of minutes), λ absorbs that drift
(fitting.crop_classifier.train_key). A tripwire (too little material) is not
cached — a new champion can yield more FPs and enough to train.

**Purity**: the background-mining threshold is base_setting's per_class
threshold — **estimated on fit_dev**. Using tau_test (the holdout estimate)
would let the holdout pick training material and break material purity. This
module never receives the holdout split.

**Training uses fit_train only** — the arm is scored by its fit_dev Δ, so fit_dev
crops in the training material would make scoring an in-sample comparison and
the margin would filter nothing. Same discipline as lora (trained on fit_train,
scored on fit_dev).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from ..eval.runner import chosen_groups, combined_detections, infer_for_groups
from ..fitting.crop_classifier import (mine_classifier_crops, train_crop_classifier,
                                       train_key)
from ..fitting.tail import key_digest

log = logging.getLogger(__name__)

# completion mark of a run-level training cache entry — written only after
# training finished and the artifact exists
CACHE_META = "key.json"


def run_classifier_candidate(cfg, ds, base_axes: dict, materials, out_dir: Path, *,
                             screen_fn, adapt_ctl, repo_root: Path,
                             restore_root: Path | None, seed: int, client,
                             restore_lambda: float | None = None, train_fn=None,
                             cache_dir: Path | None = None,
                             origin: dict | None = None) -> dict:
    """λ grid — {"decision", "adapt_root", "adapt_lambda", "grid", "setting"}.

    adapt_ctl(path, λ) sets the client's artifact and blend λ together
    (path=None is no classifier, λ ignored). If adopted, the client is left in
    the (artifact, λ*) state (the caller then runs holdout eval). If not,
    (restore_root, restore_lambda) is restored and adapt_root=None. train_fn is
    a test injection point — defaults to
    fitting.crop_classifier.train_crop_classifier.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cc = cfg.crop_classifier

    adapt_ctl(restore_root, restore_lambda)   # baseline = current serving state
    base_setting = screen_fn(materials, base_axes)
    base_f = base_setting["fit_dev_micro_f_beta"]
    grid = [{"lam": None, "f_beta": base_f, "delta": 0.0}]   # baseline: no classifier (current)

    artifact, counts, source = _train_or_reuse(
        cfg, ds, base_setting, materials, out_dir, client=client,
        repo_root=repo_root, train_fn=train_fn or train_crop_classifier,
        cache_dir=cache_dir, origin=origin)
    if artifact is None:
        # tripwire=True — the runner records this axis as empty material
        # (empty_axis). It must be told apart from a margin rejection (no effect)
        # so the planner reads the two differently
        total = counts["train_gt"] + counts["train_fp"]
        dec = {"adopted": False, "tripwire": True,
               "reason": f"{total} crops (<{cc.min_samples}) or "
                         f"{counts['train_fp']} background crops (<{cc.min_background}) "
                         "— training skipped",
               "counts": counts}
        _write(out_dir, dec, grid)
        return {"decision": dec, "adapt_root": None, "adapt_lambda": None,
                "grid": grid, "setting": base_setting}

    settings: dict = {}
    # the artifact stays loaded and only λ (the loop-side blend) changes, so the
    # detector responses are the same for every arm — the pass cache serves them
    for lam in sorted(float(x) for x in cc.lambdas):
        adapt_ctl(artifact, lam)
        setting = screen_fn(materials, base_axes)
        f = setting["fit_dev_micro_f_beta"]
        grid.append({"lam": lam, "f_beta": f, "delta": f - base_f})
        settings[lam] = setting
        log.info("crop classifier arm λ=%g: fit_dev micro %.4f (Δ%+.4f)",
                 lam, f, f - base_f)

    best = max(grid[1:], key=lambda g: (g["delta"], -g["lam"]))
    dec = {"counts": counts, "lam": best["lam"], "delta": best["delta"]}
    if source is not None:
        dec["classifier_reused"] = source
    if best["delta"] > cc.margin:
        dec |= {"adopted": True,
                "reason": f"fit_dev ΔF(β) {best['delta']:+.4f} > margin {cc.margin} "
                          f"(λ={best['lam']:g})"}
        adapt_ctl(artifact, best["lam"])      # to the adopted arm — the caller runs holdout
        _write(out_dir, dec, grid, artifact)
        return {"decision": dec, "adapt_root": Path(artifact),
                "adapt_lambda": best["lam"], "grid": grid,
                "setting": settings[best["lam"]]}
    adapt_ctl(restore_root, restore_lambda)   # not adopted — restore current
    dec |= {"adopted": False,
            "reason": f"fit_dev ΔF(β) {best['delta']:+.4f} ≤ margin {cc.margin} "
                      f"(best λ={best['lam']:g})"}
    _write(out_dir, dec, grid, artifact)
    return {"decision": dec, "adapt_root": None, "adapt_lambda": None,
            "grid": grid, "setting": base_setting}


def _train_or_reuse(cfg, ds, base_setting: dict, materials, out_dir: Path, *,
                    client, repo_root: Path, train_fn, cache_dir: Path | None,
                    origin: dict | None) -> tuple[Path | None, dict, dict | None]:
    """(artifact, mining counts, reuse origin) — artifact is None on a tripwire.

    If the entry with the same key (fitting.crop_classifier.train_key) in
    cache_dir has its completion mark, neither mining nor training runs. The
    mark is written only after training, so an entry without it is the remains
    of an interrupted training and is removed and retrained. With cache_dir
    None every call mines and trains in the candidate directory."""
    entry = meta = key = None
    if cache_dir is not None:
        key = train_key(fit_train=ds.fit_train, ontology=ds.ontology)
        entry = Path(cache_dir) / key_digest(key)[:16]
        meta = entry / CACHE_META
        if meta.exists():
            rec = json.loads(meta.read_text())
            artifact = Path(rec.get("artifact") or "")
            if rec.get("key") == key and (artifact / "classifier.pt").exists():
                log.info("crop classifier: reusing the artifact trained in this run — %s "
                         "(first trained: %s)", artifact, rec.get("origin"))
                return artifact, dict(rec.get("counts") or {}), \
                    dict(rec.get("origin") or {})
        if entry.exists():
            shutil.rmtree(entry)              # remains of an interrupted training (or a key collision)

    # fit_train pass (base_setting configs as-is) -> mine with fit_dev thresholds
    train = ds.fit_train
    paths = [train.image_path(e) for e in train.images]
    grid = {c["config_id"]: c for c in base_setting["configs"]}
    groups = chosen_groups(base_setting["per_class"])
    passes = infer_for_groups(cfg, client, groups, grid, ds.ontology, paths,
                              materials)
    floor, _ = combined_detections(len(paths), passes, groups, thresholds=None)
    thresholds = {cid: e["threshold"]
                  for cid, e in base_setting["per_class"].items()}
    manifest, counts = mine_classifier_crops(train, ds.ontology, floor,
                                             thresholds, out_dir)
    cc = cfg.crop_classifier
    total = counts["train_gt"] + counts["train_fp"]
    if total < cc.min_samples or counts["train_fp"] < cc.min_background:
        return None, counts, None

    target = entry if entry is not None else out_dir
    artifact = train_fn(repo_root, manifest=manifest, ontology=ds.ontology,
                        out_dir=target / "artifact",
                        device=cfg.mine.device, log_path=out_dir / "train.log")
    if meta is not None:
        # the mark goes last — if the process dies mid-training there is no mark
        # and the next call retrains
        entry.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"key": key, "origin": origin or {},
                                    "counts": counts,
                                    "artifact": os.path.abspath(str(artifact))},
                                   indent=2, ensure_ascii=False))
    return Path(artifact), counts, None


def _write(out_dir: Path, dec: dict, grid: list, artifact=None) -> None:
    """The decision record — the λ grid (the baseline is lam=None — no
    classifier) and the loaded artifact."""
    (out_dir / "classifier_decision.json").write_text(json.dumps(
        {"decision": dec, "grid": grid,
         "artifact": str(artifact) if artifact else None},
        indent=2, ensure_ascii=False, default=str))
