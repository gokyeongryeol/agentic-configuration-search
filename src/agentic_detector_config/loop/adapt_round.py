"""Round wiring for the crop classifier axis — trained on fit_train, scored on fit_dev.

Like lora, this is a "detector-state candidate": it loads an
artifact into the in-process detector client, so the runner handles it after
all other candidates are evaluated, and restores restore_root (the current
champion's artifact, or None) when not adopted.

**Purity**: the background-mining threshold is base_setting's per_class
threshold — **estimated on fit_dev**. Using tau_test (the holdout estimate)
would let the holdout pick training material and break material purity. This
module never receives the holdout split.

**Training uses fit_train only** — the arm is scored by its fit_dev Δ. If fit_dev
crops were in the training material (previously all of train), scoring would be
an in-sample comparison on crops the classifier had already seen and the margin
would filter nothing. Same discipline as lora (trained on fit_train, scored on
fit_dev).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..eval.runner import chosen_groups, combined_detections, infer_for_groups
from ..fitting.crop_classifier import mine_classifier_crops, train_crop_classifier

log = logging.getLogger(__name__)


def run_classifier_candidate(cfg, ds, base_axes: dict, materials, out_dir: Path, *,
                             screen_fn, adapt_ctl, repo_root: Path,
                             restore_root: Path | None, seed: int,
                             client, train_fn=None) -> dict:
    """Single arm — {"decision", "adapt_root", "setting"}.

    If adopted, the client is left with the artifact loaded (the caller then
    runs holdout eval). If not, restores restore_root and returns adapt_root=None.
    train_fn is a test injection point — defaults to
    fitting.crop_classifier.train_crop_classifier.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_impl = train_fn or train_crop_classifier

    adapt_ctl(restore_root)              # baseline = current serving state
    base_setting = screen_fn(materials, base_axes)
    base_f = base_setting["fit_dev_micro_f_beta"]

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
        # tripwire=True — the runner records this axis as empty material
        # (empty_axis). It must be told apart from a margin rejection (no effect)
        # so the planner reads the two differently
        dec = {"adopted": False, "tripwire": True,
               "reason": f"{total} crops (<{cc.min_samples}) or "
                         f"{counts['train_fp']} background crops (<{cc.min_background}) "
                         "— training skipped",
               "counts": counts}
        _write(out_dir, dec, base_f, None)
        return {"decision": dec, "adapt_root": None, "setting": base_setting}

    artifact = train_impl(repo_root, manifest=manifest, ontology=ds.ontology,
                          out_dir=out_dir / "artifact",
                          device=cfg.mine.device, log_path=out_dir / "train.log")

    adapt_ctl(artifact)                  # load artifact — rescore fit_dev
    setting = screen_fn(materials, base_axes)
    f = setting["fit_dev_micro_f_beta"]
    delta = f - base_f
    log.info("crop classifier arm: fit_dev micro %.4f (Δ%+.4f)", f, delta)
    if delta > cc.margin:
        dec = {"adopted": True, "delta": delta, "counts": counts,
               "reason": f"fit_dev ΔF(β) {delta:+.4f} > margin {cc.margin}"}
        _write(out_dir, dec, base_f, f)
        return {"decision": dec, "adapt_root": Path(artifact), "setting": setting}
    adapt_ctl(restore_root)              # not adopted — restore current
    dec = {"adopted": False, "delta": delta, "counts": counts,
           "reason": f"fit_dev ΔF(β) {delta:+.4f} ≤ margin {cc.margin}"}
    _write(out_dir, dec, base_f, f)
    return {"decision": dec, "adapt_root": None, "setting": base_setting}


def _write(out_dir: Path, dec: dict, base_f: float, f) -> None:
    (out_dir / "classifier_decision.json").write_text(json.dumps(
        {"decision": dec, "fit_dev_base": base_f, "fit_dev_with_classifier": f},
        indent=2, ensure_ascii=False, default=str))
