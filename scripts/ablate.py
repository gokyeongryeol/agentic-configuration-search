#!/usr/bin/env python
r"""The final champion's **marginal contribution** per component — leave-one-component-out.

Why. The Δ the loop records is the **increment at adoption**, so it is confounded
by adoption order: an axis adopted earlier eats the headroom first, so "the axis
with the largest Δ" tends to coincide with "the axis proposed first". This
measures instead the loss of removing each layer **from the settled recipe**, one
at a time, which does not depend on order.

Definition. Marginal contribution of C = F(champion) - F(champion \ C), both
measured the same way on holdout (τ_c re-estimated per variant — as the loop does
for every candidate). The CI is a paired bootstrap against the champion records.

What changes. The champion's setting (config assignment) is **fixed** and only C
is removed — the question is "what does that layer add in this placement".
Re-screening would ask another question, "how well can it do without that layer".

  exemplar, alias, text_negatives, visual_negatives   re-mined from train (purity kept) — every
                                                      other axis inherits the champion's
                                                      materials, and the LLM refinement and
                                                      VLM caption hooks only attach to a
                                                      running planner vLLM
  tiling                                              tiling turned off in the setting's configs
  rescorer                                            the calibration is not applied
  lora                                                the detector goes back to the base weights
  crop_classifier                                     the classifier artifact and blend λ are removed

Usage:
  .venv/bin/python scripts/ablate.py runs/<ds>/run --config <config the run used> \
      --gpu 4 --out ablate/<ds>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

for _t in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_t, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger("ablate")

# axes whose materials must be re-mined — the rest use the champion's materials as they are
MINING_AXES = ("exemplar", "alias", "text_negatives", "visual_negatives")
# Only the 8 components are ablated. Count knobs (exemplar_base_k etc.) are not
# separate layers but settings of the axes above, so they are not removed on their own.
COMPONENTS = ("exemplar", "alias", "text_negatives", "visual_negatives",
              "tiling", "rescorer", "lora", "crop_classifier")
OFF = (None, False, "none", "off", 0)


def _active(champ: dict) -> list[str]:
    """The components that are on — judged by **the served state, not the label**.

    champion.json's axes can disagree with what is served:
      · crop_classifier=False with adapt_root still set keeps the artifact loaded.
      · tiling=g (grid) with no class assigned to a tiled config is void.
    Trusting the label would report "no contribution" for a layer that is
    actually on without ever touching it.
    """
    axes = champ["axes"]
    on = [k for k in COMPONENTS if axes.get(k) not in OFF]
    setting = champ.get("setting") or {}
    grid = {c["config_id"]: c for c in setting.get("configs", [])}
    used = {grid[v["config_id"]].get("tiling")
            for v in setting.get("per_class", {}).values() if v["config_id"] in grid}
    if "tiling" in on and not any(u for u in used if u):
        log.warning("tiling is on but no class is assigned to a tiled config — void, "
                    "so it is not ablated")
        on.remove("tiling")
    if "crop_classifier" not in on and champ.get("adapt_root"):
        log.warning("crop_classifier is off but adapt_root is loaded — it is actually "
                    "served, so it is ablated (%s)", champ["adapt_root"])
        on.append("crop_classifier")
    return on


# Dependencies between axes — the same relations planner.axes.validate_axes
# enforces. Turning a parent off must turn its dependents off for the combination
# to be valid, so some axes **cannot be removed on their own**: the measured
# contribution is "the axis + its dependents", and the report says so.
DEPENDENTS = {
    "exemplar": ("exemplar_base_k", "exemplar_fn_k", "visual_negatives",
                 "visual_negative_k"),
    "visual_negatives": ("visual_negative_k",),
    "text_negatives": ("text_negative_k",),
}


def _axes_without(axes: dict, drop: str) -> tuple[dict, list[str]]:
    from agentic_configuration_search.planner.axes import BASELINE
    out = dict(axes)
    out[drop] = BASELINE[drop]        # the axis's safe point = exactly the baseline behavior
    also = []
    for dep in DEPENDENTS.get(drop, ()):
        if out.get(dep) not in OFF:
            also.append(dep)
        out[dep] = BASELINE[dep]
    return out, also


def apply_detector_state(detector_hooks: dict, weights, adapt_root, lam) -> None:
    """Load the detector state (weights, classifier artifact) together with the
    loop-side blend λ.

    - λ: loading only the artifact without λ serves argmax reclassification, not
      the prediction that won adoption (the same rule as scripts/predict.py).
    - pass cache: the loop wraps the detector-state hooks to empty it on every
      swap (runner._invalidating_hooks). Here the hooks are called directly, so it
      is emptied directly — otherwise a pass of another state could be reused.
    """
    from agentic_configuration_search.infer import gateway
    if adapt_root and lam is None:
        raise SystemExit(f"adapt_root ({adapt_root}) without a blend λ — the "
                         "adopted prediction cannot be replayed")
    detector_hooks["weights_ctl"](Path(weights) if weights else None)
    detector_hooks["adapt_ctl"](Path(adapt_root) if adapt_root else None)
    gateway.set_classifier_blend(lam if adapt_root else None)
    gateway.invalidate_pass_cache()


def _setting_without(setting: dict, drop: str) -> dict:
    """Turn off an axis baked into the setting itself — config and assignment, not materials."""
    out = copy.deepcopy(setting)
    if drop == "tiling":
        for c in out.get("configs", []):
            c["tiling"] = None
    elif drop == "visual_negatives":
        for c in out.get("configs", []):
            c["visual_negatives"] = False
            c["neg_eta"] = None
    elif drop == "exemplar":
        # modality "semantic" queries with the visual exemplars — without exemplars
        # that assignment does not hold. Drop it to the safe point (text prompts).
        for v in out["per_class"].values():
            if v.get("modality") == "semantic":
                v["modality"] = "text"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, required=True,
                    help="GPU for the detector, embedding and training workers")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import yaml
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.core.data import load_dataset
    from agentic_configuration_search.eval.evaluate import evaluate_on_holdout, gate
    from agentic_configuration_search.fitting.materials import Materials
    from agentic_configuration_search.loop import rescore
    from agentic_configuration_search.loop.hooks import build_hooks
    from agentic_configuration_search.loop.materialize import materialize
    from agentic_configuration_search.loop.runner import _setting_load

    run_dir = Path(args.run_dir).resolve()
    raw = yaml.safe_load(Path(args.config).read_text())
    # pin every worker to this GPU so a running experiment is not touched
    dev = f"cuda:{args.gpu}"
    for section in ("detector", "mine", "lora"):
        raw[section] = {**raw.get(section, {}), "device": dev}
    # No planner is used (kind=deterministic — no new vLLM is launched). The
    # re-mining LLM hooks (alias and negative refinement, VLM captions) must match
    # the loop, so if a planner vLLM is up it only attaches to that endpoint
    # (base_url) — it neither launches nor stops one.
    planner = {**raw.get("planner", {}), "kind": "deterministic"}
    if not planner.get("base_url"):
        launch = planner.get("launch") or {}
        url = f"http://{launch.get('host', '127.0.0.1')}:{launch.get('port', 8104)}"
        import requests
        try:
            if requests.get(f"{url}/health", timeout=5).ok:
                planner["base_url"] = url
        except requests.RequestException:
            pass
        if not planner.get("base_url"):
            log.warning("could not attach to the planner vLLM (%s) — re-mining runs "
                        "without LLM refinement and VLM captions (materials may differ "
                        "from the loop's)", url)
    raw["planner"] = planner
    cfg = config_from_dict(raw)

    ds = load_dataset(Path(cfg.dataset).resolve(), seed=cfg.split.seed,
                      fit_train_ratio=cfg.split.fit_train_ratio)
    champ = json.loads((run_dir / "champion.json").read_text())
    axes = champ["axes"]
    active = _active(champ)
    log.info("champion components on = %s", active)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / (run_dir.parent.name + "_work")
    work.mkdir(parents=True, exist_ok=True)

    cat_ids = [c.id for c in ds.ontology.classes]
    results = {"dataset": run_dir.parent.name, "champion_axes": active,
               "holdout_split": ds.holdout_split, "ablations": []}

    hooks = build_hooks(cfg, REPO, work, llm_base_url=planner.get("base_url"))
    client, detector_hooks = hooks["client"], hooks["detector_hooks"]

    def evaluate(setting, materials, rescorer):
        if rescorer is not None:
            return rescore.rescored_evaluate(cfg, client, ds, setting, materials, rescorer)
        return evaluate_on_holdout(cfg, client, ds.holdout, ds.ontology, setting,
                                   materials)

    def load_rescorer():
        p = champ.get("rescorer_path")
        if not p:
            return None
        from agentic_configuration_search.fitting.rescorer import (Rescorer,
                                                                   dataset_fingerprint)
        return Rescorer.load(Path(p), expect=dataset_fingerprint(ds.root.name,
                                                                 ds.ontology))

    # champion.json stores per_class with string keys — restore int keys (score_setting
    # uses them as category ids)
    setting_full = _setting_load(champ["setting"])
    mats_full = Materials.load(champ["materials_dir"])
    resc_full = load_rescorer()

    recorded = json.loads((run_dir / "champion_eval.json").read_text()).get("micro_f_beta")
    # ---- reference: re-measure the full champion on holdout (left side of the paired comparison) ----
    lam_full = champ.get("adapt_lambda")
    apply_detector_state(detector_hooks, champ.get("weights"), champ.get("adapt_root"),
                         lam_full)
    ev_full = evaluate(setting_full, mats_full, resc_full)
    log.info("champion holdout micro = %.4f (recorded %s)", ev_full["micro_f_beta"],
             recorded)
    results["champion_micro"] = ev_full["micro_f_beta"]
    results["champion_micro_recorded"] = recorded
    # a wrong reference makes every marginal contribution wrong — do not go on silently
    if recorded is not None and abs(ev_full["micro_f_beta"] - recorded) > 0.02:
        log.error("the reference re-measurement disagrees with the record: %.4f vs %.4f "
                  "— stopping", ev_full["micro_f_beta"], recorded)
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        return 3

    # ---- remove one axis at a time ----
    for drop in active:
        log.info("=== ablate: %s", drop)
        axes2, also = _axes_without(axes, drop)
        if also:
            log.info("  turning %s off also turns off its dependents: %s", drop, also)
        setting2 = _setting_without(setting_full, drop)
        if "visual_negatives" in also:
            setting2 = _setting_without(setting2, "visual_negatives")
        mats2, resc2 = mats_full, resc_full
        weights2, adapt2, lam2 = champ.get("weights"), champ.get("adapt_root"), lam_full

        if drop in MINING_AXES or any(a in MINING_AXES for a in also):
            # every axis but the removed one inherits the champion's materials (as
            # runner._parent_kw does) — re-mining other axes would mix their
            # difference into the contribution
            mats2 = materialize(cfg, ds, axes2, work / f"ablate_{drop}", client=client,
                                seed=cfg.run_seed,      # the loop's mining seed
                                parent=champ.get("materials_dir"), parent_axes=axes,
                                **{k: hooks[k] for k in
                                   ("encode_fn", "caption_fn", "vlm_caption_fn",
                                    "refine_alias_fn", "refine_negatives_fn")})
        elif drop == "rescorer":
            resc2 = None
        elif drop == "lora":
            weights2 = None                    # base weights
        elif drop == "crop_classifier":
            adapt2, lam2 = None, None

        apply_detector_state(detector_hooks, weights2, adapt2, lam2)
        try:
            ev2 = evaluate(setting2, mats2, resc2)
        except Exception as e:                 # one axis failing does not stop the rest
            log.exception("evaluating ablation %s failed", drop)
            results["ablations"].append({"component": drop, "error": repr(e)})
            continue

        # The argument order is the direction of the question. (ablated, full)
        # makes Δ = "full - ablated" = the marginal contribution, and the one-sided
        # lower bound reads "removing this layer loses at least this much".
        g = gate(ev2["records"], ev_full["records"], cat_ids=cat_ids,
                 iou_thr=cfg.detector.iou_threshold, margin=0.0,
                 n_boot=cfg.gate.bootstrap, seed=cfg.run_seed,
                 alpha=1.0 - cfg.gate.confidence, beta=cfg.beta)
        results["ablations"].append({
            "component": drop,
            "also_dropped": also,
            "micro_without": ev2["micro_f_beta"],
            "marginal": ev_full["micro_f_beta"] - ev2["micro_f_beta"],
            "marginal_boot": g["delta"],
            "ci_lo": g["lo"], "significant": bool(g["adopted"]),
        })
        log.info("%s: without=%.4f  marginal=%+.4f  lower=%+.4f  significant=%s",
                 drop, ev2["micro_f_beta"], ev_full["micro_f_beta"] - ev2["micro_f_beta"],
                 g["lo"], g["adopted"])

    apply_detector_state(detector_hooks, champ.get("weights"), champ.get("adapt_root"),
                         lam_full)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"→ {args.out}")
    for a in results["ablations"]:
        if "marginal" in a:
            dep = f" (+{','.join(a['also_dropped'])})" if a.get("also_dropped") else ""
            print(f"  {a['component']:18}{dep} marginal {a['marginal']:+.4f}  "
                  f"lower {a['ci_lo']:+.4f}  "
                  f"{'significant' if a['significant'] else 'not significant'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
