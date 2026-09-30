#!/usr/bin/env python
"""Predict with the champion setting — replays the run artifacts as-is.

`champion.json` is the complete replay recipe:
  weights       tail arm descriptor (detector weights)
  adapt_root    crop classifier reclassification artifact
  materials_dir exemplar / alias / negative materials
  rescorer_path score recalibrator (client-side post-processing)
  setting       config grid + per-class (config_id, modality) assignment

Thresholds come from **tau_test** in `champion_eval.json`, not from champion.json —
the latter holds the screen (fit_dev) thresholds; the adopted performance was
achieved with the former (estimated on holdout; the field name is legacy).

Usage:
  .venv/bin/python scripts/predict.py runs/<ds>/run --config config/generated/par_<ds>.yaml
  .venv/bin/python scripts/predict.py runs/<ds>/run --config ... --images /path/to/dir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# cap worker/BLAS threads — must run before numpy is imported (same reason as cli.py)
for _t in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_t, "1")


def _image_split(root: Path, ontology):
    """Wrap a directory of images without GT as a Split (same shape as core.data._subset)."""
    from PIL import Image

    from agentic_detector_config.core.coco import ImageEntry, Split
    split = Split.__new__(Split)
    split.root, split.raw, split.ontology = root, {}, ontology
    split.images = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for i, p in enumerate(sorted(root.iterdir())):
        if p.suffix.lower() not in exts:
            continue
        with Image.open(p) as im:
            w, h = im.size
        split.images.append(ImageEntry(id=i, file_name=p.name, width=w, height=h))
    if not split.images:
        raise SystemExit(f"no images in {root}")
    return split


def _report_split(dataset_root: Path, ontology):
    """Open `test` **directly** — the path reserved for final reporting.

    When the dataset has valid/, the loop never reads test anywhere:
    core.data.load_dataset does not open it, and tests/test_split_roles.py
    enforces that. So test is the only split that never took part in search,
    thresholding or adoption.

    holdout (valid) is different. τ_c is chosen to maximize F(β) on holdout, the
    metric is measured on the same split (eval.evaluate.score_setting), and the
    adoption gate re-tests that split every round. Holdout numbers are therefore
    **optimistic estimates at an oracle threshold**; the finalized champion must be
    re-measured here to get an unbiased value.

    **Without valid/ that premise breaks** — test was promoted to the loop's holdout
    (core.data.HOLDOUT_CANDIDATES), so the value measured here is merely a
    re-measurement on the same split, not a debiased number. Passing silently would
    mix both kinds of numbers in one table, so a warning is emitted.

    That is why this calls Split.load directly instead of load_dataset — not going
    through the loop's data gate keeps the guard valid.
    """
    from agentic_detector_config.core.coco import ANNOTATIONS_FILENAME, Split
    if not (dataset_root / "valid" / ANNOTATIONS_FILENAME).exists():
        print("warning: this dataset has no valid/, so test served as the loop's "
              "holdout — the numbers below are a re-measurement of the split used "
              "for selection, not of an unseen split (optimistic bias remains).",
              file=sys.stderr)
    return Split.load(dataset_root / "test", ontology)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="runs/<dataset>/run")
    ap.add_argument("--config", required=True,
                    help="config the run used (config/generated/par_<ds>.yaml)")
    ap.add_argument("--split", default="test",
                    choices=("test", "holdout", "train", "fit_dev"),
                    help="split to predict on (when --images is not given). The default, test,"
                         " is a report-only split the loop never opened, provided the"
                         " dataset has valid/. Without valid/, test was the loop's"
                         " holdout, so bias remains — a warning is printed then"
                         " (see the _report_split docstring)")
    ap.add_argument("--rewrite-root", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite absolute paths in champion.json (when replaying after"
                         " moving the run directory). e.g. --rewrite-root /a/runs=/a/archives3")
    ap.add_argument("--images", help="image directory without GT — instead of a split")
    ap.add_argument("--out", default=None, help="predictions JSON path (default: <run_dir>/predictions.json)")
    ap.add_argument("--coco-map", action="store_true",
                    help="also compute COCO mAP@[.5:.95] and AP50 (pycocotools). mAP is"
                         " rank-based, so it is measured on **all floor detections"
                         " before the τ cut** — τ_c is only the F(β) operating point"
                         " and has nothing to do with the PR curve. For comparison"
                         " with mAP-reporting baselines such as DetPO.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from agentic_detector_config.config import load_config
    from agentic_detector_config.core import metrics
    from agentic_detector_config.core.data import load_dataset
    from agentic_detector_config.eval.runner import (chosen_groups, combined_detections,
                                                     infer_for_groups)
    from agentic_detector_config.fitting.materials import Materials
    from agentic_detector_config.loop import rescore

    run_dir = Path(args.run_dir).resolve()
    champ = json.loads((run_dir / "champion.json").read_text())
    for rule in args.rewrite_root:
        old, _, new = rule.partition("=")
        champ = json.loads(json.dumps(champ).replace(old, new))
    tau = {int(k): float(v) for k, v in
           json.loads((run_dir / "champion_eval.json").read_text())["tau_test"].items()}
    cfg = load_config(args.config)
    repo_root = Path.cwd()
    ds = load_dataset(Path(cfg.dataset).resolve())
    if args.images:
        split = _image_split(Path(args.images).resolve(), ds.ontology)
    elif args.split == "test":
        split = _report_split(Path(cfg.dataset).resolve(), ds.ontology)
    else:
        split = getattr(ds, args.split)
    paths = [split.image_path(e) for e in split.images]
    log = logging.getLogger("predict")
    log.info("champion round %s (%s) — axes %s", champ["round"], champ["id"],
             {k: v for k, v in champ["axes"].items() if v})

    # The detector is an in-process client — build it through the same seam as the
    # loop (build_hooks) and apply serving state (weights, reclassification artifact)
    # via the same hooks. Replay wired differently from the loop would itself be a
    # reproducibility hole.
    weights = Path(champ["weights"]) if champ.get("weights") else None
    adapt_root = Path(champ["adapt_root"]) if champ.get("adapt_root") else None
    from agentic_detector_config.loop.hooks import build_hooks
    hooks = build_hooks(cfg, repo_root, run_dir.parent)
    client = hooks["client"]
    deps = hooks["detector_hooks"]
    deps["weights_ctl"](weights)
    deps["adapt_ctl"](adapt_root)
    materials = Materials.load(champ["materials_dir"])
    setting = champ["setting"]
    grid = {c["config_id"]: c for c in setting["configs"]}
    groups = chosen_groups(setting["per_class"])
    need_emb = bool(champ.get("rescorer_path"))   # inputs for rescorer features
    passes = infer_for_groups(cfg, client, groups, grid, ds.ontology, paths,
                              materials, return_embeddings=need_emb)
    if need_emb:
        from agentic_detector_config.fitting.rescorer import Rescorer, dataset_fingerprint
        rescorer = Rescorer.load(
            Path(champ["rescorer_path"]),
            expect=dataset_fingerprint(ds.root.name, ds.ontology))
        preds = rescore.preds_dict(split, passes, groups, "px")
        preds = rescore.apply_to_preds(cfg, ds, setting, rescorer, preds)
        floor = rescore.dets_by_image(split, preds)
    else:
        floor, _ = combined_detections(len(paths), passes, groups, thresholds=None)

    # apply τ_test — cut on the same channel that won adoption (rescored or raw scores)
    dets = [[d for d in per if d["score"] >= tau[d["category_id"]]] for per in floor]
    names = ds.ontology.id_to_name
    out = [{"image": e.file_name, "category_id": d["category_id"],
            "category": names[d["category_id"]], "score": round(float(d["score"]), 6),
            "box_xyxy": [round(float(v), 2) for v in d["box"]]}
           for e, per in zip(split.images, dets) for d in per]
    out_path = Path(args.out) if args.out else run_dir / "predictions.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{len(out)} detections / {len(paths)} images → {out_path}")

    if args.coco_map and not args.images:
        # COCO eval over all floor detections before the τ cut — procedure:
        #   1) load the split's original _annotations.coco.json as GT
        #   2) convert floor detections to (image_id, category_id, xywh, score)
        #   3) COCOeval bbox — stats[0]=mAP@[.5:.95], stats[1]=AP50
        import contextlib, io
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        ann_path = split.root / "_annotations.coco.json"
        name_to_id = {e.file_name: e.id for e in split.images}
        results = []
        for e, per in zip(split.images, floor):
            iid = name_to_id[e.file_name]
            for d in per:
                x1, y1, x2, y2 = d["box"]
                results.append({"image_id": iid, "category_id": int(d["category_id"]),
                                "bbox": [float(x1), float(y1),
                                         float(x2 - x1), float(y2 - y1)],
                                "score": float(d["score"])})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):        # swallow pycocotools' verbose output
            gt = COCO(str(ann_path))
            dt = gt.loadRes(results) if results else None
        if dt is None:
            print("coco-map: 0 floor detections — recording mAP 0")
            map_stats = {"map": 0.0, "ap50": 0.0}
        else:
            with contextlib.redirect_stdout(buf):
                ev = COCOeval(gt, dt, "bbox")
                ev.evaluate(); ev.accumulate(); ev.summarize()
            map_stats = {"map": round(float(ev.stats[0]), 4),
                         "ap50": round(float(ev.stats[1]), 4)}
        map_stats["n_floor_detections"] = len(results)
        side = out_path.with_suffix(".map.json")
        side.write_text(json.dumps(map_stats, indent=2))
        print(f"COCO mAP@[.5:.95] {map_stats['map']:.4f}  AP50 {map_stats['ap50']:.4f}"
              f"  ({len(results)} floor detections) → {side}")

    if not args.images:      # with GT, report metrics too
        recs = metrics.per_image_records(split, paths, dets)
        micro = metrics.records_micro_f_beta(recs, [c.id for c in ds.ontology.classes],
                                             iou_thr=cfg.detector.iou_threshold, beta=cfg.beta)
        print(f"{args.split} micro F(β) {micro:.4f}  (holdout at adoption {champ['micro_f_beta']:.4f})")
        # Metrics sidecar — printing alone would mean scraping logs. Boxes in
        # predictions.json are rounded to 2 decimals, flipping a few IoU-0.5 borderline
        # matches, so values recomputed from that artifact differ slightly (≲5e-4).
        # **This value is canonical.**
        side = out_path.with_suffix(".metrics.json")
        side.write_text(json.dumps({"split": args.split, "micro_f_beta": float(micro),
                                    "beta": float(cfg.beta),
                                    "iou_threshold": float(cfg.detector.iou_threshold),
                                    "micro_f_beta_at_adoption": float(champ["micro_f_beta"]),
                                    "n_images": len(paths)}, indent=2))
        print(f"→ {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
