# agentic-detector-config

An agentic loop that tunes inference settings of YOLOE (ultralytics'
open-vocabulary detector) for a target dataset. Each round an LLM planner reads
an FP/FN diagnosis and proposes candidate settings, materials are built from
`train`, and a candidate is adopted only if it beats the current champion on a
holdout split (micro F(β) with a bootstrap CI gate). The loop stops at
saturation.

**License: AGPL-3.0** — the package imports ultralytics (AGPL-3.0) in-process.
Research baseline only. ultralytics and torch are not declared dependencies;
install them in the venv separately.

## Install and test

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . --no-build-isolation --no-deps
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

## Running

```bash
HF_HUB_OFFLINE=1 .venv/bin/adc run --config config/yoloe-soccer.yaml
.venv/bin/adc run --config config/yoloe-soccer.yaml --resume   # replay, no LLM calls
.venv/bin/adc run --config ... --rounds 1 --candidates candidates.yaml  # inject your own candidates
.venv/bin/adc report runs/<ds>/run      # trajectory report (report.md)
.venv/bin/adc ab runs/a runs/b          # compare two runs (e.g. deterministic vs llm)
scripts/run_parallel.sh                 # demo datasets in parallel across GPUs
```

YOLOE runs in-process; the only server is the vLLM planner, launched
automatically unless `planner.base_url` points to one. All config keys and
defaults are in `config/default.yaml`; `config/_smoke-full.yaml` is a one-round
smoke run.

```yaml
dataset: demo_data/soccer-players-5fuqs
beta: 1.0                # F-beta of the decision metric (1.0 = F1)
detector:
  model: weights_yoloe/yoloe-26x-seg.pt
  device: "cuda:0"
planner:
  kind: llm              # or deterministic
mine:
  device: "cuda:1"       # DINOv3 embedding / crop-classifier training worker
```

## Features

### Data splits

Datasets use the Roboflow layout (`train/`, `valid/`, `test/`).

| split | role |
|---|---|
| `train` (80% `fit_train` / 20% `fit_dev`) | builds all materials — prompts from all of `train`, tail fine-tune / rescorer / crop classifier trained on `fit_train`; `fit_dev` scores candidates |
| `valid` (holdout) | per-class thresholds τ_c, adoption decision, saturation |
| `test` | never read by the loop — measures the final champion once |

Without `valid/`, `test/` becomes the holdout and no unopened split remains;
the run records this as `holdout_split` in `run_summary.json`.

### Search space

A setting combines the axes below. Every axis has an "off" value that equals
the baseline behavior.

| axis | values |
|---|---|
| alias | class name only / VLM-proposed alias from crops |
| visual exemplar | text only / GT-crop exemplars (`base`, `fn_arm`) |
| visual_negatives, neg_eta | negative crops emulated by IoU suppression × threshold grid |
| text_negatives | off / `fp_crop` / `gt_crop` (other classes' GT — not on single-class datasets) / `background` |
| tiling | off / grid g ∈ 2–6 (image-relative g×g, zoom (g−1)(1−overlap)+1, 1+g² passes per image) — the planner raises it one step at a time |
| rescorer | per-class score calibration α_c (α=0 is off) |
| lora | YOLOE tail fine-tune with λ interpolation (λ=0 is the base) |
| crop_classifier | SigLIP2+LoRA re-classification of detections |
| counts k | how many exemplars/negatives to use |

Axes not to search on a dataset go in `search.disabled_axes` (e.g. `[lora, tiling]`;
`search.disabled_values` bans single values). Only candidates that newly turn one
on are blocked — a value the champion already has is kept.

### Round loop

```
propose → materialize → screen (fit_dev) → evaluate (holdout) → decide → diagnose → record
```

- **screen** passes only the top-K candidates to holdout.
- **decide** adopts a candidate only if the paired-bootstrap lower bound of its
  holdout Δ clears the margin; otherwise the champion stays.
- **saturation** stops the run after `saturation.rounds` rounds without a
  meaningful adoption, or when the planner has nothing left to try.
- lora and crop_classifier change the detector state (weights / artifact), so
  they are processed after the other candidates, and the state is restored if
  they are not adopted.

### Diagnosis and planner

- Every round classifies train and holdout errors (missed, below threshold,
  misclassified, localization, background FP, duplicate); this is the planner's
  only input. Holdout is exposed as aggregates only.
- The LLM planner (vLLM, guided JSON) only proposes — the holdout gate decides.
  Prompts and responses are saved, so `--resume` replays them without the LLM.
- `--planner deterministic` maps error causes to remedy axes by rule; it backs
  the tests and serves as the A/B control.

### Outputs

```
runs/<dataset>/run/
  history.jsonl        # round ledger
  run_summary.json     # stop reason, final champion, holdout_split
  champion.json        # champion setting (axes, materials, weights, adapt_root, forwards_per_image)
  champion_eval.json   # champion holdout evaluation, incl. tau_test
  rounds/round_NNN/    # proposal, settings/<cand>/ materials, screen, eval, decision, diagnosis
```

### Prediction

`scripts/predict.py` replays a run's champion:

```bash
.venv/bin/python scripts/predict.py runs/<ds>/run --config <run config> --split test
.venv/bin/python scripts/predict.py runs/<ds>/run --config <run config> --images /path/to/dir
```

It uses `tau_test` from `champion_eval.json` as the threshold (not the
`fit_dev` threshold in `champion.json`). Paths in `champion.json` are absolute,
so keep the run directory in place.

## Code layout

`src/agentic_detector_config/`, layered
`core ← clients ← infer ← fitting ← eval ← diagnose ← planner ← loop ← cli`
(enforced by `tests/test_architecture.py`). All detector inference goes through
`infer/gateway.py`; torch code lives only in `workers/`. Guard tests
(`test_materialize_purity`, `test_split_roles`, `test_decision_channel`,
`test_replay`) pin the split roles and must keep passing.
