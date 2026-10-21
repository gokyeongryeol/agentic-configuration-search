# agentic-configuration-search

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
HF_HUB_OFFLINE=1 .venv/bin/acs run --config config/yoloe-soccer.yaml
.venv/bin/acs run --config config/yoloe-soccer.yaml --resume   # replay, no LLM calls
.venv/bin/acs run --config ... --rounds 1 --candidates candidates.yaml  # inject your own candidates
.venv/bin/acs report runs/<ds>/run      # trajectory report (report.md)
.venv/bin/acs ab runs/a runs/b          # compare two runs (e.g. deterministic vs llm)
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

A setting combines the 8 axes below. Every axis has an "off" value (the first
line) that equals the baseline behavior exactly.

| axis | values |
|---|---|
| `alias` — Prompt alias | `false`: class name only (manual aliases in `mine.text_aliases` are always included)<br>`vlm`: alias proposed by a multimodal LLM that looks at the crops |
| `exemplar` — Visual exemplar | `none`: text prompts only<br>`base`: GT crops as image prompts (by default chosen for diversity by embedding k-center — `mine.exemplar_selection`)<br>`fn_arm`: `base` + crops of GT objects missed when train is inferred with the `base` exemplars<br>counts `exemplar_k` ∈ {1, 2, 4, 8} (default 4), `fn_k` ∈ {1, 2, 4} (default 2) |
| `text_negatives` — Negative text | `false`: no suppression phrases<br>`fp_crop`: phrases from false-positive crops of the mining pass — the mining pass infers train with the candidate's exemplars only (a GT k-center selection if it has none), without tiling or negatives<br>`gt_crop`: phrases from other classes' GT crops (suppress confusion — not valid on single-class datasets)<br>`background`: phrases from the same mining pass's false positives that overlap no GT of any class (suppress background)<br>count `text_negative_k` ∈ {1, 2, 4} (default 1) |
| `visual_negatives` — Negative visual | `false`: no suppression crops<br>`true`: false-positive crops of the mining pass (the same as `fp_crop`) as negative image prompts (needs `exemplar`), emulated as IoU suppression. The suppression IoU threshold is chosen on fit_dev by screen, one config shared across classes, among a config without negatives and `search.neg_eta_grid` (0.25, 0.35, 0.5, 0.65, 0.75 — lower suppresses more)<br>count `visual_negative_k` ∈ {1, 2, 4} (default 1) |
| `rescorer` — Score calibration | `false`: detection scores as-is<br>`true`: a discriminator recalibrates scores. The per-class mixing weight α_c is chosen on fit_dev from `rescorer.alphas` (0, 0.1, 0.25, 0.5, 1, 2, 3, 4), and is non-zero only when the gain exceeds `rescorer.margin` (α_c = 0 is the original score) |
| `crop_classifier` — Re-classification | `false`: detected classes as-is<br>`true`: a SigLIP2+LoRA classifier re-classifies detection boxes or drops them as background |
| `lora` — Tail fine-tune | `false`: base weights<br>`true`: fine-tunes the YOLOE tail on fit_train and interpolates with the base over `lora.lambdas` (0, 0.25, 0.5, 0.75, 1) — λ = 0 is the base. Not searched when fit_train has fewer than `lora.min_images` (20) images. The key keeps the name `lora` for history compatibility |
| `tiling` — Tiling | `false`: full image only<br>g ∈ {2, 3, 4, 5, 6}: a config that also infers the image as g×g tiles and merges. Zoom (g−1)(1−overlap)+1 (at overlap 0.5, g=2 is 1.5×, g=3 is 2×), 1+g² forwards per image per pass. The planner is told to raise it one step at a time |

An axis that is on may still go unused by the setting — screen picks whichever
is better on fit_dev (exemplar against text prompts per class, tiling against
the untiled config).

Axes not to search on a dataset go in `search.disabled_axes` by the axis keys
in the table (e.g. `[lora, tiling]`). The count keys (`exemplar_k`, `fn_k`,
`text_negative_k`, `visual_negative_k`) can be disabled on their own too (the
count stays at its default or the champion's value), and
`search.disabled_values` bans single values (e.g. `{text_negatives: [gt_crop]}`).
An unknown key or value fails at the start of the run. Disabled axes and values
are left out of the planner's value list, and candidates that turn them on or
change them away from the champion's value are rejected — candidates that turn
them off or inherit the champion's value are kept.

### Round loop

```
propose → materialize → screen (fit_dev) → evaluate (holdout) → decide → diagnose → record
```

- **screen** passes only the top-K candidates to holdout.
- **decide** adopts a candidate only if the paired-bootstrap lower bound of its
  holdout Δ clears the margin; otherwise the champion stays.
- **saturation** stops the run after `saturation.rounds` rounds without a
  meaningful adoption, or when the planner has nothing left to try.
- A round adopts one candidate, the one with the largest Δ. Candidates that
  passed the gate but lost to it are marked `gate_passed_but_lost` and carried
  into the next round as `carry_<id>`: the axes they changed are applied on top
  of the **new** champion. That makes each one a single increment over the new
  champion, and holdout is not reused within the round. Carried candidates go
  through the same filters as planner and human candidates, whichever planner
  is used.
- The same setting is not evaluated twice. After screen, a candidate's served
  state (per class: config content and modality, the materials that modality
  uses, rescorer on/off, server state) is compared with the champion, its
  siblings in the round, and candidates evaluated on holdout before; a match
  is rejected without holdout evaluation. Axis labels are not compared,
  because materials can differ under the same label. If an axis the candidate
  turned on (tiling, exemplar, visual_negatives) went unused and that made it
  equal the champion, the reason is `axis_unused` (which axes in
  `unused_axes`); otherwise `same_setting`. Rescorer candidates are judged
  before the discriminator refit, the most expensive step. An evaluated
  candidate with an unused axis gets its label corrected to the measured state,
  with the request kept in `requested_axes`.
- Sizing signals are recorded in history: for tiling candidates, whether
  screen actually picked a tile config (`tiling_used`); for visual-negative
  candidates, the suppression IoU thresholds picked for semantic classes
  (`neg_eta_used`) and the grid searched (`neg_eta_grid`). The planner reads
  them before widening a count axis: tiling used on an adopted grid suggests
  g+1, negatives picked at the low (strongest) end suggest raising
  `visual_negative_k`, and picks at the high end suggest folding the axis.
  Candidates that ran a semantic pass also record each class's fit_dev F(β)
  for the semantic and text passes (`modality_f`), including candidates
  filtered before holdout, so the planner can tell whether exemplar lost
  narrowly (try another k or fn_arm) or badly. The prompt carries only a
  summary whose size does not depend on the class count (classes text won and
  the narrowest margin); the full per-class record would make the budget
  guard shrink the history window on datasets with many classes.
- lora and crop_classifier change the detector state (weights / artifact), so
  they are processed after the other candidates, and the state is restored if
  they are not adopted. Once adopted they are not turned off: the state stays
  loaded while other candidates are measured, so a candidate turning one off
  would be measured with it still on. Such candidates are rejected as
  `detector_state_off` without evaluation.

### Diagnosis and planner

- Every round classifies train and holdout errors (missed, below threshold,
  misclassified, localization, background FP, duplicate); this is the planner's
  only input. Holdout is exposed as aggregates only.
- Candidate axes are read as an increment over the current champion. The LLM
  planner and human candidates list only the axes they change; omitted axes
  keep the champion's value (turning one off takes an explicit `false`).
  Filling them from the baseline would evaluate the champion with its axes
  silently off.
- The LLM planner (vLLM, guided JSON) only proposes — the holdout gate decides.
  Prompts and responses are saved, so `--resume` replays them without the LLM.
  The prompt is rebuilt every round; when it exceeds the estimated-token budget
  (server context − output cap), the history window shrinks first. Response
  length is bounded by the schema (rationale, id, candidate count). One call
  per round, plus at most one follow-up in the same conversation when the
  planner repeats carried candidates: those are dropped as
  `carried_duplicate` and the freed slots are asked for again.
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

`src/agentic_configuration_search/`, layered
`core ← clients ← infer ← fitting ← eval ← diagnose ← planner ← loop ← cli`
(enforced by `tests/test_architecture.py`). All detector inference goes through
`infer/gateway.py`; torch code lives only in `workers/`. Guard tests
(`test_materialize_purity`, `test_split_roles`, `test_decision_channel`,
`test_replay`) pin the split roles and must keep passing.
