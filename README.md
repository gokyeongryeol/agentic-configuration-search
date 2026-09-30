# agentic-detector-config

A loop that searches for good inference settings for YOLOE (an open-vocabulary
detector from ultralytics) on `train`, confirms them on holdout (`valid`) with
micro F(β) plus a CI gate, and proposes the next candidates from an FP/FN
diagnosis, repeating until saturation. An LLM planner proposes; the holdout gate
decides adoption. The loop never reads `test` — it is used once, to measure the
final champion (on datasets without `valid/`, `test` stands in as holdout — see
"When there are not three splits").

This document is the repo's normative reference.

## Install and test

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . --no-build-isolation --no-deps
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

## Running

The detector (YOLOE) is an in-process client, so there is no detector server.
The only server is the vLLM planner: if `planner.base_url` is unset,
ManagedServer launches it (persistent), otherwise the loop attaches to that
endpoint. The lora (tail fine-tune) and crop classifier axes are wired through
the detector client's state controls (weight swap, artifact loading).

**License: AGPL-3.0.** The package imports ultralytics (AGPL-3.0) in-process, so
the whole package is AGPL-3.0. This repo is a research baseline only and is not
merged into product code. ultralytics is not a declared dependency; install it
separately in the venv.

```bash
HF_HUB_OFFLINE=1 .venv/bin/adc run --config config/yoloe-soccer.yaml
.venv/bin/adc run --config config/yoloe-soccer.yaml --resume   # zero LLM calls on replay
.venv/bin/adc run --config ... --rounds 1 --candidates candidates.yaml  # inject human candidates
scripts/run_parallel.sh                  # demo datasets in parallel across GPUs (list at top of script)
.venv/bin/adc report runs/<ds>/run       # trajectory report
.venv/bin/adc ab runs/a runs/b           # compare two runs (deterministic vs llm)
```

`config/_smoke-full.yaml` is a one-round smoke run; `config/_smoke-candidates.yaml`
is a matching candidates file.

Example config (all keys and defaults are in `config/default.yaml`; a test
blocks drift from the code defaults):

```yaml
dataset: demo_data/soccer-players-5fuqs
beta: 1.0                # β of the decision metric F-beta — 1.0=F1, >1 weights recall
detector:
  model: weights_yoloe/yoloe-26x-seg.pt
  device: "cuda:0"
planner:
  kind: llm              # default model Qwen/Qwen3.8-27B-FP8 (launch.gpu: 3)
mine:
  device: "cuda:1"       # DINOv3 embedding and crop-classifier training worker
```

## Data and split roles

Datasets are taken as-is in Roboflow format (`train/`, `valid/`, `test/`).
If `valid/` is missing, `test/` stands in as holdout — see "When there are not
three splits" below.

| split | name | role |
|---|---|---|
| train 80% | `fit_train` | tail fine-tune training (lora axis), rescorer discriminator training |
| train 20% | `fit_dev` | config grid scoring, lora/classifier arm scoring, rescorer α grid |
| all of train | `train` | prompt mining (exemplar, alias, visual_negatives), crop classifier training, FP/FN diagnosis |
| valid | `holdout` | τ_c estimation (`tau_test`), adoption decision (micro F(β) + CI), saturation, FP/FN diagnosis |
| test | — | **off-limits to the loop** (when valid exists) — report-only, measures the final champion once (via `scripts/predict.py --split test`, not `adc`) |

**Why valid is the selection split.** Every round the loop estimates τ_c and
decides adoption on holdout — it is reused a dozen-plus times per run. A reused
split is contaminated accordingly and cannot serve the final report. With
Roboflow's 70/20/10, valid is about twice the size of test, so **the larger split
goes to repeated use** and test is kept for a single measurement. This also
matches convention.

The 80/20 split is image-level stratified (preserving class distribution) with a
fixed seed; the boundary does not move across rounds.

**Core invariant**: the crops and phrases put into prompts and the
lora/rescorer/classifier **weights (materials) come from train only.** Holdout can
change only two things: **τ_c and "what to try next".**

**Holdout is reused every round, so numbers on it are optimistic.** τ_c is
chosen to maximize F(β) on holdout and the metric is measured on the same split
(oracle threshold), and the gate re-tests the same split a dozen-plus times per
run (selection bias). So **the final report is produced once, on `test`** — a
split the loop never read; `test_split_roles.py` enforces that ban (when valid
exists; see below).

**When there are not three splits.** Without `valid/`, `test/` is promoted to
holdout (`core.data.HOLDOUT_CANDIDATES` — earlier entries win). This fallback
lets non-3-split datasets in, at a clear cost: **no sealed split remains.** There
is nowhere to remove the optimism bias described above (oracle threshold +
repeated testing), so the value from `predict.py --split test` is a re-measurement
of the split used for selection, not a measurement of an unopened split. A
warning is emitted at load time and by `predict.py`, and the split used is
recorded as `holdout_split` in `run_summary.json` — **never mix numbers from the
two cases in one table.** As long as valid exists, test is never opened anywhere,
and `test_test_is_never_read_when_valid_exists` keeps enforcing that.

τ_c is not a shipped material — the champion setting's materials are complete
from train alone, and τ_c belongs to this experiment's evaluation instruments.
Whoever deploys a setting and needs τ is responsible for estimating it on their
own data.

## Search space (setting axes)

A "setting" is a combination of the axes below. Each axis has a safe point where
**turning it off is exactly the current behavior**. Modality and config follow
"shared config + per-class thresholds" — choosing configs per class would
multiply inference cost by the number of classes.

| axis | value space | safe point | material source |
|---|---|---|---|
| alias (text prompt) | off / `vlm` (multimodal LLM looking at crops) | class name alone | train (cross-pool filter + verbatim refine) |
| visual exemplar | off / `base` (GT crop k-center) / `fn_arm` | text-only | train |
| visual_negatives / neg_eta | suppression crops on/off × suppression IoU threshold grid (emulated negatives, `core.suppress`) | none | train |
| text_negatives | off / `fp_crop` / `gt_crop` (other-class GT) / `background` (FPs not overlapping GT) | off | train (captions + safety-rule refine) |
| tiling | off / slice grid | off | — |
| rescorer α_c | per-class α grid (includes 0) | α=0 | discriminator on `fit_train`, α on `fit_dev` |
| lora (name kept) | YOLOE tail fine-tune + λ interpolation grid (includes λ=0), **off by default** | λ=0 (bit-exact base) | trained on `fit_train`, scored on `fit_dev` |
| crop classifier | detection reclassification (SigLIP2+LoRA) on/off, **off by default** | off (no reclassification) | trained on train (positive=GT crops, background=FPs passing the fit_dev threshold), scored on `fit_dev` |
| counts k (exemplar_k, fn_k, visual_negative_k, text_negative_k) | None, 1, 2, 4 (exemplar_k up to 8) | None (config quota) | train |
| τ_c | per-class threshold | — | **estimated on holdout** (`tau_test`) |

Re-proposal is prevented by **dedup on the full axes combination** — only
combinations already tried are permanently blocked; retrying the same axis after
the champion changes is allowed.

## Loop — round structure, gate, saturation

```
round_r/
  propose     planner reads diagnosis_{r-1} and proposes N candidates
              + merges human-injected candidates (candidates.yaml)   → proposal.json
  materialize build materials per candidate: mining/training under train rules → settings/*/
  screen      score candidates on fit_dev, only top K pass              → screen.json
  evaluate    infer passing candidates on holdout → estimate tau_test → F(β)+CI → eval.json
  decide      gate: adopt only if the CI margin vs. current champion passes → decision.json
  diagnose    train and holdout FP/FN diagnosis regardless of adoption    → diagnosis_*.json
  record      append round summary to history.jsonl
```

- **Screen saves holdout inference**: only `screen.top_k` candidates per round
  reach holdout.
- **Adoption gate**: Δ of holdout micro F(β) is tested with a paired bootstrap
  (image-level resampling) — adopt only if **the one-sided lower bound ≥ margin
  and Δ > 0**; otherwise keep the champion, i.e. exactly the current behavior.
  It is one-sided because the gate asks only one question: "is Δ larger than
  margin?". The boundary (lower bound = margin) is split by Δ because on a small
  holdout the Δ distribution collapses to 0, and a real improvement and no change
  at all would otherwise be rejected for the same reason.
  Separately, the rescorer's α_c must pass a per-class non-degradation margin
  **at fitting time** (fit_dev gain > margin) to be nonzero.
- **Saturation**: stop when there is no adoption, or the Δ CI lower bound ≤ ε,
  for `saturation.rounds` consecutive rounds. A planner declaration of "nothing
  to try" also stops the run (resumable by injecting human candidates).
- **Detector-state candidates** (lora, crop classifier) swap the detector
  client's weights/artifact (`weights_ctl`/`adapt_ctl` in the code), so they are
  processed after other candidates are evaluated and the current state is
  restored if they are not adopted.
- **Inherited axes take their materials from the champion** — the champion's
  score was achieved with those materials, so re-mining an axis with the same
  value would make the gate's Δ the sum of "increment + material swap". A
  candidate mines only the axes it changes (recorded in
  `mine_reports.inherited`).
- **Empty materials are not evaluated** — if mining for an enabled axis comes
  back empty, the candidate equals the current setting and is rejected at once
  with `skipped: "no_material"`.
- **Upstream arms do not strip downstream layers** — three axes stack in the
  order lora (detector) → crop classifier (box reclassification) → rescorer
  (score calibration). Measuring with the downstream removed would make the
  gate's Δ a composite of "+upstream, −downstream", so an upstream axis would
  fight carrying the loss of the downstream gain — violating the single-increment
  principle. Instead each layer is handled according to its nature.
  - **The rescorer is refit inside the arm.** It is calibrated to the detector's
    score distribution (the detection score is a feature; the label is category
    match), so the old one cannot be reused once upstream changes. If the refit
    ends as identity (α=0 for all classes), axes are corrected to
    `rescorer: False` so the label matches the measured state.
  - **The classifier is kept as-is.** It is a crop → class function, not tied to
    the detector's score distribution — its only detector dependency is that the
    background class was mined from old FPs, and the gate judges whether that
    staleness hurts.

## FP/FN diagnosis

Diagnosis runs every round regardless of adoption and is the planner's only
input.

- **Cause classes**: missed / below_threshold (candidate exists but below τ) /
  misclassified / localization error (IoU too low) / background false positive /
  near-duplicate.
- **Cause → remedy axis mapping**: the deterministic planner's `REMEDY` is the
  source of truth. It is a **coverage relation**, not a decision rule — the
  deterministic planner uses its order as policy, and the LLM prompt inverts the
  same relation into "errors hit per axis this round". Shipping a static table
  would make the union of remedies cover every axis whenever several causes are
  observed, narrowing nothing. Near-duplicates and tiling merge defects have
  **no remedy** — the former is handled by the gateway's within-class folding;
  the latter is left to human candidate injection, not an axis.
- Train and holdout diagnoses go to separate files (`diagnosis_train.json` /
  `diagnosis_test.json` — the latter's filename is kept for artifact
  compatibility). The planner reads both, but the holdout side exposes aggregates
  only — structurally blocking proposals overfit to individual images. Those
  aggregates still influence which axes get proposed.
- **VLM crop descriptions cover the whole sample** — describing only part of it
  would let the call budget, not significance, decide which are described, and
  the planner would misread "no note" as a signal of lower importance. Crops per
  call are handled by batching.
- **Image identifiers are not included in samples** — once descriptions are
  folded into samples there is nothing to join on, and the planner has no way to
  point at images (the response schema has no such field). Boxes are given by
  size (`wh`) rather than coordinates, and classes by name rather than id —
  `per_class` is keyed by name, so ids alone would make individual errors
  unattributable to classes. Dataset-specific context goes through
  `dataset_notes`.

## LLM planner

- Served by vLLM, single guided-JSON calls. The prompt is rebuilt every round
  from champion axes + history + tried combinations + diagnosis; if it exceeds
  the character budget, the history window is trimmed first.
- **Role separation makes nondeterminism harmless**: the LLM **only proposes**;
  the holdout CI gate adopts. Bad proposals die at the gate.
- **Replay as the reproducibility substitute**: prompt and response are saved in
  `proposal.json`, and `--resume` replays proposals without calling the LLM. The
  resume key is the **proposal artifact**, not a config hash.
- **Deterministic planner** (`--planner deterministic`): a rule implementation of
  the cause → remedy mapping. It is the fixed execution path for guard tests and
  the control arm for LLM A/B.
- Other paths using the same server: alias refine (pool-membership check,
  verbatim only — no synthesis), mining captions for alias/text_negatives, and
  FP/FN crop descriptions. VLM descriptions are diagnostic signals only and are
  never used for decisions.

## Architecture

The package lives under `src/agentic_detector_config/` (src layout — tests and
scripts import the installed package, never the repo root). Layers:

```
core ← clients ← infer ← fitting ← eval ← diagnose ← planner ← loop ← cli
                                                      (services orthogonal)
```

- `core`: coco, data, metrics, tiling, io, prompts, suppress — pure functions.
- `clients`: YOLOE in-process client, response contract (wire), vLLM
  (OpenAI-compatible) client.
- `infer`: **gateway seam** — all detector inference goes through
  `infer/gateway.py`. Tests monkeypatch only the gateway.
- `fitting`: mine, materials, rescorer, verifier, tail, crop_classifier —
  material producers. **They read train only.**
- `eval`: screen (fit_dev), evaluate (holdout), τ estimation — consumers.
- `diagnose`: FP/FN classification, aggregation, reports.
- `planner`: deterministic / llm, two implementations of one interface.
- `loop`: round orchestration, gate, saturation, resume.
- `services`: subprocess/server management. `workers/` is the only place torch is
  allowed — workers run as one-shot subprocesses (dinov3/adapt in the vLLM venv,
  where the package is not installed, so they get `src/` as PYTHONPATH —
  `services.paths.PACKAGE_PARENT`), and boundary arguments are absolute paths only.

Import direction is enforced by `tests/test_architecture.py`.

**The four guard tests** — every change must keep them passing:

| test | what it pins |
|---|---|
| `test_materialize_purity.py` | materials are independent of holdout (`tau_test` is the only exception) |
| `test_split_roles.py` | fixed 80/20 boundary, stratification and seed, **no test access**, no role violations |
| `test_decision_channel.py` | shuffling holdout must change the decision (adoption really comes from holdout) |
| `test_replay.py` | resume in LLM mode makes zero LLM calls — replays only the proposal artifact |

## Outputs

```
runs/<dataset>/run/
  history.jsonl           # round ledger (trajectory, ts, dur_s)
  run_summary.json        # stop reason (stopped), adoption count, final champion, holdout_split
  champion.json           # current champion setting (axes, setting, materials, weights, adapt_root)
  champion_eval.json      # champion's holdout evaluation — tau_test, micro, records
  rounds/round_NNN/
    proposal.json         # planner proposal (with prompt/response for LLM — the replay unit)
    crop_notes.json       # VLM crop descriptions (kind=llm — diagnostic only, replay unit)
    settings/<cand>/      # mined materials (exemplars/visual_negatives/aliases/phrases)
      mine_reports.json   #   mining/inheritance rationale (parent, inherited, filter reasons)
      rescorer.json       #   discriminator and α (rescorer axis, or refit inside an arm)
      artifact/           #   reclassification artifact trained by the crop classifier arm
      arms/               #   per-λ arm descriptors for the lora arm (tail interpolation)
    screen.json           # fit_dev scoring
    eval.json             # holdout micro F(β) + tau_test + gate + candidate records
                          #   (records allow re-judging if gate criteria change)
    decision.json         # adopt/reject with rationale
    diagnosis_train.json / diagnosis_test.json
```

## Prediction — replaying the champion setting

To run inference with the setting the loop found, `scripts/predict.py` replays
the run artifacts (it is not a CLI subcommand — its lifecycle differs from the
loop's, so it is kept separate).

```bash
.venv/bin/python scripts/predict.py runs/<ds>/run \
  --config config/generated/par_<ds>.yaml --split test   # final report (never read by the loop)
.venv/bin/python scripts/predict.py runs/<ds>/run \
  --config config/generated/par_<ds>.yaml --images /path/to/dir
```

(`config/generated/par_<ds>.yaml` is written by `scripts/run_parallel.sh`; pass
whatever config the run used.)

Four pieces of state must be restored, all in `champion.json`: detector weights
(`weights`, tail arm descriptor), reclassification artifact (`adapt_root`, crop
classifier), materials (`materials_dir`), and score calibrator (`rescorer_path`).
The first two are loaded into the client through the same hooks as the loop
(`weights_ctl`, `adapt_ctl`).

Two things break reproduction if ignored:

- **The threshold is `tau_test` from `champion_eval.json`.** `champion.json`'s
  `setting.per_class[*].threshold` is the screen (fit_dev) threshold, a
  different value. That τ is **fit on holdout and is used unchanged for the
  `--split test` final report** — it is not re-estimated on test (`predict.py`
  applies `tau_test` from `champion_eval.json` as-is). Re-estimating would argmax
  the threshold on the report split, defeating the point of an unopened split.
- **A champion with the rescorer on must infer with `return_embeddings=True`**
  and threshold on the calibrated score channel — the calibration features come
  from embeddings, and `tau_test` was estimated on that channel.

`--config` is required because the run directory does not snapshot the config.
Paths in `champion.json` are absolute, so moving the run directory breaks them.
