# agentic-configuration-search

An agentic loop that searches the inference settings of YOLOE (ultralytics'
open-vocabulary detector) for each dataset. A planner reads the FP/FN diagnosis
and proposes candidates; only a candidate that passes the holdout gate becomes
the champion.
License: AGPL-3.0 (ultralytics is imported in-process — research baseline
only) · Design: [docs/design.md](docs/design.md)

## Quick start

```bash
python3 -m venv --system-site-packages .venv          # ultralytics and torch are installed separately, not declared
.venv/bin/pip install -e . --no-build-isolation --no-deps
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Datasets use the Roboflow layout (`train/`, `valid/`, `test/`). A config lists
only the keys it changes (the rest come from
[`config/default.yaml`](config/default.yaml); an unknown key is an error).

```yaml
dataset: demo_data/soccer-players-5fuqs
detector: {model: weights_yoloe/yoloe-26x-seg.pt, device: "cuda:0"}
mine: {device: "cuda:1"}
```
```bash
HF_HUB_OFFLINE=1 .venv/bin/acs run --config c.yaml             # → runs/<dataset>/run/
.venv/bin/acs run --config c.yaml --resume                      # continue (replays saved proposals, no LLM calls)
.venv/bin/acs run --config c.yaml --rounds 1 --candidates cands.yaml   # inject human candidates
.venv/bin/acs report runs/<ds>/run [--config c.yaml]            # redraw report.md
.venv/bin/acs ab runs/a runs/b                                  # compare two runs
.venv/bin/python scripts/predict.py runs/<ds>/run --config c.yaml --split test   # final measurement
scripts/run_parallel.sh --rounds 10                             # demo datasets in parallel across GPUs
```

YOLOE runs in-process; the only server is the vLLM planner, which a run
attaches to when a resident (persist) server is healthy and launches
otherwise. With `planner.base_url` it attaches to that endpoint.
`config/_smoke-full.yaml` is a one-round smoke run.

## One round

Round 0 is the baseline with every axis off. Each later round runs in this
order ([round loop](docs/design.md#round-loop)).

```
diagnose     count the champion's FP/FN by cause          train (holdout aggregates only)
propose      the planner proposes increments on the champion
materialize  mine / train the changed axes' materials     train / fit_train
screen       pick configs and thresholds, top_k pass      fit_dev
evaluate     estimate τ_c + micro F(β)                     holdout
decide       adopt the gate-passing candidate with max Δ  holdout
```

| split | role ([split roles](docs/design.md#split-roles)) |
|---|---|
| `train` = `fit_train` 80% + `fit_dev` 20% | material mining (all of train), training (`fit_train` — a subset of it above `subsample.max_images`), candidate scoring (`fit_dev`) |
| `valid` (holdout) | τ_c estimation, adoption, stop decisions. Without it, `test` takes its place |
| `test` | never read by the loop. Measured once with `predict.py` |

Adoption: the one-sided paired-bootstrap lower bound of Δ ≥ `gate.margin`,
Δ > 0, and Δ ≥ `gate.min_delta`.
Stop: `saturation.rounds` stagnant rounds, `saturation.empty_rounds` empty
rounds, planner stop, or the `rounds` cap.

## Search axes

Every axis has a value that equals the current behavior exactly when off (the
first value). To leave axes out, `search.disabled_axes: [lora, tiling]`;
details in the [axis reference](docs/design.md#axis-reference).

| axis | values | what it does |
|---|---|---|
| `alias` | `false` · `vlm` | class aliases a multimodal LLM makes from GT crops, added to the text prompt |
| `exemplar` | `none` · `base` · `fn_arm` | GT crops (+ crops of missed GT) as image prompts. Counts `exemplar_base_k`, `exemplar_fn_k` |
| `text_negatives` | `false` · `fp_crop` · `gt_crop` · `background` | suppression phrases from false-positive or other-class crops. Count `text_negative_k` |
| `visual_negatives` | `false` · `fp_crop` · `gt_crop` · `background` | crops from the same mining sources as `text_negatives` as negative image prompts (needs `exemplar`). Count `visual_negative_k` |
| `rescorer` | `false` · `true` | discriminator score calibration (per-class α, `rescorer.alphas`) |
| `crop_classifier` | `false` · `true` | SigLIP2+LoRA class probabilities blended into each box's class and score over a λ grid |
| `lora` | `false` · `true` | YOLOE tail fine-tune interpolated with the base (`lora.lambdas`, λ=0 is the base) |
| `tiling` | `false` · 2–6 | g×g tile inference, then merge. 1+g² forwards per image |

## Common settings

| key | default | meaning |
|---|---|---|
| `dataset` · `rounds` · `beta` | required · 10 · 1.0 | dataset, round cap, β of F(β) |
| `planner.kind` | `llm` | the vLLM planner. `deterministic` is the rule planner (A/B control, environments without an LLM) |
| `planner.max_candidates` | 3 | candidates per round (1–3) |
| `dataset_notes` | `""` | dataset description carried verbatim in the LLM prompt |
| `search.disabled_axes` | `[]` | axes not to search |
| `gate.min_delta` | 0.005 | minimum Δ for adoption |
| `saturation.rounds` | 5 | stagnant rounds before stopping |
| `detector.device` · `mine.device` · `lora.device` | `cuda:0` · `cuda:0` · `cuda:0` | GPU placement (keep off the planner's GPU) |
| `subsample.max_images` · `subsample.center_ratio` | 500 · 0.1 | when fit_train is large, train lora, the rescorer and the classifier on a subset that covers the GT embedding clusters. null = off |
| `rescorer.max_train_rows` | 100000 | rescorer discriminator training row cap (every TP, FPs stratified + weighted). null = no cap |

The lora and crop_classifier axes swap the in-process detector's state (tail
weights, classifier artifact) through controls the CLI wires in.

## Outputs

```
runs/<dataset>/run/
  history.jsonl       round ledger (per-candidate axes, outcome, gate, wall_min)
  run_summary.json    stop reason, final champion, holdout_split, fit_train (training images, subset record), dataset
  fit_train_subset.json  fit_train subset (picked images, centers, coverage) — only when fit_train exceeds the cap
  champion.json       champion setting (axes, materials, weight paths, classifier blend λ)
  champion_eval.json  holdout evaluation, tau_test
  rounds/round_NNN/   proposal, settings/<cand>/, screen, eval, decision, diagnosis
  rounds/final/       final champion diagnosis + residual.json
  lora_cache/         tail training output — lora candidates with the same training inputs reuse it
  classifier_cache/   crop classifier artifact — trained once per run, later proposals reuse it
  report.md           run report without technical terms — data → how it improved → what is left → more data
  report_assets/      report figures (chart PNGs, photo JPEGs)
```

`predict.py` uses `tau_test` from `champion_eval.json`. Options are in the
[outputs reference](docs/design.md#outputs-reference).

## Operations (shared GPUs)

- Before touching anything, check running experiments with `nvidia-smi` and
  `pgrep -af "run_parallel|agentic_configuration_search|vllm"`.
- Do not take down the vLLM planner server (GPU 3, port 8104). Start it with
  `scripts/start_vllm.sh`; check it with `curl 127.0.0.1:8104/health`. When
  sharing the GPU, lower its preallocation with e.g. `--gpu-mem 0.5` (default
  0.9; 27B-FP8 weights are ~29 GB, so 0.5 ≈ 40 GB is near the floor).
- `scripts/run_parallel.sh` runs with `--resume` inside, so it is safe to
  restart. Logs go to `runs/_log_<ds>.txt`.
- Deleting `runs/`: on NFS an open fd blocks it. Stop monitors, mv, then rm.
- Secrets (HF_TOKEN) only through env. For cached models, `HF_HUB_OFFLINE=1`.

## Code map

Layers `core ← clients ← infer ← fitting ← eval ← diagnose ← planner ← loop ← cli`
(enforced by `tests/test_architecture.py`); torch only in `workers/`.

| package | contents |
|---|---|
| `core/` | dataset loading and split roles, COCO I/O, metrics and threshold estimation, tiling |
| `clients/` | in-process YOLOE client, vLLM chat client, the detector response contract |
| `infer/` | `gateway.py` (the single path for all detector inference), query and phrase assembly, embedder |
| `fitting/` | material mining (`mine.py`), rescorer, tail fine-tune, crop classifier |
| `eval/` | screen (fit_dev), holdout evaluation and the adoption gate (`evaluate.py`) |
| `diagnose/` | FP/FN cause classification, VLM notes, residual analysis, ledger readers (`report.py`), run report (`story.py`) |
| `planner/` | axes schema, deterministic and LLM planners, tiling gate, axis ledger |
| `loop/` | round orchestration (`runner.py`), materialize, per-axis round wiring |
| `services/` · `workers/` | subprocess and server management · one-shot GPU workers (`ACS_VLLM_VENV`; DINOv3 worker CPU threads and loaders via `ACS_EMBED_THREADS`, `ACS_EMBED_LOADERS`, default min(8, cores), min(4, cores)) |

Guard tests `test_materialize_purity`, `test_split_roles`,
`test_decision_channel` and `test_replay` must not be weakened.
