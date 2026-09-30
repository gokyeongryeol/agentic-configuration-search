#!/usr/bin/env bash
# YOLOE seed wave over 3 datasets (aerial, aquarium, soccer).
#
# Protects existing outputs: the root is runs_yoloe_seed<S>/ — runs_yoloe/ is never
# touched. Derived configs are written only to config/generated_seed<S>/; the base
# config/yoloe-*.yaml files are read-only.
#
# **split.seed is pinned to 0** — if the fit_train/fit_dev boundary changed per wave,
# the measurement would mix split variance into search variance. Only run_seed is
# varied; it governs material-mining sampling, lora and crop classifier training,
# and gate bootstrap resampling.
#
# GPU: datasets are placed 1:1 onto the slots given as arguments. YOLOE is in-process,
# so there is no separate server; detector.device, mine.device and lora.device share one GPU.
# GPU 3 is the resident vLLM planner — never use it.
#
# Usage: scripts/run_seeds_yoloe.sh <seed> [gpu1 gpu2 gpu3]
#        scripts/run_seeds_yoloe.sh <seed> --only <suffix> --gpu <g>
set -uo pipefail
cd "$(dirname "$0")/.."

SEED=${1:?usage: run_seeds_yoloe.sh <seed> [gpu...] | <seed> --only <ds> --gpu <g>}
shift
ROUNDS=10

# HF cache — agent/cron shells have no HF_HOME and fall back to the default cache
# (/root/.cache/huggingface). Weights missing there (e.g. the dinov3 embedder) then
# fail with OSError under HF_HUB_OFFLINE=1. An existing HF_HOME is respected.
export HF_HOME="${HF_HOME:-/data/krgo/hf_cache}"

# Block ultralytics auto-install. During the 2026-09-11 seed1 run (04:37–04:39)
# ultralytics auto-installed dependencies and bumped torch to 2.14.0+cu130, which does
# not match this box's driver (570/CUDA 12.8), so **seed2/seed3, started afterwards,
# died at CUDA init**. Processes already running keep the old torch in memory and pass
# silently — the failure shows up in the next job, which makes it hard to trace.
# Never let the environment change mid-experiment.
export YOLO_AUTOINSTALL=false
# --only/--gpu: run a single dataset (config suffix) on the given GPU and exit —
# an entry point for an external queue driver. Passing a GPU list as positional
# arguments still runs the full wave.
ONLY=""; ONLY_GPU=""; POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY=$2; shift 2 ;;
        --gpu)  ONLY_GPU=$2; shift 2 ;;
        *) POS+=("$1"); shift ;;
    esac
done
GPUS=(${POS[@]+"${POS[@]}"}); [ ${#GPUS[@]} -eq 0 ] && GPUS=(0 1 2)

ROOT="runs_yoloe_seed$SEED"
CFGDIR="config/generated_seed$SEED"
DATASETS=(aerial aquarium soccer)      # base config suffixes

if [ -n "$ONLY" ]; then
    DATASETS=("$ONLY")
    GPUS=("${ONLY_GPU:?--only requires --gpu}")
fi

mkdir -p "$ROOT" "$CFGDIR"

# ---- code fingerprint — so these outputs can be traced back to the code that produced them ----
if [ -z "$ONLY" ] || [ ! -f "$ROOT/_code_fingerprint.txt" ]; then
{
  echo "commit: $(git rev-parse HEAD 2>/dev/null)"
  echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "started: $(date -Is)"
  echo "backend: yoloe  seed(run_seed): $SEED  split.seed: 0  rounds: $ROUNDS"
  echo "gpus: ${GPUS[*]}"
  echo "--- git status --porcelain ---"
  git status --porcelain 2>/dev/null
} > "$ROOT/_code_fingerprint.txt"
git diff > "$ROOT/_code_uncommitted.diff" 2>/dev/null
fi

# ---- shared vLLM ----
# Launched only from scripts/start_vllm.sh (idempotent — no-op if already up).
bash scripts/start_vllm.sh || {
    echo "[yoloe-seed$SEED] failed to bring up vLLM — check runs/_vllm.log"; exit 1; }

gen_cfg() {  # $1=base suffix $2=gpu
    local base="config/yoloe-$1.yaml" f="$CFGDIR/yoloe-$1.yaml"
    # base is read-only — the derived copy rewrites the 3 devices and out_dir and appends both seeds
    sed -e "s|cuda:[0-9]\+|cuda:$2|g" \
        -e "s|^out_dir: .*|out_dir: $ROOT|" "$base" > "$f"
    {
      echo ""
      echo "run_seed: $SEED   # the only thing varied — mining sampling, training, bootstrap"
      echo "split:"
      echo "  seed: 0        # fixed — the fit_train/fit_dev boundary must be identical across waves"
    } >> "$f"
    echo "$f"
}

declare -A PID_BY_GPU
LAST_RC=0
next=0
n=${#DATASETS[@]}

while :; do
    for i in "${!GPUS[@]}"; do
        g=${GPUS[$i]}
        if [ -z "${PID_BY_GPU[$g]:-}" ] && [ "$next" -lt "$n" ]; then
            ds=${DATASETS[$next]}; next=$((next + 1))
            cfg=$(gen_cfg "$ds" "$g")
            echo "[yoloe-seed$SEED] $(date +%H:%M:%S) $ds → GPU $g"
            HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
                OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                PYTORCH_ALLOC_CONF=expandable_segments:True \
                .venv/bin/adc run --config "$cfg" --resume \
                > "$ROOT/_log_$ds.txt" 2>&1 &
            PID_BY_GPU[$g]=$!
        fi
    done
    for g in "${!PID_BY_GPU[@]}"; do
        if ! kill -0 "${PID_BY_GPU[$g]}" 2>/dev/null; then
            wait "${PID_BY_GPU[$g]}" 2>/dev/null
            wrc=$?
            [ "$wrc" != 0 ] && LAST_RC=$wrc      # propagate so a queue driver notices the failure
            echo "[yoloe-seed$SEED] $(date +%H:%M:%S) GPU $g slot freed (exit $wrc)"
            unset "PID_BY_GPU[$g]"
        fi
    done
    [ "$next" -ge "$n" ] && [ "${#PID_BY_GPU[@]}" -eq 0 ] && break
    sleep 20
done

echo "[yoloe-seed$SEED] search done — reports"
for d in "$ROOT"/*/run; do
    [ -f "$d/history.jsonl" ] && .venv/bin/adc report "$d" > /dev/null && echo "report: $d/report.md"
done

exit $LAST_RC
