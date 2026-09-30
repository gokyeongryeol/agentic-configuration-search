#!/usr/bin/env bash
# Run the demo_data datasets in parallel across GPUs.
#
# Layout: GPU 3 = shared vLLM planner (persist — every run attaches), the other 7
# GPUs are dataset slots. When a slot frees up, the next queued dataset is started.
# Per dataset, the detector (in-process YOLOE) and workers (mine/lora device) share that one GPU.
#
# Usage: scripts/run_parallel.sh [--rounds N]
set -uo pipefail
cd "$(dirname "$0")/.."

ROUNDS=10
[ "${1:-}" = "--rounds" ] && ROUNDS=$2

DATASETS=(soda-bottles cable-damage cell-towers
          water-meter-jbktv-7vz5k-axdg-1 aquarium-qlnqy aerial-spheres
          soccer-players-5fuqs)   # largest first — long runs make the most of the parallel phase
GPUS=(0 1 2 4 5 6 7)              # GPU 3 is the shared vLLM

mkdir -p runs config/generated

# ---- shared vLLM ----
# Launched only from scripts/start_vllm.sh (idempotent — no-op if already up).
# An inline copy would let flags like --max-model-len silently drift.
bash scripts/start_vllm.sh || {
    echo "[scheduler] failed to bring up vLLM — check runs/_vllm.log"; exit 1; }

gen_cfg() {  # $1=dataset $2=gpu
    local f="config/generated/par_$1.yaml"
    cat > "$f" <<EOF
dataset: demo_data/$1
rounds: $ROUNDS
planner:
  kind: llm
  model: Qwen/Qwen3.8-27B-FP8   # default planner model (weight-only dequant on A100)
  launch: {gpu: 3, port: 8104, persist: true}
detector:
  device: "cuda:$2"
mine:
  device: "cuda:$2"
lora:
  device: "cuda:$2"
EOF
    echo "$f"
}

declare -A PID_BY_GPU
next=0
n=${#DATASETS[@]}

while :; do
    # fill empty slots
    for g in "${GPUS[@]}"; do
        if [ -z "${PID_BY_GPU[$g]:-}" ] && [ "$next" -lt "$n" ]; then
            ds=${DATASETS[$next]}; next=$((next + 1))
            cfg=$(gen_cfg "$ds" "$g")
            echo "[scheduler] $(date +%H:%M:%S) $ds → GPU $g"
            HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
                OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                .venv/bin/adc run --config "$cfg" --out "runs/$ds/run" --resume \
                > "runs/_log_$ds.txt" 2>&1 &
            PID_BY_GPU[$g]=$!
        fi
    done
    # reap finished runs
    for g in "${!PID_BY_GPU[@]}"; do
        if ! kill -0 "${PID_BY_GPU[$g]}" 2>/dev/null; then
            wait "${PID_BY_GPU[$g]}" 2>/dev/null
            echo "[scheduler] $(date +%H:%M:%S) GPU $g slot freed (exit $?)"
            unset "PID_BY_GPU[$g]"
        fi
    done
    [ "$next" -ge "$n" ] && [ "${#PID_BY_GPU[@]}" -eq 0 ] && break
    sleep 20
done

echo "[scheduler] all done — generating reports"
for ds in "${DATASETS[@]}"; do
    [ -f "runs/$ds/run/history.jsonl" ] && \
        .venv/bin/adc report "runs/$ds/run" > /dev/null && \
        echo "report: runs/$ds/run/report.md"
done
