#!/usr/bin/env bash
# Start the shared vLLM planner on GPU 3 — every run attaches to it.
#
# Why a standalone script: so launch arguments (--max-model-len etc.) do not split
# into several copies, launching happens in exactly one place (run_parallel.sh calls this).
#
# Properties:
#   · **Idempotent** — if health already returns 200, exit without doing anything.
#     Calling it twice by mistake never kills the running server or starts a second one.
#   · **setsid-detached** — stopping a background task kills its whole process tree.
#     A resident server must live outside that tree to survive.
#   · Model load takes several minutes and the port is not bound until then, so success is
#     reported only after health comes up — starting runs without waiting would race:
#     each sees a free port and launches its own vLLM.
#
# Usage: scripts/start_vllm.sh [--gpu 3] [--port 8104] [--max-len 32768] [--model NAME]
#                              [--gpu-mem 0.9]
#        scripts/start_vllm.sh --stop     # shut down (only when no run is using it)
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=3
PORT=8104
MAXLEN=32768                       # the same as the config default planner.launch.max_model_len
MODEL=Qwen/Qwen3.8-27B-FP8
# Share of GPU memory vLLM preallocates (weights + KV cache pool). The default
# 0.9 equals planner.launch.gpu_memory_utilization when a run launches it. On a
# shared GPU the floor is the weights — 27B-FP8 weights alone are ~29 GB, so 0.5
# (~40 GB) on an 80 GB A100 leaves ~10 GB of KV cache (one 32k-context request
# takes ~9 GB — use 0.6 or more with many concurrent requests).
GPU_MEM=0.9
# Planner server venv. This is a **different variable** from the GPU worker venv
# (ACS_VLLM_VENV, services/paths.py) — the two may diverge on this box. While the
# resident server is up, the worker side needs no vllm at all (VllmServer only attaches).
VENV="${ACS_VLLM_SERVE_VENV:-/data/krgo/vllm2}/bin/vllm"
STOP=0

# HF cache location. Interactive shells have it set but agent/cron shells don't, so
# they look at the default cache (/root/.cache/huggingface), miss the weights and die —
# with HF_HUB_OFFLINE=1 nothing is downloaded and it ends in LocalEntryNotFoundError.
# An existing HF_HOME is respected.
export HF_HOME="${HF_HOME:-/data/krgo/hf_cache}"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)      GPU=$2; shift 2 ;;
        --port)     PORT=$2; shift 2 ;;
        --max-len)  MAXLEN=$2; shift 2 ;;
        --model)    MODEL=$2; shift 2 ;;
        --gpu-mem)  GPU_MEM=$2; shift 2 ;;
        --stop)     STOP=1; shift ;;
        *) echo "unknown argument: $1"; exit 2 ;;
    esac
done

# 0 < GPU_MEM <= 1 — vLLM may reject an out-of-range value only after the long load
if ! awk -v m="$GPU_MEM" 'BEGIN { exit !(m + 0 > 0 && m + 0 <= 1 && m ~ /^[0-9]*\.?[0-9]+$/) }'; then
    echo "[vllm] --gpu-mem must be a share greater than 0 and at most 1 (got: $GPU_MEM)"; exit 2
fi

URL="http://127.0.0.1:$PORT"
LOG="runs/_vllm.log"
mkdir -p runs

if [ "$STOP" = 1 ]; then
    # Don't shut down while a run uses it — that would pull the planner out from under a live experiment.
    # The pattern also matches `za run` (legacy CLI name) since the vLLM may be shared with other checkouts.
    if pgrep -f "(za|acs) run --config" >/dev/null 2>&1; then
        echo "[vllm] an acs run is active — not stopping the planner. Stop the run first"; exit 1
    fi
    pids=$(pgrep -f "vllm serve $MODEL" | tr '\n' ' ')
    [ -z "$pids" ] && { echo "[vllm] not running"; exit 0; }
    echo "[vllm] stopping: $pids"; kill $pids 2>/dev/null
    for _ in $(seq 1 30); do pgrep -f "vllm serve $MODEL" >/dev/null || break; sleep 2; done
    pgrep -f "vllm serve $MODEL" >/dev/null && kill -9 $(pgrep -f "vllm serve $MODEL") 2>/dev/null
    echo "[vllm] stopped"; exit 0
fi

if curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "[vllm] already up ($URL) — nothing to do"
    exit 0
fi

# check the GPU is free — stacking onto someone else's job OOM-kills both
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
if [ -n "$used" ] && [ "$used" -gt 2000 ]; then
    echo "[vllm] GPU $GPU already has ${used}MiB in use — it may be another job."
    echo "       To launch anyway, free the GPU or pick another one with --gpu."
    exit 1
fi

# without cached weights we'd only fail after a 30-min health wait — check up front
snap="$HF_HOME/hub/models--$(echo "$MODEL" | sed 's|/|--|')"
if [ "${HF_HUB_OFFLINE:-1}" = 1 ] && [ ! -d "$snap" ]; then
    echo "[vllm] $MODEL not in cache: $snap"
    echo "       Check that HF_HOME is correct (current value: $HF_HOME)"
    exit 1
fi

echo "[vllm] launching: GPU $GPU, $MODEL, port $PORT, max-model-len $MAXLEN, gpu-mem $GPU_MEM (HF_HOME=$HF_HOME)"
CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    setsid nohup "$VENV" serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" --max-model-len "$MAXLEN" \
    > "$LOG" 2>&1 &

echo "[vllm] waiting for load (several minutes; the port opens after loading) — log: $LOG"
for i in $(seq 1 180); do
    curl -sf "$URL/health" >/dev/null 2>&1 && { echo "[vllm] ready ($((i*10))s)"; exit 0; }
    sleep 10
done
echo "[vllm] health did not come up within 30 min — check $LOG"
exit 1
