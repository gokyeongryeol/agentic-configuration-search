#!/usr/bin/env bash
# Replay the baseline and champion of a YOLOE seed wave on test — micro F(β) + COCO mAP.
#
#   · No server (in-process backend) → no port; the GPU is reassigned via three
#     keys: detector.device + mine.device + lora.device.
#   · Dataset names (aerial-spheres) differ from config suffixes (aerial) → mapped below.
#   · Output root is runs_yoloe_seed<S>/, configs are config/generated_seed<S>/yoloe-*.yaml
#     (both written by scripts/run_seeds_yoloe.sh).
#
# Idempotent: skips anything that already has a .map.json. Also skips datasets whose
# search is still running (detected by an acs run holding that config file).
#
# Usage: scripts/map_batch_seed.sh <seed> [gpu...]
set -u

# HF cache — agent/cron shells have no HF_HOME, so they look at the default cache
# and miss the weights (e.g. the dinov3 embedder). Replay runs in a different
# process from the search and does not inherit run_seeds_yoloe.sh's export — so set it here too.
export HF_HOME="${HF_HOME:-/data/krgo/hf_cache}"
cd "$(dirname "$0")/.."
SEED=${1:?usage: map_batch_seed.sh <seed> [gpu...]}
shift
POOL=("$@"); [ ${#POOL[@]} -eq 0 ] && POOL=(0 1 2 4 5 6 7)

BACKEND=yoloe
ROOT="runs_${BACKEND}_seed$SEED"
CFGDIR="config/generated_seed$SEED"
LOG="$ROOT/_map_batch.log"
# dataset directory name → config suffix
declare -A SUFFIX=( [aerial-spheres]=aerial [aquarium-qlnqy]=aquarium
                    [soccer-players-5fuqs]=soccer )

log(){ echo "[map-$BACKEND-seed$SEED] $(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
[ -d "$ROOT" ] || { echo "$ROOT does not exist — run the search first"; exit 1; }

# Pick one GPU from the pool with <1000MiB used; wait until one frees up —
# stacking onto someone else's job OOM-kills both.
pick_gpu(){ for g in "${POOL[@]}"; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -n "$m" ] && [ "$m" -lt 1000 ] && { echo "$g"; return; }
  done; echo ""; }

for s in "$ROOT"/*/run/run_summary.json; do
  [ -f "$s" ] || continue
  ds=$(basename "$(dirname "$(dirname "$s")")")
  suf=${SUFFIX[$ds]:-}
  [ -z "$suf" ] && { log "skip $ds (no config suffix mapping)"; continue; }
  cfg="$CFGDIR/$BACKEND-$suf.yaml"
  [ -f "$cfg" ] || { log "skip $ds ($cfg missing)"; continue; }
  pgrep -f "$cfg" >/dev/null && { log "skip $ds (search in progress)"; continue; }

  g=$(pick_gpu); while [ -z "$g" ]; do log "waiting for a GPU…"; sleep 60; g=$(pick_gpu); done

  # 1) baseline — reconstruct round 0 → verify on holdout → test. Skipped if already done.
  if [ ! -f "$ROOT/$ds/baseline/predictions_test.map.json" ]; then
      log "baseline $ds (gpu $g)"
      ACS_RUNS="$ROOT" ACS_CFG="$cfg" \
          bash scripts/baseline_one.sh "$ds" "$g" >> "$LOG" 2>&1 \
          || log "FAILED baseline $ds"
  fi

  # 2) champion
  out="$ROOT/$ds/run/predictions_test.json"
  if [ -f "${out%.json}.map.json" ]; then log "skip champ $ds (already measured)"; continue; fi
  tmp=$(mktemp /tmp/maps_XXXX.yaml)
  MAP_CFG="$cfg" .venv/bin/python - "$g" "$tmp" <<'PY'
import sys, os, yaml
g, tmp = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(os.environ["MAP_CFG"]))
c["detector"]["device"] = f"cuda:{g}"
c["mine"]["device"] = f"cuda:{g}"
c["lora"]["device"] = f"cuda:{g}"
yaml.safe_dump(c, open(tmp, "w"), allow_unicode=True)
PY
  log "champion $ds (gpu $g)"
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/predict.py "$ROOT/$ds/run" \
      --config "$tmp" --split test --coco-map --out "$out" >> "$LOG" 2>&1 \
      || log "FAILED champion $ds"
  rm -f "$tmp"
done
log "seed$SEED test replay done"
