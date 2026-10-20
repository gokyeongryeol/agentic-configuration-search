#!/usr/bin/env bash
# Replay one dataset's **round-0 baseline** on test.
#
# The detector (YOLOE) runs in-process, so there is no launch (gpu/port) to
# retarget — the GPU is reassigned via three keys: detector.device +
# mine.device + lora.device.
#
# Why re-measure the baseline: the recorded BASE is a holdout number, so the
# improvement is only defined on the split the gate looked at. Re-measuring the
# baseline on test makes it a same-split difference against the champion
# (already measured on test), closing that gap.
#
# The baseline has no artifact of its own, but it is **fully reconstructible**:
#   all axes off   = a single config (config_id 0, no tiling/visual_neg) +
#                    modality "text" for every class
#   τ_c            = tau_test from rounds/round_000/decision.json (fixed on holdout)
#   materials      = rounds/round_000/settings/baseline (as round 0 left them)
#
# The reconstruction is not taken on faith: it is first **replayed on holdout and
# checked against the recorded BASE** (±0.002); on mismatch we abort without touching test.
#
# Usage: ACS_RUNS=<root> ACS_CFG=<config> scripts/baseline_one.sh <dataset> <gpu>
#   e.g. ACS_RUNS=runs_yoloe_seed1 ACS_CFG=config/generated_seed1/yoloe-aerial.yaml
set -u
cd "$(dirname "$0")/.."
ds=$1; gpu=$2
RUNS=${ACS_RUNS:?ACS_RUNS required (e.g. runs_yoloe_seed1)}
CFG=${ACS_CFG:?ACS_CFG required (e.g. config/generated_seed1/yoloe-aerial.yaml)}
base="$RUNS/$ds/baseline"
log(){ echo "[base-$ds] $(date '+%m-%d %H:%M:%S') $*"; }

mkdir -p "$base/_logs"
.venv/bin/python - "$ds" "$base" "$RUNS" <<'PY' || exit 1
import json, sys
from pathlib import Path
ds, base, runs = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
run = Path(runs) / ds / "run"
r0 = json.loads((run / "rounds/round_000/decision.json").read_text())
hist0 = json.loads((run / "history.jsonl").read_text().splitlines()[0])
champ0 = hist0["champion"]
tau = r0["tau_test"]
setting = {
    "configs": [{"config_id": 0, "visual_negatives": False,
                 "neg_eta": None, "tiling": None}],
    # same config and text modality for every class — this is what "all axes off" means
    "per_class": {c: {"config_id": 0, "threshold": float(t),
                      "f_beta": 0.0, "modality": "text"} for c, t in tau.items()},
}
mats = (run / "rounds/round_000/settings/baseline").resolve()
assert mats.is_dir(), f"round-0 materials missing: {mats}"
(base / "champion.json").write_text(json.dumps({
    "round": 0, "id": "baseline", "axes": champ0["axes"],
    "materials_dir": str(mats), "setting": setting,
    "micro_f_beta": champ0["micro_f_beta"],
    "weights": None, "adapt_root": None, "rescorer_path": None,
}, indent=2))
(base / "champion_eval.json").write_text(json.dumps({"tau_test": tau}, indent=2))
print(f"recorded BASE (holdout) = {champ0['micro_f_beta']:.4f}")
PY

tmp=$(mktemp /tmp/basecfg_XXXX.yaml)
BG=$gpu .venv/bin/python - "$CFG" "$tmp" <<'PY'
import sys, os, yaml
cfg, tmp = sys.argv[1], sys.argv[2]
g = os.environ["BG"]
c = yaml.safe_load(open(cfg))
# in-process detector — move detector.device, not launch (gpu/port)
c["detector"]["device"] = f"cuda:{g}"
c["mine"]["device"] = f"cuda:{g}"
c["lora"]["device"] = f"cuda:{g}"
yaml.safe_dump(c, open(tmp, "w"), allow_unicode=True)
PY

# 1) verify the reconstruction — the recorded BASE must reproduce on holdout
log "verify (holdout, gpu $gpu)"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/predict.py "$base" --config "$tmp" \
    --split holdout --out "$base/predictions_holdout.json" > "$base/_logs/verify.log" 2>&1
got=$(grep -oE "holdout micro F\(β\) [0-9.]+" "$base/_logs/verify.log" | tail -1 | grep -oE "[0-9.]+$")
want=$(.venv/bin/python -c "import json;print(f\"{json.load(open('$base/champion.json'))['micro_f_beta']:.4f}\")")
if [ -z "$got" ]; then log "FAIL verification replay produced no number"; tail -5 "$base/_logs/verify.log"; rm -f "$tmp"; exit 1; fi
ok=$(.venv/bin/python -c "print(1 if abs($got-$want)<=0.002 else 0)")
if [ "$ok" != "1" ]; then log "FAIL reconstruction mismatch: holdout $got vs recorded $want — aborting test measurement"; rm -f "$tmp"; exit 1; fi
log "verify ok: holdout $got == recorded $want"

# 2) actual measurement — baseline on test (micro F(β) + COCO mAP)
log "measure (test)"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/predict.py "$base" --config "$tmp" \
    --split test --coco-map --out "$base/predictions_test.json" > "$base/_logs/test.log" 2>&1 \
    || { log "FAILED test"; tail -5 "$base/_logs/test.log"; rm -f "$tmp"; exit 1; }
rm -f "$tmp"
log "done: $(grep -oE 'test micro F\(β\) [0-9.]+' "$base/_logs/test.log" | tail -1)  $(grep -oE 'COCO mAP.*' "$base/_logs/test.log" | tail -1)"
