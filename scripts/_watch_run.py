"""Run progress watcher — one line per new round in history.jsonl; exits on completion or crash.

usage: _watch_run.py <run_dir> <log_file>
"""
import json
import sys
import time
from pathlib import Path

run = Path(sys.argv[1])
logf = Path(sys.argv[2])
seen = 0
while True:
    hist = run / "history.jsonl"
    if hist.exists():
        lines = hist.read_text().splitlines()
        for line in lines[seen:]:
            e = json.loads(line)
            d = e.get("decision") or {}
            ch = e.get("champion", {})
            axes = {k: v for k, v in (ch.get("axes") or {}).items()
                    if v not in (False, None, "none")}
            print(f"round {e['round']}: adopted={d.get('adopted')} "
                  f"id={d.get('id')} champion_f1={ch.get('micro_f_beta'):.4f} "
                  f"axes={axes}", flush=True)
        seen = len(lines)
    summary = run / "run_summary.json"
    if summary.exists() and seen:
        s = json.loads(summary.read_text())
        print(f"RUN DONE: {s['stopped']} adopted={s['adopted_rounds']} "
              f"final={s['champion']['micro_f_beta']}", flush=True)
        break
    if logf.exists():
        txt = logf.read_text(errors="replace")
        for sig in ("CUDA error", "out of memory", "OutOfMemoryError"):
            if sig in txt:
                print(f"CRASH SIGNATURE: {sig}", flush=True)
                sys.exit(1)
        # process died without a summary — print the tail (last traceback)
        if txt.rstrip().endswith("exit=1"):
            print("PROCESS EXITED 1 — tail:", flush=True)
            print("\n".join(txt.splitlines()[-15:]), flush=True)
            sys.exit(1)
    time.sleep(20)
