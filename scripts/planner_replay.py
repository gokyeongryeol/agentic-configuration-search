#!/usr/bin/env python
"""Ask the planner again for every round of a recorded run, comparing the proposals
before and after the axis status (OFF) is turned on.

No detector or GPU — only the planner vLLM is called. Round r's prompt is rebuilt
from its history (before r) and its diagnosis (rounds/round_r/diagnosis_*.json,
crop notes), and asked of a planner with `search.axis_retire_after` off (before)
and on (after).

Metrics (per round and variant):
  off_hit   candidates changing only OFF axes — by the axis status computed from
            the same history (retire_after=K)
  invalid   response rows dropped for wrong axes or values (invalid_axes)
  hit       whether it proposed a candidate with the same increment (over the
            round's starting champion, by request) as the one actually adopted
            that round — a round won by a carried candidate cannot hit by design
  prompt    first-prompt length (characters)

before's off_hit is "candidates it would have saved knowing OFF", after's off_hit
is "candidates proposed despite knowing OFF" (in a real run the runner drops those
as axis_retired). The two variants' prompts differ only in retire_after — before
has no OFF verdict (no OFF marks, nothing left out) and the axis board's run
column is a record only. So it measures the effect of OFF alone.

The prompt is assembled as the runner does: token_budget (the max_model_len the
server reports, else planner.launch.max_model_len), the budget card (rounds left,
stop counters), the prescreen note and the fit_train count after subsampling
(DatasetFacts). Re-asks are not replayed.

Usage:
  .venv/bin/python scripts/planner_replay.py runs/<ds>/run --config c.yaml \\
      --llm-url http://127.0.0.1:8104 --out replay/<ds>.json [--retire-after 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")       # before numpy is imported (same as cli)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger("planner_replay")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def round_diagnosis(run_dir: Path, r: int) -> dict:
    """Round r's diagnosis — in the shape the loop gave the planner (crop notes on train)."""
    d = run_dir / "rounds" / f"round_{r:03d}"
    diag = {"train": _load(d / "diagnosis_train.json") or {},
            "test": _load(d / "diagnosis_test.json") or {}}
    notes = (_load(d / "crop_notes.json") or {}).get("notes")
    if notes and isinstance(diag["train"], dict):
        diag["train"]["crop_notes"] = notes
    return diag


def adopted_delta(entry: dict, champion: dict) -> dict | None:
    """The adopted candidate's increment (by request) — compared with what the planner proposed even if relabelled."""
    from agentic_configuration_search.planner.llm import _delta_axes
    for c in entry.get("candidates", []):
        if c.get("adopted"):
            return _delta_axes(c.get("requested_axes") or c.get("axes"), champion)
    return None


def score(candidates: list[dict], status, champion: dict, adopted: dict | None) -> dict:
    from agentic_configuration_search.planner.axis_status import _changed_axes
    from agentic_configuration_search.planner.llm import _delta_axes
    off = 0
    for c in candidates:
        changed = _changed_axes(c["axes"], champion)
        off += bool(changed) and changed <= status.off
    hit = adopted is not None and any(_delta_axes(c["axes"], champion) == adopted
                                      for c in candidates)
    return {"n": len(candidates), "off_hit": off, "hit": hit}


def n_fit_train(run_dir: Path, ds) -> int:
    """The same count as the runner's DatasetFacts — the subset's if one was recorded."""
    from agentic_configuration_search.fitting.subsample import SUBSET_FILE
    rec = _load(run_dir / SUBSET_FILE) or {}
    return len(rec.get("images") or ()) or len(ds.fit_train.images)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8104")
    ap.add_argument("--retire-after", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=None, help="only this many rounds from the start")
    ap.add_argument("--rounds", type=int, default=None,
                    help="total rounds on the budget card — the run's --rounds (default: config rounds)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from agentic_configuration_search.clients.llm import LLMClient
    from agentic_configuration_search.config import load_config
    from agentic_configuration_search.core.data import load_dataset, prescreen_subset
    from agentic_configuration_search.diagnose.report import _history
    from agentic_configuration_search.loop.runner import (_token_budget, _trailing_empty,
                                                          _trailing_stagnant)
    from agentic_configuration_search.planner.axes import DatasetFacts, effective_disabled
    from agentic_configuration_search.planner.axis_status import axis_status
    from agentic_configuration_search.planner.llm import LLMPlanner

    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(args.config)
    ds = load_dataset(Path(cfg.dataset).resolve(), seed=cfg.split.seed,
                      fit_train_ratio=cfg.split.fit_train_ratio)
    facts = DatasetFacts(n_classes=len(ds.ontology.classes),
                         n_fit_train=n_fit_train(run_dir, ds),
                         lora_min_images=cfg.lora.min_images)
    llm = LLMClient(args.llm_url, cfg.planner.model)
    if not llm.ping():
        log.error("the planner vLLM (%s) is not responding", args.llm_url)
        return 2
    chat = lambda m, **kw: llm.chat_json(m, **kw)  # noqa: E731
    s = cfg.screen
    sub = (prescreen_subset(ds.fit_dev, s.prescreen_fraction, s.prescreen_min_images,
                            max_share=s.prescreen_max_share)
           if s.prescreen_fraction is not None else None)
    prescreen = {"on": bool(s.lost_value_prescreen), "of": len(ds.fit_dev.images),
                 "n_images": len(sub.images) if sub is not None else len(ds.fit_dev.images)}
    sat = cfg.saturation
    hist = _history(run_dir)
    rows = []
    for i, entry in enumerate(hist):
        r = entry.get("round", 0)
        if r == 0:
            continue
        if args.max_rounds and r > args.max_rounds:
            break
        past = hist[:i]
        champion = (past[-1].get("champion") or {}).get("axes") or {}
        diag = round_diagnosis(run_dir, r)
        budget = {"round": r, "rounds": args.rounds or cfg.rounds,
                  "stagnant": _trailing_stagnant(past, sat.epsilon),
                  "saturation_rounds": sat.rounds, "empty": _trailing_empty(past),
                  "empty_rounds": sat.empty_rounds}
        status = axis_status(past, args.retire_after)
        adopted = adopted_delta(entry, champion)
        row = {"round": r, "off": sorted(status.off), "adopted": adopted is not None}
        for name, retire in (("before", None), ("after", args.retire_after)):
            with tempfile.TemporaryDirectory() as tmp:
                planner = LLMPlanner(
                    chat=chat, artifact_dir=Path(tmp),
                    max_candidates=cfg.planner.max_candidates,
                    notes=cfg.dataset_notes, facts=facts,
                    disabled_axes=effective_disabled(cfg.search.disabled_axes, champion),
                    disabled_values=cfg.search.disabled_values,
                    tiling_overlap=cfg.search.tiling_overlap,
                    tiling_tp_quantile=cfg.search.tiling_tp_quantile,
                    tiling_tp_min=cfg.search.tiling_tp_min, retire_after=retire,
                    token_budget=_token_budget(cfg, llm.max_model_len()), budget=budget,
                    prescreen=prescreen, min_delta=cfg.gate.min_delta,
                    margin=cfg.gate.margin)
                prop = planner.propose(past, diag)
                art = _load(Path(tmp) / "proposal.json") or {}
            prompt = (art.get("prompt") or [{}])[-1].get("content", "")
            # a fallback round's candidates are deterministic remedies — not counted in the LLM metrics
            llm_cands = [c for c in prop["candidates"] if c.get("source") != "fallback"]
            row[name] = {**score(llm_cands, status, champion, adopted),
                         "invalid": sum(c.get("skipped") == "invalid_axes"
                                        for c in prop.get("dropped") or []),
                         "fallback": art.get("fallback"),
                         "prompt": len(prompt),
                         "axes": [c["axes"] for c in prop["candidates"]],
                         "dropped": [{k: c.get(k) for k in ("axes", "skipped", "note")}
                                     for c in prop.get("dropped") or []]}
        log.info("R%d off=%s before off_hit %d/%d · after off_hit %d/%d", r, row["off"],
                 row["before"]["off_hit"], row["before"]["n"],
                 row["after"]["off_hit"], row["after"]["n"])
        rows.append(row)

    def total(name, key):
        return sum(int(x[name][key]) for x in rows)
    summary = {name: {"candidates": total(name, "n"), "off_hit": total(name, "off_hit"),
                      "invalid": total(name, "invalid"),
                      "hit": total(name, "hit"),
                      "adopted_rounds": sum(x["adopted"] for x in rows),
                      "mean_prompt": round(sum(x[name]["prompt"] for x in rows)
                                           / max(len(rows), 1))}
               for name in ("before", "after")}
    out = {"run": str(run_dir), "retire_after": args.retire_after,
           "summary": summary, "rounds": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
