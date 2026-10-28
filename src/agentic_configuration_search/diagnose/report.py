"""Run ledger readers — trajectory, candidate attempts, the final diagnosis
bundle, A/B comparison and the retrospective hit rate.

Inputs are only the run directory's ledger (history.jsonl) and round
artifacts — no inference, so it runs without a GPU. The run report (report.md)
is drawn from these by diagnose.story.

"test" in artifact names (diagnosis_test.json) is the legacy name for the
holdout split.
"""

from __future__ import annotations

import json
from pathlib import Path


def _history(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "history.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — not a completed run directory")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _axes_delta(axes: dict, base: dict) -> dict:
    """Axis delta vs. the champion — a candidate is identified by its delta, not its name."""
    return {k: v for k, v in (axes or {}).items() if (base or {}).get(k) != v}


def trajectory(run_dir: Path) -> dict:
    """Per-round champion trajectory and adoption history — shared input for
    the report and A/B.

    Candidates are recorded **by axis delta, not by name**: names are
    free-form strings from the planner and cannot be interpreted in a report.
    """
    hist = _history(run_dir)
    rounds = []
    prev_axes: dict = {}
    for e in hist:
        d = e.get("decision", {})
        evaluated = d.get("evaluated") or {}
        cands = []
        for c in e.get("candidates", []):
            ev = evaluated.get(c.get("id")) or {}
            gate = ev.get("gate") or {}
            cands.append({
                "delta": _axes_delta(c.get("axes"), prev_axes),
                "adopted": bool(c.get("adopted")),
                "skipped": c.get("skipped"),
                "fit_dev": c.get("fit_dev_micro_f_beta"),
                "test": ev.get("micro_f_beta"),
                "gate_delta": gate.get("delta"),
                "gate_ci_lo": gate.get("lo"),
            })
        rounds.append({
            "round": e["round"],
            "champion_micro_f_beta": e["champion"]["micro_f_beta"],
            "champion_id": e["champion"]["id"],
            "champion_axes": e["champion"].get("axes") or {},
            "adopted": bool(d.get("adopted")),
            "adopted_id": d.get("id"),
            "adopted_delta": next((c["delta"] for c in cands if c["adopted"]), {}),
            "candidates": cands,
            "skipped": d.get("skipped", []),
        })
        prev_axes = e["champion"].get("axes") or {}
    first = rounds[0]["champion_micro_f_beta"] if rounds else 0.0
    last = rounds[-1]["champion_micro_f_beta"] if rounds else 0.0
    champion_meta = {}
    cj = Path(run_dir) / "champion.json"
    if cj.exists():
        champion_meta = json.loads(cj.read_text())
    return {"rounds": rounds, "baseline_micro_f_beta": first,
            "final_micro_f_beta": last, "gain": last - first,
            "n_adopted": sum(1 for r in rounds if r["adopted"] and r["round"] > 0),
            "champion": champion_meta}


# --------------------------------------------------------------- candidate attempts

# count keys roll up to their axis — a candidate that only changes exemplar_base_k is
# an exemplar attempt too (diagnose.story)
_AXIS_PARENT = {"exemplar_base_k": "exemplar", "exemplar_fn_k": "exemplar",
                "visual_negative_k": "visual_negatives",
                "text_negative_k": "text_negatives"}


_OFF = (False, None, "none")


def _outcome(c: dict, evaluated: dict) -> str:
    """One candidate row's outcome — adopted, holdout gate, rejected before
    holdout, or outside screen's top K."""
    if c.get("adopted"):
        return "adopted"
    if c.get("skipped"):
        return str(c["skipped"])
    ev = evaluated.get(c.get("id"))
    if ev is not None:
        gate = ev.get("gate") or {}
        return ("gate_passed_but_lost"
                if c.get("gate_passed_but_lost") or gate.get("adopted")
                else "gate_failed")
    return "screen_top_k"


def attempts(hist: list[dict]) -> list[dict]:
    """Candidate attempts per round — increments against the champion at the
    **start of the round** (by requested axes).

    A candidate left as two rows (lora/classifier arms: a screen row and a
    rejection row) is merged into the rejection row — so the outcome is not
    counted twice."""
    out = []
    for i, e in enumerate(hist):
        if e.get("round", 0) == 0 or i == 0:
            continue
        start = hist[i - 1].get("champion") or {}
        base = start.get("axes") or {}
        d = e.get("decision") or {}
        evaluated = d.get("evaluated") or {}
        rows: dict = {}
        for c in e.get("candidates", []):
            cid = c.get("id")
            if cid in rows and not c.get("skipped"):
                continue
            rows[cid] = c
        for c in rows.values():
            out.append({
                "round": e["round"], "id": c.get("id"),
                "delta": _axes_delta(c.get("requested_axes") or c.get("axes"), base),
                "outcome": _outcome(c, evaluated)})
    return out


def _load_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _final_is_current(final: Path, t: dict) -> bool:
    """Whether rounds/final is the current champion's diagnosis — read from the
    champion id in the diagnosis key (diagnosis_key.json; the id holds the
    round). The files existing is not enough: if the new champion's final
    diagnosis fails after a resume, the old champion's files are left behind."""
    key = _load_json(final / "diagnosis_key.json") or {}
    champ = key.get("champion") or {}
    last = t["rounds"][-1] if t["rounds"] else {}
    return bool(champ) and champ.get("id") == last.get("champion_id") \
        and (final / "diagnosis_test.json").exists()


def _final_diag(run_dir: Path, t: dict) -> dict | None:
    """The final champion's diagnosis bundle — the current champion's
    rounds/final if present, else the last round's diagnosis. stale = the
    diagnosis used instead is not the final champion's."""
    run_dir = Path(run_dir)
    final = run_dir / "rounds" / "final"
    last = max((r["round"] for r in t["rounds"]), default=0)
    if _final_is_current(final, t):
        d, source, stale = final, "final", False
    else:
        d = run_dir / "rounds" / f"round_{last:03d}"
        source = "round"
        stale = next((r["adopted"] for r in t["rounds"] if r["round"] == last), False) \
            and last > 0
    test = _load_json(d / "diagnosis_test.json")
    train = _load_json(d / "diagnosis_train.json")
    if not test and not train:
        return None
    notes = (_load_json(d / "crop_notes.json") or {}).get("notes") or []
    # a run whose final diagnosis failed may hold an old champion's residual.json — not used
    failed = ((_load_json(run_dir / "run_summary.json") or {})
              .get("final_diagnosis") or {}).get("error")
    return {"source": source, "round": last, "stale": stale,
            "dir": d.relative_to(run_dir).as_posix(), "train": train or {},
            "test": test or {},
            "residual": None if failed else _load_json(d / "residual.json"),
            "notes": [n for n in notes if isinstance(n, dict)]}


def _note_key(image, box) -> tuple:
    return (image, tuple(round(float(v)) for v in box or ()))


def ab_compare(run_a: Path, run_b: Path, *, label_a: str = "A",
               label_b: str = "B") -> dict:
    """Compare the trajectories of two runs (e.g. deterministic vs. llm planner).

    Assumes both runs used the same dataset and seed. The verdict is the final
    micro plus a side-by-side per-round table — no statistical test (a single
    pair of runs gives no sample).
    """
    ta, tb = trajectory(run_a), trajectory(run_b)
    rounds = []
    for i in range(max(len(ta["rounds"]), len(tb["rounds"]))):
        ra = ta["rounds"][i] if i < len(ta["rounds"]) else None
        rb = tb["rounds"][i] if i < len(tb["rounds"]) else None
        rounds.append({
            "round": i,
            label_a: ra["champion_micro_f_beta"] if ra else None,
            label_b: rb["champion_micro_f_beta"] if rb else None,
        })
    return {"rounds": rounds,
            label_a: {"final": ta["final_micro_f_beta"], "gain": ta["gain"],
                      "n_adopted": ta["n_adopted"]},
            label_b: {"final": tb["final_micro_f_beta"], "gain": tb["gain"],
                      "n_adopted": tb["n_adopted"]},
            "winner": (label_a if ta["final_micro_f_beta"] > tb["final_micro_f_beta"]
                       else label_b if tb["final_micro_f_beta"] > ta["final_micro_f_beta"]
                       else "tie")}


def retrospective_hit_rate(run_dir: Path, propose_fn) -> dict:
    """Retrospective planner hit rate — replays an existing run's diagnoses
    without a GPU and measures how often propose_fn(history, diagnosis)
    proposes the candidate that was actually adopted.

    "Hit" = the axes of the candidate that passed the gate that round appear in
    the proposal list. Rounds without an adoption are excluded from the
    denominator (no ground truth).
    """
    run_dir = Path(run_dir)
    hist = _history(run_dir)
    hits, total, detail = 0, 0, []
    for e in hist:
        r = e["round"]
        if r == 0 or not e["decision"].get("adopted"):
            continue
        adopted_axes = next((c["axes"] for c in e.get("candidates", [])
                             if c.get("adopted")), None)
        if adopted_axes is None:
            continue
        dl = run_dir / "rounds" / f"round_{r:03d}" / "diagnosis_train.json"
        dt = run_dir / "rounds" / f"round_{r:03d}" / "diagnosis_test.json"
        diagnosis = {}
        if dl.exists():
            diagnosis["train"] = json.loads(dl.read_text())
        if dt.exists():
            diagnosis["test"] = json.loads(dt.read_text())
        prior = [h for h in hist if h["round"] < r]
        proposal = propose_fn(prior, diagnosis)
        proposed = [tuple(sorted(c["axes"].items()))
                    for c in proposal.get("candidates", [])]
        hit = tuple(sorted(adopted_axes.items())) in proposed
        hits += int(hit)
        total += 1
        detail.append({"round": r, "hit": hit,
                       "adopted_axes": adopted_axes,
                       "n_proposed": len(proposed)})
    return {"hits": hits, "total": total,
            "hit_rate": (hits / total) if total else None, "detail": detail}
