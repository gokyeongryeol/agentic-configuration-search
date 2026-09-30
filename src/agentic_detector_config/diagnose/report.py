"""Run trajectory report and A/B comparison.

Inputs are only the run directory's ledger (history.jsonl) and round
artifacts — no inference, so it runs without a GPU. The output is a single
markdown file kept in the run directory (the repo's only doc stays the README).

"test" in artifact names and report columns (diagnosis_test.json, the `test`
column) is the legacy name for the holdout split.
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


def _fmt_delta(delta: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(delta.items())) or "(same as current)"


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


def _num(v, *, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+.4f}" if signed else f"{v:.4f}"


def _round_diag(run_dir: Path, rnd: int, split: str) -> dict | None:
    p = Path(run_dir) / "rounds" / f"round_{rnd:03d}" / f"diagnosis_{split}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _error_trend_section(run_dir: Path, t: dict) -> list[str]:
    """Per-round FP/FN counts by cause and their changes (holdout aggregates).

    The final number alone does not say what shrank to produce the gain —
    which causes an adoption cut is input for designing the next experiment."""
    rows = []
    for r in t["rounds"]:
        d = _round_diag(run_dir, r["round"], "test")
        if d and d.get("summary"):
            counts = {k: v for side in ("fn", "fp")
                      for k, v in (d["summary"].get(side) or {}).items() if v}
            rows.append((r["round"], counts))
    if not rows:
        return []
    kinds = sorted({k for _, c in rows for k in c})
    out = ["", "## FP/FN causes over rounds (holdout)", "",
           "Each row is the champion's diagnosis **on entering** that round — the round 1",
           "row is the baseline, and the last row equals the final champion only if",
           "that round adopted nothing. Parentheses show the change from the previous row.", "",
           "| entering round | " + " | ".join(kinds) + " |",
           "|---" * (len(kinds) + 1) + "|"]
    prev: dict = {}
    for rnd, counts in rows:
        cells = []
        for k in kinds:
            now = counts.get(k, 0)
            if prev:
                diff = now - prev.get(k, 0)
                cells.append(f"{now} ({diff:+d})" if diff else f"{now} (—)")
            else:
                cells.append(str(now))
        out.append(f"| {rnd} | " + " | ".join(cells) + " |")
        prev = counts
    first, last = rows[0][1], rows[-1][1]
    moved = sorted(((last.get(k, 0) - first.get(k, 0), k) for k in kinds),
                   key=lambda kv: kv[0])
    if moved:
        best, worst = moved[0], moved[-1]
        out += ["", f"- most reduced cause: **{best[1]}** ({best[0]:+d})",
                f"- most increased cause: **{worst[1]}** ({worst[0]:+d})"]
    return out


# Terms that, when recurring in notes, hint at collection conditions — capture
# and subject conditions only
_CONDITION_TERMS = (
    "blurry", "blur", "motion", "small", "tiny", "distant", "far", "occluded",
    "occlusion", "overlap", "dark", "shadow", "bright", "glare", "reflection",
    "low-resolution", "low resolution", "cluttered", "partial", "cropped",
    "angle", "viewpoint", "similar", "look-alike",
)


def _note_terms(run_dir: Path, rnd: int) -> list[tuple[str, int]]:
    """Condition-term frequencies in the last round's crop notes — clues to collection conditions."""
    p = Path(run_dir) / "rounds" / f"round_{rnd:03d}" / "crop_notes.json"
    if not p.exists():
        return []
    notes = json.loads(p.read_text()).get("notes") or []
    text = " ".join(str(n.get("note", "")).lower() for n in notes)
    hits = [(term, text.count(term)) for term in _CONDITION_TERMS]
    return sorted(((t, n) for t, n in hits if n), key=lambda kv: -kv[1])[:8]


def _sourcing_section(run_dir: Path, t: dict) -> list[str]:
    """Data-sourcing insights — states **only what the artifacts can back up**.

    Three angles: per-class residual miss_rate and GT count (few GT + high miss
    suggests missing data), the residual cause distribution at saturation
    (missed-dominant → data; below_threshold-dominant → calibration), and
    recurring condition terms in notes (collection conditions).

    It does not say how many more images to collect — no learning curve is
    measured, so the current artifacts cannot estimate it.
    """
    last = max((r["round"] for r in t["rounds"]), default=0)
    test, train = _round_diag(run_dir, last, "test"), _round_diag(run_dir, last, "train")
    if not test:
        return []
    out = ["", "## Additional data sourcing insights", "",
           "The following is read from the **structure of residual errors**; it does not",
           "estimate how much data is needed (no learning curve is measured, so these artifacts cannot tell).", ""]
    last_adopted = next((r["adopted"] for r in t["rounds"] if r["round"] == last),
                        False)
    if last_adopted:
        out += [f"> Note: round {last} adopted a candidate, so the diagnosis below is of the "
                "champion **just before that adoption** — residual errors of the final "
                "champion are not recorded.", ""]

    per_class = (test.get("per_class") or {})
    if per_class:
        out += ["| class | holdout GT | miss_rate | top residual FN | reading |",
                "|---|---|---|---|---|"]
        for name, s in sorted(per_class.items(),
                              key=lambda kv: -(kv[1].get("miss_rate") or 0)):
            gt, mr = s.get("gt", 0), s.get("miss_rate") or 0.0
            fn = sorted((s.get("fn") or {}).items(), key=lambda kv: -kv[1])
            top = ", ".join(f"{k} {v}" for k, v in fn[:2]) or "—"
            if mr >= 0.5 and gt < 100:
                read = "few GT and many misses — priority for collection"
            elif mr >= 0.5:
                read = "enough GT but many misses — likely a method problem"
            elif mr < 0.2:
                read = "stable"
            else:
                read = "moderate"
            out.append(f"| {name} | {gt} | {mr:.3f} | {top} | {read} |")
        out.append("")

    counts = {k: v for side in ("fn", "fp")
              for k, v in (test.get("summary") or {}).get(side, {}).items() if v}
    if counts:
        dom = max(counts, key=counts.get)
        reading = {
            "missed": "no detection candidates at all — beyond reach of prompts/detector, so "
                      "collecting data for these conditions is the most direct fix.",
            "missed_small": "small objects are missed — needs higher-resolution or close-up captures, "
                            "or a human should tune the tiling settings.",
            "below_threshold": "detected but scored low — closer to a calibration/threshold "
                               "problem than a data problem.",
            "class_confusion": "classes are confused — needs examples that separate the confused pairs.",
            "background": "background is detected — needs examples of the backgrounds causing false positives.",
            "localization": "boxes are misplaced — check label quality (box alignment) first.",
        }.get(dom)
        out += [f"- **top residual error: {dom} ({counts[dom]})** — "
                + (reading or "no entry for it in the remedy mapping."), ""]

    terms = _note_terms(run_dir, last)
    if terms:
        out += ["- **conditions recurring in notes**: "
                + ", ".join(f"`{t_}` {n}" for t_, n in terms),
                "  (from the last round's crop notes — clues for narrowing collection conditions.)", ""]
    elif train:
        out += ["- no crop notes, so condition terms are left empty (the planner is deterministic "
                "or the description call failed this round).", ""]
    return out


def render_report(run_dir: Path, *, title: str | None = None) -> str:
    """Markdown report for one run — also written to `report.md`."""
    run_dir = Path(run_dir)
    t = trajectory(run_dir)
    name = title or run_dir.name
    lines = [f"# {name} — setting search report", ""]
    ch = t["champion"]
    lines += [
        f"- **final holdout micro F(β)**: {t['final_micro_f_beta']:.4f} "
        f"(baseline {t['baseline_micro_f_beta']:.4f}, Δ{t['gain']:+.4f})",
        f"- **rounds adopted**: {t['n_adopted']} / {max(r['round'] for r in t['rounds'])}"
        if t["rounds"] else "- (no rounds)",
    ]
    if ch:
        axes_on = {k: v for k, v in ch.get("axes", {}).items()
                   if v not in (False, None, "none")}
        lines += [f"- **champion**: round {ch.get('round')} — axes on "
                  f"{_fmt_delta(axes_on) if axes_on else 'none (baseline)'}"]
        if ch.get("weights"):
            lines += [f"- tail arm weights: `{ch['weights']}`"]
        if ch.get("rescorer_path"):
            lines += [f"- rescorer: `{ch['rescorer_path']}`"]
    lines += ["", "## Round trajectory", "",
              "Candidates are listed by axis delta vs. the champion (candidate names are",
              "free-form strings from the planner and cannot be interpreted).", "",
              "| round | champion micro F(β) | adopted delta |",
              "|---|---|---|"]
    for r in t["rounds"]:
        if r["round"] == 0:
            adopted = "baseline"
        elif r["adopted"]:
            adopted = f"✅ {_fmt_delta(r['adopted_delta'])}"
        else:
            adopted = "—"
        lines.append(f"| {r['round']} | {r['champion_micro_f_beta']:.4f} | {adopted} |")

    lines += ["", "### Candidates tried", "",
              "| round | axis delta | fit_dev | holdout | gate Δ | gate CI lower | result |",
              "|---|---|---|---|---|---|---|"]
    for r in t["rounds"]:
        for c in r["candidates"]:
            if c["skipped"]:
                result = f"skip: {c['skipped']}"
            elif c["adopted"]:
                result = "adopted"
            else:
                result = "rejected"
            lines.append(
                f"| {r['round']} | {_fmt_delta(c['delta'])} | {_num(c['fit_dev'])} "
                f"| {_num(c['test'])} | {_num(c['gate_delta'], signed=True)} "
                f"| {_num(c['gate_ci_lo'], signed=True)} | {result} |")

    lines += _error_trend_section(run_dir, t)
    lines += _sourcing_section(run_dir, t)
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text)
    return text


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
