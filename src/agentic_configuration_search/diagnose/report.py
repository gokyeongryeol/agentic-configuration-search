"""Run trajectory report and A/B comparison.

Inputs are only the run directory's ledger (history.jsonl) and round
artifacts — no inference, so it runs without a GPU. The output is a single
markdown file kept in the run directory (the repo's only doc stays the README).

"test" in artifact names and report columns (diagnosis_test.json, the `test`
column) is the legacy name for the holdout split.
"""

from __future__ import annotations

import json
from collections import Counter
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
_AXIS_PARENT = {"exemplar_k": "exemplar", "fn_k": "exemplar",
                "visual_negative_k": "visual_negatives",
                "text_negative_k": "text_negatives"}


_OFF = (False, None, "none")


def _axis_label(delta: dict) -> str:
    """The axes an increment turns on or changes (count keys roll up to their
    axis). An increment that only turns axes off is "off:axis".

    Older runs have candidates written from the baseline rather than on top of
    the champion, so their increment mixes in turning champion axes off —
    putting those into the axis name too would scatter one attempt across
    combinations."""
    on = {_AXIS_PARENT.get(k, k) for k, v in delta.items() if v not in _OFF}
    off = {_AXIS_PARENT.get(k, k) for k in delta} - on
    if on:
        return "+".join(sorted(on))
    return ("off:" + "+".join(sorted(off))) if off else "(none)"


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
    rejection row) is merged into the rejection row — so outcome and wall clock
    are not counted twice."""
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
            delta = _axes_delta(c.get("requested_axes") or c.get("axes"), base)
            gate = (evaluated.get(c.get("id")) or {}).get("gate") or {}
            out.append({
                "round": e["round"], "champ_round": start.get("round"),
                "champ_f": start.get("micro_f_beta"), "id": c.get("id"),
                "delta": delta, "key": tuple(sorted(
                    (k, json.dumps(v)) for k, v in delta.items())),
                "axis": _axis_label(delta), "outcome": _outcome(c, evaluated),
                "gate_delta": gate.get("delta"), "gate_lo": gate.get("lo"),
                "wall": c.get("wall_min") or {}, "reask": c.get("reask"),
                "carry": (str(c.get("id", "")).startswith("carry_")
                          or c.get("source") == "carryover"),
                "unused": c.get("unused_axes") or []})
    return out


def _why(a: dict) -> str:
    """The outcome in a word — gate outcomes get Δ and the CI lower bound."""
    if a["outcome"] in ("gate_failed", "gate_passed_but_lost", "adopted") \
            and a["gate_delta"] is not None:
        return (f"{a['outcome']} (Δ{_num(a['gate_delta'], signed=True)}, "
                f"CI lower {_num(a['gate_lo'], signed=True)})")
    return a["outcome"]


def _counts(items) -> str:
    c = Counter(items)
    return ", ".join(f"{k} {n}" for k, n in c.most_common()) or "—"


def _adoption_section(run_dir: Path, t: dict, hist: list[dict]) -> list[str]:
    """Adoption path — re-adoptions, per-axis contribution, carry-over results.

    A re-adoption is an increment that came back after the champion changed and
    was adopted. A champion change releases the folded axes and the tried
    combinations; whether that pays off is told by these re-adoption versus
    re-rejection counts."""
    out = ["", "## Adoption path", ""]
    att = attempts(hist)
    if not att:
        return out + ["_No candidate attempts recorded — this section is skipped._"]
    by_key: dict = {}
    re_adopt, re_reject = [], []
    for a in att:
        if not a["key"]:
            continue
        prior = [p for p in by_key.get(a["key"], [])
                 if p["champ_round"] != a["champ_round"]]
        if prior and a["outcome"] == "adopted":
            re_adopt.append((a, prior))
        elif prior:
            re_reject.append(a)
        by_key.setdefault(a["key"], []).append(a)
    changes = sum(1 for r in t["rounds"] if r["round"] > 0 and r["adopted"])
    out += [f"{changes} champion change(s). Re-proposals of an **increment already tried "
            f"under an earlier champion**: {len(re_adopt) + len(re_reject)} → adopted "
            f"{len(re_adopt)}, rejected again {len(re_reject)}"
            + (f" ({_counts(a['outcome'] for a in re_reject)})" if re_reject else "")
            + ".", ""]
    if re_adopt:
        out += ["### Re-adopted — increments that failed under an earlier champion", "",
                "| adopted | increment | adopted Δ | earlier attempts (champion → outcome) |",
                "|---|---|---|---|"]
        for a, prior in re_adopt:
            was = "; ".join(f"R{p['round']} (champion R{p['champ_round']}) {_why(p)}"
                            for p in prior)
            out.append(f"| R{a['round']} (champion R{a['champ_round']})"
                       f"{' carried' if a['carry'] else ''} "
                       f"| {_fmt_delta(a['delta'])} "
                       f"| {_num(a['gate_delta'], signed=True)} | {was} |")
        out.append("")

    adopted = [a for a in att if a["outcome"] == "adopted"]
    if adopted:
        f_after = {r["round"]: r["champion_micro_f_beta"] for r in t["rounds"]}
        out += ["### Holdout contribution of adopted increments", "",
                "| round | increment | holdout Δ | CI lower | F after adoption |",
                "|---|---|---|---|---|"]
        for a in adopted:
            out.append(f"| {a['round']} | {_fmt_delta(a['delta'])} "
                       f"| {_num(a['gate_delta'], signed=True)} "
                       f"| {_num(a['gate_lo'], signed=True)} "
                       f"| {_num(f_after.get(a['round']))} |")
        out.append("")

    groups: dict = {}
    for a in att:
        if a["key"]:
            groups.setdefault(a["axis"], []).append(a)
    if groups:
        out += ["### Results by axis", "",
                "The axis is what the increment turned on or changed (count keys count "
                "toward their axis; axes turned off alongside are left out). Outcomes: "
                "gate_failed = missed the holdout gate, gate_passed_but_lost = passed "
                "but lost to a winner with a larger Δ, screen_top_k = outside screen's "
                "top K, anything else = rejected before holdout for that reason.", "",
                "| axis | tried | adopted | adopted Δ sum | outcomes |", "|---|---|---|---|---|"]
        for axis, rows in sorted(groups.items(),
                                 key=lambda kv: (-sum(a["outcome"] == "adopted"
                                                      for a in kv[1]), kv[0])):
            gain = sum(a["gate_delta"] or 0.0 for a in rows if a["outcome"] == "adopted")
            n_ad = sum(a["outcome"] == "adopted" for a in rows)
            out.append(f"| {axis} | {len(rows)} | {n_ad} "
                       f"| {_num(gain, signed=True) if n_ad else '—'} "
                       f"| {_counts(a['outcome'] for a in rows)} |")
        never = [axis for axis, rows in groups.items()
                 if not any(a["outcome"] == "adopted" for a in rows)]
        if never:
            out += ["", "- Axes never adopted: " + ", ".join(
                f"**{axis}** ({_counts(a['outcome'] for a in groups[axis])})"
                for axis in sorted(never))]
        out.append("")

    carry = [a for a in att if a["carry"]]
    if carry:
        out += ["### Carry-over results", "",
                "Candidates that passed the gate in the previous round but lost, re-run "
                f"on top of the new champion — {len(carry)}, of which adopted "
                f"{sum(a['outcome'] == 'adopted' for a in carry)}.", ""]
        for a in carry:
            res = "adopted" if a["outcome"] == "adopted" else f"rejected — {_why(a)}"
            out.append(f"- R{a['round']} `{a['id']}` {_fmt_delta(a['delta'])}: {res}")
        out.append("")
    return out


# --------------------------------------------------------------- time

_STAGES = ("materialize", "screen", "holdout")


def _time_section(run_dir: Path, t: dict, hist: list[dict]) -> list[str]:
    """Time use — written only from the keys recorded (dur_s, wall_min,
    pass_cache, reask)."""
    out = ["", "## Time use", ""]
    att = attempts(hist)
    durs = {e["round"]: e["dur_s"] for e in hist if e.get("dur_s") is not None}
    walled = [a for a in att if a["wall"]]
    caches = [(e["round"], e["pass_cache"]) for e in hist
              if isinstance(e.get("pass_cache"), dict)]
    summary = _load_json(Path(run_dir) / "run_summary.json") or {}
    final = summary.get("final_diagnosis") or {}
    if not (durs or walled or caches or final):
        return out + ["_This run predates the time keys (dur_s, wall_min, pass_cache) — "
                      "this section is skipped._"]
    if durs:
        total = sum(durs.values()) / 60
        out += [f"- **run wall clock**: {total:.1f} min ({total / 60:.2f} h) — per round (min) "
                + " · ".join(f"R{r} {s / 60:.1f}" for r, s in sorted(durs.items()))]
    if walled:
        stage = {k: sum(float(a["wall"].get(k) or 0.0) for a in walled) for k in _STAGES}
        cand_total = sum(float(a["wall"].get("total") or 0.0) for a in walled)
        out += ["", "| stage | min | share |", "|---|---|---|"]
        for k in _STAGES:
            out.append(f"| {k} | {stage[k]:.1f} | {stage[k] / cand_total:.0%} |"
                       if cand_total else f"| {k} | {stage[k]:.1f} | — |")
        out.append(f"| candidates total | {cand_total:.1f} | |")
        rest = sum(s for r, s in durs.items() if r > 0) / 60 - cand_total
        if durs and rest > 0:
            out.append(f"| outside candidates (diagnosis, notes, planner, decision) "
                       f"| {rest:.1f} | |")
        by_out = Counter()
        for a in walled:
            o = a["outcome"]
            bucket = ("adopted" if o == "adopted" else
                      "rejected on holdout" if o in ("gate_failed", "gate_passed_but_lost")
                      else "outside screen top K" if o == "screen_top_k"
                      else "rejected before holdout")
            by_out[bucket] += float(a["wall"].get("total") or 0.0)
        out += ["", "- Minutes by candidate outcome: " + " · ".join(
            f"{k} {v:.1f}" for k, v in by_out.most_common())]
        groups: dict = {}
        for a in walled:
            groups.setdefault(a["axis"], []).append(a)
        out += ["", "| axis | tried | min | adopted Δ sum | adopted Δ / hour |",
                "|---|---|---|---|---|"]
        for axis, rows in sorted(groups.items(), key=lambda kv: -sum(
                float(a["wall"].get("total") or 0.0) for a in kv[1])):
            mins = sum(float(a["wall"].get("total") or 0.0) for a in rows)
            gain = sum(a["gate_delta"] or 0.0 for a in rows if a["outcome"] == "adopted")
            rate = f"{gain / (mins / 60):+.3f}" if gain and mins > 0 else "—"
            out.append(f"| {axis} | {len(rows)} | {mins:.1f} "
                       f"| {_num(gain, signed=True) if gain else '—'} | {rate} |")
    if caches:
        hits = sum(c.get("hits", 0) for _, c in caches)
        misses = sum(c.get("misses", 0) for _, c in caches)
        rate = hits / (hits + misses) if hits + misses else 0.0
        out += ["", f"- **pass cache**: hits {hits} · misses {misses} (hit rate {rate:.1%}), "
                f"{sum(c.get('invalidations', 0) for _, c in caches)} invalidations — "
                "per round " + " · ".join(
                    f"R{r} {c.get('hit_rate', 0.0):.0%}" for r, c in caches)]
    reasks = {e["round"]: (e.get("decision") or {}).get("reasks")
              for e in hist if (e.get("decision") or {}).get("reasks")}
    asked = [a for a in att if a["reask"]]
    if reasks or asked:
        out += [f"- **re-asks**: {sum(reasks.values())} ("
                + ", ".join(f"R{r} {n}" for r, n in sorted(reasks.items()))
                + f") — of {len(asked)} re-asked candidates, adopted "
                f"{sum(a['outcome'] == 'adopted' for a in asked)}"]
    empty = [e["round"] for e in hist if e.get("round", 0) > 0
             and not (e.get("decision") or {}).get("evaluated")]
    out += [f"- **empty rounds** (nothing measured on holdout): {len(empty)}"
            + (" — " + ", ".join(f"R{r}" for r in empty) if empty else "")
            + (f"; stop reason `{summary['stopped']}`" if summary.get("stopped") else "")]
    if final:
        if final.get("error"):
            out.append(f"- **final champion diagnosis**: failed ({final['error']})")
        else:
            how = ("reused the last round's diagnosis (no inference)" if final.get("reused")
                   else "computed fresh (one train inference)")
            out.append(f"- **final champion diagnosis**: {how}, "
                       f"{final.get('dur_s', 0) / 60:.1f} min")
    return out


# --------------------------------------------------------------- residual misses

# terms that, when repeated in notes, suggest collection conditions — only
# capture and subject conditions
_CONDITION_TERMS = (
    "blurry", "blur", "motion", "small", "tiny", "distant", "far", "occluded",
    "occlusion", "overlap", "dark", "shadow", "bright", "glare", "reflection",
    "low-resolution", "low resolution", "cluttered", "partial", "cropped",
    "angle", "viewpoint", "similar", "look-alike",
)
_FN_KINDS = ("missed", "missed_small", "below_threshold")
SAMPLES_PER_CLASS = 5     # cap on representative miss samples (per class)
NOTE_MAX = 140            # cap on the note length in the sample table
# thresholds of the data-collection suggestions — so a weakly supported bucket
# is not read as dominant
SKEW = 1.5                # a bucket's miss rate at this multiple of the class's overall one is dominant
MIN_BUCKET_GT = 5         # buckets with fewer GT than this are not read as rates
LOC_SHARE = 0.3           # a localization share of FPs at or above this suggests checking box alignment
LOC_MIN = 10


def _load_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _terms(notes: list[dict]) -> list[tuple[str, int]]:
    """Number of notes per condition term — counted once per note. A short term
    contained in a longer one (blur in blurry) is not counted in the same note."""
    c = Counter()
    for n in notes:
        text = str(n.get("note", "")).lower()
        hit = {t_ for t_ in _CONDITION_TERMS if t_ in text}
        hit = {t_ for t_ in hit if not any(t_ != o and t_ in o for o in hit)}
        c.update(hit)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def _final_diag(run_dir: Path, t: dict) -> dict | None:
    """The final champion's diagnosis bundle — rounds/final if present, else
    (older runs) the last round's diagnosis. stale = the diagnosis used instead
    is not the final champion's."""
    run_dir = Path(run_dir)
    final = run_dir / "rounds" / "final"
    last = max((r["round"] for r in t["rounds"]), default=0)
    if (final / "diagnosis_test.json").exists():
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
    return {"source": source, "round": last, "stale": stale,
            "dir": d.relative_to(run_dir).as_posix(), "train": train or {},
            "test": test or {}, "residual": _load_json(d / "residual.json"),
            "notes": [n for n in notes if isinstance(n, dict)]}


def _rate(pair) -> str:
    gt, fn = pair
    return f"{fn}/{gt} ({fn / gt:.0%})" if gt else (f"{fn}/0" if fn else "—")


def _px_ranges(small_ratio: float, side: int) -> dict:
    lo, hi = small_ratio * side, 3 * small_ratio * side
    return {"small": f"<{lo:.0f}px", "medium": f"{lo:.0f}–{hi:.0f}px",
            "large": f"≥{hi:.0f}px"}


def _fn_rows(fd: dict) -> list[dict]:
    """train FN detail — residual (with facts attached) if present, else the
    diagnosis rows."""
    res = fd.get("residual") or {}
    rows = (res.get("train") or {}).get("fn")
    return rows if rows is not None else (fd["train"].get("fn") or [])


def _row_size(r: dict, small_ratio: float) -> str | None:
    if r.get("size"):
        return r["size"]
    if r.get("size_ratio") is None:
        return None
    from .residual import size_bucket
    return size_bucket(float(r["size_ratio"]), small_ratio)


def _residual_section(run_dir: Path, t: dict, fd: dict | None) -> list[str]:
    """What the final champion still misses — cause, size, crowding, border,
    conditions, samples."""
    out = ["", "## Residual misses", ""]
    if fd is None:
        return out + ["_No diagnosis artifacts (rounds/*/diagnosis_*.json) — this section "
                      "is skipped._"]
    from .residual import INPUT_SIDE
    if fd["source"] == "final":
        out += [f"The final champion's diagnosis (`{fd['dir']}`). Train is detailed; "
                "holdout is aggregates only.", ""]
    elif fd["stale"]:
        out += [f"> Note: this run has no final champion diagnosis (rounds/final), so "
                f"round {fd['round']}'s diagnosis stands in. Round {fd['round']} adopted "
                "a candidate, so this is the champion **just before that adoption** — "
                "residual errors of the final champion are not recorded.", ""]
    else:
        out += [f"The diagnosis of round {fd['round']} (the run has no rounds/final) — "
                "that round adopted nothing, so it equals the final champion's diagnosis.",
                ""]
    train, test = fd["train"], fd["test"]
    res = fd.get("residual") or {}
    small_ratio = res.get("small_ratio") or 0.04
    side = res.get("input_side") or INPUT_SIDE

    names = sorted(set(train.get("per_class") or {}) | set(test.get("per_class") or {}))
    if names:
        out += ["| class | train FN/GT | holdout FN/GT | holdout residual FN causes |",
                "|---|---|---|---|"]
        for n in sorted(names, key=lambda n: -((test.get("per_class") or {})
                                               .get(n, {}).get("miss_rate") or 0)):
            tr = (train.get("per_class") or {}).get(n) or {}
            te = (test.get("per_class") or {}).get(n) or {}
            fn_tr = sum((tr.get("fn") or {}).values())
            fn_te = sum((te.get("fn") or {}).values())
            kinds = ", ".join(f"{k} {v}" for k, v in sorted(
                (te.get("fn") or {}).items(), key=lambda kv: -kv[1]) if v) or "—"
            out.append(f"| {n} | {_rate((tr.get('gt', 0), fn_tr))} "
                       f"| {_rate((te.get('gt', 0), fn_te))} | {kinds} |")
        out.append("")

    counts = {k: v for side_ in ("fn", "fp")
              for k, v in ((test or train).get("summary") or {}).get(side_, {}).items()
              if v}
    if counts:
        dom = max(counts, key=counts.get)
        reading = {
            "missed": "no detection candidates at all — beyond reach of prompts/detector, so "
                      "collecting data for these conditions is the most direct fix.",
            "missed_small": "small objects are missed — needs higher-resolution or close-up captures, "
                            "or, if the tiling grid has not reached its cap (6), raising it further.",
            "below_threshold": "detected but scored low — closer to a calibration/threshold "
                               "problem than a data problem.",
            "class_confusion": "classes are confused — needs examples that separate the confused pairs.",
            "background": "background is detected — needs examples of the backgrounds causing false positives.",
            "localization": "boxes are misplaced — check label quality (box alignment) first.",
        }.get(dom)
        out += [f"- **top residual error: {dom} ({counts[dom]}, "
                f"{'holdout' if test else 'train'})** — "
                + (reading or "no entry for it in the remedy mapping."), ""]

    px = _px_ranges(small_ratio, side)
    rtr = (res.get("train") or {}).get("per_class") or {}
    rte = (res.get("holdout") or {}).get("per_class") or {}
    if rtr or rte:
        out += [f"### Size buckets (at a {side}px detector input side: small {px['small']}, "
                f"medium {px['medium']}, large {px['large']})", "",
                "Cells are FN/GT (miss rate).", "",
                "| class | split | small | medium | large |", "|---|---|---|---|---|"]
        for n in names:
            for label, src in (("train", rtr), ("holdout", rte)):
                f = src.get(n)
                if f and f.get("gt"):
                    out.append(f"| {n} | {label} | " + " | ".join(
                        _rate(f["size"][b]) for b in ("small", "medium", "large")) + " |")
        out += ["", "### Cause × size (train FN counts)", "",
                "| class | cause | small | medium | large |", "|---|---|---|---|---|"]
        for n in names:
            for kind, b in sorted(((rtr.get(n) or {}).get("kind_size") or {}).items()):
                out.append(f"| {n} | {kind} | {b.get('small', 0)} | {b.get('medium', 0)} "
                           f"| {b.get('large', 0)} |")
        crowd_names = [name for _, name in _crowd_buckets()]
        out += ["", "### Crowding (GT per image) — FN/GT", "",
                "| class | split | " + " | ".join(crowd_names) + " |",
                "|---" * (len(crowd_names) + 2) + "|"]
        for n in names:
            for label, src in (("train", rtr), ("holdout", rte)):
                f = src.get(n)
                if f and f.get("gt"):
                    out.append(f"| {n} | {label} | " + " | ".join(
                        _rate(f["crowd"].get(b, [0, 0])) for b in crowd_names) + " |")
        out += ["", "### Border and aspect — FN/GT", "",
                "border = the box touches within 1% of an image side (an object cut by "
                "the frame, likely), elongated = long side / short side ≥ 3.", "",
                "| class | split | touches border | interior | elongated | other |",
                "|---|---|---|---|---|---|"]
        for n in names:
            for label, src in (("train", rtr), ("holdout", rte)):
                f = src.get(n)
                if f and f.get("gt"):
                    inner = [f["gt"] - f["border"][0], f["fn"] - f["border"][1]]
                    plain = [f["gt"] - f["elongated"][0], f["fn"] - f["elongated"][1]]
                    out.append(f"| {n} | {label} | {_rate(f['border'])} | {_rate(inner)} "
                               f"| {_rate(f['elongated'])} | {_rate(plain)} |")
        out.append("")
    else:
        rows = _fn_rows(fd)
        sized = Counter((r.get("category"), _row_size(r, small_ratio)) for r in rows
                        if _row_size(r, small_ratio))
        if sized:
            out += [f"### Size buckets (train FN counts — at a {side}px detector input side: "
                    f"small {px['small']}, medium {px['medium']}, large {px['large']})", "",
                    "This run has no residual.json, so GT denominators, crowding and "
                    "borders cannot be counted.", "",
                    "| class | small | medium | large |", "|---|---|---|---|"]
            for n in sorted({c for c, _ in sized}, key=str):
                out.append(f"| {n} | " + " | ".join(
                    str(sized.get((n, b), 0)) for b in ("small", "medium", "large")) + " |")
            out.append("")
        else:
            out += ["_An older run without size information (size_ratio) on its FN rows — "
                    "size, crowding and border buckets are skipped._", ""]

    notes = fd["notes"]
    fn_notes = [n for n in notes if n.get("kind") in _FN_KINDS]
    if notes:
        allt = _terms(notes)
        if allt:
            out += ["- **conditions recurring in notes** (all): "
                    + ", ".join(f"`{t_}` {n}" for t_, n in allt[:8]),
                    "  (from crop notes — clues for narrowing collection conditions; the "
                    "numbers are how many notes mention the term.)", ""]
        groups: dict = {}
        for n in fn_notes:
            groups.setdefault((n.get("category") or f"id {n.get('category_id')}",
                               n.get("kind")), []).append(n)
        if groups:
            out += ["### Condition terms in miss notes (class × cause)", "",
                    "| class | cause | notes | recurring condition terms |", "|---|---|---|---|"]
            for (cls, kind), ns in sorted(groups.items(), key=lambda kv: str(kv[0])):
                terms = ", ".join(f"{t_} {k}" for t_, k in _terms(ns)[:5]) or "—"
                out.append(f"| {cls} | {kind} | {len(ns)} | {terms} |")
            out.append("")
    else:
        out += ["- no crop notes, so condition terms are left empty (the planner is "
                "deterministic or the description call failed for this diagnosis).", ""]

    samples = _samples(_fn_rows(fd), notes)
    if samples:
        out += [f"### Representative misses (train, up to {SAMPLES_PER_CLASS} per class)", "",
                "Those with notes first, then larger objects (missing a large object is "
                "stranger). No holdout samples are shown.", "",
                "| class | image | box (x1,y1,x2,y2) | cause | input px | note |",
                "|---|---|---|---|---|---|"]
        for r in samples:
            box = ",".join(f"{float(v):.0f}" for v in r["box"])
            px_ = (f"{r['px']:.0f}" if r.get("px") is not None else
                   f"{float(r['size_ratio']) * side:.0f}" if r.get("size_ratio") is not None
                   else "—")
            note = (r.get("_note") or "").replace("|", "/")
            if len(note) > NOTE_MAX:
                note = note[:NOTE_MAX - 1] + "…"
            out.append(f"| {r.get('category')} | `{r.get('image')}` | {box} "
                       f"| {r.get('kind')} | {px_} | {note or '—'} |")
        out.append("")
    return out


def _crowd_buckets():
    from .residual import CROWD_BUCKETS
    return CROWD_BUCKETS


def _note_key(image, box) -> tuple:
    return (image, tuple(round(float(v)) for v in box or ()))


def _samples(rows: list[dict], notes: list[dict]) -> list[dict]:
    """SAMPLES_PER_CLASS per class — round-robin over causes (missed →
    missed_small → below_threshold) to cover them evenly; within a cause,
    those with notes first, then by larger area. At most two per image.
    Deterministic (ties broken by filename and box)."""
    by_note = {_note_key(n.get("image"), n.get("box")): n.get("note")
               for n in notes if n.get("note") and n.get("box")}
    per: dict = {}
    for r in rows:
        if r.get("image") and r.get("box"):
            per.setdefault(str(r.get("category")), []).append(
                {**r, "_note": by_note.get(_note_key(r["image"], r["box"]))})
    out = []
    for cls in sorted(per):
        def area(r):
            b = r["box"]
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        queues: dict = {}
        for r in sorted(per[cls], key=lambda r: (r["_note"] is None, -area(r),
                                                 str(r["image"]), tuple(r["box"]))):
            queues.setdefault(r.get("kind"), []).append(r)
        order = sorted(queues, key=lambda k: (_FN_KINDS.index(k) if k in _FN_KINDS
                                              else len(_FN_KINDS), str(k)))
        seen: Counter = Counter()
        picked = []
        while len(picked) < SAMPLES_PER_CLASS and any(queues.values()):
            for k in order:
                while queues[k]:
                    r = queues[k].pop(0)
                    if seen[r["image"]] < 2:
                        seen[r["image"]] += 1
                        picked.append(r)
                        break
                if len(picked) >= SAMPLES_PER_CLASS:
                    break
        out += picked
    return out


# --------------------------------------------------------------- data collection

_SIZE_PHRASE = {"small": "small objects", "medium": "medium objects", "large": "large objects"}


def _skewed(rf: dict, pair) -> bool:
    """Is a [GT, FN] bucket's miss rate at least SKEW× the class's overall rate
    (with GT ≥ MIN_BUCKET_GT)?"""
    gt, fn = rf.get("gt", 0), rf.get("fn", 0)
    g, f = pair
    return bool(gt and fn and g >= MIN_BUCKET_GT and f and f / g >= SKEW * fn / gt)


def _suggest(name: str, tr: dict, te: dict, rf: dict | None, rows: list[dict],
             notes: list[dict], small_ratio: float, side: int) -> tuple[str, str, str, str]:
    """(main cause, main size, conditions, suggestion) — only what has evidence
    goes into the sentence."""
    kinds = Counter(te.get("fn") or {}) or Counter(tr.get("fn") or {})
    kinds = Counter({k: v for k, v in kinds.items() if v})
    total = sum(kinds.values())
    if not total:
        return "—", "—", "—", "no residual misses"
    dom_kinds = ", ".join(f"{k} {v / total:.0%}" for k, v in kinds.most_common(2))
    mine = [r for r in rows if r.get("category") == name
            and r.get("kind") != "below_threshold"]
    sizes = Counter(_row_size(r, small_ratio) for r in mine if _row_size(r, small_ratio))
    size_txt, size_phrase = "—", ""
    if sizes:
        b, n = sizes.most_common(1)[0]
        size_txt = f"{b} {n / sum(sizes.values()):.0%}"
        if rf and rf["size"][b][0]:
            size_txt += f" (miss rate {rf['size'][b][1] / rf['size'][b][0]:.0%})"
        rng = _px_ranges(small_ratio, side)[b]
        size_phrase = f"{_SIZE_PHRASE[b]} (input {rng})"
    cls_notes = [n for n in notes if n.get("category") == name
                 and n.get("kind") in _FN_KINDS]
    terms = [t_ for t_, _ in _terms(cls_notes)[:3]]
    cond = ", ".join(terms) or "—"
    # conditions of the misses data can fix (missed, missed_small) — notes on
    # detections below the threshold are left out
    data_terms = [t_ for t_, _ in _terms(
        [n for n in cls_notes if n.get("kind") != "below_threshold"])[:3]]
    parts = [size_phrase or "missed objects"]
    if rf:
        rng = _px_ranges(small_ratio, side)
        for b, pair in rf["size"].items():
            if b != (sizes.most_common(1)[0][0] if sizes else None) and _skewed(rf, pair):
                parts.append(f"especially {_SIZE_PHRASE[b]} (input {rng[b]}, miss rate "
                             f"{pair[1] / pair[0]:.0%})")
        crowd = [b for b, pair in (rf.get("crowd") or {}).items() if _skewed(rf, pair)]
        if crowd:
            parts.append(f"crowded scenes (GT per image {'/'.join(crowd)})")
        if _skewed(rf, rf["border"]):
            parts.append("objects cut by the frame border")
        if _skewed(rf, rf["elongated"]):
            parts.append("elongated objects (3:1 or more)")
    data = f"{name}: " + ", ".join(parts)
    if data_terms:
        data += f" — conditions: {', '.join(data_terms)}"
    below = kinds.get("below_threshold", 0) / total
    if below >= 0.5:
        text = (f"mainly a method problem — {below:.0%} of misses are detections below "
                "the threshold (calibration/threshold axes)")
        if below < 1.0:
            text += f". Collection direction for the other {1 - below:.0%}: {data}"
        return dom_kinds, size_txt, cond, text
    text = data
    gt_tr, mr = tr.get("gt", 0), te.get("miss_rate") or 0.0
    if mr >= 0.5 and gt_tr < 100:
        text = "few GT and many misses (collection first) — " + text
    elif mr >= 0.5:
        text += " (enough GT but many misses — also consider a method problem)"
    return dom_kinds, size_txt, cond, text


def _collection_section(run_dir: Path, t: dict, fd: dict | None) -> list[str]:
    """Data-collection suggestions — states **only what the artifacts can back up**.

    Per class, a one-line suggestion from GT count, residual miss rate, and the
    dominant cause, size bucket and condition terms. When below_threshold
    dominates it reads as a method problem, not a data one. It does not say how
    many more images to collect — no learning curve is measured, so the current
    artifacts cannot estimate it.
    """
    out = ["", "## Data collection suggestions", "",
           "The following is read from the **structure of residual errors**; it does not",
           "estimate how much data is needed (no learning curve is measured, so these artifacts cannot tell).", ""]
    if fd is None:
        return out + ["_No diagnosis artifacts, so no suggestions can be made._"]
    from .residual import INPUT_SIDE
    train, test = fd["train"], fd["test"]
    res = fd.get("residual") or {}
    small_ratio = res.get("small_ratio") or 0.04
    side = res.get("input_side") or INPUT_SIDE
    rtr = (res.get("train") or {}).get("per_class") or {}
    rows = _fn_rows(fd)
    names = sorted(set(train.get("per_class") or {}) | set(test.get("per_class") or {}),
                   key=lambda n: -((test.get("per_class") or {}).get(n, {})
                                   .get("miss_rate") or 0))
    if names:
        out += ["| class | GT train/holdout | holdout miss rate | main cause | main size (train) "
                "| condition terms | suggestion |", "|---|---|---|---|---|---|---|"]
        for n in names:
            tr = (train.get("per_class") or {}).get(n) or {}
            te = (test.get("per_class") or {}).get(n) or {}
            kinds, size_txt, cond, text = _suggest(n, tr, te, rtr.get(n), rows,
                                                   fd["notes"], small_ratio, side)
            out.append(f"| {n} | {tr.get('gt', '—')}/{te.get('gt', '—')} "
                       f"| {_num(te.get('miss_rate'))} | {kinds} | {size_txt} "
                       f"| {cond} | {text} |")
        out.append("")

    out += ["### Label quality flags", ""]
    flags = (res.get("train") or {}).get("label_flags")
    hflags = (res.get("holdout") or {}).get("label_flags") or {}
    if flags is not None:
        for key, label in (("tiny_gt", "GT with a short side ≤ 3px (annotation error suspected)"),
                           ("duplicate_gt", "duplicate GT (same class, IoU ≥ 0.95)")):
            rows_ = flags.get(key) or []
            line = (f"- **{label}**: train {len(rows_)} · holdout "
                    f"{hflags.get(key, 0)}")
            if rows_:
                line += " — e.g. " + "; ".join(
                    f"`{r['image']}` {r.get('category')} "
                    f"[{','.join(f'{float(v):.0f}' for v in r['box'])}]"
                    for r in rows_[:5])
            out.append(line)
    else:
        out.append("- This run has no residual.json, so the GT size and duplicate checks "
                   "are skipped.")
    loc = []
    for n, s in sorted((train.get("per_class") or {}).items()):
        fp = s.get("fp") or {}
        tot, lc = sum(fp.values()), fp.get("localization", 0)
        if tot and lc >= LOC_MIN and lc / tot >= LOC_SHARE:
            loc.append(f"{n} {lc}/{tot} ({lc / tot:.0%})")
    if loc:
        out.append("- **classes with a high localization share** (of train FPs; check box "
                   "alignment): " + ", ".join(loc))
    out.append("")
    return out


# --------------------------------------------------------------- render


def render_report(run_dir: Path, *, title: str | None = None) -> str:
    """Markdown report for one run — also written to `report.md`.

    Sections are drawn as far as artifacts exist — a missing one is noted in a
    line and skipped, and a section that breaks does not stop the others (an
    older run's artifacts may have a different shape)."""
    run_dir = Path(run_dir)
    t = trajectory(run_dir)
    hist = _history(run_dir)
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

    fd = None
    try:
        fd = _final_diag(run_dir, t)
    except Exception:  # noqa: BLE001 — a broken diagnosis bundle only empties the diagnosis sections
        fd = None
    sections = [
        ("Adoption path", lambda: _adoption_section(run_dir, t, hist)),
        ("Time use", lambda: _time_section(run_dir, t, hist)),
        ("FP/FN causes over rounds (holdout)", lambda: _error_trend_section(run_dir, t)),
        ("Residual misses", lambda: _residual_section(run_dir, t, fd)),
        ("Data collection suggestions", lambda: _collection_section(run_dir, t, fd)),
    ]
    for head, fn in sections:
        try:
            lines += fn()
        except Exception as exc:  # noqa: BLE001 — one section must not block the whole report
            lines += ["", f"## {head}", "",
                      f"_This section could not be generated ({type(exc).__name__}: "
                      f"{exc}) — the artifacts have an unexpected shape._"]
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
