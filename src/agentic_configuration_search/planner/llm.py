"""LLM planner — the model served by vLLM reads the diagnosis and proposes the
next candidates.

Why non-determinism is harmless: the LLM **only proposes**; adoption is
decided by the holdout F(β) CI gate. Reproducibility comes from **replay** —
prompt, response, and candidates are saved to `proposal.json`, and rebuilding
with the same artifact_dir (= resume) replays them without calling the LLM.
The resume key is this artifact, not a config hash (guard: tests/test_replay.py).

Failed or invalid responses fall back to the deterministic planner — a planner
failure never kills the loop (the gate protects against poor proposals).
"""

from __future__ import annotations

import copy
import json
import logging
import statistics
from pathlib import Path

from ..diagnose.sample import pick_detail
from . import deterministic
from .axes import DatasetFacts, axis_values, sanitize_candidates
from .gates import INPUT_SIDE, tiling_gate, tp_floor, under_champion

log = logging.getLogger(__name__)

ARTIFACT = "proposal.json"
# Proposals re-asked within the same round (k = 1, 2, …) — the same replay
# contract as proposal.json
REASK_ARTIFACT = "proposal_reask{k}.json"

SYSTEM = ("You plan the next inference-setting experiments for an open-vocabulary "
          "object detector. Answer with JSON only.")

SCHEMA = {
    "name": "proposal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "maxLength": 64},
                        "axes": {"type": "object"},
                        # rationales run to about 1,700 characters at most
                        # (p99 about 1,050). Without a cap one rationale could
                        # run on until the response hit the output cap
                        "rationale": {"type": "string", "maxLength": 2000},
                    },
                    "required": ["axes", "rationale"],
                },
            },
            "stop": {"type": "boolean"},
        },
        "required": ["candidates", "stop"],
    },
}

# Caps on diagnosis detail and history in the prompt — the first defense against
# overflowing the server context. The last is the budget guard that measures the
# assembled prompt (DEFAULT_TOKEN_BUDGET).
MAX_HISTORY = 8          # recent rounds in the prompt (the budget guard may cut further)
# Response cap — the schema bounds the response length (candidates ≤
# max_candidates, rationale ≤ RATIONALE_MAX_CHARS characters, id ≤ 64) and this
# is set above the longest English response (axes is a free object and is not
# bounded — see _schema). A truncated response breaks the JSON and the planner
# silently falls back to deterministic.
RATIONALE_MAX_CHARS = SCHEMA["schema"]["properties"]["candidates"]["items"][
    "properties"]["rationale"]["maxLength"]
OUTPUT_MAX_TOKENS = 4096
# Prompt token estimate — counted generously without the server tokenizer.
# ASCII counts at 1.85 chars/token (whole planner prompts measured no lower than
# 2.68; number-heavy JSON fragments about 1.85), and other characters (Hangul,
# Han, ...) count 1.5 tokens each — a character budget overflowed the server
# context on CJK-heavy prompts (around 1.8 chars/token). It is not a bound:
# text that is a long run of digits tokenizes below it, since the Qwen
# tokenizer splits digits one by one (synthetic numeric JSON about 1.2). Real
# prompts mix in prose, and on all measured planner prompts the estimate came
# out above the actual count.
ASCII_CHARS_PER_TOKEN = 1.85
NON_ASCII_TOKENS = 1.5
CONTEXT_MARGIN = 256      # chat template tokens etc.
# Prompt token budget — overflow is fixed by shrinking history (the same
# formula as runner._token_budget). Overflow makes vLLM return 400 and the
# planner silently falls back to deterministic.
DEFAULT_TOKEN_BUDGET = 16384 - OUTPUT_MAX_TOKENS - CONTEXT_MARGIN
MAX_NOTE = 80            # prefix length of the skip note carried in the prompt
NDIGITS = 4              # float digits in the prompt — at margin scale (0.005) the
                         # gate Δ/CI verdict turns on the 4th digit (3 is too few)


def _round_floats(obj, ndigits: int = NDIGITS):
    """Round floats in the prompt payload only — stored artifacts (history.jsonl
    etc.) keep full precision (rounding is a display choice, not a record choice)."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _det_key(row: dict) -> tuple:
    """Detection identity — (image, box). Crop notes and FP/FN samples come from
    the same records, so matching both means the same detection."""
    return (str(row.get("image", "")),
            tuple(round(float(v), NDIGITS) for v in (row.get("box") or ())))


def _merge_notes(c: dict) -> None:
    """Fold crop notes into their sample rows as `note`.

    Described crops are drawn from the same records as the samples, so most are
    the same detections — left as-is, image/box/score/category would appear
    twice. Only unmatched notes stay in crop_notes."""
    notes = c.get("crop_notes")
    if not isinstance(notes, list) or not notes:
        c.pop("crop_notes", None)
        return
    index = {_det_key(r): r for side in ("fp_sample", "fn_sample")
             for r in c.get(side, [])}
    leftover = []
    for n in notes:
        note, target = n.get("note"), index.get(_det_key(n))
        if not note:
            continue
        if target is None:
            leftover.append({k: v for k, v in n.items() if k != "crop"})
        else:
            target["note"] = note
    if leftover:
        c["crop_notes"] = leftover
    else:
        c.pop("crop_notes")


def _slim_rows(c: dict) -> None:
    """Trim sample rows for the prompt — call after _merge_notes.

    - category_id → category (name). per_class is keyed by name, so a name lets
      individual errors be attributed to a class.
    - box (xyxy) → wh (integer size). The planner can use size, not coordinates.
    """
    for key in ("fp_sample", "fn_sample", "crop_notes"):
        rows = c.get(key)
        if not isinstance(rows, list):
            continue
        for r in rows:
            if r.get("category") is not None:
                r.pop("category_id", None)
            box = r.pop("box", None)
            if box and len(box) == 4:
                r["wh"] = [round(box[2] - box[0]), round(box[3] - box[1])]
            # Drop image — once notes are merged there is nothing to join on and
            # the planner cannot point at an image. image_index exists only in
            # old artifacts.
            r.pop("image", None)
            r.pop("image_index", None)


def _nonzero(summary: dict) -> dict:
    """Drop zero-count kinds — kept in artifacts, but noise in the prompt."""
    return {side: {k: v for k, v in (kinds or {}).items() if v}
            for side, kinds in (summary or {}).items()}


def _compact_diagnosis(diagnosis: dict) -> dict:
    """All aggregates, but only severity-based representative samples of the
    detail (diagnose.sample).

    Samples are copies — the caller's diagnosis (the object saved as an
    artifact) is not mutated.
    """
    out = {}
    for split, d in (diagnosis or {}).items():
        if not isinstance(d, dict):
            continue
        c = {k: d[k] for k in ("n_images", "summary", "per_class") if k in d}
        if "summary" in c:
            c["summary"] = _nonzero(c["summary"])
        if isinstance(d.get("crop_notes"), list):
            c["crop_notes"] = [dict(n) for n in d["crop_notes"]]
        for side in ("fp", "fn"):
            detail = d.get(side)
            if isinstance(detail, list) and detail:
                c[f"{side}_sample"] = [dict(r) for r in pick_detail(detail)]
                c[f"{side}_total"] = len(detail)
        _merge_notes(c)          # needs the image/box join, so before _slim_rows
        _slim_rows(c)
        out[split] = c
    return _round_floats(out)


def _kind_counts(diagnosis: dict) -> dict[str, int]:
    """Per-kind counts — sum of train and holdout ("test" key) aggregates (same as
    deterministic._error_ranking)."""
    counts: dict[str, int] = {}
    for split in ("train", "test"):
        d = diagnosis.get(split) or {}
        for side in ("fn", "fp"):
            for kind, n in (d.get("summary", {}).get(side) or {}).items():
                if n:
                    counts[kind] = counts.get(kind, 0) + int(n)
    return counts


def _small_block(diagnosis: dict, overlap: float, *, tp_quantile: float = 0.1,
                 tp_min: int = 20) -> str:
    """Input-px quantiles of train missed_small and the median after each grid's
    magnification — evidence for picking the first tiling grid. Samples are a
    coverage selection, not a distribution, so this counts the full fn list.
    Holdout shows only aggregates, so only train is used.

    Also carries s*, the size of small objects the champion already detects
    (the `tp_quantile` of TP sizes of the class with the most missed_small —
    planner.gates.tp_floor). It only advises that the smallest grid whose
    magnified median reaches s* is the first candidate. With fewer TPs than
    `tp_min` in that class the line is left out (the same criterion as the fold
    gate's exception). Its numbers come from the gate's material (the train
    diagnosis's tiling_sizes — sizes counted at fit_dev screen thresholds), so
    its counts can differ slightly from the quantiles above (the fn list
    classified at tau_test).

    INPUT_SIDE (planner.gates, from diagnose.residual) converts size_ratio to
    input px: YOLOE letterboxes the long side to 640, exact for square images
    (the same assumption as DiagnoseCfg's 0.04 ≈ 26 px)."""
    from .axes import TILING_GRIDS

    fn = (diagnosis.get("train") or {}).get("fn") or []
    sizes = sorted(float(e["size_ratio"]) * INPUT_SIDE for e in fn
                   if isinstance(e, dict) and e.get("kind") == "missed_small"
                   and e.get("size_ratio") is not None)
    if not sizes:
        return ""

    def q(p):
        return round(sizes[min(len(sizes) - 1, int(p * len(sizes)))])
    p50 = sizes[len(sizes) // 2]
    grids = "\n".join(
        f"- grid {g}: {(g - 1) * (1 - overlap) + 1:.2g}x -> median "
        f"{round(p50 * ((g - 1) * (1 - overlap) + 1))} px, cost {1 + g * g} passes"
        for g in TILING_GRIDS)
    floor = tp_floor(diagnosis.get("train"), quantile=tp_quantile, tp_min=tp_min)
    tp_line = ""
    if floor and floor["available"]:
        tp_line = (f"\nSmallest objects the champion already detects: champion TP "
                   f"p{round(tp_quantile * 100)} ≈ {floor['tp_px']:.0f} px for "
                   f"{floor['class']} (the class with the most small misses: "
                   f"{floor['n_missed']}, their median {floor['missed_px']:.0f} px; "
                   f"{floor['n_tp']} TPs). Advisory: the smallest grid whose "
                   "magnified median reaches that size is a sensible first grid.")
    return SMALL_BLOCK.format(n=len(sizes), side=INPUT_SIDE, p25=q(0.25),
                              p50=round(p50), p75=q(0.75), overlap=overlap,
                              grids=grids).rstrip("\n") + tp_line + "\n"


def _leverage_lines(diagnosis: dict) -> list[str]:
    """One line per axis with **the error count it touches this round**, largest first.

    A static "kind → axis" table unions to every axis when several kinds are
    observed and narrows nothing. The join happens here and the model gets only
    the per-round counts, as evidence.

    Axis names come from REMEDY (canonical), so no nonexistent axis can appear.
    """
    counts = _kind_counts(diagnosis)
    per_axis: dict[str, dict[str, int]] = {}
    for kind, n in counts.items():
        for axis, _value in deterministic.REMEDY.get(kind, ()):
            per_axis.setdefault(axis, {})[kind] = n
    lines = []
    for axis, kinds in sorted(per_axis.items(),
                              key=lambda kv: (-sum(kv[1].values()), kv[0])):
        detail = ", ".join(f"{k} {v}" for k, v in
                           sorted(kinds.items(), key=lambda kv: -kv[1]))
        lines.append(f"- {axis}: {detail} (total {sum(kinds.values())})")
    return lines


def _dead_axis_lines(history: list[dict]) -> list[str]:
    """Axes rejected for empty material, or too little to train on (empty_axis)
    — collected over the full history.

    Once pushed out of the history window (MAX_HISTORY) the model proposes the
    same axis again. It is not a permanent ban — once the champion changes, the
    mining source changes and material may appear.
    """
    dead: dict[str, set] = {}
    for e in history:
        for c in e.get("candidates", []):
            axis = c.get("empty_axis")
            if not axis:
                continue
            value = (c.get("axes") or {}).get(axis)
            dead.setdefault(axis, set()).add(value)
    lines = []
    for axis in sorted(dead):
        vals = sorted(str(v) for v in dead[axis] if v not in (None, False))
        lines.append(f"- {axis}" + (f" (values tried: {', '.join(vals)})"
                                    if vals else ""))
    return lines


MAX_COST_LINES = 12      # cap on per-axis observed-cost lines in the prompt


def _cost_label(axis: str, value) -> str:
    """Group name in the cost summary — tiling includes the value since cost
    grows as 1+g² per grid; other axes use the axis name only (cost differs
    little between their values)."""
    return f"{axis}={value}" if axis == "tiling" and value else axis


def _cost_lines(history: list[dict]) -> list[str]:
    """Per-candidate wall clock observed in this run (wall_min.total, minutes),
    by axis — median and count.

    A candidate is attributed to each axis of its increment (by request) over
    the champion at the **start** of its round (the same basis as
    _compact_history). A candidate changing several axes counts once for each.
    Collected over the whole history — cost changes little when the champion
    changes, and forgetting an expensive attempt that slid out of the window
    means paying it again."""
    costs: dict[str, list[float]] = {}
    for i, e in enumerate(history):
        champ = ((history[i - 1].get("champion", {}).get("axes") if i > 0 else None)
                 or e.get("champion", {}).get("axes") or {})
        for c in e.get("candidates", []):
            total = (c.get("wall_min") or {}).get("total")
            if total is None:
                continue
            delta = _delta_axes(c.get("requested_axes") or c.get("axes"), champ)
            for axis in {_cost_label(k, v) for k, v in delta.items()}:
                costs.setdefault(axis, []).append(float(total))
    rows = sorted(costs.items(), key=lambda kv: -statistics.median(kv[1]))
    return [f"- {label}: median {statistics.median(v):.0f} min per candidate "
            f"(n={len(v)}, max {max(v):.0f})"
            for label, v in rows[:MAX_COST_LINES]]


FOLD_MIN_UNUSED = 2      # axes unused this many times under the current champion are listed as folded


def _folded_axis_lines(history: list[dict]) -> list[str]:
    """Axes screen repeatedly left unused under the current champion
    (unused_axes) — a hint to fold them.

    Unlike empty_axis (material exhausted), the material was there but the
    setting did not pick it. At FOLD_MIN_UNUSED or more, other values likely
    end the same way. A champion change changes the context, so only records
    under it count (the same principle as planner.gates). Tiling is folded
    after one unused grid by the gate, which the runner enforces, so its state
    is listed instead (TilingGate.describe). Other axes are a hint only, not
    blocked."""
    gate = tiling_gate(history)
    counts: dict[str, int] = {}
    values: dict[str, set] = {}
    for e in under_champion(history):
        for c in e.get("candidates", []):
            asked = c.get("requested_axes") or c.get("axes") or {}
            for axis in c.get("unused_axes") or ():
                counts[axis] = counts.get(axis, 0) + 1
                values.setdefault(axis, set()).add(str(asked.get(axis)))
    lines = []
    for axis in sorted(counts):
        if counts[axis] < FOLD_MIN_UNUSED or (axis == "tiling"
                                              and gate.folded_at is not None):
            continue
        lines.append(f"- {axis}: screened and not used by {counts[axis]} candidates "
                     f"under the current champion (values: "
                     f"{', '.join(sorted(values[axis]))}) — treat as folded until "
                     "the champion changes")
    return lines + gate.describe()


def _on_axes(axes: dict) -> dict:
    """Only axes that are on — off axes (False/None/"none") are safe points and carry no information."""
    return {k: v for k, v in (axes or {}).items()
            if v not in (False, None, "none")}


def _delta_axes(axes: dict, base: dict) -> dict:
    """Delta vs. the champion — a candidate is identified by its delta, not its full axes."""
    return {k: v for k, v in (axes or {}).items() if (base or {}).get(k) != v}


MAX_TRIED = 40           # cap on tried combos in the prompt (most recent first)


def _tried_lines(history: list[dict], champion: dict) -> list[str]:
    """Already-tried axes combinations, compressed to **deltas vs. the champion**.

    A combination outside the window (MAX_HISTORY) that is re-proposed is dropped
    by the runner's combo filter and wastes the candidate budget, so collect from
    the full history. List only what differs from the champion to keep it short.
    """
    seen: dict[tuple, None] = {}
    for e in history:
        for c in e.get("candidates", []):
            # a relabelled candidate's original request is a tried combination
            # too — the runner's combo filter (tried_combos) blocks it, so the
            # prompt must list it or the planner re-proposes the same request
            for axes in (c.get("axes"), c.get("requested_axes")):
                if isinstance(axes, dict):
                    seen[tuple(sorted(axes.items()))] = None
    lines = []
    for key in list(seen)[-MAX_TRIED:]:
        axes = dict(key)
        delta = _delta_axes(axes, champion)
        lines.append(", ".join(f"{k}={v}" for k, v in sorted(delta.items()))
                     or "(baseline = current champion)")
    return lines


def _schema(max_candidates: int) -> dict:
    """The schema with the candidate cap. With rationale, id and the candidate
    count bounded, the longest English response fits within OUTPUT_MAX_TOKENS —
    axes is a free object and is not bounded, and a rationale written in CJK
    takes more tokens per character and can reach the cap (a warning is logged
    then)."""
    out = copy.deepcopy(SCHEMA)
    out["schema"]["properties"]["candidates"]["maxItems"] = max_candidates
    return out


def _modality_summary(modality_f: dict) -> dict:
    """Summary of per-class semantic and text F(β) — its size does not depend on
    the number of classes.

    Carrying the full per-class record (history's modality_f) would make one
    row exceed 1KB on a 19-class dataset, and the budget guard would shrink the
    history window. The planner needs the number of classes text won and the
    narrowest of those margins. Wins follow screen's pick (chosen) — the
    recorded F is rounded and cannot tell a narrow gap (old records without
    chosen are counted by F; screen picks semantic on a tie)."""
    def text_won(v):
        return v["chosen"] == "text" if v.get("chosen") else v["text"] > v["semantic"]
    margins = [v["text"] - v["semantic"] for v in modality_f.values() if text_won(v)]
    out = {"text_won": len(margins), "semantic_won": len(modality_f) - len(margins)}
    if margins:
        out["closest_text_margin"] = round(min(margins), 4)
    return out


def _free_id(base: str, taken: set) -> str:
    """An id not in taken — appends _2, _3, ... on a clash."""
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def _axes_key(axes: dict) -> tuple:
    return tuple(sorted(axes.items()))


def _drop_carried(candidates: list[dict], carried_keys: set,
                  dropped: list) -> tuple[list[dict], list[dict]]:
    """Drop combinations equal to a carried candidate — (kept, dropped). The
    dropped ones are recorded in dropped."""
    keep, dups = [], []
    for c in candidates:
        if _axes_key(c["axes"]) in carried_keys:
            dups.append(c)
            dropped.append({**c, "skipped": "carried_duplicate",
                            "note": "same combination as an automatically carried "
                                    "candidate — handled as the carry-over (see the "
                                    "carry_<id> row)"})
        else:
            keep.append(c)
    return keep, dups


def _outcome_lines(rows: list[dict], champion: dict) -> list[str]:
    """Candidate lines of the re-ask message — the increment (by request), the
    rejection reason, the start of the note, and when present the served
    material (same_setting) and unused axes."""
    lines = []
    for c in rows[:MAX_REASK_LINES]:
        delta = _delta_axes(c.get("requested_axes") or c.get("axes"), champion)
        line = (f"- {c.get('id')} ("
                + (", ".join(f"{a}={v}" for a, v in sorted(delta.items()))
                   or "no change") + f"): skipped={c.get('skipped')}")
        if c.get("unused_axes"):
            line += f", unused_axes={c['unused_axes']}"
        if c.get("note"):
            line += f" — {str(c['note'])[:MAX_NOTE]}"
        if c.get("material"):
            line += f" [served material: {c['material']}]"
        lines.append(line)
    if len(rows) > MAX_REASK_LINES:
        lines.append(f"- … and {len(rows) - MAX_REASK_LINES} more")
    return lines


def _compact_history(history: list[dict], window: int = MAX_HISTORY) -> list[dict]:
    """Per round: trajectory, candidate deltas, and verdicts only — large
    record-like fields are left out.

    axes are carried **as deltas only** — listing them all overflows the context
    in later rounds.

    *How badly* a candidate was rejected is included too: holdout score (keys
    are legacy test_*) and gate Δ / CI lower bound, fit_dev score, and the skip
    note prefix — so a near miss can be told from a clear regression."""
    out = []
    # window=0 means "no history" — history[-0:] is the whole list in Python.
    start = max(len(history) - window, 0) if window > 0 else len(history)
    for i in range(start, len(history)):
        e = history[i]
        d = e.get("decision", {})
        evaluated = d.get("evaluated") or {}
        # deltas are against the champion at the **start** of the round — the
        # row's champion is the one after it, so comparing with that would mix
        # in the axes adopted that round
        champ_axes = ((history[i - 1].get("champion", {}).get("axes") if i > 0
                       else None) or e.get("champion", {}).get("axes") or {})
        cands = []
        for c in e.get("candidates", []):
            # a relabelled candidate shows its requested increment — against the
            # corrected label, an inherited axis that went unused would look
            # turned off
            row = {"id": c.get("id"),
                   "axes_delta": _delta_axes(c.get("requested_axes") or c.get("axes"),
                                             champ_axes),
                   "adopted": c.get("adopted"), "skipped": c.get("skipped")}
            if c.get("fit_dev_micro_f_beta") is not None:
                row["fit_dev_micro_f_beta"] = c["fit_dev_micro_f_beta"]
            if "tiling_used" in c:
                row["tiling_used"] = c["tiling_used"]
            if c.get("tiling_f"):
                row["tiling_f"] = c["tiling_f"]      # best tiled vs untiled F — how far short
            if "neg_eta_used" in c:
                row["neg_eta_used"] = c["neg_eta_used"]
                row["neg_eta_grid"] = c.get("neg_eta_grid")
            if c.get("unused_axes"):
                row["unused_axes"] = c["unused_axes"]
            if c.get("modality_f"):
                row["modality"] = _modality_summary(c["modality_f"])
            if c.get("gate_passed_but_lost"):
                row["gate_passed_but_lost"] = True   # carried over — see CARRY_BLOCK
            if (c.get("wall_min") or {}).get("total") is not None:
                row["wall_min"] = c["wall_min"]["total"]   # per stage only in the artifact
            if c.get("note"):
                row["note"] = str(c["note"])[:MAX_NOTE]
            if c.get("material"):
                row["material"] = c["material"]   # same_setting — what was the same
            ev = evaluated.get(c.get("id"))
            if ev:
                row["test_micro_f_beta"] = ev.get("micro_f_beta")
                g = ev.get("gate") or {}
                row["gate_delta"] = g.get("delta")
                row["gate_ci_lo"] = g.get("lo")
            cands.append(row)
        out.append({
            "round": e.get("round"),
            "champion_micro_f_beta": e.get("champion", {}).get("micro_f_beta"),
            # the same yardstick as candidates' fit_dev_micro_f_beta — whether
            # tiling "helped"
            "champion_fit_dev_micro_f_beta":
                e.get("champion", {}).get("fit_dev_micro_f_beta"),
            "champion_axes": _on_axes(e.get("champion", {}).get("axes")),
            "adopted": d.get("adopted"), "adopted_id": d.get("id"),
            "candidates": cands,
        })
    return _round_floats(out)


PROMPT = """A zero-shot detector is being adapted to one dataset by searching
inference settings. **Objective: maximize micro F(beta) on a held-out
validation split (the holdout; fields and keys named "test" below refer to
it).** Each candidate toggles axes from the current champion:
{axes_doc}

Rules:
- Change as FEW axes as possible per candidate (single increments attribute
  gains cleanly). List only the axes you change: omitted axes keep the
  champion's value; to turn an axis off, set it to false explicitly.
  "visual_negatives" requires exemplar != "none".
- Materials are mined from the train split only; adoption is decided by a
  held-out gate afterwards — propose boldly, the gate filters.
- The gate adopts only if a paired-bootstrap LOWER BOUND on the holdout delta
  clears a margin AND the delta is positive. A candidate can therefore score
  higher on the holdout and still be rejected for being within noise: gate_delta > 0
  with gate_ci_lo < 0 means "close, but unproven" — a variant of that direction
  is often worth one more try, while a clearly negative gate_delta is not.
- "tiling" is a grid g: every image is also cut into g x g tiles and each tile
  is detected at magnification (g-1)(1-overlap)+1 (1.5x at g=2, 2x at g=3 with
  overlap 0.5). Cost: 1 + g^2 detector passes per image (5 at g=2, 10 at g=3,
  17 at g=4, 26 at g=5, 37 at g=6) — each step up costs much more than the
  last. Pick the first grid from the missed-small sizes below: the grid whose
  magnification brings the typical missed object to a detectable size, no
  larger. Raising g further is worth it only when the previous grid was used
  and helped (tiling_used=true and its fit_dev score beat the champion's)
  while small-object misses remain. tiling_f compares the mean per-class
  fit_dev F of the best tiled vs untiled config. A row with tiling_used=false
  means the screen preferred no tiling at that grid (the setting fell back to
  no tiling): fold the axis — a larger grid costs more and rarely helps. The
  loop enforces the fold: under the current champion, a grid at or above an
  unused one is dropped as skipped="tiling_blocked" without being screened. A
  note starting "prescreen" means the grid was first screened on a subset of
  fit_dev and not used there, so the full screen was skipped.
  tiling_f.gap_ub is a paired-bootstrap upper bound on (tiled - untiled) over
  the fit_dev images it was screened on; gap_ub >= 0 means the loss is within
  noise. One exception to the fold: when the unused grid g was such a near
  miss AND fell short on magnification (its magnified missed-small median
  stayed below the smallest objects the champion already detects, while g+1
  reaches them), grid g+1 is unblocked exactly once under the current
  champion — the fold-status lines below say when. It is an option, not a
  request; grids above g+1 stay blocked, and after g+1 is screened the fold at
  g holds.
- A row with unused_axes lists axes the candidate had on that the screened
  setting did not use (exemplar: the text pass beat the semantic pass for every
  class; visual_negatives: no semantic class kept a negative config). Other
  variants of such an axis (another k, fn_arm) rarely change the setting
  while that stays true. A row's modality counts the classes where the text
  pass beat the semantic (exemplar) pass on fit_dev F (text_won) and the
  rest (semantic_won); closest_text_margin is the smallest F gap among the
  classes text won. A small margin means another exemplar_k or fn_arm may
  flip a class to semantic; a large one means exemplar variants are unlikely
  to win.
- A visual_negatives row carries neg_eta_used (the suppression IoU thresholds
  the screen picked for semantic classes) and neg_eta_grid (the thresholds it
  tried). Lower means stronger suppression. Picks at the low end of the grid
  mean negatives are pulling hard, so more negatives (raise visual_negative_k)
  may help; picks at the high end mean they barely help, so fold the axis
  rather than raising k.
- Once "lora" or "crop_classifier" is on in the champion, keep it on: they are
  detector state (weights / classifier artifact) that stays loaded while
  other candidates are measured, so a candidate turning one off is rejected as
  skipped="detector_state_off" without evaluation.
- If nothing promising remains, return stop=true with no candidates.
- History rows may carry test_micro_f_beta / gate_delta / gate_ci_lo (how close
  a rejected candidate came) and a skip note. skipped="combo_tried" means that
  exact axes combination was already tried — never repeat it verbatim; vary it
  or move on. skipped="carried_duplicate" means it repeated a re-queued
  candidate; that combination is handled by the carry_<id> row instead.
  skipped="same_setting" means the screened setting served exactly the same
  thing as the champion or an already-evaluated candidate; its material field
  shows what was served (e.g. text negatives that a larger k refined down to
  the same phrases) — another value of that axis will likely collapse the same
  way.

Current champion axes (only axes that are ON are listed; an empty object means
the baseline — every axis at its safe point):
{champion}

Round history (adopted/rejected candidates and the holdout micro F(beta) trajectory):
{history}

Already tried — these exact axes combinations are permanently blocked, listed
as their difference from the current champion. Proposing one again wastes a
candidate slot:
{tried}

Error diagnosis. "train" carries individual error samples; "test" (the
holdout) is aggregate-only — read its stats as evidence that a problem also holds outside
the split the materials come from. In a sample, "wh" is the box size in pixels
and "note" is a vision model's one-line observation of that error's crop —
diagnostic evidence, not ground truth.

**Samples are a coverage selection, not a random draw**: they round-robin over
error kinds and classes, most severe first, capped per kind. Their mix therefore
does NOT reflect how frequent each kind is — read frequency from "summary" and
"per_class" only, and read the samples for what the errors look like.
{diagnosis}

Axis leverage this round — for each axis, the observed errors it could plausibly
address and their counts (train + holdout combined). These are HEURISTIC search hints, not rules: the count
says how much is at stake if the axis works, not that it will. Weigh them
against the diagnosis above (aggregate counts, then per-class, then the
individual samples and their notes) rather than picking the largest number.
{leverage}
{small_block}{dead_block}{fold_block}{cost_block}{carry_block}{notes_block}
Propose up to {max_candidates} candidates."""

# Inserted only when train has missed_small — evidence for the first tiling grid.
SMALL_BLOCK = """
Missed-small sizes (train, {n} objects) — box side at the detector input
(sqrt of box/image area x {side} px, long side resized to {side}): p25 {p25} px,
median {p50} px, p75 {p75} px. After tiling magnification (overlap {overlap}):
{grids}
"""

# Inserted only when some axis had empty material — otherwise omitted.
DEAD_BLOCK = """
Axes whose mined material came out empty, or too small to train on, on this
dataset (the candidate was rejected without evaluation, so it cost a slot for
nothing). Their material pool is exhausted here — re-propose one only with a
different source value, or after the champion has changed:
{dead}
"""

# Inserted only when an axis went unused repeatedly under the current champion
# — otherwise omitted.
FOLD_BLOCK = """
Axes folded under the current champion — the screen had them on but the chosen
setting did not use them, repeatedly (or, for tiling, at a grid the loop now
blocks). Other values of these axes are unlikely to change the setting; spend
slots elsewhere until the champion changes:
{folded}
"""

# Inserted only once candidate wall clocks (wall_min) are recorded — otherwise
# omitted.
COST_BLOCK = """
Observed wall-clock cost per candidate in this run (materialize + fit_dev
screen + held-out eval, minutes), grouped by the axes a candidate changed from
the champion of its round. The run's time budget is shared by every candidate
— weigh the expected gain of an expensive axis value against what it has cost:
{costs}
"""

# Inserted only when some candidate is carried over — otherwise omitted.
CARRY_BLOCK = """
Re-queued automatically this round — each passed the gate last round but lost
to the adopted candidate, so it is retried on top of the new champion (listed
as its difference from the current champion). Do not spend a slot on these —
a candidate repeating one is dropped as a duplicate:
{carried}
"""

# Follow-up message asking again for the slots freed when the planner repeated
# carried candidates.
FOLLOWUP = """{n} of your candidates repeated re-queued ones that are queued anyway ({dups}),
so they were dropped. Kept from your answer: {kept}. Propose up to {k} further
candidates that differ from both, under the same rules, in the same JSON format."""

# Follow-up asked in the same conversation when every candidate of the round
# was rejected before holdout (up to planner.max_reasks times, by the runner).
# It carries what happened to each candidate.
REASK = """None of the candidates so far this round reached the held-out evaluation,
so the round has measured nothing yet. What happened to each (difference from
the current champion, then why it was dropped):
{outcomes}
Propose up to {k} new candidates that differ from all of the above and from the
already-tried combinations, under the same rules, in the same JSON format. Avoid
the same failure: a folded tiling grid is blocked, an axis the screen left unused
stays unused for other values of it, and a quantity whose material collapsed to
the same setting collapses again. If nothing promising remains, return stop=true
with no candidates."""

MAX_REASK_LINES = 24     # cap on candidate lines in the re-ask message

# Inserted only when dataset_notes (config) is set — otherwise notes_block=""
# and the prompt is byte-identical to the no-notes form (pinned by
# tests/test_planner.py).
NOTES_BLOCK = """
Dataset notes from the operator (verbatim context — weigh against the diagnosis):
{notes}
"""

# Inserted only when disabled_axes/disabled_values are set — banned axes/values
# are not just omitted from the list but explicitly forbidden (so the LLM does
# not reintroduce them from its own world knowledge).
DISABLED_BLOCK = """
Unavailable in this run (never turn these on; the champion keeps any it already has): {disabled}.
"""


def _est_tokens(messages: list[dict]) -> int:
    """A generous estimate of prompt tokens (ASCII_CHARS_PER_TOKEN, NON_ASCII_TOKENS)."""
    text = "".join(m["content"] if isinstance(m.get("content"), str)
                   else "".join(str(x) for x in m.get("content", []))
                   for m in messages)
    n_ascii = sum(1 for ch in text if ord(ch) < 128)
    return round(n_ascii / ASCII_CHARS_PER_TOKEN
                 + (len(text) - n_ascii) * NON_ASCII_TOKENS)


def _strip_samples(diag: dict) -> dict:
    """Strip detail samples from the diagnosis — aggregates (summary, per_class) stay."""
    return {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_sample")}
            if isinstance(v, dict) else v for k, v in (diag or {}).items()}

class LLMPlanner:
    """chat: (messages, **kw) -> dict (parsed JSON). In production this is a
    partial application of clients.llm.LLMClient.chat_json."""

    def __init__(self, chat, artifact_dir: Path, *, max_candidates: int = 3,
                 notes: str = "", token_budget: int = DEFAULT_TOKEN_BUDGET,
                 disabled_axes=(), disabled_values=None,
                 facts: DatasetFacts | None = None, tiling_overlap: float = 0.5,
                 tiling_tp_quantile: float = 0.1, tiling_tp_min: int = 20):
        self.chat = chat
        self.tiling_overlap = tiling_overlap   # search.tiling_overlap — magnification
        # search.tiling_tp_quantile and tiling_tp_min — the s* line of the
        # missed-small block
        self.tiling_tp_quantile = tiling_tp_quantile
        self.tiling_tp_min = tiling_tp_min
        self.artifact_dir = Path(artifact_dir)
        self.max_candidates = max_candidates
        self.notes = notes or ""
        self.token_budget = token_budget
        self.disabled_axes = tuple(disabled_axes or ())
        self.disabled_values = {k: tuple(v) for k, v in
                                (disabled_values or {}).items() if v}
        # Values whose outcome the data already decides (gt_crop on a single class,
        # lora whose training would be skipped) are left out of the prompt and
        # filtered from the response — None means the full value space
        self.facts = facts

    def _refill(self, build, raw, dups, candidates, champion, carried_keys,
                dropped, carried_ids) -> dict:
        """Ask once more, in the same conversation, for the slots freed by
        candidates that repeated carried ones (extends candidates and dropped in
        place). Returns the record kept in the artifact.

        The conversation grows by the first response and the follow-up message —
        if the first prompt nearly filled the budget, shrink the history window
        (build) to fit (_fit). With dups present there is at least one free slot."""
        k = self.max_candidates - len(candidates)

        def delta(c):
            return ", ".join(f"{a}={v}" for a, v in
                             sorted(_delta_axes(c["axes"], champion).items()))
        note = FOLLOWUP.format(n=len(dups), dups="; ".join(delta(c) for c in dups),
                               kept="; ".join(delta(c) for c in candidates) or "none",
                               k=k)
        tail = [{"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
                {"role": "user", "content": note}]
        msgs, window = self._fit(build, tail)
        record = {"message": note, "response": None, "error": None, "window": window}
        if _est_tokens(msgs) > self.token_budget:
            record["error"] = "over the prompt budget — not asking again"
            return record
        try:
            raw2 = self.chat(msgs, schema=_schema(k), max_tokens=OUTPUT_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001 — the follow-up is a bonus
            record["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("follow-up after carried duplicates failed (%s) — using the "
                        "first response only", record["error"])
            return record
        record["response"] = raw2
        # ids are fixed before filtering — an id equal to a kept, dropped or
        # carried candidate's would put two rows in one round, and the next
        # round's history would attach the evaluation score to the dropped row too
        taken = ({c["id"] for c in candidates} | {d["id"] for d in dropped}
                 | set(carried_ids))
        renamed = []
        for i, c in enumerate((raw2 or {}).get("candidates") or []):
            if isinstance(c, dict):
                c = {**c, "id": _free_id(str(c.get("id") or f"cand_{i}"), taken)}
                taken.add(c["id"])
            renamed.append(c)
        more = sanitize_candidates(renamed, max_candidates=k, dropped=dropped,
                                   disabled_axes=self.disabled_axes,
                                   disabled_values=self.disabled_values,
                                   base=champion or None, facts=self.facts)
        more, _ = _drop_carried(more, carried_keys, dropped)
        seen = {_axes_key(c["axes"]) for c in candidates}
        for c in more:
            if _axes_key(c["axes"]) in seen:
                dropped.append({**c, "skipped": "duplicate_axes",
                                "note": "same combination as a candidate in the first response"})
                continue
            seen.add(_axes_key(c["axes"]))
            candidates.append(c)
        return record

    @property
    def artifact_path(self) -> Path:
        return self.artifact_dir / ARTIFACT

    def reask_path(self, k: int) -> Path:
        return self.artifact_dir / REASK_ARTIFACT.format(k=k)

    @staticmethod
    def _replay(path: Path) -> dict:
        """Replay a stored proposal — zero LLM calls (resume contract). The drop
        record is restored too — so a resumed run does not silently lose budget
        where a fresh one would not. Old artifacts without it read as an empty
        list."""
        log.info("proposal replay: %s", path)
        stored = json.loads(path.read_text())
        out = {k: stored[k] for k in ("planner", "candidates", "stop")}
        out["dropped"] = stored.get("dropped") or []
        return out

    def _builder(self, history: list[dict], diagnosis: dict):
        """(build, carried) — build(window, strip=False) is the first prompt
        assembled with history window `window`. strip removes the diagnosis
        detail (samples), the last line of defense. propose and reask must
        assemble the same prompt for a re-ask to continue the same
        conversation."""
        champion = (history[-1].get("champion", {}).get("axes")
                    if history else None) or {}
        axes_doc = "\n".join(
            f"- {k}: one of "
            f"{[x for x in v if x not in self.disabled_values.get(k, ())]}"
            for k, v in axis_values(self.facts).items()
            if k not in self.disabled_axes)
        banned = list(self.disabled_axes) + [
            f"{k}={x!r}" for k, xs in sorted(self.disabled_values.items())
            for x in xs]
        compact = _compact_diagnosis(diagnosis)
        diag_full = json.dumps(compact, ensure_ascii=False, default=str)
        tried_lines = _tried_lines(history, champion)
        leverage_lines = [
            ln for ln in _leverage_lines(diagnosis)
            if not any(ln.startswith(f"- {a}:") for a in self.disabled_axes)]
        dead_lines = _dead_axis_lines(history)
        cost_lines = _cost_lines(history)
        fold_lines = _folded_axis_lines(history)
        carried = deterministic.carryover(history, self.disabled_axes,
                                          self.disabled_values, facts=self.facts)
        carry_lines = [", ".join(f"{k}={v}" for k, v in
                                 sorted(_delta_axes(c["axes"], champion).items()))
                       for c in carried]

        def build(window: int, strip: bool = False) -> list[dict]:
            diag_json = (json.dumps(_strip_samples(compact), ensure_ascii=False,
                                    default=str) if strip else diag_full)
            return [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(
                    axes_doc=axes_doc,
                    champion=json.dumps(_on_axes(champion), ensure_ascii=False),
                    history=json.dumps(_compact_history(history, window),
                                       ensure_ascii=False, default=str),
                    tried="\n".join(f"- {x}" for x in tried_lines) or "- (none)",
                    diagnosis=diag_json,
                    leverage="\n".join(leverage_lines) or "- (no errors observed)",
                    small_block=_small_block(diagnosis, self.tiling_overlap,
                                             tp_quantile=self.tiling_tp_quantile,
                                             tp_min=self.tiling_tp_min),
                    fold_block=(FOLD_BLOCK.format(folded="\n".join(fold_lines))
                                if fold_lines else ""),
                    cost_block=(COST_BLOCK.format(costs="\n".join(cost_lines))
                                if cost_lines else ""),
                    dead_block=(
                        (DISABLED_BLOCK.format(disabled=", ".join(banned))
                         if banned else "")
                        + (DEAD_BLOCK.format(
                            dead="\n".join(dead_lines)) if dead_lines else "")),
                    carry_block=(CARRY_BLOCK.format(
                        carried="\n".join(f"- {x}" for x in carry_lines))
                        if carry_lines else ""),
                    notes_block=(NOTES_BLOCK.format(notes=self.notes)
                                 if self.notes else ""),
                    max_candidates=self.max_candidates)},
            ]
        return build, carried

    def _fit(self, build, tail: list[dict] = ()) -> tuple[list[dict], int]:
        """Budget guard — drop the oldest rounds first. If still over at window=0,
        strip the diagnosis samples as the last line of defense. tail is the end
        of the conversation (earlier responses, follow-up messages) — it counts
        toward the budget too."""
        tail = list(tail)
        window = MAX_HISTORY
        messages = build(window) + tail
        while _est_tokens(messages) > self.token_budget and window > 0:
            window -= 1
            messages = build(window) + tail
        if _est_tokens(messages) > self.token_budget:
            messages = build(0, strip=True) + tail
        return messages, window

    def reask(self, history: list[dict], diagnosis: dict, outcomes: list[dict], *,
              k: int, taken: set) -> dict | None:
        """The k-th re-ask of the round — called by the runner when every
        candidate was rejected before holdout. history and diagnosis are exactly
        what propose got (diagnosis and champion are not recomputed). outcomes
        are this round's rejection rows (history-row shape), and taken holds the
        candidate ids already used this round — a repeated id would collide in
        materials directories and history rows, so _free_id renames it.

        Continues the conversation of the first proposal (proposal.json) and the
        earlier re-asks, and stores the result as proposal_reask<k>.json — resume
        replays it (zero calls). Without a first response (the call failed and
        the round fell back to deterministic) there is no conversation to
        continue, so it does not ask (None). A failed call is an empty proposal —
        not filled by the deterministic fallback (this round's first proposal
        already ran)."""
        path = self.reask_path(k)
        if path.exists():
            return self._replay(path)
        first = (json.loads(self.artifact_path.read_text())
                 if self.artifact_path.exists() else {})
        if first.get("response") is None:
            log.info("re-ask skipped — no first planner response (fallback round)")
            return None

        def said(raw):
            return {"role": "assistant",
                    "content": json.dumps(raw, ensure_ascii=False, default=str)}
        tail = [said(first["response"])]
        follow = first.get("followup") or {}
        if follow.get("response") is not None:
            tail += [{"role": "user", "content": follow["message"]},
                     said(follow["response"])]
        for j in range(1, k):
            prev = (json.loads(self.reask_path(j).read_text())
                    if self.reask_path(j).exists() else {})
            if prev.get("response") is not None:
                tail += [{"role": "user", "content": prev["message"]},
                         said(prev["response"])]

        champion = (history[-1].get("champion", {}).get("axes")
                    if history else None) or {}
        message = REASK.format(outcomes="\n".join(_outcome_lines(outcomes, champion))
                               or "- (none)", k=self.max_candidates)
        tail.append({"role": "user", "content": message})
        build, _ = self._builder(history, diagnosis)
        messages, window = self._fit(build, tail)
        record = {"reask": k, "message": message, "response": None, "error": None,
                  "window": window}
        raw = None
        if _est_tokens(messages) > self.token_budget:
            record["error"] = "over the prompt budget — not asking again"
        else:
            try:
                raw = self.chat(messages, schema=_schema(self.max_candidates),
                                max_tokens=OUTPUT_MAX_TOKENS)
            except Exception as exc:  # noqa: BLE001 — the re-ask is a bonus
                record["error"] = f"{type(exc).__name__}: {exc}"
                log.warning("re-ask %d failed (%s) — closing the round", k, record["error"])
        record["response"] = raw
        taken = set(taken)
        renamed = []
        for i, c in enumerate((raw or {}).get("candidates") or []):
            if isinstance(c, dict):
                c = {**c, "id": _free_id(str(c.get("id") or f"reask{k}_{i}"), taken)}
                taken.add(c["id"])
            renamed.append(c)
        dropped: list[dict] = []
        candidates = sanitize_candidates(renamed, max_candidates=self.max_candidates,
                                         dropped=dropped,
                                         disabled_axes=self.disabled_axes,
                                         disabled_values=self.disabled_values,
                                         base=champion or None, facts=self.facts)
        stop = bool((raw or {}).get("stop")) and not candidates
        proposal = {"planner": "llm", "candidates": candidates, "stop": stop,
                    "dropped": dropped}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**proposal, "prompt": messages, **record},
                                   indent=2, ensure_ascii=False, default=str))
        return proposal

    def propose(self, history: list[dict], diagnosis: dict) -> dict:
        if self.artifact_path.exists():
            return self._replay(self.artifact_path)

        champion = (history[-1].get("champion", {}).get("axes")
                    if history else None) or {}
        build, carried = self._builder(history, diagnosis)
        messages, window = self._fit(build)
        if window < MAX_HISTORY:
            log.info("prompt budget (estimated %d tokens) — history window shrunk %d→%d rounds",
                     self.token_budget, MAX_HISTORY, window)
        raw, error = None, None
        try:
            raw = self.chat(messages, schema=_schema(self.max_candidates),
                            max_tokens=OUTPUT_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001 — falling back is the contract
            error = f"{type(exc).__name__}: {exc}"
            log.warning("LLM planner call failed (%s) — falling back to deterministic", error)

        dropped: list[dict] = []
        candidates = sanitize_candidates((raw or {}).get("candidates"),
                                         max_candidates=self.max_candidates,
                                         dropped=dropped,
                                         disabled_axes=self.disabled_axes,
                                         disabled_values=self.disabled_values,
                                         base=champion or None,
                                         facts=self.facts)
        # a candidate equal to a carried one does not take a slot — the runner
        # runs the carry-overs anyway, so candidates repeated against the
        # prompt's instruction are dropped and the freed slots asked for again
        carried_keys = {_axes_key(c["axes"]) for c in carried}
        candidates, dups = _drop_carried(candidates, carried_keys, dropped)
        followup = None
        if dups and raw is not None and not raw.get("stop"):
            followup = self._refill(build, raw, dups, candidates, champion,
                                    carried_keys, dropped,
                                    [c["id"] for c in carried])
        if dropped:
            log.info("sanitizer dropped %d candidates: %s", len(dropped),
                     [(c["id"], c["skipped"]) for c in dropped])
        # proposing only carried candidates is still proposing — not a stop
        # (same as before)
        stop = bool((raw or {}).get("stop")) and not candidates and not dups
        fallback = None
        # proposing only carried candidates makes an empty proposal correct (the
        # runner runs the carry-overs) — no fallback
        if raw is None or (not candidates and not (raw or {}).get("stop")
                           and not dups):
            # Call failed or every candidate invalid — fill this round deterministically
            det = deterministic.propose(history, diagnosis,
                                        max_candidates=self.max_candidates,
                                        disabled_axes=self.disabled_axes,
                                        disabled_values=self.disabled_values,
                                        facts=self.facts)
            candidates, stop = det["candidates"], det["stop"]
            fallback = "deterministic"

        proposal = {"planner": "llm", "candidates": candidates, "stop": stop,
                    "dropped": dropped}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(
            {**proposal, "prompt": messages, "response": raw,
             "fallback": fallback, "error": error, "followup": followup},
            indent=2, ensure_ascii=False, default=str))
        return proposal
