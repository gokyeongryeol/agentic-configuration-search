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
from .axes import (BASELINE, PLAN_DIFFERS_MAX, DatasetFacts, axis_values,
                   sanitize_candidates)
from .gates import INPUT_SIDE, champion_round, tiling_gate, tp_floor, under_champion
from .ledger import K_PARENT, axis_ledger, champion_requests, tried_under_champion
from .ledger import label as value_label

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
                    # property order is generation order — the expected gain,
                    # cost and variant note are written before the rationale.
                    # The parser keeps a candidate even if the three are missing
                    # or malformed (planner.axes._plan_fields)
                    "properties": {
                        "id": {"type": "string", "maxLength": 64},
                        "axes": {"type": "object"},
                        "expected_gain": {"type": "number"},
                        "cost_min": {"type": "number"},
                        "differs": {"type": "string", "maxLength": PLAN_DIFFERS_MAX},
                        # rationales run to about 1,700 characters at most
                        # (p99 about 1,050). Without a cap one rationale could
                        # run on until the response hit the output cap. Since
                        # the expected gain, cost and differs became structured
                        # it is cut to 1,200 so the longest schema response
                        # fits in OUTPUT_MAX_TOKENS
                        "rationale": {"type": "string", "maxLength": 1200},
                    },
                    "required": ["axes", "expected_gain", "cost_min", "differs",
                                 "rationale"],
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
# Recent rounds carried in detail (the budget guard may cut further). Older
# rounds fold into one-line summaries (_older_lines), and results under the
# current champion into the axis ledger (_ledger_block) — carrying every round
# as JSON lets history take a large share of the prompt
MAX_HISTORY = 3
MAX_OLDER_LINES = 24     # cap on old rounds carried as one-line summaries (most recent first)
MAX_LEDGER_ROWS = 24     # cap on rows of the ledger table
# Response cap — the schema bounds the response length (candidates ≤
# max_candidates, rationale ≤ RATIONALE_MAX_CHARS characters, id ≤ 64) and this
# is set above the longest English response (axes is a free object and is not
# bounded — see _schema). A truncated response breaks the JSON and the planner
# silently falls back to deterministic.
RATIONALE_MAX_CHARS = SCHEMA["schema"]["properties"]["candidates"]["items"][
    "properties"]["rationale"]["maxLength"]
OUTPUT_MAX_TOKENS = 4096
# Prompt token estimate — counted generously without the server tokenizer.
# ASCII counts at 1.85 chars/token (prose-heavy prompts tokenize at more chars
# per token, number-heavy JSON close to this), and other characters (Hangul,
# Han, ...) count 1.5 tokens each — a plain character budget can overflow the
# server context on CJK-heavy prompts. It is not a bound: text that is a long
# run of digits tokenizes below it, since the Qwen tokenizer splits digits one
# by one. Real prompts mix in prose, so the estimate is meant to land above the
# actual count.
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


def _digest_lines(diagnosis: dict, disabled=()) -> list[str]:
    """Diagnosis digest — counts per cause (train and holdout separately) and the
    axes that target the cause, most first.

    A static "cause → axis" table narrows nothing when there are several kinds
    of cause, since the union of remedies becomes every axis. The join is done
    here and the model only gets the counts that change each round as
    evidence. One line per cause, so a count is not repeated per axis (the old
    per-axis lines carried below_threshold 69 four times on four axes).

    Axis names come from REMEDY (the canonical source), so a nonexistent axis
    cannot appear. Axes turned off on this dataset (disabled) are left out —
    they cannot be proposed."""
    split_counts: dict[str, dict[str, int]] = {}
    for split, name in (("train", "train"), ("test", "held-out")):
        d = diagnosis.get(split) or {}
        for side in ("fn", "fp"):
            for kind, n in ((d.get("summary") or {}).get(side) or {}).items():
                if n:
                    split_counts.setdefault(kind, {})[name] = int(n)
    lines = []
    for kind, by in sorted(split_counts.items(),
                           key=lambda kv: (-sum(kv[1].values()), kv[0])):
        axes = list(dict.fromkeys(a for a, _v in deterministic.REMEDY.get(kind, ())
                                  if a not in disabled))
        where = ", ".join(f"{k} {v}" for k, v in by.items())
        lines.append(f"- {kind}: {sum(by.values())} ({where}) -> "
                     + (", ".join(axes) or "no axis (not addressable by a setting)"))
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


# rows dropped without being measured — they do not count as trying the axis
# (hygiene, combo dedup, not wirable)
_UNMEASURED = {"combo_tried", "no_change", "champion_combo", "axis_disabled",
               "max_candidates", "invalid_axes", "duplicate_axes",
               "carried_duplicate", "missing_hook", "lora_deps", "adapt_ctl",
               "detector_state_off", "compound_server_state"}

UNTRIED_RULE = ("While untried axes remain, prefer one of them over another variant "
                "of an already-tried value, unless the diagnosis gives a concrete "
                "reason for the variant.")


def _axes_touched(axes: dict, base: dict) -> set[str]:
    """Function axes changed from base — quantity axes (exemplar_k etc.) fold into their function axis."""
    return {K_PARENT.get(k, k) for k in _delta_axes(axes, base)}


def _untried_axis_lines(history: list[dict], values: dict, disabled=()) -> list[str]:
    """Axes never measured under the current champion — with few candidates per
    round the order is the budget, so this keeps the planner from repeating
    variants and reaching other axes late (guidance only, nothing is blocked).

    Targets are function axes that are not disabled (search.disabled_axes) and
    still have a value to turn on besides the champion's — axes the data blocks
    (values axis_values left out), boolean axes the champion already turned on,
    and visual_negatives while the champion has no exemplar (a layer on top of
    exemplars) are left out. A try is a candidate row that changed the axis (by
    request; quantity axes count for their function axis); rows dropped without
    being measured (_UNMEASURED) do not count. Axes never proposed in the whole
    run are marked separately (each round against the champion it started
    with)."""
    if not history:
        return []
    champ = (history[-1].get("champion") or {}).get("axes") or {}
    open_axes = [k for k, vals in values.items()
                 if k in BASELINE and k not in K_PARENT and k not in disabled
                 and any(v != BASELINE[k] and v != champ.get(k) for v in vals)
                 # visual negatives are a layer on exemplars — cannot be turned on alone
                 and not (k == "visual_negatives"
                          and champ.get("exemplar", "none") == "none")]
    if not open_axes:
        return []
    recent: set[str] = set()
    for e in under_champion(history):
        for c in e.get("candidates", []):
            if c.get("skipped") not in _UNMEASURED:
                recent |= _axes_touched(c.get("requested_axes") or c.get("axes"), champ)
    ever = {k for k, v in champ.items() if v != BASELINE.get(k, v)}
    for i, e in enumerate(history):
        start = ((history[i - 1].get("champion") or {}).get("axes") if i > 0
                 else None) or (e.get("champion") or {}).get("axes") or {}
        ever |= {k for k, v in start.items() if v != BASELINE.get(k, v)}
        for c in e.get("candidates", []):
            if c.get("skipped") not in _UNMEASURED:
                ever |= _axes_touched(c.get("requested_axes") or c.get("axes"), start)
    untried = [k for k in open_axes if k not in recent]
    if not untried:
        return []
    never = [k for k in untried if k not in ever]
    return [f"- Untried axes under the current champion: {', '.join(untried)}"
            + (f" (never tried in this run: {', '.join(never)})" if never else "")
            + f". {UNTRIED_RULE}"]


def _on_axes(axes: dict) -> dict:
    """Only axes that are on — off axes (False/None/"none") are safe points and carry no information."""
    return {k: v for k, v in (axes or {}).items()
            if v not in (False, None, "none")}


def _delta_axes(axes: dict, base: dict) -> dict:
    """Delta vs. the champion — a candidate is identified by its delta, not its full axes."""
    return {k: v for k, v in (axes or {}).items() if (base or {}).get(k) != v}


MAX_TRIED = 40           # cap on tried combos in the prompt (most recent first)


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
        cmp = c.get("fit_dev_cmp")
        if cmp and cmp.get("ub") is not None:
            # the note is cut at MAX_NOTE — the prescreen verdict's numbers go separately
            line += f", fit_dev delta {cmp['delta']:+.4f} (ub {cmp['ub']:+.4f})"
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
                                             champ_axes)}
            # empty values are left out — every row used to carry "adopted": null, "skipped": null
            if c.get("adopted") is not None:
                row["adopted"] = c["adopted"]
            if c.get("skipped"):
                row["skipped"] = c["skipped"]
            if c.get("fit_dev_micro_f_beta") is not None:
                row["fit_dev_micro_f_beta"] = c["fit_dev_micro_f_beta"]
            cmp = c.get("fit_dev_cmp") or (c.get("prescreen") or {}).get("fit_dev_cmp")
            if cmp:
                # (candidate − champion) fit_dev micro F and its bootstrap upper bound — how badly it lost
                row["fit_dev_cmp"] = {k: cmp[k] for k in ("delta", "ub") if k in cmp}
            if c.get("lost_values"):
                row["lost_values"] = c["lost_values"]   # it was a variant of a clearly-lost value
            if (c.get("plan") or {}).get("expected_gain") is not None:
                # the planner's expected gain — lets it calibrate against the actual gate_delta
                row["expected_gain"] = c["plan"]["expected_gain"]
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


# Cost of axes not observed yet — minutes differ per dataset, so only relative
# size is given
STATIC_COST = {
    "tiling": "tiling=g: every detector pass costs 1+g^2 forwards per image "
              "(champion: {forwards})",
    # Facts only — no value judgment ("the slowest axis") or time estimate for an
    # axis not observed yet. Durations differ per dataset; once observed, the
    # measured minutes above (_cost_lines) take over
    "lora": "lora: fine-tunes the detector tail before screening",
    "crop_classifier": "crop_classifier: trains a crop classifier once per run, "
                       "then screens a grid of blend weights per proposal",
    "rescorer": "rescorer: re-infers fit_train and fit_dev to refit the rescorer",
}


def _budget_lines(budget: dict | None, history: list[dict], disabled=()) -> list[str]:
    """Budget card — rounds left, stop counters, per-axis cost (observed median,
    otherwise a static note).

    The counters are what the runner counted from the tail of history (the same
    way as run_rounds' stop checks)."""
    b = budget or {}
    lines = []
    r, total = b.get("round"), b.get("rounds")
    if r is not None and total:
        left = max(int(total) - int(r) + 1, 1)
        lines.append(f"- Round {r} of {total}: {left} round"
                     f"{'s' if left != 1 else ''} left including this one.")
    if b.get("saturation_rounds"):
        lines.append(f"- Rounds in a row without adoption: {b.get('stagnant', 0)}; "
                     f"the run stops at {b['saturation_rounds']}.")
    if b.get("empty_rounds"):
        lines.append(f"- Rounds in a row that measured nothing on the held-out split "
                     f"(a round where a candidate was dropped for losing on fit_dev "
                     f"does not count): {b.get('empty', 0)}; the run stops at "
                     f"{b['empty_rounds']}.")
    cost = _cost_lines(history)
    if cost:
        lines.append("- Measured wall-clock per candidate in this run (materialize + "
                     "fit_dev screen + held-out eval), by the axes it changed from the "
                     "champion of its round:")
        lines += ["  " + c for c in cost]
    measured = {ln[2:].split(":")[0].split("=")[0] for ln in cost}
    forwards = ((history[-1].get("champion") or {}).get("forwards_per_image")
                if history else None) or 1
    static = [text.format(forwards=forwards) for axis, text in STATIC_COST.items()
              if axis not in measured and axis not in disabled]
    if static:
        lines.append("- Not measured yet: " + "; ".join(static) + ".")
    return lines


def _champion_block(history: list[dict]) -> str:
    """The current champion in one line — axes on, holdout and fit_dev micro F,
    adoption round, serving cost."""
    ch = (history[-1].get("champion") or {}) if history else {}
    parts = [json.dumps(_on_axes(ch.get("axes")), ensure_ascii=False)
             + " (only axes that are ON; {} = the baseline, every axis at its safe point)"]
    nums = []
    if ch.get("micro_f_beta") is not None:
        nums.append(f"held-out micro F {ch['micro_f_beta']:.4f}")
    if ch.get("fit_dev_micro_f_beta") is not None:
        nums.append(f"fit_dev micro F {ch['fit_dev_micro_f_beta']:.4f}")
    if ch.get("round") is not None:
        nums.append(f"adopted in round {ch['round']} ({ch.get('id')})")
    if ch.get("forwards_per_image"):
        nums.append(f"{ch['forwards_per_image']} detector passes per image")
    if nums:
        parts.append("; ".join(nums))
    return "\n".join(parts)


_STATUS_RANK = {"clearly-lost (prescreened)": 0, "clearly-lost": 1, "near-miss": 2,
                "folded": 3}


def _fmt_delta(d: dict | None, bound: str) -> str:
    if not d or d.get("delta") is None:
        return "-"
    b = d.get(bound)
    return f"{d['delta']:+.4f}" + (f" ({b:+.4f})" if b is not None else "")


def _ledger_block(history: list[dict]) -> str:
    """The axis ledger table (under the current champion, planner.ledger) and one
    line of combinations tried under it."""
    led = axis_ledger(history)
    cr = champion_round(history)
    champ = ((history[-1].get("champion") or {}).get("axes") or {}) if history else {}
    head = (f"Rounds after round {cr} (when the current champion was adopted); "
            "resets when the champion changes. fit_dev delta is (candidate - "
            "champion) micro F with its bootstrap upper bound ub; held-out delta is "
            "the gate delta with its lower bound lo (shown, never used to "
            "prescreen). Minutes are summed over candidates containing the value."
            if cr is not None else "")
    lines = [head] if head else []
    if not led:
        lines.append("- (nothing tried under this champion yet)")
    else:
        lines.append("value | tries (alone) | best fit_dev delta (ub) | "
                     "best held-out delta (lo) | status | minutes")

        def status(rec):
            if champ.get(rec.key[0]) == rec.key[1]:
                return "champion value (k variants)"
            return rec.status
        rows = sorted(led.values(), key=lambda r: (_STATUS_RANK.get(status(r), 9),
                                                   -r.tries, value_label(r.key)))
        for rec in rows[:MAX_LEDGER_ROWS]:
            lines.append(f"{value_label(rec.key)} | {rec.tries} ({rec.single}) | "
                         f"{_fmt_delta(rec.best_fit_dev, 'ub')} | "
                         f"{_fmt_delta(rec.best_holdout, 'lo')} | {status(rec)} | "
                         f"{rec.minutes:.0f}")
        if len(rows) > MAX_LEDGER_ROWS:
            lines.append(f"... and {len(rows) - MAX_LEDGER_ROWS} more values")
    tried = tried_under_champion(history)[-MAX_TRIED:]
    # the current champion's own combination (empty increment) comes first —
    # stating it is not a candidate. If the request that adopted the champion
    # differs from its label (unused-axis correction, rescorer identity refit),
    # that is the champion too — stated so it is not read as a new combination
    asked = [", ".join(f"{k}={v}" for k, v in sorted(_delta_axes(a, champ).items()))
             for a in champion_requests(history)]
    itself = [CHAMPION_REQUEST.format(f"{{{a}}}") for a in dict.fromkeys(asked) if a]
    lines.append("Combos already tried, as differences from the champion (an exact "
                 "repeat is dropped as skipped=\"combo_tried\"): "
                 + "; ".join([CHAMPION_COMBO] + itself
                             + [f"{{{t}}}" for t in tried if t not in asked]))
    return "\n".join(lines)


# first item of the tried-combination list — the current champion itself.
# Re-proposing it has no increment and is dropped
CHAMPION_COMBO = ("{} = the current champion itself (not a candidate: re-proposing "
                  "it changes nothing and is dropped as skipped=\"no_change\")")
# when the request that adopted the champion differs from its label — that
# request is the champion too
CHAMPION_REQUEST = ("{} = the request that adopted the current champion (an axis it "
                    "asked for was unused or came out identity, so it IS the champion; "
                    "dropped as skipped=\"champion_combo\")")


def _allowed_lines(history: list[dict], *, prescreen: dict | None, dead: list[str],
                   folded: list[str], carried: list[str],
                   untried: list[str] = ()) -> list[str]:
    """What may be proposed this round — what is blocked (tiling fold, folded
    axes, exhausted material, carried candidates) and which values are
    prescreened. Disabled axes are already stated by the axis list (axes_doc)."""
    lines = list(folded)
    if dead:
        lines.append(DEAD_BLOCK.strip("\n").format(dead="\n".join(dead)))
    lost = [r for r in axis_ledger(history).values() if r.prescreened]
    if lost:
        ev = "; ".join(
            f"{value_label(r.key)} (round {r.evidence['round']} {r.evidence['id']}: fit_dev "
            f"delta {r.evidence['delta']:+.4f}, ub {r.evidence['ub']:+.4f})"
            for r in lost if r.evidence)
        p = prescreen or {}
        if p.get("on", True):
            where = (f"{p['n_images']} of {p['of']} fit_dev images" if p.get("of")
                     else "a fixed subset of fit_dev")
            lines.append(
                f"- Prescreened values (clearly lost under this champion): {ev}. A "
                f"candidate containing one is first screened on {where} and dropped as "
                "skipped=\"lost_value_prescreen\" unless the upper bound of "
                "(candidate - champion) fit_dev micro F is >= 0 there. Say in "
                "\"differs\" what is different this time.")
        else:
            lines.append(f"- Clearly lost under this champion (not prescreened on "
                         f"this run): {ev}.")
    if carried:
        lines.append(CARRY_BLOCK.strip("\n").format(
            carried="\n".join(f"- {x}" for x in carried)))
    return (lines or ["- No blocked axes or prescreened values right now."]) \
        + list(untried)


def _older_lines(history: list[dict], window: int) -> list[str]:
    """Rounds outside the detailed window — one line per round (the adopted
    increment and Δ, or no adoption and the number of candidates)."""
    end = max(len(history) - window, 0) if window > 0 else len(history)
    lines = []
    for i in range(end):
        e = history[i]
        d = e.get("decision") or {}
        ch = e.get("champion") or {}
        cands = [c for c in e.get("candidates", [])]
        if e.get("round", 0) == 0 or d.get("kind") == "baseline":
            lines.append(f"R{e.get('round', 0)}: baseline, held-out micro F "
                         f"{ch.get('micro_f_beta', 0.0):.4f}")
            continue
        evaluated = d.get("evaluated") or {}
        start = ((history[i - 1].get("champion") or {}).get("axes") if i > 0
                 else None) or ch.get("axes") or {}
        if d.get("adopted"):
            row = next((c for c in cands if c.get("id") == d.get("id")), {})
            delta = _delta_axes(row.get("requested_axes") or row.get("axes"), start)
            g = (evaluated.get(d.get("id")) or {}).get("gate") or {}
            lines.append(
                f"R{e['round']}: adopted {d.get('id')} ("
                + (", ".join(f"{k}={v}" for k, v in sorted(delta.items())) or "-")
                + f") -> held-out micro F {ch.get('micro_f_beta', 0.0):.4f}"
                + (f", gate delta {g['delta']:+.4f}" if g.get("delta") is not None else "")
                + f"; {len({c.get('id') for c in cands})} candidates")
        else:
            gates = [(v.get("gate") or {}).get("delta") for v in evaluated.values()]
            gates = [x for x in gates if x is not None]
            lines.append(f"R{e['round']}: no adoption; "
                         f"{len({c.get('id') for c in cands})} candidates, "
                         f"{len(evaluated)} evaluated"
                         + (f" (best gate delta {max(gates):+.4f})" if gates else ""))
    return lines[-MAX_OLDER_LINES:]


PROMPT = """A zero-shot detector is being adapted to one dataset by searching
inference settings. **Objective: maximize micro F(beta) on a held-out
validation split (the holdout; fields and keys named "test" refer to it).**
Each candidate changes axes of the current champion:
{axes_doc}

Rules:
- Change as FEW axes as possible per candidate (single increments attribute
  gains cleanly). List only the axes you change: omitted axes keep the
  champion's value; to turn an axis off, set it to false explicitly.
  "visual_negatives" requires exemplar != "none".
- Materials are mined from the train split only. Each candidate is screened on
  fit_dev (a slice of train) and only the best go to the held-out gate. The
  gate adopts only if a paired-bootstrap LOWER BOUND on the holdout delta
  clears a margin AND the delta is positive, so a candidate can score higher
  on the holdout and still be rejected as noise: gate_delta > 0 with
  gate_ci_lo < 0 means "close, but unproven"; a clearly negative gate_delta is
  not worth repeating.
- fit_dev_cmp compares a candidate with the champion on fit_dev: delta =
  (candidate - champion) micro F, ub = its paired-bootstrap upper bound. ub < 0
  means it clearly lost on fit_dev; ub >= 0 is a near miss (lost within noise,
  or won). A value whose single-increment candidate clearly lost under this
  champion is "clearly-lost": its variants (another k, or the value combined
  with other axes) are not blocked but prescreened (see Allowed moves), so
  propose one only with a concrete reason it should behave differently.
- "tiling" is a grid g: every image is also cut into g x g tiles and each tile
  is detected at magnification (g-1)(1-overlap)+1 (1.5x at g=2, 2x at g=3 with
  overlap 0.5). Cost: 1 + g^2 detector passes per image (5 at g=2, 10 at g=3,
  17 at g=4, 26 at g=5, 37 at g=6). Pick the first grid from the missed-small
  sizes below: the grid whose magnification brings the typical missed object
  to a detectable size, no larger. Raising g further is worth it only when the
  previous grid was used and helped (tiling_used=true and its fit_dev score
  beat the champion's) while small-object misses remain. tiling_f compares the
  mean per-class fit_dev F of the best tiled vs untiled config;
  tiling_f.gap_ub is a paired-bootstrap upper bound on (tiled - untiled),
  gap_ub >= 0 meaning the loss is within noise. tiling_used=false means the
  screen preferred no tiling at that grid: fold the axis. The loop enforces
  the fold: under the current champion, a grid at or above an unused one is
  dropped as skipped="tiling_blocked" without being screened. A note starting
  "prescreen" means a candidate was first screened on a subset of fit_dev and
  dropped there. One exception to the fold: when the unused grid g was such a
  near miss AND fell short on magnification (its magnified missed-small median
  stayed below the smallest objects the champion already detects, while g+1
  reaches them), grid g+1 is unblocked exactly once under the current
  champion — Allowed moves says when.
- unused_axes lists axes a candidate had on that the screened setting did not
  use (exemplar: the text pass beat the semantic pass for every class;
  visual_negatives: no semantic class kept a negative config); other variants
  of such an axis rarely change the setting while that stays true. A row's
  modality counts the classes where the text pass beat the semantic (exemplar)
  pass on fit_dev F (text_won) and the rest (semantic_won);
  closest_text_margin is the smallest F gap among the classes text won — small
  means another exemplar_k or fn_arm may flip a class, large means exemplar
  variants are unlikely to win.
- A visual_negatives row carries neg_eta_used (the suppression IoU thresholds
  the screen picked; lower means stronger suppression) and neg_eta_grid (the
  thresholds tried): picks at the low end of the grid mean more negatives
  (raise visual_negative_k) may help; picks at the high end mean fold the axis
  rather than raising k.
- Once "lora" or "crop_classifier" is on in the champion, keep it on: they are
  detector state, so a candidate turning one off is rejected as
  skipped="detector_state_off" without evaluation.
- skipped="combo_tried": that exact combination was already tried — never
  repeat it verbatim. skipped="carried_duplicate": it repeated a re-queued
  candidate (handled by the carry_<id> row). skipped="same_setting": the
  screened setting served exactly what the champion or an evaluated candidate
  serves; its material field shows what (e.g. text negatives that a larger k
  refined to the same phrases) — another value of that axis will likely
  collapse the same way. skipped="fit_dev_lost" / "fit_dev_identical": it was
  screened but clearly lost on fit_dev (ub < 0) / matched the champion image
  for image there, so it never reached the held-out gate.
- If nothing promising remains, return stop=true with no candidates.

## Budget
{budget}

## Current champion
{champion}

## Axis ledger under the current champion
{ledger}

## Allowed moves this round
{allowed}

## Round history
Recent rounds in detail (axes_delta is the change from the champion at the
start of that round; test_micro_f_beta / gate_delta / gate_ci_lo are holdout
results; expected_gain is what the planner predicted):
{history}
{older}
## Diagnosis digest
Error kinds this round (train + holdout) and the axes that can address them.
HEURISTIC hints, not rules: a count says how much is at stake if the axis
works, not that it will — weigh it against per-class counts and the samples.
{digest}
{small_block}
Error detail. "train" carries individual error samples; "test" (the holdout)
is aggregate-only. In a sample, "wh" is the box size in pixels and "note" is a
vision model's one-line observation of the crop — evidence, not ground truth.
**Samples are a coverage selection, not a random draw** (round-robin over
kinds and classes, most severe first): read frequency from the digest and
"per_class" only, and the samples for what the errors look like.
{diagnosis}
{notes_block}
## Response
Propose up to {max_candidates} candidates. For each give "axes" (only the
changes), "expected_gain" (your estimate of the holdout micro F delta over
the champion, e.g. 0.01), "cost_min" (expected minutes, from the budget),
"differs" (for a candidate containing a clearly-lost value: what is different
this time; otherwise "") and a short "rationale"."""

# Inserted only when train has missed_small — evidence for the first tiling grid.
SMALL_BLOCK = """
Missed-small sizes (train, {n} objects) — box side at the detector input
(sqrt of box/image area x {side} px, long side resized to {side}): p25 {p25} px,
median {p50} px, p75 {p75} px. After tiling magnification (overlap {overlap}):
{grids}
"""

# Carried in the allowed moves only when some axis had empty material.
DEAD_BLOCK = """
Axes whose mined material came out empty, or too small to train on, on this
dataset (the candidate was rejected without evaluation, so it cost a slot for
nothing). Their material pool is exhausted here — re-propose one only with a
different source value, or after the champion has changed:
{dead}
"""

# Carried in the allowed moves only when some candidate is carried over.
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
the same failure: re-proposing the current champion (skipped="champion_combo")
changes nothing, a folded tiling grid is blocked, an axis the screen left unused
stays unused for other values of it, a quantity whose material collapsed to
the same setting collapses again, and a variant of a clearly-lost value that the
prescreen dropped (lost_value_prescreen) needs a real difference to pass. If
nothing promising remains, return stop=true with no candidates."""

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
                 tiling_tp_quantile: float = 0.1, tiling_tp_min: int = 20,
                 budget: dict | None = None, prescreen: dict | None = None):
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
        # The budget card (rounds left and stop counters — counted by the runner)
        # and the lost-value prescreen settings (on, subset size). Without them
        # only those lines are left out
        self.budget = budget
        self.prescreen = prescreen

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
        # Counts per cause are carried by the digest — dropped from the JSON so
        # they are not carried twice
        for d in compact.values():
            if isinstance(d, dict):
                d.pop("summary", None)
        diag_full = json.dumps(compact, ensure_ascii=False, default=str)
        digest = _digest_lines(diagnosis, self.disabled_axes)
        carried = deterministic.carryover(history, self.disabled_axes,
                                          self.disabled_values, facts=self.facts)
        carry_lines = [", ".join(f"{k}={v}" for k, v in
                                 sorted(_delta_axes(c["axes"], champion).items()))
                       for c in carried]
        allowed = _allowed_lines(history, prescreen=self.prescreen,
                                 dead=_dead_axis_lines(history),
                                 folded=_folded_axis_lines(history),
                                 carried=carry_lines,
                                 untried=_untried_axis_lines(
                                     history,
                                     {k: [x for x in v
                                          if x not in self.disabled_values.get(k, ())]
                                      for k, v in axis_values(self.facts).items()},
                                     self.disabled_axes))
        budget = _budget_lines(self.budget, history, self.disabled_axes)
        champion_text = _champion_block(history)
        ledger = _ledger_block(history)
        small = _small_block(diagnosis, self.tiling_overlap,
                             tp_quantile=self.tiling_tp_quantile,
                             tp_min=self.tiling_tp_min)
        if banned:
            axes_doc += "\n" + DISABLED_BLOCK.strip("\n").format(
                disabled=", ".join(banned))

        def build(window: int, strip: bool = False) -> list[dict]:
            diag_json = (json.dumps(_strip_samples(compact), ensure_ascii=False,
                                    default=str) if strip else diag_full)
            older = _older_lines(history, window)
            return [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(
                    axes_doc=axes_doc,
                    budget="\n".join(budget) or "- (no budget information)",
                    champion=champion_text,
                    ledger=ledger,
                    allowed="\n".join(allowed),
                    history=json.dumps(_compact_history(history, window),
                                       ensure_ascii=False, default=str),
                    older=("Earlier rounds, one line each:\n" + "\n".join(older) + "\n"
                           if older else ""),
                    digest="\n".join(digest) or "- (no errors observed)",
                    small_block=small,
                    diagnosis=diag_json,
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
