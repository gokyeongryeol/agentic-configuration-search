"""LLM planner — the model served by vLLM reads the diagnosis and proposes the
next candidates.

Why non-determinism is harmless: the LLM **only proposes**; adoption is
decided by the holdout F(β) CI gate. Reproducibility comes from **replay** —
prompt, response, and candidates are saved to `proposal.json`, and rebuilding
with the same artifact_dir (= resume) replays them without calling the LLM.
The resume key is this artifact, not a config hash (guard: tests/test_replay.py).

Failed or invalid responses fall back to the deterministic planner — a planner
failure never kills the loop (the gate protects against poor proposals).

The prompt is English only. Candidate rows' `note` is a sentence for people and
is not carried — a skip reason reaches the prompt as an English `reason` built
from its code and structured fields (_REASON).
"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path

from ..config import GateCfg
from ..core.metrics import FP_KINDS
from ..diagnose.sample import pick_detail
from . import deterministic
from .axes import PLAN_DIFFERS_MAX, DatasetFacts, axis_values, sanitize_candidates
from .axis_status import NOT_RETRIED, axis_status
from .board import board_lines
from .gates import INPUT_SIDE, tiling_gate, tp_floor
from .ledger import axis_ledger, champion_requests, kv, tried_under_champion

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
            # no stop field — only saturation, empty_rounds and the rounds cap stop
            # a run. At least one candidate (constrained decoding rules out an empty proposal)
            "candidates": {
                "type": "array",
                "minItems": 1,
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
                        # Without a cap one rationale can run on until the
                        # response hits the output cap. The expected gain,
                        # cost and differs are structured separately, so 1,200
                        # is enough and the longest schema response fits in
                        # OUTPUT_MAX_TOKENS
                        "rationale": {"type": "string", "maxLength": 1200},
                    },
                    "required": ["axes", "expected_gain", "cost_min", "differs",
                                 "rationale"],
                },
            },
        },
        "required": ["candidates"],
    },
}

# Caps on diagnosis detail and history in the prompt — the first defense against
# overflowing the server context. The last is the budget guard that measures the
# assembled prompt (DEFAULT_TOKEN_BUDGET).
# Recent rounds carried in detail (the budget guard may cut further). Older
# rounds fold into one-line summaries (_older_lines), and results under the
# current champion into the champion column of the axis board (planner.board)
MAX_HISTORY = 3
MAX_OLDER_LINES = 24     # cap on old rounds carried as one-line summaries (most recent first)
# Response cap — the schema bounds the response length (candidates ≤
# max_candidates, rationale ≤ RATIONALE_MAX_CHARS characters, id ≤ 64) and this
# is set above the longest English response (axes is a free object and is not
# bounded — see _schema). A truncated response breaks the JSON and falls back to
# deterministic (an ERROR log and run_summary.planner_fallbacks).
RATIONALE_MAX_CHARS = SCHEMA["schema"]["properties"]["candidates"]["items"][
    "properties"]["rationale"]["maxLength"]
OUTPUT_MAX_TOKENS = 4096
# Prompt token estimate — counted generously without the server tokenizer.
# ASCII counts at 1.85 chars/token, and other characters (Hangul, Han, ...)
# count 1.5 tokens each — a plain character budget can overflow the server
# context on CJK-heavy prompts. It is not a bound: text that is a long run of
# digits tokenizes below it, since the Qwen tokenizer splits digits one by one.
# Real prompts mix in prose, so the estimate lands above the actual count.
ASCII_CHARS_PER_TOKEN = 1.85
NON_ASCII_TOKENS = 1.5
CONTEXT_MARGIN = 256      # chat template tokens etc.
# Prompt token budget — overflow is fitted by the budget ladder (BUDGET_LADDER:
# history window, diagnosis samples) (the same formula as runner._token_budget —
# from the attached server's context). If even the last step is over, it is not
# sent to the server and falls back with the reason (PromptOverBudget) — kept in
# run_summary.
DEFAULT_TOKEN_BUDGET = 32768 - OUTPUT_MAX_TOKENS - CONTEXT_MARGIN
NDIGITS = 4              # float digits in the prompt — at min_delta scale (0.005) the
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
            # the planner cannot point at an image.
            r.pop("image", None)


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
        # the stored key "test" is legacy — the prompt carries it as held_out
        out["held_out" if split == "test" else split] = c
    return _round_floats(out)


def _cell(v) -> str:
    """One table cell — line breaks and runs of spaces become one space, the column separator (|) becomes /."""
    return " ".join(str(v).split()).replace("|", "/")


def _kinds_cell(kinds: dict | None) -> str:
    rows = sorted(((k, n) for k, n in (kinds or {}).items() if n), key=lambda kv: -kv[1])
    return ", ".join(f"{k} {n}" for k, n in rows) or "-"


def _is_fp(r: dict) -> bool:
    return r.get("score") is not None or r.get("kind") in FP_KINDS


def _sample_row(label: str, r: dict) -> str:
    """One sample row — the detection score for an FP, the size (sqrt(box/image))
    for an FN. Unmatched crop-note rows tell FP from FN by their fields."""
    if _is_fp(r):
        what = f"score {r['score']:.3f}" if r.get("score") is not None else "-"
    else:
        what = f"size {r['size_ratio']:.2f}" if r.get("size_ratio") is not None else "-"
    wh = r.get("wh")
    cls = r.get("category") or (f"id {r['category_id']}" if r.get("category_id") is not None
                                else "-")
    return (f"{label} | {_cell(r.get('kind') or '-')} | {_cell(cls)} | {what} | "
            + (f"{wh[0]}x{wh[1]}" if wh else "-") + f" | {_cell(r.get('note') or '-')}")


def _diagnosis_text(compact: dict) -> str:
    """The diagnosis detail as tables — per-class aggregates (per split) and train
    error samples. Column names appear once. Each split's FP and FN totals go in
    the Per class header, so they stay without samples. With nothing to carry, one
    line says so."""
    if not any(isinstance(d, dict) and (d.get("per_class") or d.get("fp_sample")
                                        or d.get("fn_sample") or d.get("crop_notes"))
               for d in compact.values()):
        return "- (no error detail)"
    lines = []
    sizes = []
    for split, d in compact.items():
        if not isinstance(d, dict):
            continue
        part = [f"{d['n_images']} images"] if d.get("n_images") is not None else []
        part += [f"{d[f'{side}_total']} {side.upper()}" for side in ("fp", "fn")
                 if d.get(f"{side}_total")]
        if part:
            sizes.append(f"{split} " + ", ".join(part))
    lines.append("Per class" + (f" ({'; '.join(sizes)})" if sizes else "") + ": gt = "
                 "ground-truth boxes, miss = miss rate, FN and FP counts by kind.")
    lines.append("split | class | gt | miss | FN | FP")
    for split, d in compact.items():
        for name, pc in sorted(((d or {}).get("per_class") or {}).items()):
            miss = pc.get("miss_rate")
            lines.append(f"{split} | {_cell(name)} | {pc.get('gt', '-')} | "
                         + (f"{miss:.3f}" if miss is not None else "-")
                         + f" | {_kinds_cell(pc.get('fn'))} | {_kinds_cell(pc.get('fp'))}")
    for split, d in compact.items():
        rows = [("FP", r) for r in (d or {}).get("fp_sample", [])] \
            + [("FN", r) for r in (d or {}).get("fn_sample", [])]
        # crop notes without a matching sample — not counted in the sample count (x of N)
        rows += [("FP-note" if _is_fp(r) else "FN-note", r)
                 for r in (d or {}).get("crop_notes", [])]
        if not rows:
            continue
        shown = ", ".join(f"{sum(1 for lab, _ in rows if lab == side.upper())} of "
                          f"{d[f'{side}_total']} {side.upper()}"
                          for side in ("fp", "fn") if d.get(f"{side}_total"))
        notes = any(lab.endswith("-note") for lab, _ in rows)
        lines.append(f"Error samples ({split}" + (f"; {shown}" if shown else "")
                     + "): score = detector score, size = sqrt(box/image area), wh = box "
                     "size in pixels"
                     + ("; *-note rows are crop observations without a matching sample"
                        if notes else "") + ".")
        lines.append("side | kind | class | score/size | wh | note")
        lines += [_sample_row(label, r) for label, r in rows]
    return "\n".join(lines)


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
                 tp_min: int = 20, grids_open=None) -> str:
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
    classified at tau_test). With grids_open only those grids are listed (the
    grids proposable now — the same verdict as the board).

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
        for g in TILING_GRIDS if grids_open is None or g in grids_open)
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


def _remedy_label(kind: str, values: dict | None, disabled=(),
                  champion: dict | None = None, off=(), empty=(), gate=None) -> list[str]:
    """The axes (down to values) that target one cause — a boolean as the axis
    name, tiling as the axis name only, anything else as axis=v1|v2. Only what
    can be proposed now: values the dataset blocks and disabled axes, values the
    champion already uses, axes whose status is OFF (off) and (axis, value) whose
    material came out empty under the current champion (empty) are left out.
    Without an exemplar in the champion visual_negatives is left out too — it
    cannot be turned on alone, and carrying it gets it proposed as is."""
    champion = champion or {}
    no_exemplar = champion.get("exemplar", "none") == "none"
    picked: dict[str, list] = {}
    for axis, v in deterministic.REMEDY.get(kind, ()):
        if axis in disabled or axis in off or champion.get(axis) == v \
                or (axis, v) in empty \
                or (values is not None and v not in values.get(axis, (v,))):
            continue
        if axis == "visual_negatives" and no_exemplar:
            continue
        # only tiling grids larger than the champion's are a remedy (step-up), and folded grids are dropped
        if axis == "tiling" and ((champion.get("tiling") and v <= champion["tiling"])
                                 or (gate is not None and gate.block({"tiling": v}))):
            continue
        vals = picked.setdefault(axis, [])
        if v not in vals:
            vals.append(v)
    return [a if a == "tiling" or vals == [True] else f"{a}={'|'.join(map(str, vals))}"
            for a, vals in picked.items()]


# a cause with no remedy at all, apart from one whose remedies cannot be proposed now
NO_REMEDY = "no axis (not addressable by a setting)"
NONE_NOW = ("no axis to propose now (its axes are on in the champion, OFF, blocked or "
            "out of material)")


def _digest_lines(diagnosis: dict, disabled=(), values: dict | None = None,
                  champion: dict | None = None, off=(), empty=(), gate=None) -> list[str]:
    """Diagnosis digest — counts per cause (train and holdout separately) and the
    axes that target the cause, most first.

    A static "cause → axis" table narrows nothing when there are several kinds
    of cause, since the union of remedies becomes every axis. The join is done
    here and the model only gets the counts that change each round as evidence.
    One line per cause, so a count is not repeated per axis.

    Axes and values come from REMEDY (the canonical source), so a nonexistent axis
    cannot appear. Axes turned off on this dataset (disabled), blocked values (not
    in values), values the champion already uses and OFF axes (off,
    planner.axis_status) are left out — they cannot be proposed or are dropped if
    proposed. (axis, value) with empty material under the current champion
    (empty) and tiling grids at or below the champion's or folded (gate,
    planner.gates) are left out too. Without an exemplar in the champion
    visual_negatives is left out (a layer on exemplars). When a cause has
    remedies but all are left out it says "nothing to propose now" (NONE_NOW);
    only a cause with no remedy at all is NO_REMEDY."""
    split_counts: dict[str, dict[str, int]] = {}
    for split, name in (("train", "train"), ("test", "held_out")):
        d = diagnosis.get(split) or {}
        for side in ("fn", "fp"):
            for kind, n in ((d.get("summary") or {}).get(side) or {}).items():
                if n:
                    split_counts.setdefault(kind, {})[name] = int(n)
    lines = []
    for kind, by in sorted(split_counts.items(),
                           key=lambda kv: (-sum(kv[1].values()), kv[0])):
        axes = _remedy_label(kind, values, disabled, champion, off, empty, gate)
        where = ", ".join(f"{k} {v}" for k, v in by.items())
        none = NONE_NOW if deterministic.REMEDY.get(kind) else NO_REMEDY
        lines.append(f"- {kind}: {sum(by.values())} ({where}) -> "
                     + (", ".join(axes) or none))
    return lines


def _retire_line(status) -> str:
    """The axis retirement rule in one line — how the board's OFF and PROBE are decided."""
    if not status.retire_after:
        return ("The run column is a record only on this run: nothing is turned OFF "
                "automatically.")
    return (f"An axis turns OFF after {status.retire_after} rounds in a row where it was "
            "measured and never won (each round counts once per axis: held-out gate "
            "failed, clearly lost or identical on fit_dev, served the champion's setting, "
            "unused by the screened setting, or its own arm gate — lora, crop_classifier, "
            "rescorer — failed; empty material (including an arm not trained for lack of "
            "data) and a setting equal to a sibling's or an earlier candidate's do not "
            "count). When the champion changes, OFF axes reopen as PROBE.")


def _constraint_lines(disabled, status, champion: dict | None = None) -> list[str]:
    """Conditions whose breach drops a candidate without evaluation — up front, so
    they are not buried in the long rules. The tiling fold and per-axis
    circumstances are on the axis board (planner.board)."""
    # values that are blocked themselves (gt_crop on a single class etc.) are already
    # left out of the board's propose column
    state = [a for a in ("lora", "crop_classifier") if a not in disabled]
    no_ex = (champion or {}).get("exemplar", "none") == "none"
    lines = [('- visual_negatives needs exemplar != "none": set an exemplar value in the '
              'same candidate (allowed while exemplar is OFF unless visual_negatives is OFF '
              'too).' if no_ex else '- visual_negatives needs exemplar != "none".')
             if "visual_negatives" not in disabled else None,
             (f"- Once {' or '.join(state)} is on in the champion, keep it on (detector "
              "state; turning one off is dropped as skipped=\"detector_state_off\").")
             if state else None,
             ("- Do not turn lora and crop_classifier on together in one candidate "
              "(skipped=\"compound_server_state\"); add one at a time.")
             if len(state) == 2 else None,
             "- A tiling grid folded under the current champion (see the Axis board) is "
             "dropped as skipped=\"tiling_blocked\"."
             if "tiling" not in disabled else None]
    off = sorted(status.off)
    if off and status.retire_after:
        lines.append(f"- OFF axes on the Axis board ({', '.join(off)}): a candidate changing "
                     "only OFF axes is dropped as skipped=\"axis_retired\".")
    return [x for x in lines if x]


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
    row large on a dataset with many classes, and the budget guard would shrink
    the history window. The planner needs the number of classes text won and the
    narrowest of those margins. Wins follow screen's pick (chosen) — the
    recorded F is rounded and cannot tell a narrow gap."""
    margins = [v["text"] - v["semantic"] for v in modality_f.values()
               if v["chosen"] == "text"]
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


# English reasons for skips — a row's note is a free sentence written for people
# and is not carried in the prompt. Reasons are built from the skip code and
# structured fields; fields whose numbers are carried separately (fit_dev_cmp,
# unused_axes, material, same_as, tiling_used) are not repeated here
_REASON = {
    "axis_disabled": "the axis is disabled on this dataset",
    "disabled_axes": "it uses an axis or value disabled on this dataset",
    "tiling_blocked": "the tiling grid is folded under the current champion",
    "axis_retired": "it changes only axes that are OFF on the Axis board",
    "champion_combo": "it is the current champion itself",
    "no_change": "it changes nothing",
    "combo_tried": "this exact combination was already tried",
    "carried_duplicate": "it repeats a re-queued candidate",
    "duplicate_axes": "it repeats another candidate in the same answer",
    "max_candidates": "over the candidate limit",
    "invalid_axes": "the axes break the value lists or Hard constraints",
    "detector_state_off": "it turns off a detector-state axis of the champion",
    "compound_server_state": "it turns lora and crop_classifier on together",
    "missing_hook": "this axis is not wired in this run",
    "lora_deps": "the tail fine-tune is not wired in this run",
    "adapt_ctl": "the crop classifier is not wired in this run",
    "no_material": "the mined material came out empty",
    "lost_value_prescreen": "it contains a clearly-lost value and lost again on the "
                            "prescreen subset of fit_dev",
    "axis_unused": "the screened setting did not use the axis",
    "same_setting": "the screened setting is the same as one already served",
    "fit_dev_lost": "it clearly lost on fit_dev",
    "fit_dev_identical": "it matched the champion image for image on fit_dev",
    "rescorer_identity": "the rescorer refit came out as the identity",
    "error": "it failed with a runtime error",
}


def _reason(c: dict) -> str:
    """One-line skip reason (English) — the skip code plus the numbers that go with
    it (arm verdict, prescreen, the empty axis, the exception type)."""
    skip = c.get("skipped")
    gate = c.get("arm_gate") or {}
    if skip in ("lora_gate", "classifier_gate"):
        arm = "tail fine-tune" if skip == "lora_gate" else "crop classifier"
        if gate.get("tripwire"):
            return (f"{arm} not trained: " + ("too few fit_train images"
                    if skip == "lora_gate" else "too few training or background crops"))
        out = f"{arm} arm gate failed"
        if gate.get("delta") is not None and gate.get("margin") is not None:
            out += (f": best fit_dev delta {gate['delta']:+.4f} <= margin "
                    f"{gate['margin']}")
        if gate.get("lam") is not None:
            out += f" (best lambda {gate['lam']:g})"
        return out
    out = _REASON.get(skip, str(skip))
    if skip == "invalid_axes" and c.get("invalid"):
        out += f" ({c['invalid']})"            # planner.axes.AxesError.en
    if skip == "no_material" and c.get("empty_axis"):
        out += f" ({c['empty_axis']})"
    exc = re.match(r"[A-Za-z_]\w*(?=:)", c.get("note") or "") if skip == "error" else None
    if exc:
        out += f" ({exc.group(0)})"           # the exception type only — the message is free text
    pre = c.get("prescreen")
    if pre and pre.get("of") and not pre.get("passed"):   # once passed, the full fit_dev decided
        out += f"; decided on a prescreen of {pre['n_images']}/{pre['of']} fit_dev images"
    return out


def _outcome_lines(rows: list[dict], champion: dict) -> list[str]:
    """Candidate lines of the re-ask message — the increment (by request), the skip
    code with its English reason (_reason), and when present the served material
    (same_setting) and unused axes."""
    lines = []
    for c in rows[:MAX_REASK_LINES]:
        delta = _delta_axes(c.get("requested_axes") or c.get("axes"), champion)
        line = (f"- {c.get('id')} ("
                + (", ".join(kv(a, v) for a, v in sorted(delta.items()))
                   or "no change") + f"): skipped={c.get('skipped')}")
        if c.get("unused_axes"):
            line += f", unused_axes={json.dumps(c['unused_axes'])}"
        cmp = c.get("fit_dev_cmp")
        if cmp and cmp.get("ub") is not None:
            line += f", fit_dev delta {cmp['delta']:+.4f} (ub {cmp['ub']:+.4f})"
        line += f" — {_reason(c)}"
        if c.get("same_as"):
            line += f", same_as={c['same_as']}"
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

    *How badly* a candidate was rejected is included too: holdout score
    (held_out_micro_f_beta) and gate Δ / CI lower bound, fit_dev score, and the
    skip reason (reason — English, _reason) — so a near miss can be told from a
    clear regression."""
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
            # empty values are left out — not "adopted": null, "skipped": null on every row
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
                row["gate_passed_but_lost"] = True   # carried over (GLOSSARY)
            if (c.get("wall_min") or {}).get("total") is not None:
                row["wall_min"] = c["wall_min"]["total"]   # per stage only in the artifact
            if c.get("skipped"):
                row["reason"] = _reason(c)          # the note (free text) is not carried
            if c.get("material"):
                row["material"] = c["material"]   # same_setting — what was the same
            if c.get("same_as"):
                row["same_as"] = c["same_as"]     # "champion" counts as a failure of the axis
            ev = evaluated.get(c.get("id"))
            if ev:
                row["held_out_micro_f_beta"] = ev.get("micro_f_beta")
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


def _budget_lines(budget: dict | None) -> list[str]:
    """Budget card — rounds left and stop counters. Per-axis cost is on the axis
    board (min column).

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
        lines.append(f"- Stagnant rounds in a row (no adoption, or one whose held-out "
                     f"gain was not clearly positive): {b.get('stagnant', 0)}; "
                     f"the run stops at {b['saturation_rounds']}.")
    if b.get("empty_rounds"):
        lines.append(f"- Rounds in a row that measured nothing on the held-out split "
                     f"(a round where a candidate was dropped for losing on fit_dev "
                     f"does not count): {b.get('empty', 0)}; the run stops at "
                     f"{b['empty_rounds']}.")
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
        n = ch["forwards_per_image"]
        nums.append(f"{n} detector pass{'es' if n != 1 else ''} per image")
    if nums:
        parts.append("; ".join(nums))
    return "\n".join(parts)


def _combos_line(history: list[dict]) -> str:
    """One line of combinations tried under the current champion (increments by
    request) — an exact repeat is blocked by the filter."""
    champ = ((history[-1].get("champion") or {}).get("axes") or {}) if history else {}
    tried = tried_under_champion(history)[-MAX_TRIED:]
    # the current champion's own combination (empty increment) comes first —
    # stating it is not a candidate. If the request that adopted the champion
    # differs from its label (unused-axis correction, rescorer identity refit),
    # that is the champion too — stated so it is not read as a new combination
    asked = [", ".join(kv(k, v) for k, v in sorted(_delta_axes(a, champ).items()))
             for a in champion_requests(history)]
    itself = [CHAMPION_REQUEST.format(f"{{{a}}}") for a in dict.fromkeys(asked) if a]
    return ("Combos already tried under this champion, as differences from it (an exact "
            "repeat is dropped as skipped=\"combo_tried\"): "
            + "; ".join([CHAMPION_COMBO] + itself
                        + [f"{{{t}}}" for t in tried if t not in asked]))


# first item of the tried-combination list — the current champion itself.
# Re-proposing it has no increment and is dropped
CHAMPION_COMBO = ("{} = the current champion itself (not a candidate: re-proposing "
                  "it changes nothing and is dropped as skipped=\"no_change\")")
# when the request that adopted the champion differs from its label — that
# request is the champion too
CHAMPION_REQUEST = ("{} = the request that adopted the current champion (an axis it "
                    "asked for was unused or came out identity, so it IS the champion; "
                    "dropped as skipped=\"champion_combo\")")


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
                + (", ".join(kv(k, v) for k, v in sorted(delta.items())) or "-")
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
split.** Each candidate changes axes of the current champion; the Axis board
below lists every axis, the values you may set now and its record.

Hard constraints (a candidate breaking one is dropped without evaluation):
{constraints}

Rules:
- Change as FEW axes as possible per candidate (single increments attribute
  gains cleanly). List only the axes you change: omitted axes keep the
  champion's value; to turn an axis off, set it to its off value (Axis board legend).
- Materials are mined from the train split only. Each candidate is screened on
  fit_dev (a slice of train) and only the best go to the held-out gate. The
  gate adopts only if a paired-bootstrap LOWER BOUND on the held-out delta
  clears {margin} AND the delta is positive and at least {min_delta}, so a
  candidate can score higher on the held-out split and still be rejected as noise:
  gate_delta > 0 with gate_ci_lo below {margin}, or with gate_delta below {min_delta},
  means "close, but unproven"; a clearly negative gate_delta is not worth
  repeating.
- A value whose single-increment candidate clearly lost on fit_dev under this
  champion is "clearly-lost": its variants (another k, or the value combined
  with other axes) are not blocked but prescreened (see the Axis board), so
  propose one only with a concrete reason it should behave differently.
{tiling_rule}- While some axis on the Axis board is "never tried in this run" (axes that cannot
  be proposed alone right now say "not tried" instead), prefer one of them over
  another variant of an already-tried value, unless the diagnosis
  gives a concrete reason for the variant. Re-measuring an axis marked
  "{not_retried}" is not such a variant: its last result came from another champion.
- An axis the Axis board marks "{not_retried}" last failed under an earlier
  champion; the champion has changed since, so its result may differ now
  (detector-state axes react to upstream changes). Prefer re-proposing one of them
  over repeating an axis that was left unused.
- Always propose at least one candidate. The run stops on its own once rounds stop
  improving, so when nothing looks clearly promising, propose the most informative
  remaining change (an axis "{not_retried}", then one "never tried in this run").

## Budget
{budget}

## Current champion
{champion}

## Axis board
{board}

## Round history
Recent rounds in detail (axes_delta is the change from the champion at the
start of that round; held_out_micro_f_beta / gate_delta / gate_ci_lo are held-out
results; expected_gain is what the planner predicted; reason says why a skipped
candidate was dropped):
{history}
{glossary}
{older}
## Diagnosis digest
{error_kinds}
Error kinds this round (train + held-out) and the axes you can propose now for them.
HEURISTIC hints, not rules: a count says how much is at stake if the axis
works, not that it will — weigh it against per-class counts and the samples.
{digest}
{small_block}
Error detail. Only train carries individual samples; held_out is aggregate-only.
A sample's note is a vision model's one-line observation of the crop — evidence,
not ground truth. **Samples are a coverage selection, not a random draw**
(round-robin over kinds and classes, most severe first): read frequency from the
digest and the per-class table only, and the samples for what the errors look like.
{diagnosis}
{notes_block}
## Response
Propose up to {max_candidates} candidates. For each give "axes" (only the
changes), "expected_gain" (your estimate of the held-out micro F delta over
the champion, e.g. 0.01), "cost_min" (expected minutes, from the min column of the
Axis board),
"differs" (for a candidate containing a clearly-lost value: what is different
this time; otherwise "") and a short "rationale"."""

# carried in the Rules only when tiling is not disabled — what a grid means, its
# cost and how to pick the first grid
TILING_RULE = """- "tiling" is a grid g: every image is also cut into g x g tiles and each tile
  is detected at magnification (g-1)(1-overlap)+1 ({mag2}x at g=2, {mag3}x at g=3
  with overlap {overlap}). Cost: 1 + g^2 detector passes per image. Pick the
  first grid from the missed-small sizes in the diagnosis: the grid whose
  magnification brings the typical missed object to a detectable size, no
  larger. Raise g only when the previous grid was used and helped while
  small-object misses remain; a grid the screen did not use folds that grid and
  larger ones until the champion changes (the Axis board shows the fold and a
  one-time exception).
"""


def tiling_rule(overlap: float) -> str:
    """TILING_RULE filled from search.tiling_overlap — the magnification example follows the setting."""
    def mag(g):
        return f"{(g - 1) * (1 - overlap) + 1:g}"
    return TILING_RULE.format(mag2=mag(2), mag3=mag(3), overlap=f"{overlap:g}")

# meaning of the error kinds the diagnosis digest and detail use — all of
# core.metrics FP_KINDS and FN_KINDS
ERROR_KINDS = (
    "Error kinds: background = a false detection where no object of interest is; "
    "class_confusion = detected as the wrong class; localization = right class, but the "
    "box misses the object; duplicate = an extra box on an already-detected object; "
    "tiling_artifact = a false detection from the tiled passes only; missed = a "
    "ground-truth object not detected at all; missed_small = a small object not "
    "detected; below_threshold = detected, but scored below the class threshold.")

# explanations carried only when a history row has that field — next to the data, only what is needed
GLOSSARY = (
    ("fit_dev_cmp", "fit_dev_cmp: delta = (candidate - champion) micro F on fit_dev, "
                    "ub = its paired-bootstrap upper bound; ub < 0 means it clearly "
                    "lost, ub >= 0 is a near miss (lost within noise, or won)."),
    ("tiling_used", "tiling_used=false: the screen preferred no tiling at that grid. "
                    "tiling_f compares the mean per-class fit_dev F of the best tiled vs "
                    "untiled config; gap_ub >= 0 means the tiled loss is within noise."),
    ("unused_axes", "unused_axes: axes the screened setting did not use (exemplar: the "
                    "text pass beat the semantic pass for every class; visual_negatives: "
                    "no semantic class kept a negative config) — other variants of such "
                    "an axis rarely change the setting while that stays true."),
    ("modality", "modality: classes where the text pass beat the semantic (exemplar) "
                 "pass on fit_dev F (text_won) and the rest (semantic_won); "
                 "closest_text_margin is the smallest F gap among the classes text won — "
                 "small means another exemplar_base_k or fn_arm may flip a class."),
    ("neg_eta_used", "neg_eta_used / neg_eta_grid: the suppression IoU thresholds the "
                     "screen picked (lower = stronger) and the thresholds tried — picks "
                     "at the low end of the grid mean more negatives (raise "
                     "visual_negative_k) may help; picks at the high end mean a larger k "
                     "will not help either."),
    ("same_as", "same_as: a same_setting row served what the champion (\"champion\", "
                "counted as a failure of the axis), a sibling this round (\"sibling\") or "
                "an earlier evaluated candidate (\"tried\") serves; material shows what — "
                "another value of that axis will likely collapse the same way."),
    ("gate_passed_but_lost", "gate_passed_but_lost: it passed the held-out gate but lost to "
                             "the adopted candidate; it is re-queued on top of the new "
                             "champion automatically, so do not propose it again."),
    ("lost_values", "lost_values: the candidate contained values that clearly lost on "
                    "fit_dev under this champion (variants of clearly-lost values)."),
    ("wall_min", "wall_min: wall-clock minutes the candidate took (materialize + fit_dev "
                 "screen + held-out)."),
    ("material", "material: what the served setting contained when it matched another "
                 "(same_setting) — e.g. the text negatives a larger k refined to."),
)


def _glossary(rows: list[dict]) -> str:
    """Explains only the fields present in the detailed history's candidate rows
    (_compact_history) — judged by key, so a field name that happens to appear in
    a value string does not count."""
    keys = {k for e in rows for c in e.get("candidates", []) for k in c}
    lines = [f"- {text}" for field, text in GLOSSARY if field in keys]
    return ("Fields in these rows:\n" + "\n".join(lines) + "\n") if lines else ""


# Inserted only when train has missed_small and a grid larger than the champion's
# can be proposed — evidence for the first tiling grid.
SMALL_BLOCK = """
Missed-small sizes (train, {n} objects) — box side at the detector input
(sqrt of box/image area x {side} px, long side resized to {side}): p25 {p25} px,
median {p50} px, p75 {p75} px. After tiling magnification (overlap {overlap}):
{grids}
"""

# Carried below the axis board only when some candidate is carried over.
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
changes nothing, a folded tiling grid is blocked (tiling_blocked), a candidate
changing only OFF axes on the Axis board is dropped (axis_retired), an axis the
screen left unused stays unused for other values of it, a source value whose
material came out empty stays empty under this champion, a quantity whose material collapsed to
the same setting collapses again, and a variant of a clearly-lost value that the
prescreen dropped (lost_value_prescreen) needs a real difference to pass. Propose
at least one candidate — an axis the Axis board marks "{not_retried}" is a good
choice when nothing else is left, unless it already appears in the outcomes above
(the board was built at the start of this round)."""

MAX_REASK_LINES = 24     # cap on candidate lines in the re-ask message

# Inserted only when dataset_notes (config) is set — otherwise notes_block=""
# and nothing is left in its place (pinned by tests/test_planner.py).
NOTES_BLOCK = """
Dataset notes from the operator (verbatim context — weigh against the diagnosis):
{notes}
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
    """Strip only the detail samples (unmatched crop notes included) from the diagnosis — aggregates stay."""
    return {k: {kk: vv for kk, vv in v.items()
                if not kk.endswith("_sample") and kk != "crop_notes"}
            if isinstance(v, dict) else v for k, v in (diag or {}).items()}


def _halve_samples(diag: dict) -> dict:
    """Only the first half of the detail samples (unmatched crop notes included) —
    samples are in round-robin order over kinds and classes (diagnose.sample), so
    the first half keeps the coverage."""
    return {k: {kk: (vv[:(len(vv) + 1) // 2]
                     if kk.endswith("_sample") or kk == "crop_notes" else vv)
                for kk, vv in v.items()}
            if isinstance(v, dict) else v for k, v in (diag or {}).items()}


# The budget ladder — shrink in (history window, diagnosis samples) order. The last
# round (window 1) is worth more than the samples: why the last candidates lost is
# the most direct evidence for the next proposal, so the window drops to 0 only
# after the samples are halved and removed. Older rounds stay as one line each
# outside the window.
BUDGET_LADDER = tuple(dict.fromkeys(
    tuple((w, "full") for w in range(MAX_HISTORY, 0, -1))
    + ((min(1, MAX_HISTORY), "half"), (min(1, MAX_HISTORY), "none"), (0, "none"))))


class LLMPlanner:
    """chat: (messages, **kw) -> dict (parsed JSON). In production this is a
    partial application of clients.llm.LLMClient.chat_json."""

    def __init__(self, chat, artifact_dir: Path, *, max_candidates: int = 3,
                 notes: str = "", token_budget: int = DEFAULT_TOKEN_BUDGET,
                 disabled_axes=(), disabled_values=None,
                 facts: DatasetFacts | None = None, tiling_overlap: float = 0.5,
                 tiling_tp_quantile: float = 0.1, tiling_tp_min: int = 20,
                 budget: dict | None = None, prescreen: dict | None = None,
                 retire_after: int | None = None,
                 min_delta: float = GateCfg().min_delta,
                 margin: float = GateCfg().margin):
        self.chat = chat
        self.min_delta = min_delta             # gate.min_delta — the prompt's adoption rule
        self.margin = margin                   # gate.margin — what the lower bound must clear
        self.tiling_overlap = tiling_overlap   # search.tiling_overlap — magnification
        # search.tiling_tp_quantile and tiling_tp_min — the s* line of the
        # missed-small block
        self.tiling_tp_quantile = tiling_tp_quantile
        self.tiling_tp_min = tiling_tp_min
        self.artifact_dir = Path(artifact_dir)
        self.max_candidates = max_candidates
        self.notes = notes or ""
        if any(ord(ch) > 127 and ch.isalpha() for ch in self.notes):
            log.warning("dataset_notes has non-English text — the planner prompt is "
                        "English only (README)")
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
        # search.axis_retire_after — the consecutive failures that make an axis OFF.
        # None records only (it must be the same value the runner enforces)
        self.retire_after = retire_after

    def _sanitize(self, raw, max_candidates: int, dropped: list, champion: dict) -> list[dict]:
        return sanitize_candidates(raw, max_candidates=max_candidates, dropped=dropped,
                                   disabled_axes=self.disabled_axes,
                                   disabled_values=self.disabled_values,
                                   base=champion or None, facts=self.facts)

    def _refill(self, build, raw, dups, candidates, champion, carried_keys,
                dropped, carried_ids) -> dict:
        """Ask once more, in the same conversation, for the slots freed by
        candidates that repeated carried ones (extends candidates and dropped in
        place). Returns the record kept in the artifact.

        The conversation grows by the first response and the follow-up message —
        if the first prompt nearly filled the budget, the budget ladder shrinks it
        to fit (_fit). With dups present there is at least one free slot."""
        k = self.max_candidates - len(candidates)

        def delta(c):
            return ", ".join(kv(a, v) for a, v in
                             sorted(_delta_axes(c["axes"], champion).items()))
        note = FOLLOWUP.format(n=len(dups), dups="; ".join(delta(c) for c in dups),
                               kept="; ".join(delta(c) for c in candidates) or "none",
                               k=k)
        tail = [{"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
                {"role": "user", "content": note}]
        msgs, window, samples = self._fit(build, tail)
        record = {"message": note, "response": None, "error": None, "window": window,
                  "samples": samples}
        if _est_tokens(msgs) > self.token_budget:
            record["error"] = (f"PromptOverBudget: estimated {_est_tokens(msgs)} tokens > "
                               f"budget {self.token_budget} — not asking again")
            log.warning("follow-up after carried duplicates: %s", record["error"])
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
        more = self._sanitize(renamed, k, dropped, champion)
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
        where a fresh one would not."""
        log.info("proposal replay: %s", path)
        stored = json.loads(path.read_text())
        out = {k: stored[k] for k in ("planner", "candidates")}
        out["dropped"] = stored.get("dropped") or []
        return out

    def _builder(self, history: list[dict], diagnosis: dict):
        """(build, carried) — build(window, samples="full") is the first prompt
        assembled with history window `window`. samples is how much of the
        diagnosis detail samples to carry (full, half, none — BUDGET_LADDER).
        propose and reask must assemble the same prompt for a re-ask to continue
        the same conversation."""
        champion = (history[-1].get("champion", {}).get("axes")
                    if history else None) or {}
        values = {k: tuple(x for x in v if x not in self.disabled_values.get(k, ()))
                  for k, v in axis_values(self.facts).items()}
        banned = list(self.disabled_axes) + [
            kv(k, x) for k, xs in sorted(self.disabled_values.items()) for x in xs]
        compact = _compact_diagnosis(diagnosis)
        # counts per cause are carried by the digest — dropped from the error detail
        # so they are not carried twice
        for d in compact.values():
            if isinstance(d, dict):
                d.pop("summary", None)
        diag_full = _diagnosis_text(compact)
        status = axis_status(history, self.retire_after)
        # (axis, value) whose material came out empty under the current champion — the same again if re-proposed
        empty = {r.key for r in axis_ledger(history).values() if r.status == "no_material"}
        gate = tiling_gate(history)
        digest = _digest_lines(diagnosis, self.disabled_axes, values, champion, status.off,
                               empty, gate)
        carried = deterministic.carryover(history, self.disabled_axes,
                                          self.disabled_values, facts=self.facts)
        carry_lines = [", ".join(kv(k, v) for k, v in
                                 sorted(_delta_axes(c["axes"], champion).items()))
                       for c in carried]
        board = board_lines(history, values, status, disabled=self.disabled_axes,
                            banned=banned, facts=self.facts, prescreen=self.prescreen)
        board.append(_retire_line(status))
        board.append(_combos_line(history))
        if carry_lines:
            board.append(CARRY_BLOCK.strip("\n").format(
                carried="\n".join(f"- {x}" for x in carry_lines)))
        constraints = _constraint_lines(self.disabled_axes, status, champion)
        budget = _budget_lines(self.budget)
        champion_text = _champion_block(history)
        # when tiling cannot be proposed (disabled, OFF, every grid folded), the
        # first-grid hint is noise — not carried
        cur_grid = champion.get("tiling") or 0
        grids_open = ([g for g in values.get("tiling", ()) if g and g > cur_grid
                       and not gate.block({"tiling": g})]
                      if "tiling" not in self.disabled_axes and "tiling" not in status.off
                      else [])
        small = (_small_block(diagnosis, self.tiling_overlap,
                              tp_quantile=self.tiling_tp_quantile,
                              tp_min=self.tiling_tp_min, grids_open=grids_open)
                 if grids_open else "")

        diag_text_by = {"full": diag_full,
                        "half": _diagnosis_text(_halve_samples(compact)),
                        "none": _diagnosis_text(_strip_samples(compact))}

        def build(window: int, samples: str = "full") -> list[dict]:
            diag_text = diag_text_by[samples]
            older = _older_lines(history, window)
            hist_rows = _compact_history(history, window)
            hist_json = json.dumps(hist_rows, ensure_ascii=False, default=str)
            return [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(
                    constraints="\n".join(constraints),
                    budget="\n".join(budget) or "- (no budget information)",
                    champion=champion_text,
                    board="\n".join(board),
                    history=hist_json,
                    glossary=_glossary(hist_rows),
                    tiling_rule=(tiling_rule(self.tiling_overlap)
                                 if "tiling" not in self.disabled_axes else ""),
                    min_delta=self.min_delta,
                    margin=self.margin,
                    older=("Earlier rounds, one line each:\n" + "\n".join(older) + "\n"
                           if older else ""),
                    error_kinds=ERROR_KINDS,
                    digest="\n".join(digest) or "- (no errors observed)",
                    small_block=small,
                    diagnosis=diag_text,
                    notes_block=(NOTES_BLOCK.format(notes=self.notes)
                                 if self.notes else ""),
                    not_retried=NOT_RETRIED,
                    max_candidates=self.max_candidates)},
            ]
        return build, carried

    def _fit(self, build, tail: list[dict] = ()) -> tuple[list[dict], int, str]:
        """Budget guard — shrink in BUDGET_LADDER order and use the first step that
        fits (older rounds → half the diagnosis samples → no samples → the last
        round). If it never fits, the last step. tail is the end of the
        conversation (earlier responses, follow-up messages) — it counts toward
        the budget too."""
        tail = list(tail)
        for window, samples in BUDGET_LADDER:
            messages = build(window, samples) + tail
            if _est_tokens(messages) <= self.token_budget:
                break
        if (window, samples) != BUDGET_LADDER[0]:
            log.info("prompt budget (estimated %d tokens) — history window %d, "
                     "diagnosis samples %s", self.token_budget, window, samples)
        return messages, window, samples

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
        continue, so it does not ask (None). A failed call or a prompt over the
        budget is None too (also when replaying the failure record) — the failure
        is kept as an artifact and the round closes (the remaining max_reasks are
        not used). Nor is it filled by the deterministic fallback (this round's
        first proposal already ran)."""
        path = self.reask_path(k)
        if path.exists():
            if json.loads(path.read_text()).get("error"):
                return None
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
                               or "- (none)", k=self.max_candidates,
                               not_retried=NOT_RETRIED)
        tail.append({"role": "user", "content": message})
        build, _ = self._builder(history, diagnosis)
        messages, window, samples = self._fit(build, tail)
        record = {"reask": k, "message": message, "response": None, "error": None,
                  "window": window, "samples": samples}
        raw = None
        if _est_tokens(messages) > self.token_budget:
            record["error"] = (f"PromptOverBudget: estimated {_est_tokens(messages)} "
                               f"tokens > budget {self.token_budget} — not asking again")
            log.warning("re-ask %d: %s", k, record["error"])
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
        candidates = self._sanitize(renamed, self.max_candidates, dropped, champion)
        proposal = {"planner": "llm", "candidates": candidates, "dropped": dropped}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**proposal, "prompt": messages, **record},
                                   indent=2, ensure_ascii=False, default=str))
        return None if record["error"] else proposal

    def propose(self, history: list[dict], diagnosis: dict) -> dict:
        if self.artifact_path.exists():
            return self._replay(self.artifact_path)

        champion = (history[-1].get("champion", {}).get("axes")
                    if history else None) or {}
        build, carried = self._builder(history, diagnosis)
        messages, window, samples = self._fit(build)
        raw, error = None, None
        est = _est_tokens(messages)
        if est > self.token_budget:
            # a prompt the server would reject is not sent — the fallback reason is kept plainly
            error = (f"PromptOverBudget: estimated {est} tokens > budget "
                     f"{self.token_budget} (over even at the last ladder step — restart "
                     "the planner server with a larger --max-len or raise "
                     "planner.launch.max_model_len)")
        else:
            try:
                raw = self.chat(messages, schema=_schema(self.max_candidates),
                                max_tokens=OUTPUT_MAX_TOKENS)
            except Exception as exc:  # noqa: BLE001 — falling back is the contract
                error = f"{type(exc).__name__}: {exc}"
        if error:
            log.error("LLM planner failed (%s) — this round falls back to deterministic",
                      error)

        dropped: list[dict] = []
        candidates = self._sanitize((raw or {}).get("candidates"), self.max_candidates,
                                    dropped, champion)
        # a candidate equal to a carried one does not take a slot — the runner
        # runs the carry-overs anyway, so candidates repeated against the
        # prompt's instruction are dropped and the freed slots asked for again
        carried_keys = {_axes_key(c["axes"]) for c in carried}
        candidates, dups = _drop_carried(candidates, carried_keys, dropped)
        followup = None
        if dups and raw is not None:
            followup = self._refill(build, raw, dups, candidates, champion,
                                    carried_keys, dropped,
                                    [c["id"] for c in carried])
        if dropped:
            log.info("sanitizer dropped %d candidates: %s", len(dropped),
                     [(c["id"], c["skipped"]) for c in dropped])
        fallback = None
        # proposing only carried candidates makes an empty proposal correct (the
        # runner runs the carry-overs) — no fallback
        if raw is None or (not candidates and not dups):
            # Call failed or every candidate invalid — fill this round deterministically
            det = deterministic.propose(history, diagnosis,
                                        max_candidates=self.max_candidates,
                                        disabled_axes=self.disabled_axes,
                                        disabled_values=self.disabled_values,
                                        facts=self.facts)
            # ids are det_<i> — they do not clash with the dropped LLM rows (cand_<i>),
            # so the proposal and the history map 1:1. source=fallback — the runner's
            # axis-retirement filter does not block them (no empty round)
            candidates = [{**c, "id": f"det_{i}", "source": "fallback"}
                          for i, c in enumerate(det["candidates"])]
            fallback = "deterministic"
            if raw is not None and error is None:
                # the reply came but the sanitizer dropped every candidate — keep
                # the reason (never an empty one)
                error = "no valid candidate: " + ("; ".join(
                    f"{c.get('skipped')}: {c.get('invalid') or c.get('note') or ''}".rstrip(": ")
                    for c in dropped) or "empty candidate list")
                log.error("LLM planner failed (%s) — this round falls back to deterministic",
                          error)

        proposal = {"planner": "llm", "candidates": candidates, "dropped": dropped}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(
            {**proposal, "prompt": messages, "response": raw,
             "fallback": fallback, "error": error, "followup": followup,
             "budget": {"window": window, "samples": samples}},
            indent=2, ensure_ascii=False, default=str))
        return proposal
