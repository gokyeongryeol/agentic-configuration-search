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

import json
import logging
from pathlib import Path

from ..diagnose.sample import pick_detail
from . import deterministic
from .axes import DatasetFacts, axis_values, sanitize_candidates

log = logging.getLogger(__name__)

ARTIFACT = "proposal.json"

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
                        "id": {"type": "string"},
                        "axes": {"type": "object"},
                        "rationale": {"type": "string"},
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
# assembled prompt (DEFAULT_CHAR_BUDGET).
MAX_HISTORY = 8          # recent rounds in the prompt (the budget guard may cut further)
# Prompt character budget — overflow is fixed by shrinking history. Derived
# from max_model_len 16384 − 1024 output − slack, at 2.0 chars/token. Overflow
# makes vLLM return 400 and the planner silently falls back to deterministic.
DEFAULT_CHAR_BUDGET = 28_000
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
            if "neg_eta_used" in c:
                row["neg_eta_used"] = c["neg_eta_used"]
                row["neg_eta_grid"] = c.get("neg_eta_grid")
            if c.get("unused_axes"):
                row["unused_axes"] = c["unused_axes"]
            if c.get("gate_passed_but_lost"):
                row["gate_passed_but_lost"] = True   # carried over — see CARRY_BLOCK
            if c.get("note"):
                row["note"] = str(c["note"])[:MAX_NOTE]
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
  overlap 0.5), costing 1 + g^2 detector passes per image. Raise g one step at a
  time. A history row with tiling_used=false means no class benefited from that
  grid on fit_dev (the setting fell back to no tiling); tiling_used=true on an
  adopted grid while small-object misses remain suggests trying g+1.
- A row with unused_axes lists axes the candidate had on that the screened
  setting did not use (exemplar: the text pass beat the semantic pass for every
  class; visual_negatives: no semantic class kept a negative config). Other
  variants of such an axis (another k, fn_arm) rarely change the setting
  while that stays true.
- A visual_negatives row carries neg_eta_used (the suppression IoU thresholds
  the screen picked for semantic classes) and neg_eta_grid (the thresholds it
  tried). Lower means stronger suppression. Picks at the low end of the grid
  mean negatives are pulling hard, so more negatives (raise visual_negative_k)
  may help; picks at the high end mean they barely help, so fold the axis
  rather than raising k.
- If nothing promising remains, return stop=true with no candidates.
- History rows may carry test_micro_f_beta / gate_delta / gate_ci_lo (how close
  a rejected candidate came) and a skip note. skipped="combo_tried" means that
  exact axes combination was already tried — never repeat it verbatim; vary it
  or move on.

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
{dead_block}{carry_block}{notes_block}
Propose up to {max_candidates} candidates."""

# Inserted only when some axis had empty material — otherwise omitted.
DEAD_BLOCK = """
Axes whose mined material came out empty, or too small to train on, on this
dataset (the candidate was rejected without evaluation, so it cost a slot for
nothing). Their material pool is exhausted here — re-propose one only with a
different source value, or after the champion has changed:
{dead}
"""

# Inserted only when some candidate is carried over — otherwise omitted.
CARRY_BLOCK = """
Re-queued automatically this round — each passed the gate last round but lost
to the adopted candidate, so it is retried on top of the new champion (listed
as its difference from the current champion). Do not spend a slot on these:
{carried}
"""

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


def _chars(messages: list[dict]) -> int:
    return sum(len(m["content"]) if isinstance(m.get("content"), str)
               else sum(len(str(x)) for x in m.get("content", []))
               for m in messages)


def _strip_samples(diag: dict) -> dict:
    """Strip detail samples from the diagnosis — aggregates (summary, per_class) stay."""
    return {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_sample")}
            if isinstance(v, dict) else v for k, v in (diag or {}).items()}

class LLMPlanner:
    """chat: (messages, **kw) -> dict (parsed JSON). In production this is a
    partial application of clients.llm.LLMClient.chat_json."""

    def __init__(self, chat, artifact_dir: Path, *, max_candidates: int = 3,
                 notes: str = "", char_budget: int = DEFAULT_CHAR_BUDGET,
                 disabled_axes=(), disabled_values=None,
                 facts: DatasetFacts | None = None):
        self.chat = chat
        self.artifact_dir = Path(artifact_dir)
        self.max_candidates = max_candidates
        self.notes = notes or ""
        self.char_budget = char_budget
        self.disabled_axes = tuple(disabled_axes or ())
        self.disabled_values = {k: tuple(v) for k, v in
                                (disabled_values or {}).items() if v}
        # Values whose outcome the data already decides (gt_crop on a single class,
        # lora whose training would be skipped) are left out of the prompt and
        # filtered from the response — None means the full value space
        self.facts = facts

    @property
    def artifact_path(self) -> Path:
        return self.artifact_dir / ARTIFACT

    def propose(self, history: list[dict], diagnosis: dict) -> dict:
        if self.artifact_path.exists():
            # replay — zero LLM calls (resume contract)
            log.info("proposal replay: %s", self.artifact_path)
            stored = json.loads(self.artifact_path.read_text())
            out = {k: stored[k] for k in ("planner", "candidates", "stop")}
            # Restore the drop record too — so a resumed run does not silently
            # lose budget where a fresh one would not. Old artifacts without
            # the field read as an empty list.
            out["dropped"] = stored.get("dropped") or []
            return out

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
        diag_json = json.dumps(compact, ensure_ascii=False, default=str)
        tried_lines = _tried_lines(history, champion)
        leverage_lines = [
            ln for ln in _leverage_lines(diagnosis)
            if not any(ln.startswith(f"- {a}:") for a in self.disabled_axes)]
        dead_lines = _dead_axis_lines(history)
        carried = deterministic.carryover(history, self.disabled_axes,
                                          self.disabled_values, facts=self.facts)
        carry_lines = [", ".join(f"{k}={v}" for k, v in
                                 sorted(_delta_axes(c["axes"], champion).items()))
                       for c in carried]

        def build(window: int) -> list[dict]:
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

        # Budget guard — drop the oldest rounds first. If still over at
        # window=0, strip diagnosis samples as the last line of defense.
        window = MAX_HISTORY
        messages = build(window)
        while _chars(messages) > self.char_budget and window > 0:
            window -= 1
            messages = build(window)
        if _chars(messages) > self.char_budget:
            diag_json = json.dumps(_strip_samples(compact),
                                   ensure_ascii=False, default=str)
            messages = build(0)
        if window < MAX_HISTORY:
            log.info("prompt budget (%d chars) — history window shrunk %d→%d rounds",
                     self.char_budget, MAX_HISTORY, window)
        raw, error = None, None
        try:
            raw = self.chat(messages, schema=SCHEMA)
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
        if dropped:
            log.info("sanitizer dropped %d candidates: %s", len(dropped),
                     [(c["id"], c["skipped"]) for c in dropped])
        stop = bool((raw or {}).get("stop")) and not candidates
        fallback = None
        if raw is None or (not candidates and not (raw or {}).get("stop")):
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
             "fallback": fallback, "error": error},
            indent=2, ensure_ascii=False, default=str))
        return proposal
