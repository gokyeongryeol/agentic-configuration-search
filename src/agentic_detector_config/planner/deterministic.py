"""Deterministic planner — a rule implementation of the cause→remedy mapping.

It is the LLM planner's fallback, the A/B control, and the fixed execution
path for the guard test (test_decision_channel). The cause→remedy mapping
matches the README's FP/FN diagnosis section.
"""

from __future__ import annotations

from .axes import BASELINE, TILING_GRIDS, DatasetFacts, validate_axes

# Error kind → (axis, value) pairs that can address it, most promising first.
# If a value equals the champion's (already on), move to the next. exemplar
# steps up none → base → fn_arm.
#
# **This is a coverage relation, not a decision rule.** The deterministic
# planner uses the order as its policy, but the LLM prompt inverts the same
# relation into "error count this axis touches this round"
# (planner.llm._leverage_lines). So each list must contain everything that
# *can* address the kind, not just what to try first.
REMEDY: dict[str, tuple] = {
    # For misses, lora comes before alias — if alias eats the candidate budget
    # first, lora gets pushed out, and alias is often rejected for empty material.
    "missed": (("exemplar", "base"), ("exemplar", "fn_arm"), ("lora", True),
               ("alias", "vlm")),
    # tiling raises grid g one step at a time — a g at or below the champion's is
    # not a remedy (propose's step-up rule). The first value is the cheapest zoom.
    "missed_small": tuple(("tiling", g) for g in TILING_GRIDS),
    # below_threshold = detected but did not clear τ_c. Scores can be raised not
    # only by calibration (rescorer) but also by better prompts (exemplar), a
    # better detector (lora), and re-classification (crop_classifier). rescorer
    # is the direct remedy so it goes first, but listing only it would leave
    # this kind without a remedy once rescorer is blocked.
    "below_threshold": (("rescorer", True), ("exemplar", "base"),
                        ("exemplar", "fn_arm"), ("lora", True),
                        ("crop_classifier", True)),
    # The crop classifier drops detections it reclassifies as background (0) — it
    # is a background-FP filter too. On a single-class dataset class_confusion is
    # zero by definition, so without this entry the diagnosis reaches the axis only
    # through below_threshold. Training costs the most (classifier fit + detector
    # state swap), so it goes last.
    "background": (("visual_negatives", True),
                   ("text_negatives", "background"), ("rescorer", True),
                   ("crop_classifier", True)),
    "class_confusion": (("alias", "vlm"),
                        ("crop_classifier", True),
                        ("text_negatives", "gt_crop"),
                        ("visual_negatives", True)),
    "localization": (("lora", True),),
    "duplicate": (),          # handled by gateway's per-class folding — no proposal
    "tiling_artifact": (),    # tiling merge params are not an axis — left to human-injected candidates
}


def _error_ranking(diagnosis: dict) -> list[str]:
    """Sum per-kind counts from the train and holdout ("test" key) diagnoses
    into a descending list of kinds."""
    counts: dict[str, int] = {}
    for split_key in ("train", "test"):
        d = diagnosis.get(split_key) or {}
        for side in ("fn", "fp"):
            for kind, n in (d.get("summary", {}).get(side) or {}).items():
                counts[kind] = counts.get(kind, 0) + int(n)
    return [k for k, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n > 0]


def tried_combos(history: list[dict]) -> set[tuple]:
    """Set of **full axes combinations** tried in previous rounds.

    Keyed by combination rather than (axis, value) because a candidate is a
    single increment over the champion, so the combination carries the champion
    context — an axis rejected under an old champion (e.g. lora while
    text-only) must be retryable as a new combination once the champion changes
    (e.g. after exemplar is adopted). Only exact retries of the same
    combination are blocked for good. The runner's shared filter uses this set too.
    """
    out: set[tuple] = set()
    for entry in history:
        for c in entry.get("candidates", []):
            axes = c.get("axes")
            if not isinstance(axes, dict):
                continue
            try:
                axes = validate_axes(axes)
            except ValueError:
                pass                      # old-format record — key it as-is
            out.add(tuple(sorted(axes.items())))
    return out


def propose(history: list[dict], diagnosis: dict, *,
            max_candidates: int = 3, disabled_axes=(),
            disabled_values=None, facts: DatasetFacts | None = None) -> dict:
    """Candidates that each raise one axis from the champion's axes — single
    increments with clean attribution.

    Returns: {"planner": "deterministic", "candidates": [...], "stop": bool}.
    stop=True means the mapping is exhausted — a human can resume by injecting
    candidates.yaml. disabled_axes / disabled_values (per-run bans) are
    excluded from remedies.
    """
    disabled_values = disabled_values or {}
    champion = validate_axes(
        (history[-1].get("champion", {}).get("axes") if history else None) or BASELINE)
    tried = tried_combos(history)

    def _applicable(kind) -> list[dict]:
        out = []
        for axis, value in REMEDY.get(kind, ()):
            if axis in disabled_axes or value in disabled_values.get(axis, ()):
                continue                      # banned axis/value — do not propose at all
            if champion.get(axis) == value:
                continue
            # exemplar step-up rule: fn_arm only after base
            if axis == "exemplar" and value == "fn_arm" \
                    and champion["exemplar"] == "none":
                continue
            # tiling step-up rule: only grids larger than the adopted one
            if axis == "tiling" and champion["tiling"] and value <= champion["tiling"]:
                continue
            try:
                axes = validate_axes({**champion, axis: value},
                                     facts=facts)
            except ValueError:
                continue                      # dependency rule violated (e.g. negatives without
                                              # exemplars, gt_crop on a single class, lora whose
                                              # training would be skipped — DatasetFacts)
            if tuple(sorted(axes.items())) in tried:
                continue                      # block only exact retries of the same combination
            out.append({"id": f"{kind}:{axis}={value}", "axes": axes,
                        "rationale": f"{axis} axis remedy for {kind} errors"})
        return out

    # Pick **interleaved** across kinds — if the dominant kind's remedy list
    # took the whole budget, the next kind's top remedy (usually more
    # promising) would be crowded out. Go round by round from depth 0 (each
    # kind's top remedy).
    ranked = _error_ranking(diagnosis)
    per_kind = [(k, _applicable(k)) for k in ranked]
    candidates: list[dict] = []
    seen: set[tuple] = set()
    max_depth = max((len(lst) for _, lst in per_kind), default=0)
    for depth in range(max_depth):
        for _kind, lst in per_kind:
            if depth >= len(lst):
                continue
            c = lst[depth]
            key = tuple(sorted(c["axes"].items()))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(c)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    return {"planner": "deterministic", "candidates": candidates,
            "stop": not candidates}
