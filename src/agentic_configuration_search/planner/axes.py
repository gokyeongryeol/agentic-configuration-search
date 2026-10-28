"""axes schema for candidate settings — the shared contract between the
planner (proposal) and the loop (materialize).

Every axis has a safe point where turning it off is exactly the current
behavior. BASELINE is round 0's starting point and every axis's safe point:
a text prompt of the class name alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# tiling axis value = grid g — each image is cut into g×g tiles
# (core.tiling.grid_tiles). Zoom (g-1)(1-o)+1, 1+g² detector passes per image
# (o = search.tiling_overlap). The cap 6 (37 passes) is set by cost — when a grid
# falls short the planner raises it one step at a time.
TILING_GRIDS: tuple[int, ...] = (2, 3, 4, 5, 6)

AXES: dict[str, tuple] = {
    "exemplar": ("none", "base", "fn_arm"),   # visual exemplar (fn_arm ⊃ base)
    # The value is the **crop→text converter** that produces alias phrases.
    #   vlm = a multimodal LLM that looks at the crop (free-form, costs calls)
    "alias": (False, "vlm"),
    # The value is the **mining source** for suppression phrases — FP crop
    # captions alone mostly trip the safety rules (own parts, same kind,
    # category name), so the material often ends up empty.
    #   fp_crop    = captions of regions mistaken for the target (suppress the FP itself)
    #   gt_crop    = captions of other classes' GT crops (suppress confused classes)
    #   background = captions of FPs not overlapping any GT (suppress background FPs)
    "text_negatives": (False, "fp_crop", "gt_crop", "background"),
    # visual negatives take the same three sources — the crops themselves go in
    # as negative image prompts instead of captions (semantic queries only;
    # screen picks the neg_eta grid)
    "visual_negatives": (False, "fp_crop", "gt_crop", "background"),
    "tiling": (False, *TILING_GRIDS),         # grid g (see TILING_GRIDS above)
    "rescorer": (False, True),                # α_c grid (fit_train/fit_dev)
    "lora": (False, True),                    # YOLOE tail fine-tune + λ arms (name kept)
    "crop_classifier": (False, True),         # re-classify / blend detections (fixes class_confusion, suppresses background FPs)
    # --- count axes (grid 1·2·4; exemplar up to 8 to see whether it hits the cap) ---
    # None = the config's default quota (MineCfg's *_per_class) and the safe
    # point. Meaningful only when the matching feature axis is on (enforced by
    # validate_axes).
    "exemplar_base_k": (None, 1, 2, 4, 8),         # base exemplars per class
    "exemplar_fn_k": (None, 1, 2, 4),                  # extra FN-arm exemplars per class
    "visual_negative_k": (None, 1, 2, 4),            # visual negatives per class
    "text_negative_k": (None, 1, 2, 4),       # suppression phrases per class
}

# negative axes whose values are the mining source (fp_crop | gt_crop | background)
NEGATIVE_AXES = ("text_negatives", "visual_negatives")

BASELINE: dict = {"exemplar": "none", "alias": False, "text_negatives": False,
                  "visual_negatives": False, "tiling": False, "rescorer": False,
                  "lora": False, "crop_classifier": False,
                  "exemplar_base_k": None, "exemplar_fn_k": None,
                  "visual_negative_k": None, "text_negative_k": None}


@dataclass(frozen=True)
class DatasetFacts:
    """Facts used to block, up front, candidates whose **outcome is already decided
    by the data**.

    Only values that follow directly from the dataset and config, without
    inference — conditions whose outcome depends on inference (e.g. the background
    FP count behind the crop classifier tripwire) do not belong here. A None
    field means no constraint.
    """
    n_classes: int | None = None
    n_fit_train: int | None = None        # number of fit_train images
    lora_min_images: int | None = None    # config lora.min_images

    def lora_block(self) -> str | None:
        """The reason, if lora (tail fine-tune) training is certain to be skipped by its tripwire."""
        if self.n_fit_train is None or self.lora_min_images is None:
            return None
        if self.n_fit_train >= self.lora_min_images:
            return None
        return (f"fit_train {self.n_fit_train} images < lora.min_images "
                f"{self.lora_min_images} — training would be skipped, same as current")


def axis_values(facts: DatasetFacts | None = None) -> dict[str, tuple]:
    """The value space that is meaningful for the dataset — the axis list the
    planner prompt carries.

    Leaves out the values validate_axes blocks with facts — gt_crop on a single
    class, lora whose training would be skipped. Without facts this is AXES
    unchanged."""
    out = dict(AXES)
    if facts is None:
        return out
    if facts.n_classes == 1:
        for axis in NEGATIVE_AXES:
            out[axis] = tuple(v for v in AXES[axis] if v != "gt_crop")
    if facts.lora_block():
        out["lora"] = (False,)
    return out


class AxesError(ValueError):
    """An axes contract violation — str is the full message, en a short line for the planner prompt."""

    def __init__(self, msg: str, en: str):
        super().__init__(msg)
        self.en = en


def validate_axes(axes: dict, *, facts: DatasetFacts | None = None) -> dict:
    """Unknown keys/values raise. Unspecified axes are filled from BASELINE (safe point).

    With facts, combinations whose outcome is already decided on this dataset
    (material empty by definition, training that would be skipped) are rejected
    too. Callers that re-key recorded axes (tried_combos) do not pass it."""
    unknown = sorted(set(axes) - set(AXES))
    if unknown:
        raise AxesError(f"axes: unknown key {unknown} — allowed: {sorted(AXES)}",
                        f"unknown axis {', '.join(unknown)}")
    full = {**BASELINE, **axes}
    for k, v in full.items():
        # 1 == True and 2.0 == 2, so `in` alone lets types leak in — tiling is an int grid
        if k == "tiling" and v is not False and (
                isinstance(v, bool) or not isinstance(v, int)):
            raise AxesError(f"axes.tiling: {v!r} must be one of {AXES[k]} "
                            "(grid g — an integer)", f"tiling={json.dumps(v)} is not a grid value")
        if v not in AXES[k]:
            raise AxesError(f"axes.{k}: {v!r} must be one of {AXES[k]}",
                            f"{k}={json.dumps(v)} is not a valid value")
    # Visual negatives ride only on semantic queries — cannot be on without exemplars
    if full["visual_negatives"] and full["exemplar"] == "none":
        raise AxesError("axes.visual_negatives requires an exemplar "
                        "(visual negatives apply to semantic queries only — text "
                        "suppression is the text_negatives axis)",
                        'visual_negatives needs exemplar != "none"')
    # Count axes are meaningful only when their feature axis is on — meaningless
    # combos would pollute the tried set and the candidate budget, so they raise
    requires = [("exemplar_base_k", full["exemplar"] != "none", "exemplar"),
                ("exemplar_fn_k", full["exemplar"] == "fn_arm", "exemplar=fn_arm"),
                ("visual_negative_k", full["visual_negatives"], "visual_negatives"),
                ("text_negative_k", full["text_negatives"], "text_negatives")]
    for key, ok, need in requires:
        if full[key] is not None and not ok:
            raise AxesError(f"axes.{key} is only meaningful when {need} "
                            f"(got {full[key]})", f"{key} applies only with {need}")
    # gt_crop is sourced from **other classes'** GT crops — with a single class
    # the pool is empty by definition, so the candidate would be rejected as
    # no_material and only use up a round slot. fp_crop and background cover the
    # same "suppress look-alikes" role.
    if facts is None:
        return full
    for axis in NEGATIVE_AXES:
        if facts.n_classes == 1 and full[axis] == "gt_crop":
            raise AxesError(f"axes.{axis}=gt_crop is only meaningful with 2+ "
                            "classes (sourced from other classes' GT) — on a "
                            "single-class dataset use fp_crop or background",
                            f"{axis}=gt_crop needs two or more classes")
    # lora (tail fine-tune) is rejected without training if fit_train has fewer
    # than lora.min_images images — the outcome follows from the data, so block it
    # before it uses a candidate slot
    if full["lora"] and (why := facts.lora_block()):
        raise AxesError(f"axes.lora=True: {why}",
                        "lora cannot train: too few fit_train images")
    return full


def uses_disabled_axes(axes: dict, disabled_axes, disabled_values=None,
                       base: dict | None = None) -> bool:
    """Do the axes turn on a banned axis/value? (per-run proposal bans)

    - disabled_axes: ban a whole axis — only moving it off its safe point
      (BASELINE) counts.
    - disabled_values: {axis: [banned value, ...]} — ban specific values only;
      other values of the axis can still be proposed.

    A value equal to base (the current champion's) is inherited, not turned on —
    blocking it would reject every candidate once the scope is narrowed on resume.
    """
    base = base or {}

    def _new(a) -> bool:
        return a not in base or axes.get(a) != base[a]

    if any(axes.get(a) != BASELINE.get(a) and _new(a) for a in (disabled_axes or ())):
        return True
    return any(axes.get(a) in vals and _new(a)
               for a, vals in (disabled_values or {}).items())


def merge_onto(base: dict, delta: dict) -> dict:
    """Apply a candidate increment (delta) on top of base (the current champion).

    When a feature axis is turned off or changed, the dependent and count axes
    the delta does not name go back to their safe point — if the champion's
    stayed, validate_axes' dependency rules (e.g. exemplar_fn_k only with
    exemplar=fn_arm) would drop the whole candidate."""
    out = {**BASELINE, **(base or {}), **delta}

    def reset(key, value):
        if key not in delta:
            out[key] = value

    if "exemplar" in delta:
        if out["exemplar"] == "none":
            for key in ("exemplar_base_k", "exemplar_fn_k", "visual_negative_k"):
                reset(key, None)
            reset("visual_negatives", False)
        elif out["exemplar"] != "fn_arm":
            reset("exemplar_fn_k", None)
    if "visual_negatives" in delta and not out["visual_negatives"]:
        reset("visual_negative_k", None)
    if "text_negatives" in delta and not out["text_negatives"]:
        reset("text_negative_k", None)
    return out


PLAN_DIFFERS_MAX = 300      # plan.differs length cap — the same as the response schema's maxLength


def _plan_fields(c: dict) -> dict:
    """The planner's expected gain, cost and variant note — trimmed and kept if
    present; a missing or malformed field is just left out. The candidate is
    not dropped nor sent to the deterministic fallback (this is extra
    information on the proposal, not used for any decision).

    expected_gain: expected holdout micro F(β) Δ — [-1, 1]. cost_min: expected
    wall clock (minutes) — 0 or more. differs: if a variant of a clearly-lost
    value, what is different this time."""
    out = {}
    for key, lo, hi in (("expected_gain", -1.0, 1.0), ("cost_min", 0.0, 1e5)):
        try:
            v = float(c.get(key))
        except (TypeError, ValueError):
            continue
        if v == v and lo <= v <= hi:          # excludes NaN
            out[key] = round(v, 4)
    differs = c.get("differs")
    if isinstance(differs, str) and differs.strip():
        out["differs"] = differs.strip()[:PLAN_DIFFERS_MAX]
    return out


def sanitize_candidates(raw, *, max_candidates: int,
                        dropped: list | None = None,
                        disabled_axes=(), disabled_values=None,
                        base: dict | None = None,
                        facts: DatasetFacts | None = None) -> list[dict]:
    """Filter the planner's (especially the LLM's) candidate list against the contract.

    Given base (the current champion's axes), candidate axes are read as an
    **increment** over it — omitted axes keep the champion's value, and turning
    one off takes an explicit False. Without base, omitted axes are BASELINE
    (the old behavior: an LLM listing only the axes it changes silently turned
    the champion's axes off).

    Invalid candidates are removed — the planner only proposes and the gate
    adopts, so leniency here is harmless while propagating errors kills the loop.

    **No silent drops**: given a `dropped` list, record each discarded
    candidate with its reason. Without it the planner cannot tell "evaluated and
    rejected" from "never evaluated" and resubmits the same candidate. The
    runner's combo_tried follows the same principle.

    Five reasons: invalid_axes (contract violation) · disabled_axes (axis/value
    banned for this run, unless inherited from base — the champion's axes) ·
    no_change (same as base, no increment to evaluate) · duplicate_axes (repeat
    within one response) · max_candidates (over quota).
    """
    out, seen = [], set()
    merge_base = None
    if base:
        try:
            merge_base = validate_axes(base)
        except ValueError:              # a base that does not validate — read axes as-is
            merge_base = None

    def _drop(i, c, why: str, note: str, merged: dict | None = None,
              invalid: str | None = None) -> None:
        # a candidate merged onto base is recorded with the merged axes — so
        # tried_combos does not fill the increment from BASELINE and block a
        # combination that was never tried
        if dropped is None:
            return
        cand = c if isinstance(c, dict) else {}
        axes = merged or cand.get("axes")
        dropped.append({"id": str(cand.get("id") or f"cand_{i}"),
                        "axes": axes if isinstance(axes, dict) else {},
                        "rationale": str(cand.get("rationale", "")),
                        "skipped": why, "note": note}
                       | ({"invalid": invalid} if invalid else {}))

    for i, c in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(c, dict) or not isinstance(c.get("axes"), dict):
            _drop(i, c, "invalid_axes", "not a candidate — an axes dict is required")
            continue
        try:
            axes = validate_axes(merge_onto(merge_base, c["axes"]) if merge_base
                                 else c["axes"], facts=facts)
        except ValueError as exc:
            # Keep what was violated — note is the full message, invalid (a short
            # line) goes back into the next prompt as the reason and stops the
            # same violation repeating (planner.llm._reason)
            _drop(i, c, "invalid_axes", str(exc), invalid=getattr(exc, "en", None))
            continue
        if uses_disabled_axes(axes, disabled_axes, disabled_values, base=base):
            _drop(i, c, "disabled_axes", "uses an axis/value disabled for this run", axes)
            continue
        if merge_base and axes == merge_base:
            _drop(i, c, "no_change", "same combination as the current champion — "
                                     "no increment to evaluate", axes)
            continue
        key = tuple(sorted(axes.items()))
        if key in seen:
            _drop(i, c, "duplicate_axes", "duplicate combination within the same response",
                  axes)
            continue
        if len(out) >= max_candidates:
            _drop(i, c, "max_candidates", f"exceeds the cap of {max_candidates} candidates",
                  axes)
            continue
        seen.add(key)
        plan = _plan_fields(c)
        out.append({"id": str(c.get("id") or f"cand_{i}"), "axes": axes,
                    "rationale": str(c.get("rationale", ""))}
                   | ({"plan": plan} if plan else {}))
    return out
