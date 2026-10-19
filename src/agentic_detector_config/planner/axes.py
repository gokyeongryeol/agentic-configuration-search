"""axes schema for candidate settings — the shared contract between the
planner (proposal) and the loop (materialize).

Every axis has a safe point where turning it off is exactly the current
behavior. BASELINE is round 0's starting point and every axis's safe point:
a text prompt of the class name alone.
"""

from __future__ import annotations

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
    "visual_negatives": (False, True),                # visual negatives + neg_eta grid
    "tiling": (False, True),                  # tiling grid (search.tiling_configs)
    "rescorer": (False, True),                # α_c grid (fit_train/fit_dev)
    "lora": (False, True),                    # YOLOE tail fine-tune + λ arms (name kept)
    "crop_classifier": (False, True),         # re-classify detections (fixes class_confusion, drops background FPs)
    # --- count axes (grid 1·2·4; exemplar up to 8) ---
    # None = the config's default quota (MineCfg's *_per_class), which is the
    # count axes' safe point. Meaningful only when the matching feature axis is
    # on (enforced by validate_axes). config/default.yaml is canonical for the
    # concrete defaults.
    # Up to 8 so we can tell whether the grid hits its upper bound
    "exemplar_k": (None, 1, 2, 4, 8),         # base exemplars per class
    "fn_k": (None, 1, 2, 4),                  # extra FN-arm exemplars per class
    "visual_negative_k": (None, 1, 2, 4),            # visual negatives per class
    "text_negative_k": (None, 1, 2, 4),       # suppression phrases per class
}

BASELINE: dict = {"exemplar": "none", "alias": False, "text_negatives": False,
                  "visual_negatives": False, "tiling": False, "rescorer": False,
                  "lora": False, "crop_classifier": False,
                  "exemplar_k": None, "fn_k": None,
                  "visual_negative_k": None, "text_negative_k": None}


def axis_values(n_classes: int | None = None) -> dict[str, tuple]:
    """The value space that is meaningful for the dataset — the axis list the
    planner prompt carries.

    For a single-class dataset text_negatives=gt_crop is removed (see
    validate_axes). Without n_classes this is AXES unchanged."""
    if n_classes != 1:
        return dict(AXES)
    return {**AXES, "text_negatives": tuple(
        v for v in AXES["text_negatives"] if v != "gt_crop")}


def validate_axes(axes: dict, *, n_classes: int | None = None) -> dict:
    """Unknown keys/values raise. Unspecified axes are filled from BASELINE (safe point).

    With n_classes, combinations whose material is empty by definition on this
    dataset are rejected too. Callers that re-key old records (tried_combos)
    do not pass it."""
    unknown = sorted(set(axes) - set(AXES))
    if unknown:
        raise ValueError(f"axes: unknown key {unknown} — allowed: {sorted(AXES)}")
    full = {**BASELINE, **axes}
    for k, v in full.items():
        if v not in AXES[k]:
            raise ValueError(f"axes.{k}: {v!r} must be one of {AXES[k]}")
    # Visual negatives ride only on semantic queries — cannot be on without exemplars
    if full["visual_negatives"] and full["exemplar"] == "none":
        raise ValueError("axes.visual_negatives=True requires an exemplar "
                         "(visual negatives apply to semantic queries only — text "
                         "suppression is the text_negatives axis)")
    # Count axes are meaningful only when their feature axis is on — meaningless
    # combos would pollute the tried set and the candidate budget, so they raise
    requires = [("exemplar_k", full["exemplar"] != "none", "exemplar"),
                ("fn_k", full["exemplar"] == "fn_arm", "exemplar=fn_arm"),
                ("visual_negative_k", full["visual_negatives"], "visual_negatives"),
                ("text_negative_k", full["text_negatives"], "text_negatives")]
    for key, ok, need in requires:
        if full[key] is not None and not ok:
            raise ValueError(f"axes.{key} is only meaningful when {need} "
                             f"(got {full[key]})")
    # gt_crop is sourced from **other classes'** GT crops — with a single class
    # the pool is empty by definition, so the candidate would be rejected as
    # no_material and only use up a round slot. fp_crop and background cover the
    # same "suppress look-alikes" role.
    if n_classes == 1 and full["text_negatives"] == "gt_crop":
        raise ValueError("axes.text_negatives=gt_crop is only meaningful with 2+ "
                         "classes (sourced from other classes' GT) — on a "
                         "single-class dataset use fp_crop or background")
    return full


def uses_disabled_axes(axes: dict, disabled_axes, disabled_values=None) -> bool:
    """Do the axes use a banned axis/value? (per-run proposal bans)

    - disabled_axes: ban a whole axis — only moving it off its safe point
      (BASELINE) counts.
    - disabled_values: {axis: [banned value, ...]} — ban specific values only;
      other values of the axis can still be proposed.
    """
    if any(axes.get(a) != BASELINE.get(a) for a in (disabled_axes or ())):
        return True
    return any(axes.get(a) in vals for a, vals in (disabled_values or {}).items())


def sanitize_candidates(raw, *, max_candidates: int,
                        dropped: list | None = None,
                        disabled_axes=(), disabled_values=None,
                        n_classes: int | None = None) -> list[dict]:
    """Filter the planner's (especially the LLM's) candidate list against the contract.

    Invalid candidates are removed — the planner only proposes and the gate
    adopts, so leniency here is harmless while propagating errors kills the loop.

    **No silent drops**: given a `dropped` list, record each discarded
    candidate with its reason. Without it two things quietly break — only 2 of
    3 proposals get evaluated, shrinking the round budget, and the planner
    cannot tell "evaluated and rejected" from "never evaluated", so it resubmits
    the same candidate next round (observed: three rounds in a row in one run).
    The runner's combo_tried already follows the same principle.

    Four reasons: invalid_axes (contract violation) · disabled_axes (axis/value
    banned for this run) · duplicate_axes (repeat within one response) ·
    max_candidates (over quota).
    """
    out, seen = [], set()

    def _drop(i, c, why: str, note: str) -> None:
        if dropped is None:
            return
        cand = c if isinstance(c, dict) else {}
        axes = cand.get("axes")
        dropped.append({"id": str(cand.get("id") or f"cand_{i}"),
                        "axes": axes if isinstance(axes, dict) else {},
                        "rationale": str(cand.get("rationale", "")),
                        "skipped": why, "note": note})

    for i, c in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(c, dict) or not isinstance(c.get("axes"), dict):
            _drop(i, c, "invalid_axes", "not a candidate — an axes dict is required")
            continue
        try:
            axes = validate_axes(c["axes"], n_classes=n_classes)
        except ValueError as exc:
            # Carry the ValueError message verbatim — it goes back into the next
            # round's prompt and is the only signal that stops the same violation.
            _drop(i, c, "invalid_axes", str(exc))
            continue
        if uses_disabled_axes(axes, disabled_axes, disabled_values):
            _drop(i, c, "disabled_axes", "uses an axis/value disabled for this run")
            continue
        key = tuple(sorted(axes.items()))
        if key in seen:
            _drop(i, c, "duplicate_axes", "duplicate combination within the same response")
            continue
        if len(out) >= max_candidates:
            _drop(i, c, "max_candidates", f"exceeds the cap of {max_candidates} candidates")
            continue
        seen.add(key)
        out.append({"id": str(c.get("id") or f"cand_{i}"), "axes": axes,
                    "rationale": str(c.get("rationale", ""))})
    return out
