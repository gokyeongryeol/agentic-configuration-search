"""Axis value ledger — what happened to each (axis, value) under the current
champion.

Exact combination dedup (combo_tried) only blocks the same combination.
Variants of a value that lost badly (another k, a combination with other axes)
went all the way through screen and holdout only to confirm the same result
(soccer and aquarium validation runs). Variants are not blocked — they can work
with another k or another combination. Instead, variants of a **clearly-lost
value** are measured on a fit_dev subset first (the runner's lost-value
prescreen). This module computes that verdict and the planner prompt's ledger
table from history alone — the same verdict on resume.

**The verdict is made on fit_dev only.** It reads only the candidate row's
`fit_dev_cmp` (the one-sided paired-bootstrap upper bound `ub` of (candidate −
champion) fit_dev micro F(β) left by screen).

- clearly lost: a row changing only that value (a k change on the same axis is
  the same value) has `ub < 0`, and no row containing the value (combinations
  included) has `ub ≥ 0`.
- near miss: some row containing the value has `ub ≥ 0` — not prescreened.
- a row whose per-image fit_dev rows all equal the champion's
  (`fit_dev_cmp.identical`, fit_dev_identical — delta=0, ub=0) is neither a
  near miss nor a loss — a variant that changed nothing must not release a lost
  value.
- holdout results (gate Δ) are only shown in the table, never used for the
  verdict — filtering future screens with a value that came from holdout would
  reuse holdout once more.

**Only rounds under the current champion count** (from the round after its
adoption — the same rule as the tiling fold gate, planner.gates.under_champion).
A value that lost under an old champion may work in the new champion's context.

**Same value**: the axis's main value (alias=vlm, exemplar=fn_arm,
text_negatives=fp_crop …). Quantity axes (exemplar_k, fn_k, text_negative_k,
visual_negative_k) are variants of their function axis's value. A candidate
changing several axes is counted once per contained value — a combination
containing any clearly-lost value is a variant of that value (prescreened). A
combination's loss cannot be attributed to one value, so it does not make the
value "clearly lost".

**Prescreened axes** are only the prompt and inference axes (alias, exemplar,
text_negatives, visual_negatives, tiling). For rescorer, lora and
crop_classifier the expensive part (rescorer refit, LoRA training, classifier
training) comes before the fit_dev comparison, so a subset screen has nothing
to save (the lora and classifier arms already put a fit_dev margin gate before
holdout). A value that ever changed the champion (an adopted candidate's
increment) is not prescreened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gates import champion_round, tiling_gate, under_champion

# quantity axis → function axis — a candidate changing only the quantity is a
# variant of that function axis's value
K_PARENT = {"exemplar_k": "exemplar", "fn_k": "exemplar",
            "text_negative_k": "text_negatives",
            "visual_negative_k": "visual_negatives"}
# axes the lost-value prescreen can filter — those whose expensive part comes
# after the fit_dev comparison
PRESCREEN_AXES = ("alias", "exemplar", "text_negatives", "visual_negatives", "tiling")

# history skipped reason → ledger outcome group
_BLOCKED = {"tiling_blocked", "combo_tried", "axis_disabled", "carried_duplicate",
            "duplicate_axes", "max_candidates", "no_change", "invalid_axes",
            "detector_state_off", "compound_server_state", "missing_hook",
            "lora_deps", "adapt_ctl"}
_OUTCOME = {"lost_value_prescreen": "prescreened", "axis_unused": "unused",
            "same_setting": "same_setting", "no_material": "no_material",
            "rescorer_identity": "no_effect", "lora_gate": "arm_gate",
            "classifier_gate": "arm_gate", "error": "error",
            # reached screen but dropped out at the pre-holdout fit_dev verdict
            "fit_dev_lost": "screened", "fit_dev_identical": "same_setting"}


def label(key: tuple) -> str:
    axis, value = key
    return f"{axis}={value}"


def value_keys(axes: dict, champion: dict) -> tuple:
    """The (axis, main value) pairs a candidate changed from the champion — a
    sorted tuple.

    Quantity axes fold into the function axis's (candidate) value:
    exemplar=fn_arm + fn_k=4 is the single ("exemplar", "fn_arm"). Changing only
    exemplar_k is a variant of the champion's exemplar value."""
    axes, champion = axes or {}, champion or {}
    keys = set()
    for k, v in axes.items():
        if champion.get(k) == v:
            continue
        parent = K_PARENT.get(k)
        keys.add((parent, axes.get(parent)) if parent else (k, v))
    return tuple(sorted(keys, key=lambda kv: (kv[0], str(kv[1]))))


def _asked(c: dict) -> dict:
    return c.get("requested_axes") or c.get("axes") or {}


def _outcome(c: dict, evaluated: dict) -> str:
    s = c.get("skipped")
    if s is None:
        if c.get("adopted"):
            return "adopted"
        return "evaluated" if c.get("id") in evaluated else "screened"
    if s in _BLOCKED:
        return "blocked"
    return _OUTCOME.get(s, str(s))


def _merged_rows(entry: dict) -> list[dict]:
    """Merge two rows with the same id in one round (the lora and classifier arms
    leave both a screen row and an arm-gate rejection row) — the later row
    wins."""
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for c in entry.get("candidates", []):
        cid = str(c.get("id"))
        if cid not in by_id:
            order.append(cid)
            by_id[cid] = dict(c)
        else:
            by_id[cid] = {**by_id[cid], **c}
    return [by_id[i] for i in order]


@dataclass
class ValueRecord:
    """The record of one (axis, value) under the current champion."""
    key: tuple
    tries: int = 0                       # candidates containing this value (combinations included)
    single: int = 0                      # candidates changing only this value
    outcomes: dict = field(default_factory=dict)   # outcome → count
    best_fit_dev: dict | None = None     # best fit_dev_cmp (delta, ub, n)
    best_holdout: dict | None = None     # best holdout gate (delta, lo) — display only
    minutes: float = 0.0                 # summed wall clock of candidates containing it (min)
    near_miss: bool = False              # some containing row has fit_dev ub ≥ 0 (identical rows excluded)
    single_loss: bool = False            # some single-increment row has fit_dev ub < 0
    evidence: dict | None = None         # evidence of the clear loss (single row with the worst ub)
    folded: bool = False                 # a grid the tiling fold gate blocks
    protected: bool = False              # a value that ever changed the champion

    @property
    def clearly_lost(self) -> bool:
        return self.single_loss and not self.near_miss

    @property
    def prescreened(self) -> bool:
        """Whether candidates containing this value are measured on a fit_dev
        subset first."""
        return (self.clearly_lost and not self.protected
                and self.key[0] in PRESCREEN_AXES)

    @property
    def status(self) -> str:
        if self.folded:
            return "folded"
        if self.prescreened:
            return "clearly-lost (prescreened)"
        if self.clearly_lost:
            return "clearly-lost"
        if self.near_miss:
            return "near-miss"
        # outcomes that got furthest come first — reaching holdout says the most
        for kind in ("evaluated", "screened", "arm_gate", "unused", "no_material",
                     "same_setting", "no_effect", "blocked", "error"):
            if self.outcomes.get(kind):
                return kind
        return next(iter(self.outcomes), "pending")


def adopted_keys(history: list[dict]) -> set[tuple]:
    """(axis, value) pairs that ever changed the champion — the values an adopted
    candidate changed from that round's starting champion. Collected over the
    whole history."""
    out: set[tuple] = set()
    for i, e in enumerate(history):
        start = ((history[i - 1].get("champion") or {}).get("axes") if i > 0
                 else None) or (e.get("champion") or {}).get("axes") or {}
        for c in e.get("candidates", []):
            if c.get("adopted"):
                out.update(value_keys(_asked(c), start))
    return out


def axis_ledger(history: list[dict]) -> dict[tuple, ValueRecord]:
    """(axis, value) → ValueRecord under the current champion. A pure function of
    history."""
    if not history:
        return {}
    champ = (history[-1].get("champion") or {}).get("axes") or {}
    champ_fd = (history[-1].get("champion") or {}).get("fit_dev_micro_f_beta")
    protected = adopted_keys(history)
    gate = tiling_gate(history)
    out: dict[tuple, ValueRecord] = {}
    for e in under_champion(history):
        evaluated = (e.get("decision") or {}).get("evaluated") or {}
        for c in _merged_rows(e):
            keys = value_keys(_asked(c), champ)
            if not keys:
                continue
            cmp = c.get("fit_dev_cmp") or {}
            ub = cmp.get("ub")
            if ub is None and c.get("fit_dev_micro_f_beta") is not None \
                    and champ_fd is not None:
                # a screen result without comparison rows (arms etc.) — only the
                # point estimate is shown (not used for the verdict: without an
                # upper bound noise and loss cannot be told apart)
                cmp = {"delta": round(c["fit_dev_micro_f_beta"] - champ_fd, 4)}
            ev = evaluated.get(c.get("id")) or {}
            g = ev.get("gate") or {}
            minutes = float((c.get("wall_min") or {}).get("total") or 0.0)
            kind = _outcome(c, evaluated)
            for key in keys:
                rec = out.setdefault(key, ValueRecord(key=key))
                rec.tries += 1
                rec.single += len(keys) == 1
                rec.outcomes[kind] = rec.outcomes.get(kind, 0) + 1
                rec.minutes += minutes
                if cmp.get("delta") is not None and (
                        rec.best_fit_dev is None
                        or cmp["delta"] > rec.best_fit_dev["delta"]):
                    rec.best_fit_dev = dict(cmp)
                if ub is not None and not cmp.get("identical"):
                    # a row whose per-image rows all equal the champion's
                    # (fit_dev_identical) changed nothing — neither a near miss
                    # nor a loss (with delta=0, ub=0 it would otherwise release
                    # a lost value as a near miss)
                    # a prescreen-rejected row's ub is a subset value — it does not
                    # create a new loss (the value already lost) but can be a near miss
                    if ub >= 0:
                        rec.near_miss = True
                    elif len(keys) == 1 and kind != "prescreened":
                        rec.single_loss = True
                        if rec.evidence is None or ub < rec.evidence["ub"]:
                            rec.evidence = {"id": c.get("id"), "round": e.get("round"),
                                            **cmp}
                if g.get("delta") is not None and (
                        rec.best_holdout is None
                        or g["delta"] > rec.best_holdout["delta"]):
                    rec.best_holdout = {"delta": g["delta"], "lo": g.get("lo")}
    for key, rec in out.items():
        rec.protected = key in protected
        axis, value = key
        rec.folded = bool(axis == "tiling" and value and gate.block({"tiling": value}))
    return out


def lost_values(history: list[dict]) -> dict[tuple, ValueRecord]:
    """(axis, value) pairs to prescreen — clearly lost under the current champion
    and not protected."""
    return {k: r for k, r in axis_ledger(history).items() if r.prescreened}


def variant_of(axes: dict, champion: dict, lost: dict) -> list[str]:
    """Names of the clearly-lost values a candidate contains — empty means not a
    variant."""
    return [label(k) for k in value_keys(axes, champion) if k in lost]


def tried_under_champion(history: list[dict]) -> list[str]:
    """Combinations tried under the current champion (increments as requested),
    plus earlier combinations that do not turn off a champion axis — only those
    the planner might actually propose again (the runner's combination filter
    blocks the whole history). Old combinations turning an axis off are rarely
    proposed again and are prompt noise."""
    if not history:
        return []
    champ = (history[-1].get("champion") or {}).get("axes") or {}
    cr = champion_round(history)
    seen: dict[str, None] = {}
    for e in history:
        recent = cr is not None and e.get("round", 0) > cr
        for c in e.get("candidates", []):
            for axes in (c.get("axes"), c.get("requested_axes")):
                if not isinstance(axes, dict):
                    continue
                delta = {k: v for k, v in axes.items() if champ.get(k) != v}
                if not delta:
                    continue
                turns_off = any(champ.get(k) not in (None, False, "none")
                                and v in (None, False, "none")
                                and k not in K_PARENT for k, v in delta.items())
                if not recent and turns_off:
                    continue
                seen[", ".join(f"{k}={v}" for k, v in sorted(delta.items()))] = None
    return list(seen)
