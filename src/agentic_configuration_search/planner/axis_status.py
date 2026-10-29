"""Axis status — ON · OPEN · OFF · PROBE per function axis from the whole run's record.

The axis ledger (planner.ledger) looks only under the current champion and resets
when the champion changes. On its own, the same axis is tried from scratch under
every champion. This module counts consecutive failures per axis over the
**whole run** and turns an axis that keeps not working OFF, to cut search cost.
The prompt's untried-axis hint and diagnosis digest follow this status too (OFF
axes are not suggested). A pure function of history, so resume decides the same.

Rules (per axis, function axes — count axes fold into their function axis):
- Each round counts once per axis — if any candidate that changed the axis that
  round wins, it is a success (streak back to 0); if its measured outcomes are
  all failures, one failure.
  - Failure: held-out gate failed, clearly lost on fit_dev (fit_dev_lost ·
    lost_value_prescreen), equal to the champion (fit_dev_identical, and
    same_setting when it matched the champion), the arm's own verdict
    (rescorer_identity · lora_gate · classifier_gate — that arm axis only), the
    setting did not use the axis (unused_axes — the unused axis only; the row's
    other axes take the row's outcome).
  - A combination's outcome goes to every axis it changed (which one is to blame
    cannot be told apart, but retirement needs consecutive failures, so an axis
    that wins once comes back).
  - Success: adopted, or passed the gate but lost to a better candidate.
  - Not counted: no material (no_material · empty_axis — including the
    classifier and lora tripwires; a problem of the value, not the axis), a
    setting equal to a sibling's or an earlier candidate's (same_setting),
    outside screen's top K, and drops that were never measured (hygiene, combo
    dedup). A candidate that only turns an axis off does not try it.
- `retire_after` failures in a row turn the axis OFF. A planner candidate that
  changes only OFF axes is recorded by the runner as `axis_retired` without
  evaluation.
- When the champion changes, OFF axes reopen as PROBE — the streak resumes from
  retire_after − 1, so one more failure turns it OFF again (the context changed,
  so it is measured once more).
- An axis the champion has on is ON and never blocked (detector-state axes
  cannot even be turned off).
- Worth re-measuring (`stale`): an OPEN or PROBE axis whose last measured
  outcome was a failure (held-out fail, fit_dev lost, arm gate), after which the
  champion changed and the axis has not been tried under the new champion yet (a
  try that ended without an outcome — outside screen's top K, no material, same
  as a sibling — counts as a try). The context changed, so the result may differ
  (detector-state axes are sensitive to upstream changes). Unused and no-change
  are left out: they are likely the same after a champion change. The axis board
  marks it with NOT_RETRIED and the prompt suggests it before repeating unused
  axes.

It reads held-out results (gate failures) — "what to try next" is within what
the held-out split may change (the core invariant). It never touches materials
or thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .axes import BASELINE
from .ledger import K_PARENT, _merged_rows

# outcomes counted as a failure of the axis (axis_unused is counted separately, on the unused axis only)
_FAIL = {"fit_dev_lost": "fit_dev lost", "lost_value_prescreen": "fit_dev lost",
         "fit_dev_identical": "no change", "same_setting": "no change",
         "rescorer_identity": "no change", "lora_gate": "arm gate",
         "classifier_gate": "arm gate"}
# rows dropped without being measured — not counted as trying the axis either
# (hygiene, combo dedup, not wirable). error is a runtime failure, so nothing was measured
UNMEASURED = {"combo_tried", "no_change", "champion_combo", "axis_disabled",
              "axis_retired", "tiling_blocked", "error", "max_candidates",
              "invalid_axes", "duplicate_axes", "carried_duplicate", "missing_hook",
              "lora_deps", "adapt_ctl", "detector_state_off", "compound_server_state"}
# the arm's own verdict — a failure of that arm axis (not of other axes changed with it)
_ARM_SKIP = {"rescorer_identity": "rescorer", "lora_gate": "lora",
             "classifier_gate": "crop_classifier"}
_HELD_OUT_FAIL = "held-out fail"
NEVER_TRIED = "never tried in this run"   # the Rules' untried rule and the axis board point to this text
NOT_RETRIED = "not retried under this champion"   # the Rules, the re-ask and the axis board point to this text
# failures worth re-measuring after a champion change — unused and no-change are
# likely the same in a new context
_RETRY_KINDS = {_HELD_OUT_FAIL, "fit_dev lost", "arm gate"}
# on a tie between failure kinds in one round, the earlier one — no reliance on set order (PYTHONHASHSEED)
_KIND_ORDER = ("held-out fail", "fit_dev lost", "arm gate", "no change", "unused")
_UNUSED = "unused"
_OFF_VALUES = (False, None, "none")


@dataclass
class AxisState:
    axis: str
    status: str = "open"                 # on | open | off | probe
    streak: int = 0                      # consecutive failures
    fails: dict = field(default_factory=dict)      # failure kind → count (whole run)
    fail_rounds: list = field(default_factory=list)
    wins: int = 0                        # adopted, or passed the gate but lost
    measured: int = 0                    # rounds measured (once per round × axis)
    tried: int = 0                       # attempts made (without UNMEASURED drops — no material and outside top K count)
    off_since: int | None = None         # round it turned OFF
    last_round: int | None = None        # last measured round
    last_tried: int | None = None        # last round it was tried (tries without an outcome included)
    last_kind: str | None = None         # that round's outcome (None = a win)
    no_material: list = field(default_factory=list)   # values whose material came out empty

    def stale(self, champion_round: int) -> bool:
        """Was the last measurement a failure worth re-measuring, after which (at the
        end of round champion_round) the champion changed, with no try under the new
        champion yet? OPEN and PROBE axes only."""
        # a try that ended without an outcome (outside screen's top K, no material,
        # same as a sibling) was still proposed again under the new champion —
        # compare the try round, not the measured round
        return (self.status in ("open", "probe") and self.last_round is not None
                and self.last_kind in _RETRY_KINDS and champion_round > 0
                and champion_round >= max(self.last_round, self.last_tried or 0))

    def record(self) -> str:
        """One cell of the whole-run record — wins, failure kinds, empty material, whether tried, recent failed rounds."""
        rec = []
        if self.wins:
            rec.append(f"won x{self.wins}")
        for kind, n in sorted(self.fails.items(), key=lambda kv: -kv[1]):
            rec.append(f"{kind} x{n}")
        if self.fail_rounds and rec:
            # recent failed rounds right after the failure record — not read as the empty-material rounds
            rec[-1] += f" (last failed in R{', R'.join(map(str, self.fail_rounds[-4:]))})"
        # an empty source value and an arm that skipped training for too little data
        # (lora and classifier tripwires — the value is True) mean different things,
        # so they are written apart
        skipped = sum(v is True for v in self.no_material)
        empty = list(dict.fromkeys(v for v in self.no_material if v is not True))
        if empty:
            rec.append(f"empty material for {', '.join(map(str, empty))}")
        if skipped:
            rec.append(f"not trained, too little data x{skipped}")
        if not self.tried:
            rec.append(NEVER_TRIED)
        elif not self.measured and not self.no_material:
            rec.append("tried, no measured outcome yet")
        return "; ".join(rec)


def _changed_axes(axes: dict, base: dict) -> set[str]:
    """Function axes turned on or changed from base (only turning an axis off is
    not trying it). A count key folds into its function axis only while that
    axis is on — turning the axis off returns the count to its default, which is
    not an attempt."""
    out = set()
    for k, v in (axes or {}).items():
        if (base or {}).get(k) == v:
            continue
        parent = K_PARENT.get(k)
        if parent:
            if (axes or {}).get(parent, (base or {}).get(parent)) not in _OFF_VALUES:
                out.add(parent)
        elif v not in _OFF_VALUES:
            out.add(k)
    return out


def _outcome(c: dict, changed: set[str], evaluated: dict) -> dict[str, str | None]:
    """One candidate row's outcome per axis — None is a win, a string the failure
    kind. An empty dict when it does not count.

    Unused axes (unused_axes) count as unused whatever the row's outcome (the
    held-out result of a relabelled candidate is not that axis's). Arm verdicts
    (rescorer_identity · lora_gate · classifier_gate) go to that arm axis only."""
    skip = c.get("skipped")
    unused = {a for a in (c.get("unused_axes") or ()) if a in changed}
    out = {a: _UNUSED for a in unused}
    rest = changed - unused
    if bool(c.get("adopted") or c.get("gate_passed_but_lost")
            or ((evaluated.get(c.get("id")) or {}).get("gate") or {}).get("adopted")):
        return out | {a: None for a in rest}
    if skip is None and c.get("id") in evaluated:
        return out | {a: _HELD_OUT_FAIL for a in rest}
    if skip == "same_setting" and c.get("same_as") != "champion":
        return out                       # same as a sibling or an earlier candidate — not the axis's failure
    if skip in _ARM_SKIP:
        arm = _ARM_SKIP[skip]
        return out | ({arm: _FAIL[skip]} if arm in rest else {})
    if skip in _FAIL:
        return out | {a: _FAIL[skip] for a in rest}
    return out                           # unmeasured drops, outside screen's top K


@dataclass
class AxisStatus:
    retire_after: int | None
    axes: dict = field(default_factory=dict)     # axis → AxisState
    champion_round: int = 0                      # round the current champion was adopted (baseline 0)

    @property
    def off(self) -> set[str]:
        return {a for a, s in self.axes.items() if s.status == "off"}

    def block(self, axes: dict, champion: dict) -> str | None:
        """The reason when every function axis the candidate changed is OFF (else None)."""
        if not self.retire_after:
            return None
        changed = _changed_axes(axes, champion)
        if changed and changed <= self.off:
            return (f"{sorted(changed)} — turned off on this run after "
                    f"{self.retire_after} rounds in a row without effect (reopened once "
                    "when the champion changes)")
        return None


def axis_status(history: list[dict], retire_after: int | None) -> AxisStatus:
    """Compute the axis status from the whole run's history. With retire_after None nothing is turned off, only recorded."""
    st = AxisStatus(retire_after=retire_after)
    for k in BASELINE:
        if k not in K_PARENT:
            st.axes[k] = AxisState(axis=k)
    for i, e in enumerate(history):
        if i == 0 or e.get("round", 0) == 0:
            continue
        start = (history[i - 1].get("champion") or {}).get("axes") or {}
        d = e.get("decision") or {}
        evaluated = d.get("evaluated") or {}
        # grouped per round × axis — variants of the same axis in one round are one
        # attempt (a win if any wins, else one failure). No reliance on row order
        # (screened, then skipped)
        rnd: dict[str, list] = {}
        for c in _merged_rows(e):
            asked = c.get("requested_axes") or c.get("axes") or {}
            changed = _changed_axes(asked, start)
            if not changed:
                continue
            skip = c.get("skipped")
            if skip in UNMEASURED:
                continue
            for a in changed:
                if a in st.axes:
                    st.axes[a].tried += 1
                    st.axes[a].last_tried = e["round"]
            # empty or short material (including the classifier and lora tripwires) — a problem of the value
            empty = c.get("empty_axis")
            if skip == "no_material" or empty:
                # only values of axes the candidate changed — re-mining an inherited
                # axis that came out empty (the champion's value) is not that value's fault
                if empty in st.axes and empty in changed:
                    st.axes[empty].no_material.append(asked.get(empty))
                continue
            for axis, kind in _outcome(c, changed, evaluated).items():
                if axis in st.axes:
                    rnd.setdefault(axis, []).append(kind)
        for axis, kinds in rnd.items():
            s = st.axes[axis]
            s.measured += 1
            s.last_round = e["round"]
            if None in kinds:
                s.last_kind = None
                s.wins += 1
                s.streak = 0
                if s.status in ("off", "probe"):
                    s.status, s.off_since = "open", None
                continue
            kind = max(sorted(set(kinds), key=_KIND_ORDER.index), key=kinds.count)
            s.last_kind = kind
            s.fails[kind] = s.fails.get(kind, 0) + 1
            s.fail_rounds.append(e["round"])
            s.streak += 1
            if retire_after and s.streak >= retire_after and s.status != "off":
                s.status, s.off_since = "off", e["round"]
        if d.get("adopted") and e["round"] > 0:
            st.champion_round = e["round"]
            # a new champion — reopen OFF axes once (one more failure turns them OFF again)
            for s in st.axes.values():
                if s.status == "off":
                    s.status, s.off_since = "probe", None
                    s.streak = max((retire_after or 1) - 1, 0)
    champ = ((history[-1].get("champion") or {}).get("axes") or {}) if history else {}
    for axis, s in st.axes.items():
        if champ.get(axis, BASELINE[axis]) not in _OFF_VALUES:
            s.status = "on"
    return st
