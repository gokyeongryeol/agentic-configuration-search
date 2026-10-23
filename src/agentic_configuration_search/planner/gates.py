"""tiling fold gate — keeps grids larger than an unused zoom from being screened.

A grid g costs 1+g² forwards per image to screen, so each step up costs much
more (g=2 5, g=3 10, g=4 17, g=5 26, g=6 37). A planner that keeps raising g
after the previous grid went unused can spend most of a run's time on screens
whose settings fall back to no tiling. So that one case is blocked in code:

- **fold**: if a candidate asking for a grid g larger than the champion's did
  not use tiles in screen (tiling_used=false, axis_unused) under the current
  champion, grids ≥ g are blocked until the champion changes.
- **one-step near-miss exception**: if every unused row of the folded grid g
  meets two conditions, g+1 is opened **once** — (a) noise: the one-sided
  paired-bootstrap upper bound of (tiled − untiled) per-class mean F on fit_dev
  (the prescreen subset, if prescreened), `tiling_f.gap_ub`, is at or above 0
  (tiling being worse cannot be ruled out); (b) magnification shortfall: for
  the TP-size quantile s* counted at the champion's fit_dev screen thresholds,
  m·mag(g) < s* ≤ m·mag(g+1) (`tiling_shortfall.shortfall`, m = the median px
  of the class with the most missed_small). Once g+1 reaches screen (used or
  not) the exception is spent and the fold at g holds. A g+1 that the
  lost-value prescreen rejected after its (subset) screen reached it too — that
  screen's tiling_used stays on the row. Spending is counted from
  history rows, so "once" means one proposal batch — several g+1 candidates in
  the same proposal all pass. g+2 and up stay closed, and no candidate is
  created or injected.

The first grid is not blocked — the planner picks it from the missed-small
sizes. Raising past a grid that was used and helped is only recommended by the
prompt, not enforced.

**Only evidence under the current champion counts** (start over when it
changes) — a zoom unused under an old champion may work in the new champion's
context (the same principle as combination dedup). Candidates inheriting the
champion's grid, or lowering or turning tiling off, are never blocked — their
cost does not grow. A pure function of history, so resume reaches the same
verdict — the exception's two conditions also only read values recorded on
candidate rows at screen time (gap_ub, shortfall). Holdout is used nowhere —
gap_ub comes from fit_dev, and s* and m from train sizes counted at the fit_dev
screen thresholds (not tau_test).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.tiling import magnification
from ..diagnose.residual import INPUT_SIDE   # size_ratio → input px
from .axes import TILING_GRIDS


def champion_round(history: list[dict]) -> int | None:
    """The round the current champion was adopted in — later rounds are attempts
    under it."""
    if not history:
        return None
    return (history[-1].get("champion") or {}).get("round")


def under_champion(history: list[dict]) -> list[dict]:
    """Rounds run under the current champion (from the round after its adoption)."""
    cr = champion_round(history)
    if cr is None:
        return []
    return [e for e in history if e.get("round", 0) > cr]


def _q(sizes: list[float], p: float) -> float:
    """Lower p quantile of a sorted list — the same convention as the prompt's
    missed-small block."""
    return sizes[min(len(sizes) - 1, int(p * len(sizes)))]


def tp_floor(diag_train: dict | None, *, quantile: float, tp_min: int) -> dict | None:
    """s* — the size of small objects the champion already detects (input px).

    From the train diagnosis's `tiling_sizes` (diagnose.errors.size_profile —
    train TP and missed_small sizes counted at the champion's fit_dev screen
    thresholds), take the class with the most missed_small (ties by name), and
    give the lower `quantile` of its TP sizes (s*) and its missed_small median
    (m) in input px. With fewer TPs than `tp_min`, available=False — s* cannot
    be trusted. None without missed_small or material. The fn list classified at
    tau_test (holdout-estimated thresholds) is not read — this value feeds the
    fold gate's exception (a screen decision)."""
    sizes = (diag_train or {}).get("tiling_sizes") or {}
    small = {k: v for k, v in (sizes.get("missed_small") or {}).items() if v}
    if not small:
        return None
    name = min(small, key=lambda k: (-len(small[k]), k))
    missed = sorted(float(r) for r in small[name])
    tps = sorted(float(r) for r in (sizes.get("tp") or {}).get(name) or ())
    out = {"class": name, "n_missed": len(missed), "n_tp": len(tps),
           "missed_px": round(missed[len(missed) // 2] * INPUT_SIDE, 1),
           "available": len(tps) >= tp_min and len(tps) > 0,
           "thresholds": sizes.get("thresholds")}
    if out["available"]:
        out["tp_px"] = round(_q(tps, quantile) * INPUT_SIDE, 1)
    return out


def tiling_shortfall(diag_train: dict | None, grid: int, *, overlap: float,
                     quantile: float, tp_min: int) -> dict | None:
    """Did grid g lose for lack of magnification — m·mag(g) < s* ≤ m·mag(g+1).

    Recorded on the candidate row (`tiling_shortfall`) at screen time so the gate
    reads history only. None without missed_small; shortfall=False when TPs are
    too few."""
    floor = tp_floor(diag_train, quantile=quantile, tp_min=tp_min)
    if floor is None:
        return None
    out = {"grid": int(grid), **floor, "shortfall": False,
           "mag_g": round(magnification(grid, overlap), 4),
           "mag_next": round(magnification(grid + 1, overlap), 4)}
    if floor["available"]:
        m, s_star = floor["missed_px"], floor["tp_px"]
        out["shortfall"] = bool(m * magnification(grid, overlap) < s_star
                                <= m * magnification(grid + 1, overlap))
    return out


def _near_miss(c: dict) -> bool:
    ub = (c.get("tiling_f") or {}).get("gap_ub")
    return ub is not None and ub >= 0


def _short(c: dict) -> bool:
    return bool((c.get("tiling_shortfall") or {}).get("shortfall"))


@dataclass
class TilingGate:
    champion_grid: int = 0               # the champion's grid (0 if none)
    folded_at: int | None = None         # the smallest grid that went unused
    open_grid: int | None = None         # grid opened once by the near-miss exception (folded_at+1)
    evidence: dict | None = None         # the exception's evidence — gap_ub, shortfall (for describe)

    def block(self, axes: dict) -> str | None:
        """The reason (note) to block, or None. axes are the full axes merged onto
        the champion."""
        g = axes.get("tiling")
        if not g or g <= self.champion_grid:
            return None                    # off, lower, inherited — cost does not grow
        if g == self.open_grid:
            return None                    # one-step near-miss exception (once)
        if self.folded_at is not None and g >= self.folded_at:
            return (f"tiling={g}: grid {self.folded_at} was screened and not used "
                    "under the current champion (tiling_used=false) — grids ≥"
                    f"{self.folded_at} are folded until the champion changes")
        return None

    def describe(self) -> list[str]:
        """Prompt status line — only when folded (otherwise the rules paragraph
        explains it)."""
        if self.folded_at is None:
            return []
        lines = [f"- tiling: grid >= {self.folded_at} is folded under the current "
                 f"champion (grid {self.folded_at} was screened and not used) — "
                 "such candidates are dropped as tiling_blocked until the champion "
                 "changes"]
        if self.open_grid is not None and self.evidence:
            e, g = self.evidence, self.folded_at
            lines.append(
                f"- tiling: grid {self.open_grid} is open ONCE as a near-miss "
                f"exception — grid {g} lost to the untiled config only within "
                f"bootstrap noise (tiled-untiled fit_dev F upper bound "
                f"{e['gap_ub']:+.3f} >= 0) and fell short on magnification for "
                f"{e['class']}: missed-small median {e['missed_px']:.0f} px "
                f"x{e['mag_g']:.2g} = {e['missed_px'] * e['mag_g']:.0f} px < "
                f"{e['tp_px']:.0f} px (smallest objects the champion already "
                f"detects) <= x{e['mag_next']:.2g} = "
                f"{e['missed_px'] * e['mag_next']:.0f} px; once grid "
                f"{self.open_grid} is screened, grid >= {g} stays folded")
        return lines


def _screened(c: dict) -> bool:
    """Is this a tiling row that reached screen (prescreen included) — tiling_used
    is recorded. Rows rejected by the axis prescreen (axis_unused) or the
    lost-value prescreen also carry the subset screen's tiling_used."""
    return "tiling_used" in c or "tiling" in (c.get("unused_axes") or ())


def tiling_gate(history: list[dict]) -> TilingGate:
    """Collect the tiling evidence under the current champion from history."""
    champ_axes = ((history[-1].get("champion") or {}).get("axes") or {}) if history else {}
    base = int(champ_axes.get("tiling") or 0)
    gate = TilingGate(champion_grid=base)
    unused: dict[int, list[dict]] = {}
    screened: set[int] = set()
    for e in under_champion(history):
        for c in e.get("candidates", []):
            asked = c.get("requested_axes") or c.get("axes") or {}
            g = asked.get("tiling")
            if not g or int(g) <= base:
                continue
            g = int(g)
            if _screened(c):
                screened.add(g)
            used = c.get("tiling_used")
            if used is None and "tiling" in (c.get("unused_axes") or ()):
                used = False
            if used is False:
                unused.setdefault(g, []).append(c)
    if not unused:
        return gate
    g0 = min(unused)
    gate.folded_at = g0
    rows = unused[g0]
    nxt = g0 + 1
    if (nxt in TILING_GRIDS and nxt not in screened
            and all(_near_miss(c) and _short(c) for c in rows)):
        gate.open_grid = nxt
        sf = rows[0]["tiling_shortfall"]
        gate.evidence = {"gap_ub": max(c["tiling_f"]["gap_ub"] for c in rows),
                         **{k: sf[k] for k in ("class", "missed_px", "tp_px",
                                               "mag_g", "mag_next")}}
    return gate
