"""tiling fold gate — keeps grids larger than an unused zoom from being screened.

A grid g costs 1+g² forwards per image to screen, so each step up costs much
more (g=2 5, g=3 10, g=4 17, g=5 26, g=6 37). A planner that keeps raising g
after the previous grid went unused can spend most of a run's time on screens
whose settings fall back to no tiling. So that one case is blocked in code:

- **fold**: if a candidate asking for a grid g larger than the champion's did
  not use tiles in screen (tiling_used=false, axis_unused) under the current
  champion, grids ≥ g are blocked until the champion changes.

The first grid is not blocked — the planner picks it from the missed-small
sizes. Raising past a grid that was used and helped is only recommended by the
prompt, not enforced.

**Only evidence under the current champion counts** (start over when it
changes) — a zoom unused under an old champion may work in the new champion's
context (the same principle as combination dedup). Candidates inheriting the
champion's grid, or lowering or turning tiling off, are never blocked — their
cost does not grow. A pure function of history, so resume reaches the same
verdict.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class TilingGate:
    champion_grid: int = 0               # the champion's grid (0 if none)
    folded_at: int | None = None         # the smallest grid that went unused

    def block(self, axes: dict) -> str | None:
        """The reason (note) to block, or None. axes are the full axes merged onto
        the champion."""
        g = axes.get("tiling")
        if not g or g <= self.champion_grid:
            return None                    # off, lower, inherited — cost does not grow
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
        return [f"- tiling: grid >= {self.folded_at} is folded under the current "
                f"champion (grid {self.folded_at} was screened and not used) — "
                "such candidates are dropped as tiling_blocked until the champion "
                "changes"]


def tiling_gate(history: list[dict]) -> TilingGate:
    """Collect the tiling evidence under the current champion from history."""
    champ_axes = ((history[-1].get("champion") or {}).get("axes") or {}) if history else {}
    base = int(champ_axes.get("tiling") or 0)
    gate = TilingGate(champion_grid=base)
    for e in under_champion(history):
        for c in e.get("candidates", []):
            asked = c.get("requested_axes") or c.get("axes") or {}
            g = asked.get("tiling")
            if not g or int(g) <= base:
                continue
            used = c.get("tiling_used")
            if used is None and "tiling" in (c.get("unused_axes") or ()):
                used = False
            if used is False:
                gate.folded_at = (int(g) if gate.folded_at is None
                                  else min(gate.folded_at, int(g)))
    return gate
