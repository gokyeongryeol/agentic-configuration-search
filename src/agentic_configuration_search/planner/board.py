"""Axis board — the planner prompt's facts for each axis gathered in one row.

One axis's facts (values, status, record, blocks, cost) sit in one row, so the
same fact is not repeated in other words across several sections. Each function
axis gets a row with:

  status    ON (on in the champion) · OPEN · PROBE · OFF (planner.axis_status) ·
            BLOCKED (OPEN but nothing to propose now) · PROBE (nothing to propose now)
  propose   the values that may be proposed now — values the data blocks, folded
            tiling grids, the champion's value and values whose material came out
            empty under the current champion are left out
  run       the whole-run record (round × axis — planner.axis_status)
  champion  the best record per value under the current champion (planner.ledger)
  min       observed minutes per candidate (median, n, max) — per grid for tiling
            (off for candidates that turned it off)
  note      constraints and circumstances of that axis only

Below the board come only the details its rows point to (the evidence for the
one-time tiling exception, prescreened values, the source-value explanation,
disabled axes). axis_status, ledger and gates decide; the board only presents.
"""

from __future__ import annotations

import json
import statistics

from .axes import BASELINE, NEGATIVE_AXES
from .axis_status import NEVER_TRIED
from .gates import tiling_gate
from .ledger import K_PARENT, axis_ledger, fmt_value
from .ledger import label as value_label

# cost of an axis not observed yet — minutes differ per dataset, so only the facts
STATIC_COST = {
    "lora": "fine-tunes the detector tail before screening",
    "crop_classifier": "trains a crop classifier once per run, then screens blend weights",
    "rescorer": "re-infers fit_train and fit_dev to refit the rescorer",
}
DETECTOR_STATE = ("lora", "crop_classifier")
# negative axis values = the mining source
SOURCE_DOC = {"fp_crop": "fp_crop = false detections of the class",
              "gt_crop": "gt_crop = other classes' ground truth (look-alike classes)",
              "background": "background = false detections that overlap no ground truth"}
_OFF = (False, None, "none")
# order of values in the champion column — what is useful for the next move first
# (k variants of the champion's value, values that lost narrowly)
_STATUS_RANK = {"champion value (k variants)": 0, "near-miss": 1,
                "clearly-lost (prescreened)": 2, "clearly-lost": 3, "folded": 4}
MAX_CHAMPION_VALUES = 2   # values per axis carried with numbers under the current champion (the rest by name)
# the "never tried" mark (axis_status.NEVER_TRIED) — axes that cannot be proposed alone now show "not tried"

LEGEND = (
    "One row per axis. status: ON = on in the champion; OPEN = may be proposed; PROBE = "
    "reopened after "
    "the champion changed, one more failure turns it OFF; OFF = a candidate changing "
    "only OFF axes is dropped (skipped=\"axis_retired\"); BLOCKED = nothing can be "
    "proposed now (see note); \"PROBE (nothing to propose now)\" = a PROBE axis with no "
    "value left now. propose = values you may set now, with their per-class "
    "quantity keys in brackets; \"k variants: ...\" = only quantity variants of the "
    "champion's value are left. To turn an axis off, set exemplar to \"none\", any other "
    "axis to false; a quantity key set to null returns to its default. run = whole-run "
    "record, counted once per round; \"empty material for v\" means source value v mined "
    "nothing on this dataset, so re-propose v only after the champion changes; \"not "
    "trained, too little data\" means the arm skipped training for lack of data — "
    "re-propose it only after the champion changes. champion "
    "= record per value since the current champion was adopted (fd = fit_dev delta with "
    "upper bound ub, ho = held-out gate delta with lower bound lo; \"champion value (k "
    "variants)\" = quantity variants of the value the champion already uses; near-miss = "
    "lost on fit_dev within noise or won; clearly-lost = its single change clearly lost "
    "on fit_dev, (prescreened) = variants are first screened on a subset; folded = tiling "
    "grid folded; otherwise the latest outcome — adopted, evaluated, screened, arm_gate, "
    "prescreened, unused, no_material, same_setting, no_effect, blocked or error). min = "
    "measured minutes per candidate (materialize + fit_dev screen + held-out), median "
    "(n, max).")


def axis_costs(history: list[dict]) -> dict[str, list[float]]:
    """Wall clock per candidate (wall_min.total, minutes) by axis — for each
    function axis of the increment (by request) over the champion that started
    the round (count axes fold into their function axis, tiling keeps the grid).
    A candidate changing several axes counts once for each. Collected over the
    whole history — cost changes little when the champion changes, and forgetting
    it means paying for an expensive value again."""
    costs: dict[str, list[float]] = {}
    for i, e in enumerate(history):
        champ = ((history[i - 1].get("champion", {}).get("axes") if i > 0 else None)
                 or e.get("champion", {}).get("axes") or {})
        for c in e.get("candidates", []):
            total = (c.get("wall_min") or {}).get("total")
            if total is None:
                continue
            asked = c.get("requested_axes") or c.get("axes") or {}
            labels = {(f"tiling={v}" if k == "tiling" and v else K_PARENT.get(k, k))
                      for k, v in asked.items() if champ.get(k) != v}
            for lab in labels:
                costs.setdefault(lab, []).append(float(total))
    return costs


def _fmt(d: dict | None, bound: str, name: str) -> str:
    if not d or d.get("delta") is None:
        return ""
    b = d.get(bound)
    return f"{name} {d['delta']:+.4f}" + (f" ({bound} {b:+.4f})" if b is not None else "")


def _champion_cell(axis: str, ledger: dict, champ: dict) -> str:
    rows = [r for r in ledger.values() if r.key[0] == axis]
    if not rows:
        return "-"

    def status(r):
        return "champion value (k variants)" if champ.get(axis) == r.key[1] else r.status
    rows.sort(key=lambda r: (_STATUS_RANK.get(status(r), 9), -r.tries, value_label(r.key)))
    cells = []
    for r in rows[:MAX_CHAMPION_VALUES]:
        nums = "; ".join(x for x in (_fmt(r.best_fit_dev, "ub", "fd"),
                                     _fmt(r.best_holdout, "lo", "ho")) if x)
        cells.append(f"{fmt_value(r.key[1])} {status(r)}" + (f" ({nums})" if nums else ""))
    rest = rows[MAX_CHAMPION_VALUES:]
    if rest:
        cells.append("also " + ", ".join(f"{fmt_value(r.key[1])} {status(r)}"
                                         for r in rest))
    return "; ".join(cells)


def _cost(v: list[float]) -> str:
    return f"{statistics.median(v):.0f} (n={len(v)}, max {max(v):.0f})"


def _cost_cell(axis: str, costs: dict) -> str:
    if axis == "tiling":
        grids = sorted((lab for lab in costs if lab.startswith("tiling=")),
                       key=lambda s: int(s.split("=")[1]))
        cells = [f"g{lab.split('=')[1]} {_cost(costs[lab])}" for lab in grids]
        if costs.get("tiling"):                      # candidates that turned tiling off
            cells.append(f"off {_cost(costs['tiling'])}")
        return ", ".join(cells) or "-"
    v = costs.get(axis)
    return _cost(v) if v else "-"


def board_lines(history: list[dict], values: dict, status, *, disabled=(),
                banned=(), facts=None, prescreen: dict | None = None) -> list[str]:
    """The axis board — legend, table, detail lines. disabled are the axes left
    out of the board (search.disabled_axes); banned lists every disabled axis and
    axis=value (search.disabled_values) for the closing line — values already
    holds only the allowed values."""
    champ = ((history[-1].get("champion") or {}).get("axes") or {}) if history else {}
    gate = tiling_gate(history)
    ledger = axis_ledger(history)
    costs = axis_costs(history)
    forwards = ((history[-1].get("champion") or {}).get("forwards_per_image")
                if history else None) or 1
    rows = ["axis | status | propose | run | champion | min | note"]
    for axis, vals in values.items():
        if axis not in BASELINE or axis in K_PARENT or axis in disabled:
            continue
        cur = champ.get(axis, BASELINE[axis])
        st = status.axes.get(axis)
        note = []
        if axis in DETECTOR_STATE and cur not in _OFF:
            propose = "keep on"
            note.append("detector state: turning it off is dropped")
        else:
            # values whose material came out empty under the current champion are
            # left out (proposable again once the champion changes)
            empty = {r.key[1] for r in ledger.values()
                     if r.key[0] == axis and r.status == "no_material"}
            opts = [v for v in vals if v not in _OFF and v != cur and v not in empty
                    and not (axis == "tiling" and gate.block({"tiling": v}))]
            if not opts and empty:
                note.append("not trained under this champion (too little data)"
                            if axis in DETECTOR_STATE else
                            "every remaining value mined empty under this champion")
            qty = ["{} {}".format(k, "|".join(str(v) for v in values.get(k, ())
                                              if v is not None))
                   for k, p in K_PARENT.items() if p == axis and k in values]
            # values in response-JSON notation (true; strings as they are) — the
            # same notation as the champion line
            propose = ", ".join(json.dumps(v) if isinstance(v, bool) else str(v)
                                for v in opts) or "-"
            # quantity keys go with values to propose, or when the champion has the
            # axis on (k variants)
            if qty and opts:
                propose += f" ({'; '.join(qty)})"
            elif qty and cur not in _OFF:
                propose = f"k variants: {'; '.join(qty)}"
        state = (st.status if st else "open").upper()
        if st and st.status == "off":
            state = f"OFF since R{st.off_since}"
        blocked = propose == "-" and state == "OPEN"
        if blocked:
            state = "BLOCKED"
        elif propose == "-" and state == "PROBE":
            state = "PROBE (nothing to propose now)"
        # axes that cannot be proposed alone now — "never tried" is not shown, so the
        # untried rule does not recommend them
        alone_blocked = blocked or (axis == "visual_negatives"
                                    and champ.get("exemplar", "none") == "none")
        if axis == "tiling" and gate.folded_at is not None:
            note.append(f"grid >= {gate.folded_at} folded under this champion"
                        + (f"; grid {gate.open_grid} open ONCE (see below)"
                           if gate.open_grid is not None else ""))
        if axis == "tiling":
            note.append(f"cost 1+g^2 passes per image (champion: {forwards})")
        if axis == "visual_negatives" and champ.get("exemplar", "none") == "none":
            note.append("needs exemplar: only together with an exemplar value")
        lora_blocked = axis == "lora" and facts is not None and facts.lora_block()
        if lora_blocked:
            note.append("too few fit_train images to train")
        if st and st.status == "off":
            note.append("only together with an axis that is not OFF")
        if axis in STATIC_COST and axis not in costs and not lora_blocked:
            note.append(STATIC_COST[axis])
        run = st.record() if st else "-"
        if alone_blocked and NEVER_TRIED in run:
            run = run.replace(NEVER_TRIED, "not tried")
        rows.append(f"{axis} | {state} | {propose} | {run} | "
                    f"{_champion_cell(axis, ledger, champ)} | {_cost_cell(axis, costs)} | "
                    + ("; ".join(note) or "-"))
    out = [LEGEND] + rows
    once = gate.describe()[1:]
    if once:
        out += [line.replace("- tiling: ", "Tiling exception: ", 1) for line in once]
    lost = [r for r in ledger.values() if r.prescreened and r.evidence]
    if lost:
        ev = "; ".join(
            f"{value_label(r.key)} (round {r.evidence['round']} {r.evidence['id']}: fit_dev "
            f"delta {r.evidence['delta']:+.4f}, ub {r.evidence['ub']:+.4f})" for r in lost)
        p = prescreen or {}
        if p.get("on", True):
            where = (f"{p['n_images']} of {p['of']} fit_dev images" if p.get("of")
                     else "a fixed subset of fit_dev")
            out.append(f"Prescreened values (clearly lost under this champion): {ev}. A "
                       f"candidate containing one is first screened on {where} and dropped "
                       "as skipped=\"lost_value_prescreen\" unless the upper bound of "
                       "(candidate - champion) fit_dev micro F is >= 0 there. Say in "
                       "\"differs\" what is different this time.")
        else:
            out.append(f"Clearly lost under this champion (not prescreened on this run): "
                       f"{ev}.")
    doc = _axis_doc_line(values, disabled)
    if doc:
        out.append(doc)
    src = _source_line(values, disabled)
    if src:
        out.append(src)
    if banned:
        out.append("Disabled on this dataset (never turn these on; the champion keeps any "
                   "it already has): " + ", ".join(banned))
    return out


# What each axis does — by name alone, base vs fn_arm is chosen from the record
# only. The negative axes are explained by _source_line, tiling by the prompt's
# TILING_RULE
AXIS_DOC = {
    "exemplar": ('exemplar = visual example prompts from ground-truth crops ("base" = '
                 'typical crops; "fn_arm" = base plus crops of objects the detector missed)'),
    "alias": ('alias = one extra text phrase per class, mined from captions of '
              'ground-truth crops ("vlm" = a vision LLM looks at the crops and names them)'),
    "rescorer": "rescorer = per-class score recalibration learned on fit_train",
    "lora": "lora = fine-tunes the detector itself",
    "crop_classifier": "crop_classifier = re-classifies detected boxes with a trained crop "
                       "classifier",
}


def _axis_doc_line(values: dict, disabled=()) -> str:
    """One line on what each axis that is not off does."""
    docs = [doc for a, doc in AXIS_DOC.items() if a in values and a not in disabled]
    return ("What the axes do: " + "; ".join(docs) + ".") if docs else ""


def _source_line(values: dict, disabled=()) -> str:
    """One line explaining the values (mining sources) of text_negatives and visual_negatives."""
    axes = [a for a in NEGATIVE_AXES if a not in disabled and len(values.get(a, ())) > 1]
    if not axes:
        return ""
    srcs = [v for v in SOURCE_DOC if any(v in values[a] for a in axes)]
    use = {"text_negatives": "text_negatives turns the crops into suppressing phrases",
           "visual_negatives": "visual_negatives feeds the crops as negative image prompts"}
    return (" and ".join(axes) + " values are the mining source: "
            + "; ".join(SOURCE_DOC[v] for v in srcs) + ". "
            + "; ".join(use[a] for a in axes) + ".")
