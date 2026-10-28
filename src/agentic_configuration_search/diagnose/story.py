"""Run report — a story that a reader without the technical background can read
front to back.

It goes data → how it improved → what is left → what to do next, without
technical terms. Its inputs are the ledger and diagnosis artifacts that
diagnose.report reads. The output is `report.md` in the run directory and a
figure folder `report_assets/` (chart PNGs, photo JPEGs) — nothing in the loop
reads them.

- Scores and counts are holdout aggregates. The holdout is reused for adoption
  decisions, so scores are somewhat optimistic, and the report says so to the
  reader. Example photos come from train only — the same rule as
  diagnose.residual (holdout is aggregates only).
- Example captions are the VLM crop notes as written (plain English). Without a
  note, a fixed sentence built from the cause and size is used — the report is
  drawn in full without an LLM.
- It never states how many photos are needed — the loop does not measure a
  learning curve, so the artifacts cannot estimate it.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from ..core.metrics import SMALL_OBJECT_RATIO
from . import residual
from .report import (_AXIS_PARENT, _OFF, _final_diag, _history, _load_json, _note_key,
                     attempts, trajectory)

log = logging.getLogger(__name__)

REPORT_FILE = "report.md"
ASSET_DIR = "report_assets"

EXAMPLES_PER_PATTERN = 3   # example photos per pattern
MAX_PATTERNS = 4           # cap on remaining-problem patterns
PATTERN_MIN_SHARE = 0.05   # causes below this share of the remaining errors are not a pattern
SAMPLE_IMAGES = 4          # representative dataset photos
SKEW = 1.5                 # a bucket's miss rate at this multiple of the class's is "especially"
MIN_BUCKET_GT = 20         # buckets with fewer objects are not read as rates (6 of 6 = 100% noise)
MIN_WISH_RATE = 0.05       # a bucket's miss rate below this is not a collection condition (noise)
MIN_WISH_FN = 3            # a bucket with fewer misses than this is not a collection condition
DATA_SHARE_LOW = 0.3       # below this share solvable by data, "the method side is more effective"
TILE_H = 280               # example photo height (px)
THUMB = 900                # long side representative photos are shrunk to before drawing (px)

# --------------------------------------------------------------- plain-word vocabulary
# Technical terms reach the report only through these tables — a new axis or
# cause must be added here (test_story blocks missing entries).

STRATEGY = {
    "exemplar": ("Show example photos",
                 "Shows a few real photos of the object to look for."),
    "alias": ("Search by other names too",
              "Adds other words for the object, to find what one name alone missed."),
    "text_negatives": ("Name look-alikes in words",
                       "Names things that are often mistaken for the object, to cut wrong finds."),
    "visual_negatives": ("Show look-alikes in photos",
                         "Shows photos of things often mistaken for the object, to cut wrong "
                         "finds."),
    "tiling": ("Zoom into parts of the photo",
               "Splits the photo into pieces and looks at each one enlarged, to find small "
               "objects."),
    "rescorer": ("Second review",
                 "Scores borderline results once more, keeping the right ones and dropping "
                 "the wrong ones."),
    "lora": ("Extra training on this data",
             "Trains the recognition model a little more on this dataset's photos."),
    "crop_classifier": ("Double-check the kind",
                        "Cuts out each found object and checks once more what kind it is."),
}
# count keys are part of their axis — the story does not count them separately
_QUANTITY_KEYS = set(_AXIS_PARENT)

# candidate outcome → (mark, plain words). The outcome values of report.attempts.
_OUTCOME_GROUPS = {
    "adopted": ("✅", "adopted"),
    "gate_failed": ("✕", "not enough effect on the test photos"),
    "gate_passed_but_lost": ("△", "helped, but a better strategy in the same round won"),
    "screen_top_k": ("–", "ranked too low in the practice stage"),
}
_OUTCOME_ALIASES = {
    "no effect in the practice stage, so not tested": (
        "fit_dev_lost", "lost_value_prescreen", "classifier_gate", "lora_gate",
        "rescorer_identity"),
    "same as something already tried, so skipped": (
        "fit_dev_identical", "same_setting", "champion_combo", "combo_tried",
        "carried_duplicate", "duplicate_axes", "no_change"),
    "not actually used on this data, so skipped": ("axis_unused", "tiling_blocked"),
    "turned off for this run after several tries without effect, so skipped": (
        "axis_retired",),
    "the material or training data it needs was short, so skipped": ("no_material",),
    "more than one round can test, so skipped": ("max_candidates",),
    "turned off for this run, so skipped": ("axis_disabled", "disabled_axes"),
    "the proposal was malformed, so skipped": ("invalid_axes",),
    "could not run": ("error", "adapt_ctl", "missing_hook", "lora_deps",
                      "detector_state_off", "compound_server_state"),
}
for _text, _keys in _OUTCOME_ALIASES.items():
    for _k in _keys:
        _OUTCOME_GROUPS[_k] = ("–", _text)
_OUTCOME_OTHER = ("–", "could not run")

# error cause → (name, meaning, direction). Direction: data = more photos help,
# method = the decision rule needs tuning, box = check the found area or the
# label (answer marking) rules
CAUSE = {
    "missed": ("Objects not seen at all", "The object was there but was not found at all.",
               "data"),
    "missed_small": ("Objects too small to see", "The object looked so small it was not found.",
                     "data"),
    "below_threshold": ("Objects seen but dropped for low confidence",
                        "The object was found, but its confidence was below the cutoff, so it "
                        "was left out.", "method"),
    "background": ("Background taken for an object",
                   "Something was found where there is no object.", "data"),
    "class_confusion": ("Taken for another kind",
                        "The object was found in the right place but given the wrong kind.",
                        "data"),
    "localization": ("Area drawn off target",
                     "The object was found, but the marked area was far from the answer.", "box"),
    "duplicate": ("Same object counted twice", "One object was found twice.", "method"),
    "tiling_artifact": ("Error from splitting the photo",
                        "A wrong result from an object cut in two when the photo was split "
                        "into pieces.", "method"),
}
FN_KINDS = ("missed", "missed_small", "below_threshold")
# direction — a possibility read from the error type, not a tested cause (a noun phrase)
DIRECTION = {"data": "an error more photos are likely to reduce",
             "method": "an error that looks like it needs a decision-rule change",
             "box": "an error to check against the found area or the label rules"}

# condition terms in the VLM notes → plain words (used in collection conditions)
CONDITION_PLAIN = {
    "blurry": "blurry", "blur": "blurry", "motion": "blurry",
    "small": "small in frame", "tiny": "small in frame", "distant": "far away",
    "far": "far away", "occluded": "partly hidden", "occlusion": "partly hidden",
    "overlap": "overlapping", "dark": "dark", "shadow": "in shadow", "bright": "glare",
    "glare": "glare", "reflection": "reflections", "low-resolution": "low resolution",
    "low resolution": "low resolution", "cluttered": "busy background",
    "partial": "only partly visible", "cropped": "only partly visible",
    "angle": "unusual angle", "viewpoint": "unusual angle",
    "similar": "look-alike objects nearby", "look-alike": "look-alike objects nearby",
}
# run_summary.stopped → why it stopped
STOP_PLAIN = {
    "planner_stop": "it judged there was no strategy left worth trying",
    "saturation": "the score stopped rising for several rounds",
    "empty_rounds": "no new strategy came up to test",
    "rounds": "it finished all the planned rounds",
}
# mining sources of the negative axes (text and visual) → plain words
_SOURCE_PLAIN = {"fp_crop": "from wrong finds", "gt_crop": "from other kinds",
                 "background": "from the background"}
_SIZE_PLAIN = {"small": "small", "medium": "medium-size", "large": "large"}
_CROWDED = ("6-20", "21+")        # only these buckets are called "crowded scenes"
_CROWD_PLAIN = {"1": "1", "2-5": "2–5", "6-20": "6–20", "21+": "21 or more"}


def strategy_label(delta: dict) -> str:
    """Axis increment → plain strategy name. Axes turned on by name, axes turned off as "drop …"."""
    on, off = [], []
    for k, v in sorted(delta.items()):
        if k in _QUANTITY_KEYS:
            continue
        name = STRATEGY.get(k, (k,))[0]
        if v in _OFF:
            off.append(f"drop {name[0].lower()}{name[1:]}")
        elif k == "tiling" and not isinstance(v, bool):
            on.append(f"{name} ({v}×{v})")
        elif k == "exemplar" and v == "fn_arm":
            on.append(f"{name} (mostly missed objects)")
        elif v in _SOURCE_PLAIN:
            on.append(f"{name} ({_SOURCE_PLAIN[v]})")
        else:
            on.append(name)
    if not on and not off and any(k in _QUANTITY_KEYS for k in delta):
        parent = _AXIS_PARENT[next(k for k in delta if k in _QUANTITY_KEYS)]
        return f"{STRATEGY[parent][0]} (different number)"
    return " + ".join(on + off) or "Recheck the current setup"


def outcome_text(outcome: str) -> tuple[str, str]:
    return _OUTCOME_GROUPS.get(outcome, _OUTCOME_OTHER)


def _p1(v: float) -> float:
    """Score out of 100 (one decimal) — changes subtract these values, so the text adds up."""
    return round(100 * v, 1)


def _pts(v) -> str:
    return "—" if v is None else f"{_p1(v):.1f}"


def _pct(n, d) -> str:
    return f"{n / d:.0%}" if d else "—"


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else many or word + 's'}"


# --------------------------------------------------------------- figures

def _plt():
    """matplotlib with the Agg backend (no display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


_BLUE, _GREY, _GREEN, _RED, _ORANGE = "#2f6db5", "#b8bec7", "#2e9e5b", "#d64545", "#e39b2d"
_PALETTE = ["#2f6db5", "#e39b2d", "#2e9e5b", "#d64545", "#7a5cc2", "#1aa3a3",
            "#c2569b", "#8a7b3b", "#4a4a4a", "#6aa84f"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # figures with photos are JPEG — as PNG they run to several MB
    kw = {"pil_kwargs": {"quality": 85}} if path.suffix == ".jpg" else {}
    fig.savefig(path, dpi=130, bbox_inches="tight", **kw)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _chart_classes(facts: dict, path: Path) -> None:
    plt = _plt()
    names = facts["classes"]
    tr = [facts["train"]["per_class"].get(n, {}).get("objects", 0) for n in names]
    ho = [facts["holdout"]["per_class"].get(n, {}).get("objects", 0) for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.2))
    y = range(len(names))
    ax.barh(y, tr, color=_BLUE, label="training photos")
    ax.barh(y, ho, left=tr, color=_ORANGE, label="test photos")
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_xlabel("objects")
    ax.set_title("Objects per kind")
    ax.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)


def _chart_sizes_crowd(facts: dict, path: Path) -> None:
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    tr = facts["train"]
    sizes = [tr["size"].get(b, 0) for b in residual.SIZE_BUCKETS]
    labels = [f"{_SIZE_PLAIN[b]}\n({facts['size_ranges'][b]})" for b in residual.SIZE_BUCKETS]
    a1.bar(labels, sizes, color=[_RED, _ORANGE, _BLUE])
    a1.set_title("Object sizes (training photos)")
    a1.set_ylabel("objects")
    crowd = [tr["crowd"].get(name, 0) for _, name in residual.CROWD_BUCKETS]
    a2.bar([_CROWD_PLAIN[name] for _, name in residual.CROWD_BUCKETS], crowd, color=_BLUE)
    a2.set_title("Objects in one photo (training photos)")
    a2.set_ylabel("photos")
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    _save(fig, path)


def _draw_boxes(img, boxes, colors, width: int = 3, halo: bool = False):
    """Draw boxes. With halo a white border goes on first, so the box shows on a same-color object."""
    from PIL import ImageDraw
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = (float(v) for v in box)
        if halo:
            d.rectangle([x1 - width, y1 - width, x2 + width, y2 + width],
                        outline="white", width=width * 3)
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def _chart_samples(ds, names: list[str], picks: list, path: Path) -> None:
    from PIL import Image
    plt = _plt()
    color = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}
    cols = min(len(picks), 2)
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)
    id_to_name = ds.ontology.id_to_name
    for ax, e in zip(axes.flat, picks):
        try:
            with Image.open(ds.train.image_path(e)) as im:   # NFS — do not leave an open fd
                img = im.convert("RGB")
        except OSError as exc:              # one missing photo does not stop the rest
            log.warning("could not open a representative photo (%s): %s", e.file_name, exc)
            ax.set_axis_off()
            continue
        scale = min(1.0, THUMB / max(img.size))
        img = img.resize((round(img.width * scale), round(img.height * scale)))
        boxes = [[v * scale for v in a.bbox_xyxy] for a in e.annotations]
        cols_ = [color.get(id_to_name.get(a.category_id), _GREY) for a in e.annotations]
        ax.imshow(_draw_boxes(img, boxes, cols_, max(2, round(max(img.size) / 300))))
        ax.set_axis_off()
    for ax in list(axes.flat)[len(picks):]:
        ax.set_axis_off()
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=color[n], label=n) for n in names],
               loc="lower center", ncol=min(len(names), 5), frameon=False)
    fig.suptitle("Representative photos (boxes = objects marked as answers)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, path)


def grey_points(hist: list[dict]) -> list[tuple[int, float]]:
    """(round, holdout score) — attempts scored but not adopted. Attempts that
    passed the gate but lost to a better candidate are in too (only the round's
    winner decision.id is left out)."""
    out = []
    for e in hist:
        d = e.get("decision") or {}
        won = d.get("id") if d.get("adopted") else None
        for cid, ev in (d.get("evaluated") or {}).items():
            if cid != won and ev.get("micro_f_beta") is not None:
                out.append((e["round"], ev["micro_f_beta"]))
    return out


def _chart_rounds(t: dict, hist: list[dict], adopted: list[dict], path: Path) -> None:
    plt = _plt()
    xs = [r["round"] for r in t["rounds"]]
    ys = [100 * r["champion_micro_f_beta"] for r in t["rounds"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    pts = grey_points(hist)
    gx, gy = [x for x, _ in pts], [100 * y for _, y in pts]
    if gx:
        ax.scatter(gx, gy, color=_GREY, s=28, zorder=2,
                   label="tried but not adopted")
    ax.step(xs, ys, where="post", color=_BLUE, lw=2.5, zorder=3, label="best score so far")
    ax.scatter(xs[:1], ys[:1], color=_BLUE, s=40, zorder=4)
    for i, a in enumerate(adopted, 1):
        y = 100 * a["score"]
        ax.scatter([a["round"]], [y], color=_GREEN, s=90, zorder=5)
        ax.annotate(f"{i}", (a["round"], y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=11, fontweight="bold", color=_GREEN)
    ax.set_xticks(xs, ["start"] + [f"{x}" for x in xs[1:]])
    ax.set_xlabel("round")
    ax.set_ylabel("score (out of 100)")
    ax.set_title("Score by round — green numbers are adopted strategies")
    lo = min(ys + gy) if gy else min(ys)
    hi = max(ys + gy) if gy else max(ys)
    pad = max(2.0, 0.1 * (hi - lo))
    ax.set_ylim(max(0, lo - pad), min(100, hi + pad))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)


def _chart_errors(before: dict, after: dict, path: Path) -> None:
    plt = _plt()
    kinds = [k for k in CAUSE if before.get(k, 0) or after.get(k, 0)]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(kinds) + 1.4))
    y = list(range(len(kinds)))
    h = 0.38
    ax.barh([v - h / 2 for v in y], [before.get(k, 0) for k in kinds], h,
            color=_GREY, label="start")
    ax.barh([v + h / 2 for v in y], [after.get(k, 0) for k in kinds], h,
            color=_BLUE, label="final")
    ax.set_yticks(y, [("missed: " if k in FN_KINDS else "wrong find: ") + CAUSE[k][0]
                      for k in kinds])
    ax.invert_yaxis()
    ax.set_xlabel("count (test photos)")
    ax.set_title("Errors by type — start and final")
    ax.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)


def _strip(ds, by_name: dict, rows: list[dict], color: str, path: Path) -> list[dict]:
    """Join example crops side by side — with generous margin around the box so
    the surroundings show. Returns the rows actually drawn, in order (so the
    caption numbers match the photos)."""
    from PIL import Image
    tiles, drawn = [], []
    for r in rows:
        e = by_name.get(r["image"])
        if e is None:
            continue
        try:
            with Image.open(ds.train.image_path(e)) as im:
                img = im.convert("RGB")
        except OSError as exc:
            log.warning("could not open an example photo (%s): %s", r["image"], exc)
            continue
        x1, y1, x2, y2 = (float(v) for v in r["box"])
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        side = max(bw, bh) * 2.2 + 40
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        L = max(0, int(cx - side / 2))
        T = max(0, int(cy - side / 2))
        R = min(img.width, int(cx + side / 2))
        B = min(img.height, int(cy + side / 2))
        if R - L < 2 or B - T < 2:              # box outside the photo — nothing to draw
            continue
        crop = img.crop((L, T, R, B))
        scale = TILE_H / crop.height
        crop = crop.resize((max(1, round(crop.width * scale)), TILE_H))
        box = [(x1 - L) * scale, (y1 - T) * scale, (x2 - L) * scale, (y2 - T) * scale]
        tiles.append(_draw_boxes(crop, [box], [color], 3, halo=True))
        drawn.append(r)
    if not tiles:
        return []
    gap = 12
    canvas = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1),
                               TILE_H), "white")
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=85)
    return drawn


# --------------------------------------------------------------- dataset facts

def dataset_facts(ds, small_ratio: float) -> dict:
    """Photos, objects, objects per kind, sizes and crowding of train and holdout — GT annotations only."""
    names = [c.name for c in ds.ontology.classes]
    # size is √(box area / photo area) — a length ratio, so it reads as "of the photo's side"
    out = {"classes": names, "size_ranges": {
        "small": f"under {small_ratio:.0%} of the photo",
        "medium": f"{small_ratio:.0%}–{residual.LARGE_FACTOR * small_ratio:.0%} of the photo",
        "large": f"{residual.LARGE_FACTOR * small_ratio:.0%}+ of the photo"}}
    for key, split in (("train", ds.train), ("holdout", ds.holdout)):
        per = {n: {"objects": 0, "images": 0} for n in names}
        size, crowd, border = Counter(), Counter(), 0
        n_obj, empty = 0, 0
        for row in residual.image_rows(split):
            gts = [g for g in row["gts"] if g["category_id"] in ds.ontology.id_to_name]
            if not gts:
                empty += 1
            else:
                crowd[residual.crowd_bucket(len(gts))] += 1
            seen = set()
            for g in gts:
                name = ds.ontology.id_to_name[g["category_id"]]
                n_obj += 1
                per[name]["objects"] += 1
                seen.add(name)
                f = residual.object_facts(g["box"], row["width"], row["height"],
                                          n_gt=len(gts), small_ratio=small_ratio)
                size[f["size"]] += 1
                border += f["border"]
            for name in seen:
                per[name]["images"] += 1
        out[key] = {"images": len(split.images), "objects": n_obj, "empty": empty,
                    "per_class": per, "size": dict(size), "crowd": dict(crowd),
                    "border": border}
    return out


def _sample_images(ds, k: int) -> list:
    """k representative photos — among photos holding the most kinds with an
    object count inside the quartiles (typical), spread evenly in file-name
    order. File names usually follow capture order, so near-identical burst
    shots do not repeat. Deterministic."""
    imgs = [e for e in ds.train.images if e.annotations]
    if not imgs:
        return []
    counts = sorted(len(e.annotations) for e in imgs)
    lo, hi = counts[len(counts) // 4], counts[(3 * len(counts)) // 4]
    cover = max(len({a.category_id for a in e.annotations}) for e in imgs)
    pool = sorted((e for e in imgs if len({a.category_id for a in e.annotations}) == cover
                   and lo <= len(e.annotations) <= hi), key=lambda e: e.file_name)
    if len(pool) < k:
        pool = sorted(imgs, key=lambda e: e.file_name)
    k = min(k, len(pool))
    step = len(pool) / k
    return [pool[int(i * step + step / 2)] for i in range(k)]


def _dataset_bullets(facts: dict) -> list[str]:
    tr, names = facts["train"], facts["classes"]
    objs = {n: tr["per_class"][n]["objects"] for n in names}
    out = []
    if objs and max(objs.values()):
        most = max(names, key=lambda n: objs[n])
        # the fewest among the other kinds — on a tie it must not name the same kind twice
        least = min((n for n in names if n != most), key=lambda n: objs[n], default=most)
        if len(names) > 1 and objs[least]:
            ratio = objs[most] / objs[least]
            if ratio < 1.5:
                out.append("Objects are spread evenly across kinds "
                           f"(most: **{most}**, {objs[most]:,}; fewest: "
                           f"**{least}**, {objs[least]:,}).")
            else:
                out.append(f"The most common kind is **{most}** ({objs[most]:,}) and the "
                           f"least common is **{least}** ({objs[least]:,}), {ratio:.1f}× apart."
                           + (" The gap is large, so the rarer kinds may have too few "
                              "examples to learn from." if ratio >= 5 else ""))
        zero = [n for n in names if not objs[n]]
        if zero:
            out.append(f"**{', '.join(zero)}** do{'es' if len(zero) == 1 else ''} not appear "
                       "in the training photos at all.")
    n_img = max(tr["images"] - tr["empty"], 1)
    out.append(f"A photo with objects holds **{tr['objects'] / n_img:.1f} objects** on "
               "average.")
    small = tr["size"].get("small", 0)
    if tr["objects"]:
        share = small / tr["objects"]
        pct = "under 1%" if 0 < share < 0.01 else f"{share:.0%}"
        line = (f"Very small objects (width and height {facts['size_ranges']['small']}) "
                f"are **{pct}** of all objects.")
        if share >= 0.2:
            line += " With this many small objects, finding them is on the hard side."
        out.append(line)
        out.append(f"**{tr['border'] / tr['objects']:.0%}** of objects touch the photo edge "
                   "(and may be partly cut off).")
    if tr["empty"]:
        out.append(f"{_plural(tr['empty'], 'photo')} contain no object at all.")
    return out


# --------------------------------------------------------------- captions

def _fallback_caption(r: dict) -> str:
    """The fixed sentence when there is no note — built from the cause, size and kind only."""
    name, kind = r.get("category"), r.get("kind")
    label = CAUSE.get(kind, (str(kind),))[0]
    if kind in FN_KINDS:
        size = _SIZE_PLAIN.get(r.get("size") or "")
        return f"A case of '{label}' ({size + ' ' if size else ''}{name})."
    return f"A case of '{label}' (name the program gave: {name})."


# --------------------------------------------------------------- picking examples

def _pick_examples(rows: list[dict], kind: str, notes_by: dict, k: int,
                   train_names) -> list[dict]:
    """k examples of one cause — ones with a note first, then misses by large
    object (a large miss stands out more) and wrong finds by high score
    (confidently wrong). One per image, kinds spread. Only rows of train photos
    (train_names). Tie-broken by file name and box, so deterministic."""
    def area(r):
        b = r["box"]
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    cand = [{**r, "_note": notes_by.get(_note_key(r.get("image"), r.get("box")))}
            for r in rows if r.get("kind") == kind and r.get("box")
            and r.get("image") in train_names]
    weight = (lambda r: -area(r)) if kind in FN_KINDS else \
        (lambda r: -float(r.get("score") or 0))
    cand.sort(key=lambda r: (r["_note"] is None, weight(r), str(r["image"]),
                             tuple(r["box"])))
    picked, images, classes = [], set(), set()
    for one_per_class in (True, False):      # pass 1: one per kind, pass 2: the remaining slots
        for r in cand:
            if len(picked) >= k:
                return picked
            if r["image"] in images or (one_per_class and r.get("category") in classes):
                continue
            picked.append(r)
            images.add(r["image"])
            classes.add(r.get("category"))
    return picked


# --------------------------------------------------------------- sections

_STALE_NOTE = ("_There is no diagnosis record of the final result, so the diagnosis of "
               "{basis} is used instead. A strategy was adopted after that, so the real "
               "remaining errors may be fewer._")


def _summary(t: dict, before: dict, after: dict, adopted: list[dict],
             patterns: list[tuple[str, int]], total_err: int, stale: bool) -> list[str]:
    out = ["## At a glance", "",
           "The score is out of 100. It looks at both whether objects in the **test "
           "photos** (not used for training) were found without missing any, and whether "
           "what was found is right.", ""]
    gain = _p1(t["final_micro_f_beta"]) - _p1(t["baseline_micro_f_beta"])
    if gain > 0:
        out.append(f"- **The score rose from {_pts(t['baseline_micro_f_beta'])} to "
                   f"{_pts(t['final_micro_f_beta'])}** (+{gain:.1f}).")
    else:
        out.append(f"- **The score is {_pts(t['final_micro_f_beta'])}, the same as at the "
                   "start.** None of the strategies tried clearly raised it.")
    if before.get("recall") is not None and after.get("recall") is not None:
        out.append(f"- Of every 100 objects in the test photos, **{100 * before['recall']:.0f} "
                   f"→ {100 * after['recall']:.0f}** are found, and the share of finds that "
                   f"are right went **{before['precision']:.0%} → {after['precision']:.0%}**.")
    if adopted:
        top = max(a["gain"] for a in adopted)
        best = [a["label"] for a in adopted if a["gain"] == top]
        out.append(f"- The most effective strategy was **{'** and **'.join(best)}** "
                   f"({'each ' if len(best) > 1 else ''}+{top:.1f}).")
    if patterns and total_err:
        n_top = patterns[0][1]
        tied = [k for k, n in patterns if n == n_top]
        names = "** and **".join(CAUSE[k][0] for k in tied)
        dirs = {CAUSE[k][2] for k in tied}
        line = (f"- The most common remaining error is **{names}** "
                f"({'each ' if len(tied) > 1 else ''}{n_top / total_err:.0%}).")
        if len(dirs) == 1:
            d = DIRECTION[dirs.pop()]
            line += f" {'They are' if len(tied) > 1 else 'It is'} {d}" \
                    f"{'s' if len(tied) > 1 else ''}."
        if stale:
            line += " (This is the diagnosis at the start of the last round, not the final result.)"
        out.append(line)
    return out + [""]


def _score_note(holdout_split: str | None) -> list[str]:
    out = ["> **What is the score?** It looks at both 'were objects found without missing "
           "any' and 'is what was found right' on the test photos (out of 100). Doing well "
           "at only one does not raise it.", ">",
           "> The test photos were also used to decide whether to adopt each strategy, so "
           "this score may come out a little better than on photos never seen before."]
    if holdout_split and holdout_split != "valid":
        out += [">", "> **Caution:** this dataset has no photos set aside for a final check, "
                "so the final performance on never-seen photos cannot be confirmed. Take care "
                "not to read the score as better than it is."]
    return out + [""]


def _figure(rel: str, fname: str, alt: str, caption: str) -> list[str]:
    """A figure plus one line that reads on its own (what it shows, how to read it)."""
    return [f"![{alt}]({rel}/{fname})", "", f"*Figure — {caption}*", ""]


def _section_dataset(facts: dict | None, ds, assets: Path, rel: str) -> list[str]:
    out = ["## 1. What is the data?", "",
           "Before the results, this section introduces the objects the program had to find "
           "and the photos.", ""]
    if facts is None:
        return out + ["_The dataset path is unknown, so this section is skipped "
                      "(`acs report --config` can redraw it)._", ""]
    tr, ho = facts["train"], facts["holdout"]
    out += [f"There are **{_plural(len(facts['classes']), 'kind')}** of object to find "
            f"({', '.join(facts['classes'])}).", "",
            "| Set | Photos | Objects | Used for |", "|---|---|---|---|",
            f"| Training photos | {tr['images']:,} | {tr['objects']:,} | extra training and "
            "preparing strategies |",
            f"| Test photos | {ho['images']:,} | {ho['objects']:,} | scoring and deciding "
            "whether to adopt a strategy (not used for training) |", "",
            "Part of the training photos is set aside as a 'practice test' and used to check "
            "early whether a strategy really helps. Only strategies that pass the practice "
            "test are scored on the test photos.", ""]
    _chart_classes(facts, assets / "dataset_classes.png")
    _chart_sizes_crowd(facts, assets / "dataset_sizes.png")
    out += _figure(rel, "dataset_classes.png", "Objects per kind",
                   "Objects per kind. Blue bars are objects in the training photos, orange "
                   "bars in the test photos.")
    out += _figure(rel, "dataset_sizes.png", "Object sizes and objects per photo",
                   "Left, object sizes; right, how many objects one photo holds. Size is the "
                   "object's width and height as a share of the photo's.")
    out += ["**What stands out**", ""]
    out += [f"- {b}" for b in _dataset_bullets(facts)]
    out.append("")
    picks = _sample_images(ds, SAMPLE_IMAGES)
    if picks:
        _chart_samples(ds, facts["classes"], picks, assets / "dataset_samples.jpg")
        out += _figure(rel, "dataset_samples.jpg", "Representative photos",
                       f"{_plural(len(picks), 'training photo')} with several kinds together "
                       "and a typical number of objects, spread evenly in capture order. "
                       "Boxes are the answer markings.")
    out += ["The next section shows how much the program's results improved on these "
            "photos.", ""]
    return out


def _adoptions(t: dict) -> list[dict]:
    """Adoption history — gain is the difference of rounded scores out of 100."""
    out, prev = [], t["baseline_micro_f_beta"]
    for r in t["rounds"]:
        if r["round"] > 0 and r["adopted"]:
            out.append({"round": r["round"], "label": strategy_label(r["adopted_delta"]),
                        "delta": r["adopted_delta"], "score": r["champion_micro_f_beta"],
                        "prev": prev,
                        "gain": _p1(r["champion_micro_f_beta"]) - _p1(prev)})
        prev = r["champion_micro_f_beta"]
    return out


def _duration(hist: list[dict], summary: dict) -> str:
    """Sum of round wall clocks (round 0 included) plus the final diagnosis — minutes under an hour."""
    sec = sum(float(e.get("dur_s") or 0) for e in hist)
    sec += float((summary.get("final_diagnosis") or {}).get("dur_s") or 0)
    if not sec:
        return ""
    return f"{sec / 3600:.1f} hours" if sec >= 3600 else _plural(max(1, round(sec / 60)),
                                                                   "minute")


_TESTED = ("adopted", "gate_passed_but_lost", "gate_failed")


def _section_rounds(t: dict, hist: list[dict], adopted: list[dict],
                    before_err: dict, after_err: dict, summary: dict,
                    assets: Path, rel: str) -> list[str]:
    n_rounds = max((r["round"] for r in t["rounds"]), default=0)
    stopped = summary.get("stopped")
    if stopped == "rounds":
        stop = f" It finished all {_plural(n_rounds, 'planned round')}."
    elif stopped in STOP_PLAIN:
        stop = f" It stopped at round {n_rounds} because {STOP_PLAIN[stopped]}."
    else:
        stop = ""
    took = _duration(hist, summary)
    if took:
        stop += f" It took {took} in total."
    out = ["## 2. How did it improve?", "",
           "This section shows what was tried in each round and how the score changed. In "
           "each **round** the program tries a few strategies and adopts only those that "
           "clearly raise the score, as the starting point of the next round. It ran "
           f"**{_plural(n_rounds, 'round')}** and adopted "
           f"**{_plural(len(adopted), 'strategy', 'strategies')}**.{stop}", ""]
    out += _score_note(summary.get("holdout_split"))
    att = attempts(hist)
    used = sorted({k for a in att for k, v in a["delta"].items()
                   if k in STRATEGY and v not in _OFF}, key=list(STRATEGY).index)
    if used:
        out += ["**Strategies explained** — the strategies proposed in this run.", "",
                "| Strategy | What it does |", "|---|---|"]
        out += [f"| {STRATEGY[k][0]} | {STRATEGY[k][1]} |" for k in used]
        out.append("")
    _chart_rounds(t, hist, adopted, assets / "rounds.png")
    caption = ("Score by round. The blue line is the best score so far; green numbers are "
               "adopted strategies.")
    if grey_points(hist):
        caption += (" Grey dots are attempts scored on the test photos but not adopted. Even "
                    "slightly above the blue line, one is not adopted when the difference is "
                    "small enough to arise by chance.")
    out += _figure(rel, "rounds.png", "Score by round", caption)
    if adopted:
        out += ["**Adopted strategies** (the green numbers in the chart)", ""]
        for i, a in enumerate(adopted, 1):
            out.append(f"{i}. **{a['label']}** — round {a['round']}, "
                       f"{_pts(a['prev'])} → {_pts(a['score'])} (+{a['gain']:.1f})")
        out.append("")
    by_round: dict = {}
    for a in att:
        by_round.setdefault(a["round"], []).append(a)
    score = {r["round"]: r["champion_micro_f_beta"] for r in t["rounds"]}
    out += ["**Attempts and results by round** — only attempts scored on the test photos "
            "are listed; attempts filtered out before that are only counted (✅ adopted · "
            "△ helped, but a better strategy won · ✕ not enough effect).", "",
            "| Round | Scored attempts → result | Filtered out before testing | Score at "
            "round end |", "|---|---|---|---|"]
    for rnd in range(1, n_rounds + 1):
        rows = by_round.get(rnd, [])
        cells = []
        for a in rows:
            if a["outcome"] in _TESTED:
                mark, why = outcome_text(a["outcome"])
                cells.append(f"{mark} {strategy_label(a['delta'])} — {why}")
        skipped = sum(1 for a in rows if a["outcome"] not in _TESTED)
        out.append(f"| {rnd} | {'<br>'.join(cells) or 'none'} | "
                   f"{skipped or '—'} | {_pts(score.get(rnd))} |")
    out.append("")
    if before_err and after_err:
        _chart_errors(before_err, after_err, assets / "errors_before_after.png")
        out += _figure(rel, "errors_before_after.png", "Errors by type",
                       "Errors counted on the test photos. Grey is the start, blue the final "
                       "result.")
        moved = sorted(((after_err.get(k, 0) - before_err.get(k, 0), k)
                        for k in set(before_err) | set(after_err) if k in CAUSE))
        if moved and moved[0][0] < 0:
            d, k = moved[0]
            out.append(f"- The error that fell the most is **{CAUSE[k][0]}** "
                       f"({before_err.get(k, 0):,} → {after_err.get(k, 0):,}).")
        if moved and moved[-1][0] > 0:
            d, k = moved[-1]
            out.append(f"- One error rose instead: **{CAUSE[k][0]}** "
                       f"({before_err.get(k, 0):,} → {after_err.get(k, 0):,}). As other errors "
                       "fell, some of them may have moved here.")
        out.append("")
    out += ["Some errors remain at the end. The next section looks at which ones.", ""]
    return out


def _err_of(fd: dict) -> dict:
    s = (fd["test"].get("summary") or {})
    return {k: v for side in ("fn", "fp") for k, v in (s.get(side) or {}).items()
            if v and k in CAUSE}


def _basis(fd: dict) -> str:
    return f"the setup at the start of round {fd['round']}" if fd.get("stale") \
        else "the final result"


def _section_residual(fd: dict, ds, run_dir: Path, assets: Path,
                      rel: str) -> tuple[list[str], list[tuple[str, int]], int]:
    out = ["## 3. What problems are left?", "",
           "This section groups the remaining errors by cause and shows real examples of "
           "each.", ""]
    test = fd["test"]
    err = _err_of(fd)
    total = sum(err.values())
    n_fn = sum(v for k, v in err.items() if k in FN_KINDS)
    basis = _basis(fd)
    if fd.get("stale"):
        out += [_STALE_NOTE.format(basis=basis), ""]
    out += [f"Scoring the test photos with {basis} left **{_plural(n_fn, 'missed object')}** "
            f"and **{_plural(total - n_fn, 'wrong find')}**.", ""]
    per = test.get("per_class") or {}
    if per:
        out += ["| Kind | Objects in test photos | Found | Missed | Wrong finds |",
                "|---|---|---|---|---|"]
        for name in sorted(per, key=lambda n: -(per[n].get("miss_rate") or 0)):
            s = per[name]
            fn, fp = sum((s.get("fn") or {}).values()), sum((s.get("fp") or {}).values())
            gt = s.get("gt", 0)
            out.append(f"| {name} | {gt:,} | {_pct(gt - fn, gt)} | {fn:,} | {fp:,} |")
        out.append("")
    patterns = [(k, n) for k, n in sorted(err.items(), key=lambda kv: -kv[1])
                if total and n / total >= PATTERN_MIN_SHARE][:MAX_PATTERNS]
    if not patterns:
        return out, patterns, total
    by_name = {e.file_name: e for e in ds.train.images} if ds is not None else {}
    out.append("Grouped by cause, the remaining errors are as follows.")
    if ds is not None:
        out[-1] += (" Example photos are taken from the **training photos**. Misses are "
                    "picked largest object first and wrong finds most confident first; green "
                    "boxes are missed objects and red boxes are wrong finds. The example "
                    "descriptions were written automatically by an AI looking at the photo "
                    "and may be wrong.")
    out.append("")
    notes = fd.get("notes") or []
    notes_by = {_note_key(n.get("image"), n.get("box")): n.get("note")
                for n in notes if n.get("note") and n.get("box")}
    fn_rows = ((fd.get("residual") or {}).get("train") or {}).get("fn") \
        or fd["train"].get("fn") or []
    fp_rows = fd["train"].get("fp") or []
    chosen = {k: _pick_examples(fn_rows if k in FN_KINDS else fp_rows, k, notes_by,
                                EXAMPLES_PER_PATTERN, by_name) for k, _ in patterns}
    res_h = ((fd.get("residual") or {}).get("holdout") or {}).get("per_class") or {}
    size_all = Counter()                       # size mix of all test-photo objects (the reference)
    for s in res_h.values():
        size_all.update({b: pair[0] for b, pair in (s.get("size") or {}).items()})
    for i, (k, n) in enumerate(patterns, 1):
        name, meaning, direction = CAUSE[k]
        side = "missed" if k in FN_KINDS else "wrong find"
        out += [f"### 3-{i}. {name} ({side}) — {n:,}, {n / total:.0%} of remaining errors", "",
                f"{meaning} This is {DIRECTION[direction]}.", ""]
        by_cls = Counter({c: ((s.get("fn") if k in FN_KINDS else s.get("fp")) or {})
                          .get(k, 0) for c, s in per.items()})
        top = [(c, v) for c, v in by_cls.most_common(3) if v]
        if top:
            # wrong finds go by the name the program gave — not that kind's objects being wrong
            head = "Kinds missed most" if k in FN_KINDS else "Wrong name given"
            out.append(f"- {head}: " + ", ".join(f"{c} {v:,}" for c, v in top))
        if k in ("missed", "below_threshold") and res_h:   # too-small misses are always small
            sz = Counter()
            for s in res_h.values():
                sz.update((s.get("kind_size") or {}).get(k) or {})
            tot, all_ = sum(sz.values()), sum(size_all.values())
            if tot and all_:
                b, v = sz.most_common(1)[0]
                out.append(f"- Size: {v / tot:.0%} are {_SIZE_PLAIN[b]} objects (versus "
                           f"{size_all[b] / all_:.0%} of all objects in the test photos).")
        rows = chosen[k]
        if rows and ds is not None:
            fname = f"pattern_{i}_{k}.jpg"
            drawn = _strip(ds, by_name, rows, _GREEN if k in FN_KINDS else _RED,
                           assets / fname)
            if drawn:
                out += [""] + _figure(rel, fname, f"{name} examples",
                                      f"{_plural(len(drawn), 'example')} of '{name}', "
                                      "numbered from the left.")
                for j, r in enumerate(drawn, 1):
                    out.append(f"{j}. {r['_note'] or _fallback_caption(r)}")
        out.append("")
    out += ["The next section sums up what would help reduce these errors.", ""]
    return out, patterns, total


def _note_terms(text: str) -> set[str]:
    """One note's condition terms (plain words) — matched on word boundaries (not angle in triangle)."""
    text = text.lower()
    return {plain for en, plain in CONDITION_PLAIN.items()
            if re.search(rf"(?<![a-z]){re.escape(en)}(?![a-z])", text)}


def _class_wishes(name: str, facets: dict | None, notes: list[dict],
                  size_ranges: dict) -> list[str]:
    """One kind's collection conditions — buckets with unusually many misses
    that photos can reduce (not seen, too small to see) and the condition terms
    of the notes.

    Size buckets are measured from kind_size without the low-confidence misses
    (below_threshold) — counting those would turn a decision-rule problem into
    "collect more of these photos". Crowding, border and elongation are not
    split by cause, so they are used only when most of the kind's misses are the
    share photos can reduce. Rates carry their counts (so 100% of a bucket with
    few objects is not overstated)."""
    wish = []
    gt = (facets or {}).get("gt") or 0
    ks = (facets or {}).get("kind_size") or {}
    data_by_size = Counter()
    for k in ("missed", "missed_small"):
        data_by_size.update(ks.get(k) or {})
    n_data = sum(data_by_size.values())
    if gt and n_data:
        base = n_data / gt
        for b, pair in ((facets.get("size") or {}).items()):
            g, f = pair[0], data_by_size.get(b, 0)
            if g >= MIN_BUCKET_GT and f >= MIN_WISH_FN \
                    and f / g >= max(SKEW * base, MIN_WISH_RATE):
                wish.append(f"{_SIZE_PLAIN[b]} {name} ({size_ranges[b]} — {f:,} of {g:,} "
                            "not seen)")
        if n_data / max(facets.get("fn") or 0, 1) >= 0.5:
            base_all = (facets.get("fn") or 0) / gt

            def skewed(pair):
                g, f = pair
                return (g >= MIN_BUCKET_GT and f >= MIN_WISH_FN
                        and f / g >= max(SKEW * base_all, MIN_WISH_RATE))
            crowd = [b for b, pair in (facets.get("crowd") or {}).items() if skewed(pair)]
            if any(b in _CROWDED for b in crowd):
                wish.append("crowded scenes ("
                            f"{' or '.join(_CROWD_PLAIN[b] for b in crowd if b in _CROWDED)} "
                            "objects in a photo)")
            sparse = [b for b in crowd if b not in _CROWDED]
            if sparse:
                wish.append("scenes with only "
                            f"{' or '.join(_CROWD_PLAIN[b] for b in sparse)} object(s) in a "
                            "photo")
            if skewed(facets.get("border") or (0, 0)):
                wish.append("objects at the photo edge, only partly visible")
            if skewed(facets.get("elongated") or (0, 0)):
                wish.append("objects that look long and thin (lying down or at a slant)")
    terms = Counter()
    for n in notes:
        if n.get("category") == name and n.get("kind") in ("missed", "missed_small"):
            terms.update(_note_terms(str(n.get("note", ""))))
    if terms:
        order = list(dict.fromkeys(CONDITION_PLAIN.values()))     # ties in table order
        top = sorted(terms.items(), key=lambda kv: (-kv[1], order.index(kv[0])))[:3]
        wish.append(f"photos where the object is {', '.join(t_ for t_, _ in top)}")
    return wish


CONFUSION_MIN = 5          # minimum count for a kind's "wrong name given" line


def _section_collect(fd: dict, facts: dict | None) -> list[str]:
    out = ["## 4. What to do next — more data", ""]
    err = _err_of(fd)
    total = sum(err.values())
    if not total:
        return out + ["No errors are left.", ""]
    if fd.get("stale"):
        out += [_STALE_NOTE.format(basis=_basis(fd)), ""]
    share = Counter()
    for k, v in err.items():
        share[CAUSE[k][2]] += v
    out += ["This section sums up how to reduce the remaining errors. It first separates "
            "errors more photos are likely to reduce from the rest. This split is a "
            "possibility read from the type of error, not a result checked by collecting "
            "more photos.", "",
            "| Direction | Count | Share | Errors |", "|---|---|---|---|"]
    for d in ("data", "method", "box"):
        if share[d]:
            kinds = ", ".join(CAUSE[k][0] for k in CAUSE if CAUSE[k][2] == d and err.get(k))
            out.append(f"| {DIRECTION[d][0].upper()}{DIRECTION[d][1:]} | {share[d]:,} | "
                       f"{share[d] / total:.0%} | {kinds} |")
    out.append("")
    data_share = share["data"] / total
    if data_share < DATA_SHARE_LOW:
        out += [f"Only {data_share:.0%} of the errors are likely to shrink with more photos. "
                "**Refining the decision rule or the found-area and label rules first is "
                "likely to work better than collecting more photos.** The suggestions below "
                "are for the rest.", ""]
    else:
        out += [f"{data_share:.0%} of the remaining errors are likely to shrink with more "
                "photos.", ""]
    test = fd["test"]
    per = test.get("per_class") or {}
    res_h = ((fd.get("residual") or {}).get("holdout") or {}).get("per_class") or {}
    ranges = (facts or {}).get("size_ranges") or {
        "small": "small", "medium": "medium", "large": "large"}
    out += ["### Suggestions per kind (highest miss rate first)", ""]
    for name in sorted(per, key=lambda n: -(per[n].get("miss_rate") or 0)):
        s = per[name]
        fn = s.get("fn") or {}
        gt, n_fn = s.get("gt", 0), sum(fn.values())
        if not n_fn:
            continue
        data_fn = fn.get("missed", 0) + fn.get("missed_small", 0)
        out.append(f"**{name}** — {n_fn:,} of {gt:,} missed in the test photos "
                   f"({_pct(n_fn, gt)}).")
        out.append(f"- Share more photos are likely to reduce: {data_fn:,} "
                   f"({_pct(data_fn, n_fn)} of the misses)")
        wish = _class_wishes(name, res_h.get(name), fd.get("notes") or [], ranges)
        if wish and data_fn:
            out.append("- More photos like these would help: " + "; ".join(wish))
        below = fn.get("below_threshold", 0)
        if below > data_fn:
            share_b = "most" if below / n_fn >= 0.75 else "more than half" \
                if below / n_fn > 0.5 else "many"
            out.append(f"- {share_b[0].upper()}{share_b[1:]} of the misses ({below:,}) were "
                       "'seen but dropped for low confidence', so adjusting the decision "
                       "rule comes before more photos.")
        conf = (s.get("fp") or {}).get("class_confusion", 0)
        if conf >= CONFUSION_MIN:
            out.append(f"- The wrong name given was {name} in {conf:,} cases.")
        out.append("")
    common = []
    conf_all = err.get("class_confusion", 0)
    if conf_all and conf_all / total >= PATTERN_MIN_SHARE:
        common.append(f"**Taken for another kind, {conf_all:,}**: collecting photos where "
                      "similar-looking kinds appear together is likely to reduce it.")
    bg = err.get("background", 0)
    if bg and bg / total >= PATTERN_MIN_SHARE:
        common.append(f"**Background taken for an object, {bg:,}**: also collecting photos "
                      "without the object (or with a similar-looking other object) is likely "
                      "to reduce it.")
    if common:
        out += ["### Suggestions for all kinds", ""] + [f"- {c}" for c in common] + [""]
    flags = ((fd.get("residual") or {}).get("train") or {}).get("label_flags") or {}
    tiny, dup = len(flags.get("tiny_gt") or []), len(flags.get("duplicate_gt") or [])
    loc = err.get("localization", 0)
    if tiny or dup or (loc and loc / total >= PATTERN_MIN_SHARE):
        out += ["### Check found areas and labels (answer markings)", ""]
        if loc and loc / total >= PATTERN_MIN_SHARE:
            out.append(f"- 'Area drawn off target' is {loc:,}, {loc / total:.0%} of the "
                       "remaining errors. The program may draw the area too wide or too "
                       "narrow, or the rule for boxing the answer (e.g. whether hidden parts "
                       "are included) may differ between photos. The example photos in "
                       "section 3 show which.")
        if tiny:
            out.append(f"- {_plural(tiny, 'answer marking')} in the training photos "
                       f"{'is' if tiny == 1 else 'are'} drawn very small "
                       f"({residual.TINY_GT_PX} pixels or less). "
                       f"{'It' if tiny == 1 else 'They'} may be drawn by mistake.")
        if dup:
            out.append(f"- {_plural(dup, 'object')} in the training photos "
                       f"{'is' if dup == 1 else 'are'} marked twice.")
        out.append("")
    out += ["> How many photos are needed cannot be estimated from this record alone. The "
            "suggestions above are about **which photos** to collect.", ""]
    return out


def _glossary() -> list[str]:
    return ["## Appendix — terms", "",
            "| Term | Meaning |", "|---|---|",
            "| Round | One cycle of trying a few strategies and adopting the best |",
            "| Training photos | Photos used for extra training and preparing strategies |",
            "| Practice test | Training photos set aside to check strategies early |",
            "| Test photos | Photos used for scoring and for deciding whether to adopt a "
            "strategy. Not used for training |",
            "| Adopted | A strategy that clearly raised the score on the test photos and became "
            "the next round's starting point |",
            "| Decision rule | The confidence cutoff that decides whether a found result is "
            "kept or dropped |",
            "| Missed | An object in the photo that was not found |",
            "| Wrong find | Something found where there is no object, or with the wrong kind "
            "or place |",
            ""]


def _rates(ev: dict | None) -> dict:
    prf = (ev or {}).get("per_class_prf") or {}
    tp = sum(v.get("tp", 0) for v in prf.values())
    fp = sum(v.get("fp", 0) for v in prf.values())
    fn = sum(v.get("fn", 0) for v in prf.values())
    if not tp + fn or not tp + fp:
        return {}
    return {"recall": tp / (tp + fn), "precision": tp / (tp + fp)}


def _err_counts(d: dict | None) -> dict:
    s = (d or {}).get("summary") or {}
    return {k: v for side in ("fn", "fp") for k, v in (s.get(side) or {}).items()
            if k in CAUSE}


# --------------------------------------------------------------- render

def render_report(run_dir: Path, *, ds=None, title: str | None = None,
                  small_ratio: float | None = None) -> str:
    """One run's report — `report.md` and the `report_assets/` figures.

    Without ds (core.data.Dataset) the dataset section and the example photos
    are skipped."""
    run_dir = Path(run_dir)
    t = trajectory(run_dir)
    hist = _history(run_dir)
    fd = _final_diag(run_dir, t)
    res = (fd or {}).get("residual") or {}
    small_ratio = small_ratio or res.get("small_ratio") or SMALL_OBJECT_RATIO
    assets = run_dir / ASSET_DIR
    if assets.exists():
        for p in [*assets.glob("*.png"), *assets.glob("*.jpg")]:   # the last render's figures
            p.unlink()
    assets.mkdir(parents=True, exist_ok=True)
    name = title or (ds.root.name if ds is not None else run_dir.parent.name)

    facts = dataset_facts(ds, small_ratio) if ds is not None else None
    adopted = _adoptions(t)
    summary = _load_json(run_dir / "run_summary.json") or {}
    if fd is not None and not (fd.get("test") or {}).get("summary"):
        fd = None                     # without a holdout diagnosis there is nothing to say on errors
    before_err = _err_counts(_load_json(run_dir / "rounds" / "round_001"
                                        / "diagnosis_test.json"))
    # a diagnosis used instead (stale) is not the final champion's — no "final" bars
    after_err = _err_counts(fd["test"]) if fd is not None and not fd.get("stale") else {}
    before = _rates(_load_json(run_dir / "rounds" / "round_000" / "eval.json"))
    after = _rates(_load_json(run_dir / "champion_eval.json"))

    body: list[str] = []
    sections = [
        ("1. What is the data?", lambda: _section_dataset(facts, ds, assets, ASSET_DIR)),
        ("2. How did it improve?", lambda: _section_rounds(
            t, hist, adopted, before_err, after_err, summary, assets, ASSET_DIR)),
    ]
    for head, fn in sections:
        try:
            body += fn()
        except Exception as exc:  # noqa: BLE001 — one section must not block the whole report
            log.exception("report section failed: %s", head)
            body += [f"## {head}", "", f"_This part could not be generated "
                     f"({type(exc).__name__})._", ""]
    patterns, total = [], 0
    if fd is None:
        body += ["## 3. What problems are left?", "", "_There is no diagnosis record, so "
                 "this section and the next are skipped._", ""]
    else:
        try:
            part, patterns, total = _section_residual(fd, ds, run_dir, assets, ASSET_DIR)
            body += part
        except Exception as exc:  # noqa: BLE001
            log.exception("report section failed: remaining problems")
            body += ["## 3. What problems are left?", "",
                     f"_This part could not be generated ({type(exc).__name__})._", ""]
        try:
            body += _section_collect(fd, facts)
        except Exception as exc:  # noqa: BLE001
            log.exception("report section failed: more data")
            body += ["## 4. What to do next — more data", "",
                     f"_This part could not be generated ({type(exc).__name__})._", ""]
    lines = [f"# {name} — automatic improvement report", ""]
    lines += _summary(t, before, after, adopted, patterns, total,
                      bool(fd is not None and fd.get("stale")))
    lines += body + _glossary()
    text = "\n".join(lines) + "\n"
    (run_dir / REPORT_FILE).write_text(text)
    return text
