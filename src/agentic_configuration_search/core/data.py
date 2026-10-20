"""Data loading — Roboflow 3-split detection and split-role discipline.

train -> train (full), fit_train (80%), fit_dev (20%); valid -> holdout.
This module is the single gateway for data access.

**Why valid is the selection split.** Every round the loop estimates tau_c on
the holdout and decides adoption on it — the split is reused a dozen-plus times
per run. A reused split is contaminated accordingly and **cannot be used for
the final report.** In Roboflow's 70/20/10, valid is about twice the size of
test, so the larger split goes to repeated use and test is kept for measuring
the finalized champion **once** (scripts/predict.py --split test). This also
matches convention: valid is for choosing, test is for one final measurement.

**Without valid, test is promoted as a substitute.** A fallback so non-3-split
datasets are accepted. Know the cost: **no sealed split remains.** tau_c is
chosen to maximize F(beta) on test, metrics are measured on the same split,
and the adoption gate re-tests it every round — the reported holdout numbers
are optimistic estimates at an oracle threshold, with no split left to
de-bias them. So `Dataset.holdout_split` records which split was used: mixing
both cases in one table makes it uninterpretable, so reporters must be able to
tell them apart.

As long as valid exists, test is never opened anywhere — pinned by
test_test_is_never_read_when_valid_exists in tests/test_split_roles.py.

The split is image-level stratified: images are grouped by class combination
(sorted tuple) and the dev share is taken from a seeded shuffle per group.
Same (train, seed, ratio) always gives the same boundary — fit_train/fit_dev
do not move across rounds.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .coco import ANNOTATIONS_FILENAME, Ontology, Split

log = logging.getLogger(__name__)

# Holdout candidates — earlier wins. If valid exists, test stays sealed.
HOLDOUT_CANDIDATES = ("valid", "test")


@dataclass(frozen=True)
class Dataset:
    root: Path
    ontology: Ontology
    train: Split      # full train — prompt mining, FP/FN diagnosis
    fit_train: Split    # training for YOLOE tail fine-tune (lora axis), rescorer and crop classifier
    fit_dev: Split      # scoring grids, λ, α (screen)
    holdout: Split      # τ_c estimation, adoption, saturation
    holdout_split: str  # directory the holdout came from — "valid" or substitute "test"


def _subset(split: Split, keep: set[int]) -> Split:
    sub = Split.__new__(Split)
    sub.root, sub.raw, sub.ontology = split.root, split.raw, split.ontology
    # keep filename sort order (guaranteed by Split) — determinism of order-dependent outputs
    sub.images = [img for i, img in enumerate(split.images) if i in keep]
    return sub


def stratified_dev_indices(split: Split, dev_ratio: float, seed: int) -> set[int]:
    """Image indices assigned to fit_dev — stratified per class-combination group, deterministic.

    Group share is floor(n*ratio + 0.5) (avoids banker's rounding), capped at
    n-1 — a group with 2+ images never leaves fit_train empty. Single-image
    groups stay entirely in fit_train (material production takes priority). If
    dev is empty overall, one image is moved from the largest group.
    """
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = {}
    for idx, img in enumerate(split.images):
        key = tuple(sorted({a.category_id for a in img.annotations}))
        groups.setdefault(key, []).append(idx)
    dev: set[int] = set()
    for key in sorted(groups):
        lst = list(groups[key])
        rng.shuffle(lst)
        n = len(lst)
        if n < 2:
            continue
        take = min(n - 1, math.floor(n * dev_ratio + 0.5))
        dev.update(lst[:take])
    if not dev:
        biggest = max(sorted(groups), key=lambda k: len(groups[k]))
        if len(groups[biggest]) >= 2:
            dev.add(groups[biggest][0])
        else:
            raise ValueError("train is too small to carve out fit_dev (1 image per group)")
    return dev


def _holdout_name(root: Path) -> str:
    """Directory name to use as holdout — valid first, test as a substitute.

    **Distinguishes absence from corruption.** A missing directory falls through
    to the next candidate, but a directory without annotations stops here with
    an error. Treating corruption as absence turns a loud failure into a silent
    protocol change — the run completes with nobody noticing the sealed split
    was lost.
    """
    for name in HOLDOUT_CANDIDATES:
        d = root / name
        if (d / ANNOTATIONS_FILENAME).exists():
            return name
        if d.is_dir():
            raise FileNotFoundError(
                f"{d} exists but {ANNOTATIONS_FILENAME} is missing — this is corruption, "
                f"not absence. The holdout fallback applies only when the directory is absent")
    raise FileNotFoundError(
        f"{root} has neither valid/ nor test/ — a split is needed to judge adoption")


def load_dataset(root: str | Path, *, seed: int = 0,
                 fit_train_ratio: float = 0.8) -> Dataset:
    root = Path(root)
    if not (root / "train" / ANNOTATIONS_FILENAME).exists():
        raise FileNotFoundError(f"{root / 'train' / ANNOTATIONS_FILENAME} not found — "
                                "expected Roboflow layout (train/)")
    train = Split.load(root / "train")
    ontology = train.ontology

    name = _holdout_name(root)
    if name != "valid":
        log.warning(
            "no valid/ — using %s/ as holdout. Selection and reporting share one "
            "split, so this run's holdout metrics are optimistically biased "
            "(no sealed split is left).", name)
    holdout = Split.load(root / name, ontology)
    # Ontology consistency — explicit error if the holdout categories differ
    # from train (a silently misaligned id join is the worst failure mode)
    hold_onto = Ontology.from_categories(holdout.raw.get("categories", []))
    if hold_onto.id_to_name != ontology.id_to_name:
        raise ValueError(
            f"{name} ontology differs from train: train={ontology.id_to_name} "
            f"vs {name}={hold_onto.id_to_name}")

    dev_idx = stratified_dev_indices(train, 1.0 - fit_train_ratio, seed)
    train_idx = set(range(len(train.images))) - dev_idx
    return Dataset(root=root, ontology=ontology, train=train,
                   fit_train=_subset(train, train_idx),
                   fit_dev=_subset(train, dev_idx),
                   holdout=holdout, holdout_split=name)
