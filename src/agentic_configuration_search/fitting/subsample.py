"""fit_train subset — an image cap on the training material (lora, rescorer, crop classifier).

The three trainings read all of fit_train, so their time follows the dataset
size. When fit_train is larger than `subsample.max_images`, only that many
images are used — picked not at random but **in the order that covers the
appearances** (greedy set cover over embedding clusters):

1. Cluster the DINOv3 embeddings of GT object crops per class with k-center
   (farthest-point) — the number of centers is objects × `center_ratio` (at
   least 1). Each object belongs to its nearest center.
2. Collect, per image, the set of centers of its objects.
3. Pick the image holding the most not-yet-covered centers (ties go to the
   earlier image) and mark its centers covered. Once everything is covered, run
   another pass over the remaining images (a second example per appearance).
   Images with no center to cover (all GT uncroppable) only fill leftover slots.
4. Background images without GT keep their share of fit_train (file name hash
   order) — they are the FP material the rescorer and classifier learn from.

Rare classes and appearances always get in on the first pass — unlike a random
sample, a large cluster cannot take all the slots. Deterministic (sorted inputs,
argmax, hashes), so resume and the training reuse cache (whose key holds the
fit_train fingerprint) see the same subset.

**Only the training material shrinks.** Mining (all of train), the verdicts
(fit_dev) and adoption (holdout) are unchanged — verdict noise must not grow.
The picks are made within fit_train only, so material purity is unaffected.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from ..infer import embedder
from . import mine

SUBSET_FILE = "fit_train_subset.json"


def _hash_order(split, idx) -> list[int]:
    return sorted(idx, key=lambda i: hashlib.sha256(
        split.images[i].file_name.encode()).hexdigest())


def select_fit_train(split, emb_by_class: dict, *, max_images: int,
                     center_ratio: float) -> tuple[list[int], dict]:
    """Image indices of fit_train to keep (ascending) and a record.

    emb_by_class: {category_id: (annotation ids, embedding matrix)} — rows follow the ids.
    """
    n = len(split.images)
    if max_images <= 0 or n <= max_images:
        return list(range(n)), {"applied": False, "n_images": n}

    image_of = {a.id: i for i, img in enumerate(split.images) for a in img.annotations}
    centers_of: list[set] = [set() for _ in range(n)]
    n_centers: dict[str, int] = {}
    for cid in sorted(emb_by_class):
        ann_ids, E = emb_by_class[cid]
        if not len(ann_ids):
            continue
        E = np.asarray(E, dtype=np.float32)    # float32 — halves memory for large classes
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        k = max(1, math.ceil(center_ratio * len(ann_ids)))
        centers = embedder.kcenter_greedy(E, k)
        # unit vectors, so the nearest center = largest dot product (no n×k×D distance tensor)
        for a, j in zip(ann_ids, (E @ E[centers].T).argmax(axis=1)):
            centers_of[image_of[a]].add((cid, int(j)))
        n_centers[str(cid)] = len(centers)

    background = [i for i in range(n) if not split.images[i].annotations]
    n_bg = min(len(background), math.floor(max_images * len(background) / n + 0.5))
    keep = _hash_order(split, background)[:n_bg]
    budget = max_images - n_bg
    picked, passes, first_images, first_cov = _greedy_cover(
        [i for i in range(n) if split.images[i].annotations], centers_of, budget)
    total = sum(n_centers.values())
    # images with no center to cover and the remaining background fill leftover slots in hash order
    rest = _hash_order(split, [i for i in range(n) if i not in set(picked) | set(keep)
                               and split.images[i].annotations])
    rest += _hash_order(split, [i for i in background if i not in set(keep)])
    chosen = sorted(set(keep) | set(picked) | set(rest[:max_images - len(keep)
                                                        - len(picked)]))
    return chosen, {"applied": True, "n_images": n, "n_selected": len(chosen),
                    "n_background": n_bg, "centers": n_centers, "passes": passes,
                    "first_pass_images": first_images,
                    "first_pass_coverage": round(first_cov / total, 4) if total else 0.0}


def _greedy_cover(cands: list[int], centers_of: list[set], budget: int):
    """Images with the most not-yet-covered centers first — another pass once all are covered.

    Returns (picked indices, passes, first-pass image count, centers covered on the first pass)."""
    picked: list[int] = []
    done: set[int] = set()
    passes, first_images, first_cov = 0, 0, 0
    while len(picked) < budget:
        pool = [i for i in cands if i not in done and centers_of[i]]
        if not pool:
            break
        passes += 1
        images_of: dict = {}
        for i in pool:
            for c in centers_of[i]:
                images_of.setdefault(c, []).append(i)
        score = {i: len(centers_of[i]) for i in pool}
        covered: set = set()
        while len(picked) < budget:
            best = max((i for i in pool if i not in done),
                       key=lambda i: (score[i], -i), default=None)
            if best is None or score[best] == 0:
                break
            new = centers_of[best] - covered
            covered |= new
            done.add(best)
            picked.append(best)
            for c in new:
                for j in images_of[c]:
                    if j not in done:
                        score[j] -= 1
        if passes == 1:
            first_images, first_cov = len(picked), len(covered)
    return picked, passes, first_images, first_cov


def fit_train_embeddings(split, ontology, encode_fn, crop_dir: Path) -> dict:
    """Per-class embeddings of fit_train GT crops — {category_id: (annotation ids, matrix)}.

    Crops are made with the same CropCache as mining — the real encode_fn keeps
    a run-level cache keyed by crop content hash, so crops mining already
    embedded are not computed again."""
    cache = mine.CropCache(split, ontology, crop_dir)
    groups = {cid: cache.paths[cid] for cid, anns in sorted(cache.prompt_pool.items())
              if anns}
    prefetch = getattr(encode_fn, "prefetch", None)
    if prefetch is not None and groups:
        prefetch(groups)
    return {cid: ([a.id for a in cache.prompt_pool[cid]],
                  np.asarray(encode_fn(cid, paths)))
            for cid, paths in groups.items()}
