"""DINOv3 worker invocation + k-center greedy.

Embedding itself runs in `workers/dinov3.py` as a one-shot subprocess in the
GPU worker venv (`ACS_VLLM_VENV`) — the package proper has no torch. This module only calls the worker,
reads the result, and does k-center selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from ..services.paths import PACKAGE_PARENT, VLLM_VENV

log = logging.getLogger(__name__)


def encode_crops(repo_root: Path, crop_paths: list[Path], model: str, device: str,
                 log_path: Path, popen=subprocess.Popen) -> np.ndarray:
    """Embed crop images with DINOv3. Returns an (N, D) L2-normalized array.

    GPU memory is released when the worker exits — more reliable than an
    in-process load followed by `del`/`empty_cache`. `HF_TOKEN` is inherited from
    the environment (DINOv3 is gated).
    """
    if not crop_paths:
        return np.zeros((0, 1), dtype=np.float32)

    python = repo_root / VLLM_VENV / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"vLLM venv python not found: {python} — check the ACS_VLLM_VENV env var")

    # The worker's cwd is repo_root, not ours — every path crossing the boundary
    # (argv and manifest contents) must be absolute.
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Name outputs by the log_path stem — this runs once per class, and a shared
    # filename would let the second class overwrite the first, breaking post-hoc
    # analysis.
    stem = log_path.stem
    manifest = log_path.parent / f"{stem}_manifest.json"
    out_npy = log_path.parent / f"{stem}.npy"
    manifest.write_text(json.dumps([str(Path(p).resolve()) for p in crop_paths]))

    cmd = [str(python), "-m", "agentic_configuration_search.workers.dinov3",
           "--manifest", str(manifest), "--model", model,
           "--device", device, "--out", str(out_npy)]
    log.info("running DINOv3 worker: %d crops on %s (log: %s)", len(crop_paths), device, log_path)
    # The worker runs as `-m agentic_configuration_search.workers.dinov3`, so it needs
    # the package on the path (the package and workers __init__ get imported —
    # hence both stay lightweight).
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_PARENT)}
    with open(log_path, "ab") as fh:
        proc = popen(cmd, cwd=repo_root, env=env, stdout=fh, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"DINOv3 worker failed (exit {rc}) — log: {log_path}")

    emb = np.load(out_npy)
    if len(emb) != len(crop_paths):
        raise RuntimeError(f"embedding count mismatch: crop {len(crop_paths)} vs emb {len(emb)}")
    return emb


def cached_encoder(encode, cache_path: Path):
    """An encoder for encode(paths, *args) -> (N, D), cached by **crop content hash**.

    GT crops are rewritten into a different folder for every candidate, but the
    content (same source image, box, JPEG settings) is the same — keying by
    content rather than path is what lets them be reused across rounds and
    candidates. Only missing crops go to the real encoder (the DINOv3 worker),
    and the cache is rewritten atomically to cache_path (npz) so it is read back
    after resume. When the encoder model changes, the caller must pass another
    cache_path. Extra args (a log path, ...) are passed through with the misses.
    """
    cache_path = Path(cache_path)
    store: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            store = dict(zip(z["keys"].tolist(), z["vecs"]))

    def encode_cached(paths, *args) -> np.ndarray:
        paths = [Path(p) for p in paths]
        if not paths:
            return encode([], *args)
        keys = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        first_path = dict(zip(reversed(keys), reversed(paths)))   # key → first path
        todo = [k for k in dict.fromkeys(keys) if k not in store]
        if todo:
            for k, row in zip(todo, encode([first_path[k] for k in todo], *args)):
                store[k] = np.asarray(row)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.npz")
            np.savez(tmp, keys=np.array(list(store)), vecs=np.stack(list(store.values())))
            tmp.replace(cache_path)
            log.info("embedding cache: %d computed, %d reused (%s)",
                     len(todo), len(keys) - len(todo), cache_path.name)
        return np.stack([store[k] for k in keys])

    return encode_cached


# MAD → normal σ conversion factor. If the MAD is 0 (more than half the distances
# are equal), the mean absolute deviation × √(π/2) stands in — if both are 0 there
# is no spread, so nothing is trimmed.
_MAD_TO_SIGMA = 1.4826
_MEANAD_TO_SIGMA = 1.2533


def trim_outliers(embeddings: np.ndarray, outlier_mad: float,
                  min_keep: int = 1) -> np.ndarray:
    """**Original indices** (ascending) of candidates left after dropping those whose
    centroid distance stands out from the distribution.

    A distance d is an outlier if it exceeds `median(d) + outlier_mad × σ̂`
    (σ̂ = MAD × 1.4826 — a spread estimate not pulled by extremes). Unlike a
    fixed-fraction trim, nothing is dropped when there are no outliers. 0 turns
    trimming off.

    The centroid is computed once on the **full, untrimmed set** — recomputing it
    iteratively makes results vary strongly with the data, hurting
    reproducibility. Never trims below `min_keep` (usually k) — then the min_keep
    candidates nearest the centroid are kept.
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if outlier_mad < 0:
        raise ValueError(f"outlier_mad must be >= 0 (got {outlier_mad})")
    if outlier_mad == 0.0:
        return np.arange(n)
    d = np.linalg.norm(embeddings - embeddings.mean(axis=0, keepdims=True), axis=1)
    med = float(np.median(d))
    dev = np.abs(d - med)
    sigma = float(np.median(dev)) * _MAD_TO_SIGMA or float(dev.mean()) * _MEANAD_TO_SIGMA
    if sigma == 0.0:
        return np.arange(n)
    keep = d <= med + outlier_mad * sigma
    min_keep = min(min_keep, n)
    if keep.sum() < min_keep:
        keep = np.zeros(n, dtype=bool)
        keep[np.argsort(d, kind="mergesort")[:min_keep]] = True
    kept = np.flatnonzero(keep)
    if len(kept) < n:
        log.info("outlier trim (median + %.1f·MAD): kept %d/%d candidates", outlier_mad,
                 len(kept), n)
    return kept


def _farthest_point_traversal(embeddings: np.ndarray, k: int) -> list[int]:
    """Seed = point nearest the centroid; then add, k-1 times, the point whose
    minimum distance to the chosen centers is largest.

    **Never emits duplicates**: if all remaining min-distances are 0 (duplicate
    embeddings leave fewer than k distinct points), argmax would re-pick a chosen
    point — the same GT box would then appear multiple times among exemplars,
    skewing prompt weight toward one instance. Zero-distance candidates are
    excluded, and the loop stops early when nothing new remains.
    """
    if k <= 0 or len(embeddings) == 0:
        return []
    n = len(embeddings)
    center = int(np.linalg.norm(embeddings - embeddings.mean(axis=0, keepdims=True),
                                axis=1).argmin())
    centers = [center]
    distances = np.full(n, np.inf)
    for _ in range(1, k):
        current = np.linalg.norm(embeddings - embeddings[centers[-1]], axis=1)
        distances = np.minimum(distances, current)
        distances[centers[-1]] = -1.0        # exclude chosen points (and their duplicates)
        nxt = int(distances.argmax())
        if distances[nxt] <= 0.0:
            break                            # distinct points exhausted
        centers.append(nxt)
    return centers


def kcenter_greedy(embeddings: np.ndarray, k: int, outlier_mad: float = 0.0) -> list[int]:
    """Farthest-point traversal — k points covering the embedding space as widely
    as possible.

    With `outlier_mad > 0`, **first drop candidates whose centroid distance stands
    out from the distribution** (trim_outliers), then traverse only the remainder.
    Pure k-center picks the most atypical instances, but such outliers hurt as
    visual prompts — keep diversity, exclude the extremes. Returned indices
    always refer to the original set.

    If exactly k candidates remain after trimming, the result equals the k nearest
    to the centroid (`mean_nearest`).
    """
    n = len(embeddings)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    kept = trim_outliers(embeddings, outlier_mad, min_keep=k)
    local = _farthest_point_traversal(embeddings[kept], k)
    return [int(kept[i]) for i in local]


def mean_nearest(embeddings: np.ndarray, k: int) -> list[int]:
    """The k points nearest the centroid — the most **representative** instances.

    The reference is the **candidates' own** centroid; visual prompts work
    better the closer they are to the class prototype.
    """
    n = len(embeddings)
    if n == 0 or k <= 0:
        return []
    d = np.linalg.norm(embeddings - embeddings.mean(axis=0, keepdims=True), axis=1)
    return [int(i) for i in np.argsort(d)[:min(k, n)]]


def select_indices(selection: str, k: int, n: int, *, embeddings=None,
                   outlier_mad: float = 0.0, rng=None) -> list[int]:
    """Per-class exemplar selection dispatch (single entry point per exemplar_selection).

    mean/kcenter need `embeddings`; random needs `rng`. random needs no
    embeddings, so callers may skip the DINOv3 pass.
    """
    if selection == "mean":
        return mean_nearest(embeddings, k)
    if selection == "kcenter":
        return kcenter_greedy(embeddings, k, outlier_mad=outlier_mad)
    if selection == "random":
        return rng.sample(range(n), min(k, n))
    from ..config import EXEMPLAR_SELECTIONS  # noqa: PLC0415 — error message only
    raise ValueError(f"select.exemplar_selection: unknown value {selection!r} "
                     f"({' | '.join(EXEMPLAR_SELECTIONS)})")
