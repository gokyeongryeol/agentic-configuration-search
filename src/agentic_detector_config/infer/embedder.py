"""DINOv3 worker invocation + k-center greedy.

Embedding itself runs in `workers/dinov3.py` as a one-shot subprocess in the
vLLM venv — the package proper has no torch. This module only calls the worker,
reads the result, and does k-center selection.
"""

from __future__ import annotations

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
        raise RuntimeError(f"vLLM venv python not found: {python} — check the ADC_VLLM_VENV env var")

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

    cmd = [str(python), "-m", "agentic_detector_config.workers.dinov3",
           "--manifest", str(manifest), "--model", model,
           "--device", device, "--out", str(out_npy)]
    log.info("running DINOv3 worker: %d crops on %s (log: %s)", len(crop_paths), device, log_path)
    # The worker runs as `-m agentic_detector_config.workers.dinov3`, so it needs
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


def trim_outliers(embeddings: np.ndarray, trim: float, min_keep: int = 1) -> np.ndarray:
    """**Original indices** (ascending) of candidates left after trimming the
    `trim` fraction farthest from the centroid.

    The centroid is computed once on the **full, untrimmed set** — recomputing it
    iteratively makes results vary strongly with the data at the same trim ratio,
    hurting reproducibility. Never trims below `min_keep` (usually k) — trimming
    is pointless if too few remain to pick from.
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if not 0.0 <= trim < 1.0:
        raise ValueError(f"trim must be in [0, 1) (got {trim})")
    if trim == 0.0:
        return np.arange(n)
    d = np.linalg.norm(embeddings - embeddings.mean(axis=0, keepdims=True), axis=1)
    keep_n = max(min(min_keep, n), n - int(n * trim))
    kept = np.sort(np.argsort(d)[:keep_n])
    if keep_n < n:
        log.info("outlier trim=%.2f: kept %d/%d candidates (dropped the %d farthest from the centroid)",
                 trim, keep_n, n, n - keep_n)
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


def kcenter_greedy(embeddings: np.ndarray, k: int, trim: float = 0.0) -> list[int]:
    """Farthest-point traversal — k points covering the embedding space as widely
    as possible.

    With `trim > 0`, **first drop the `trim` fraction farthest from the
    centroid**, then traverse only the remainder. Pure k-center deliberately picks
    the most atypical instances to maximize coverage, but such outliers hurt as
    visual prompts. Trimming is the compromise: **keep diversity, exclude what
    cannot serve as a prompt**. Returned indices always refer to the original set.

    Larger `trim` converges to `mean_nearest` — if exactly k candidates remain
    after trimming, the result equals the k nearest to the centroid.
    """
    n = len(embeddings)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    kept = trim_outliers(embeddings, trim, min_keep=k)
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
                   trim: float = 0.0, rng=None) -> list[int]:
    """Per-class exemplar selection dispatch — the single point that makes select
    and tune_eval use **the same rule** (if they diverged, tune's adoption
    decision would be measured under a different prompt regime than serving).

    mean/kcenter need `embeddings`; random needs `rng`. random needs no
    embeddings, so callers may skip the DINOv3 pass.
    """
    if selection == "mean":
        return mean_nearest(embeddings, k)
    if selection == "kcenter":
        return kcenter_greedy(embeddings, k, trim=trim)
    if selection == "random":
        return rng.sample(range(n), min(k, n))
    from ..config import EXEMPLAR_SELECTIONS  # noqa: PLC0415 — error message only
    raise ValueError(f"select.exemplar_selection: unknown value {selection!r} "
                     f"({' | '.join(EXEMPLAR_SELECTIONS)})")
