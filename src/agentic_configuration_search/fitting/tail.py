"""Material production for the lora axis — YOLOE tail fine-tune + λ-interpolation arms.

- Training reads **fit_train only** (material purity). The training itself runs
  in a worker subprocess (`workers/yoloe_tail.py` — torch-allowed zone). The
  worker runs in this repo's venv (sys.executable) because it needs
  ultralytics (AGPL-3.0).
- The λ grid is an **interpolation**: θ_λ = (1-λ)·θ_base + λ·θ_tuned (tuned
  subset keys only). An arm is not a heavy weight copy but a single
  **descriptor JSON** —
  {"kind": "yoloe_tail_arm", "tuned": <absolute path to subset .pt>, "lam": λ}.
  Loading (applying the interpolation) is done by the weight swap in
  clients.yoloe. By convention λ=0 has no descriptor and is the base as-is
  (bit-exact).
- Labels are boxes only, so seg labels are exported as box-rectangle polygons
  (yoloe-26x is a seg model, so the trainer requires seg labels — only boxes
  are consumed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..clients.yoloe import ARM_KIND

log = logging.getLogger(__name__)


def export_yolo_dataset(split, ontology, out_dir: Path) -> Path:
    """fit_train split -> ultralytics YOLO-seg dataset. Returns the data.yaml path.

    - images/train: symlinks to the original images (no copy, absolute targets).
    - labels/train: contiguous class index + box-rectangle polygon (8 normalized coords).
    - val points at train — training runs with val=False, but the data schema requires it.
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images" / "train"
    lbl_dir = out_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    cls_index = {c.id: i for i, c in enumerate(ontology.classes)}

    for e in split.images:
        src = Path(split.image_path(e)).resolve()
        dst = img_dir / e.file_name
        if not dst.exists():
            os.symlink(src, dst)
        lines = []
        for a in e.annotations:
            x1, y1, x2, y2 = a.bbox_xyxy
            xs = [max(0.0, min(1.0, v / e.width)) for v in (x1, x2)]
            ys = [max(0.0, min(1.0, v / e.height)) for v in (y1, y2)]
            poly = (f"{xs[0]:.6f} {ys[0]:.6f} {xs[1]:.6f} {ys[0]:.6f} "
                    f"{xs[1]:.6f} {ys[1]:.6f} {xs[0]:.6f} {ys[1]:.6f}")
            lines.append(f"{cls_index[a.category_id]} {poly}")
        (lbl_dir / (Path(e.file_name).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))

    names = "\n".join(f"  {i}: {c.name}" for i, c in enumerate(ontology.classes))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"
        f"names:\n{names}\n")
    return data_yaml


# Tail training shares its GPU with the in-process detector — once fragmentation
# piles up reserved-but-unallocated memory, a large allocation fails with OOM
# even though memory is left. The allocator setting only changes allocation, so
# results are the same. A value the operator set in the env wins.
ALLOC_CONF = "PYTORCH_ALLOC_CONF"
EXPANDABLE = "expandable_segments:True"


# cfg.lora fields that change the training output — part of the reuse key
# (train_key). The λ grid, margin and min_images (post-training verdict) and
# device (placement) are left out
TRAIN_FIELDS = ("epochs", "batch_size", "lr")


def split_identity(split) -> str:
    """Content fingerprint of a split — image names and sizes plus GT boxes (in a
    sorted, deterministic order)."""
    rows = [[e.file_name, e.width, e.height,
             sorted([a.category_id, [round(float(x), 3) for x in a.bbox_xyxy]]
                    for a in e.annotations)]
            for e in split.images]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def train_key(cfg, *, fit_train, ontology, base_weights: Path, seed: int) -> dict:
    """Every input of one train_tail call — equal keys reuse the tuned subset.

    Training always starts from the base weights on fit_train GT, regardless of
    the champion, so within a run an equal key is the same training. Training
    itself is not deterministic (GPU), but reuse only means "use what was
    trained once again", so the decision discipline is unchanged. base_weights
    is identified by path (realpath) — hashing the file would read the whole
    checkpoint."""
    lo = cfg.lora
    return {"seed": int(seed),
            "lora": {k: getattr(lo, k) for k in TRAIN_FIELDS},
            "base_weights": os.path.realpath(str(base_weights)),
            "fit_train": split_identity(fit_train),
            "ontology": ontology.to_dict() if ontology is not None else None}


def key_digest(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def train_tail(cfg, *, fit_train, ontology, base_weights: Path, out_dir: Path,
               seed: int, log_path: Path,
               popen=subprocess.Popen) -> Path:
    """Train the tail on fit_train and return the tuned subset (.pt) path.

    Hyperparameters come from cfg.lora (epochs, batch_size, lr, device). Fixed seed, few epochs.

    The ultralytics trainer always writes full-model checkpoints (last.pt,
    best.pt) at the final epoch, whatever `save` says. Only the tail subset is
    used — arms reference it and are interpolated in memory — so once the
    subset exists the checkpoints are deleted (_drop_trainer_checkpoints). On a
    failed run they are kept for diagnosis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = export_yolo_dataset(fit_train, ontology, out_dir / "dataset")
    tuned = out_dir / "tuned_tail.pt"
    cmd = [sys.executable, "-m", "agentic_configuration_search.workers.yoloe_tail",
           "--model", str(Path(base_weights).resolve()),
           "--data", str(data_yaml.resolve()),
           "--out", str(tuned.resolve()),
           "--project", str((out_dir / "ultralytics").resolve()),
           "--epochs", str(int(cfg.lora.epochs)),
           "--batch", str(int(cfg.lora.batch_size)),
           "--lr0", str(float(cfg.lora.lr)),
           "--seed", str(int(seed)),
           "--device", str(cfg.lora.device)]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("yoloe tail training: %d images, epochs=%d, device=%s (log: %s)",
             len(fit_train.images), cfg.lora.epochs, cfg.lora.device, log_path)
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)   # defensively clear any inherited GPU masking
    env.setdefault(ALLOC_CONF, EXPANDABLE)
    with open(log_path, "ab") as fh:
        proc = popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"yoloe tail training failed (exit {rc}) — log: {log_path}")
    if not tuned.exists():
        raise RuntimeError(f"tuned subset was not produced: {tuned}")
    _drop_trainer_checkpoints(out_dir / "ultralytics")
    return tuned


def _drop_trainer_checkpoints(project: Path) -> int:
    """Delete the trainer's full-model checkpoints (<project>/*/weights/*.pt) and
    return the bytes freed. The rest of the run directory (args.yaml and the
    like) is small and stays as a record."""
    freed = 0
    for wdir in Path(project).glob("*/weights"):
        if not wdir.is_dir():
            continue
        freed += sum(f.stat().st_size for f in wdir.glob("*.pt") if f.is_file())
        shutil.rmtree(wdir)
    if freed:
        log.info("yoloe tail: removed trainer checkpoints under %s (%.1f MB)",
                 project, freed / 1e6)
    return freed


def build_tail_arms(tuned: Path, lambdas, out_dir: Path) -> dict[float, Path]:
    """λ -> arm descriptor path. λ=0 is the base as-is, so no descriptor is written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms: dict[float, Path] = {}
    for lam in sorted({float(x) for x in lambdas}):
        if lam == 0.0:
            continue
        p = out_dir / f"arm_lam{lam:g}.json"
        p.write_text(json.dumps({"kind": ARM_KIND,
                                 "tuned": str(Path(tuned).resolve()),
                                 "lam": lam}, indent=1))
        arms[lam] = p
    return arms


def select_candidate(grid: list[dict], margin: float) -> tuple[dict, dict]:
    """Pick the arm to adopt from each λ arm's fit_dev ΔF(β).

    Takes the best candidate with ΔF(β) > `margin` (ties go to the smaller λ —
    the conservative choice, closest to base). λ=0 is on the grid, so without
    proven gain it naturally falls back to identity — same structure as the
    rescorer's α selection. champion (lam=None) re-enters the weights adopted
    in a previous round.
    """
    scored = [{"source": r.get("source", "grid"), "rank": r["rank"],
               "preset": r.get("preset"), "lam": r["lam"], "delta": r["delta"]}
              for r in grid]
    # λ=0 is the baseline (Δ=0), so it is excluded — it equals not adopting.
    # champion (lam=None) is not the base, so it is a candidate.
    cand = [s for s in scored if s["source"] == "champion" or s["lam"] != 0.0]
    if not cand:
        return {}, {"adopted": False, "reason": "no λ>0 candidates", "scored": scored}
    # Ties: (1) prefer a new grid candidate over champion, (2) within grid, smaller λ.
    # (1) partly offsets the selection bias champion gains by re-challenging the
    # same scoring split every round.
    best = max(cand, key=lambda s: (s["delta"], s["source"] == "grid",
                                    -(s["lam"] if s["lam"] is not None else 1.0)))
    champ = next((s for s in scored if s["source"] == "champion"), None)
    info = {"scored": scored,
            "champion": ({"delta": champ["delta"], "won": best is champ}
                         if champ else None)}
    if best["delta"] > margin:
        return best, {**info, "adopted": True, "rank": best["rank"],
                      "lam": best["lam"], "source": best["source"],
                      "delta": best["delta"],
                      "reason": f"fit_dev ΔF(β) {best['delta']:+.4f} > margin {margin}"
                                + (" (champion reused)" if best["source"] == "champion" else "")}
    return best, {**info, "adopted": False, "delta": best["delta"],
                  "source": best["source"],
                  "reason": f"best candidate fit_dev ΔF(β) {best['delta']:+.4f} "
                            f"≤ margin {margin}"}
