"""YOLOE tail fine-tune worker — one-shot subprocess (torch allowed here).

Trainer for the `lora` axis (a tail fine-tune, not LoRA adapters):
- **Freeze boundary**: freeze most of backbone and neck (modules 0–15, 17, 18,
  20, 21) and train only **the three PAN-output C3k2 blocks feeding the head
  (modules 16, 19, 22) + the head (module 23, YOLOESegment26)** — "detection/seg
  head + last neck blocks".
- **Preserve promptability**: the YOLOEPE (linear probing) trainer fuses text
  embeddings into the head, destroying open-vocab ability and parameter shapes.
  Use the non-fusing YOLOESegTrainer instead and feed txt_feats (without_reprta —
  reprta is trained inside the model) in the batch (same as
  WorldTrainer.preprocess_batch).
- **Output**: only the **training-prefix keys** of the trained state_dict
  (parameters + buffers), in float32 — the material for the interpolation
  θ_λ = (1−λ)·θ_base + λ·θ_tuned. Frozen sections (including backbone BN buffer
  drift) are discarded, so λ=0 is bit-exact base.
- Labels are boxes only, so seg labels are **box-rectangle polygons** (made on
  the export side). Mask quality is irrelevant — only boxes are consumed.

Path arguments crossing the process boundary must be absolute (the worker's cwd
may differ from the caller's).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

# Indices of trained (unfrozen) modules — the freeze boundary in the docstring
TAIL_MODULES = (16, 19, 22, 23)
FROZEN_MODULES = tuple(i for i in range(23) if i not in TAIL_MODULES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="base checkpoint (.pt, absolute path)")
    ap.add_argument("--data", required=True, help="data.yaml (absolute path)")
    ap.add_argument("--out", required=True, help="output path for the tuned tail state_dict (.pt)")
    ap.add_argument("--project", required=True, help="ultralytics run directory")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from ultralytics import YOLOE
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.yoloe import YOLOESegTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class TailSegTrainer(YOLOESegTrainer):
        """YOLOESegTrainer + txt_feats injection — promptable training without fusing."""

        _pe_cache: dict = {}

        def preprocess_batch(self, batch):
            batch = DetectionTrainer.preprocess_batch(self, batch)
            texts = list(itertools.chain(*batch["texts"]))
            missing = sorted({t for t in texts if t not in self._pe_cache})
            if missing:
                pe = unwrap_model(self.model).get_text_pe(
                    missing, batch=len(missing), without_reprta=True,
                    cache_clip_model=True)
                self._pe_cache.update(dict(zip(missing, pe.squeeze(0))))
            txt_feats = torch.stack([self._pe_cache[t] for t in texts]).to(
                self.device)
            batch["txt_feats"] = txt_feats.reshape(
                len(batch["texts"]), -1, txt_feats.shape[-1])
            # The (lazy) criterion snapshots head.nc — the pretrained head has
            # nc=80 (max text slots), but our batch's text slots equal the dataset
            # class count. Set it before the first loss so the BCE target width
            # matches pred.
            unwrap_model(self.model).model[-1].nc = batch["txt_feats"].shape[1]
            return batch

    model = YOLOE(args.model)
    base_state = {k: v.detach().clone().float()
                  for k, v in model.model.state_dict().items()}
    model.train(
        data=args.data, trainer=TailSegTrainer,
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        optimizer="AdamW", lr0=args.lr0,   # 'auto' ignores lr0 — pin explicitly
        seed=args.seed, device=args.device, freeze=list(FROZEN_MODULES),
        val=False, plots=False, amp=False, workers=2, compile=False,
        project=args.project, name="train", exist_ok=True, verbose=False)

    trained = unwrap_model(model.trainer.model).float().state_dict()
    prefixes = tuple(f"model.{i}." for i in TAIL_MODULES)
    subset: dict = {}
    skipped_shape: list[str] = []
    for k, v in trained.items():
        if not k.startswith(prefixes):
            continue
        b = base_state.get(k)
        if b is None or b.shape != v.shape:
            # The trainer model is built with the data's nc, so nc-dependent
            # shapes (e.g. Proto26's semantic branch — training loss only; serving
            # uses return_semantic=False) differ from base (nc=80). These keys are
            # unused in serving, so drop them.
            skipped_shape.append(k)
            continue
        subset[k] = v.detach().clone().float().cpu()
    if not subset:
        raise RuntimeError("tail subset is empty — freeze boundary / key prefix mismatch")
    changed = sum(1 for k in subset
                  if not torch.equal(subset[k], base_state[k].cpu()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, out)
    (out.parent / "tail_meta.json").write_text(json.dumps({
        "tail_modules": list(TAIL_MODULES), "keys": len(subset),
        "changed_keys": changed, "shape_skipped": skipped_shape,
        "epochs": args.epochs, "lr0": args.lr0,
        "batch": args.batch, "seed": args.seed}, indent=1))
    print(f"saved tail: {out} (keys={len(subset)}, changed={changed}, "
          f"shape_skipped={len(skipped_shape)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
