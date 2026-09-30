"""One-shot worker that trains the SigLIP2+LoRA crop classifier.

**Runs with the vLLM venv python (`services.paths.VLLM_VENV`, env ADC_VLLM_VENV)**,
which has torch and transformers; peft must also be installed there (peft,
Apache-2.0). The package proper is torch-free, so in-process training is
impossible; exiting also releases the GPU entirely.
**Imports only stdlib + torch + transformers + peft + PIL.**

Hyperparameters are fixed to the universal-best from a prior sweep — not a
search axis. The output checkpoint contract matches the loader in
clients.yoloe._CropReclassifier:
    {"state_dict", "num_classes", "classifier_model": "siglip"}
label 0 = background, label i+1 = class_names[i] (ascending category id).

Input manifest (jsonl): rows of {"crop_path": str, "label": int}.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict

# Universal-best from a prior sweep — fixed constants, not a search axis
# (tests/test_crop_classifier.py keeps these in sync with the fitting-side copy)
MODEL_NAME = "google/siglip2-base-patch16-224"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "out_proj"]   # "qkvo"
LORA_DROPOUT = 0.05
LR = 5e-4
EPOCHS = 20
PATIENCE = 7
BATCH_SIZE = 8
VAL_FRACTION = 0.2
MIN_RECORDS_FOR_VAL = 20


def build_model(num_classes: int):
    """SigLIP2 classifier + LoRA (qkvo). Must match the rebuild in clients.yoloe
    for state_dict keys to line up (LoRA r/alpha/target determine key structure)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True)
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=LORA_TARGET, bias="none")
    model = get_peft_model(model, lora_cfg)
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name or "classifier" in name)
    return model, processor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="crop manifest jsonl")
    parser.add_argument("--num-classes", type=int, required=True, help="number of classes, including background")
    parser.add_argument("--out", required=True, help="output path for classifier.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    with open(args.manifest, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records:
        raise SystemExit("manifest is empty — no crops to train on")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Stratified train/val split (per-label ratio) — with few samples, no val (no early stopping)
    use_val = len(records) >= MIN_RECORDS_FOR_VAL and VAL_FRACTION > 0
    if use_val:
        by_label = defaultdict(list)
        for r in records:
            by_label[int(r["label"])].append(r)
        rng = random.Random(args.seed)
        train_records, val_records = [], []
        for _, items in sorted(by_label.items()):
            shuffled = list(items)
            rng.shuffle(shuffled)
            n_val = min(max(1, int(round(len(shuffled) * VAL_FRACTION))),
                        max(0, len(shuffled) - 1))
            val_records.extend(shuffled[:n_val])
            train_records.extend(shuffled[n_val:])
        use_val = bool(val_records)
    if not use_val:
        train_records, val_records = records, []

    model, processor = build_model(args.num_classes)
    model = model.to(args.device)

    class CropDataset(Dataset):
        def __init__(self, recs):
            self.recs = recs

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            r = self.recs[i]
            image = Image.open(r["crop_path"]).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"][0]
            return pixel_values, torch.tensor(int(r["label"]), dtype=torch.long)

    # Class imbalance is corrected by the sampler only (no loss weighting) so
    # minority classes learn when background dominates. Falls back to shuffle if
    # there is only one label
    labels = [int(r["label"]) for r in train_records]
    counts = Counter(labels)
    sampler = None
    if len(counts) > 1:
        weights = torch.as_tensor([1.0 / counts[l] for l in labels], dtype=torch.double)
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True, generator=gen)
    train_loader = DataLoader(CropDataset(train_records), batch_size=BATCH_SIZE,
                              shuffle=sampler is None, sampler=sampler, num_workers=4)
    val_loader = (DataLoader(CropDataset(val_records), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4) if val_records else None)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)

    best_val_acc, best_state, stale = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for pixel_values, y in train_loader:
            pixel_values, y = pixel_values.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(pixel_values=pixel_values, labels=y)
            out.loss.backward()
            optimizer.step()
        if val_loader is None:
            print(f"epoch {epoch + 1}/{EPOCHS} (no val)", flush=True)
            continue
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for pixel_values, y in val_loader:
                pixel_values, y = pixel_values.to(args.device), y.to(args.device)
                preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
                correct += int((preds == y).sum().item())
                total += y.numel()
        val_acc = correct / max(total, 1)
        print(f"epoch {epoch + 1}/{EPOCHS}: val_acc={val_acc:.3f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc, stale = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop (best val_acc={best_val_acc:.3f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"state_dict": model.state_dict(),
                "num_classes": args.num_classes,
                "classifier_model": "siglip"}, args.out)
    print(f"saved: {args.out} (num_classes={args.num_classes}, "
          f"train={len(train_records)}, val={len(val_records)}, "
          f"best_val_acc={best_val_acc if best_val_acc >= 0 else 'n/a'})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
