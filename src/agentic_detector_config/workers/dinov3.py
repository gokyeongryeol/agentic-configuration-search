"""One-shot worker for DINOv3 crop embeddings.

**Runs with the vLLM venv python (`services.paths.VLLM_VENV`)**, where torch and
transformers already exist as vLLM dependencies. The package proper has no torch,
so in-process loading is impossible; exiting also releases the GPU entirely.

Hence this file **imports only stdlib + torch + transformers + numpy + PIL** — it
must not import agentic_detector_config modules.

Encoding: pooler_output, else the CLS token → L2 normalization, in mini-batches.
"""

from __future__ import annotations

import argparse
import json
import sys

# Feeding all crops to ViT-L at once OOMs on ~60GB of activations.
BATCH = 64


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON list of crop image paths")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="output path for the embeddings .npy")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    with open(args.manifest, encoding="utf-8") as fh:
        paths = json.load(fh)
    if not paths:
        np.save(args.out, np.zeros((0, 1), dtype=np.float32))
        return 0

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(args.device)

    chunks = []
    for start in range(0, len(paths), BATCH):
        images = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH]]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
        if getattr(out, "pooler_output", None) is not None:
            emb = out.pooler_output
        elif getattr(out, "last_hidden_state", None) is not None:
            emb = out.last_hidden_state[:, 0]          # CLS token
        else:
            raise RuntimeError("DINOv3 returned neither pooler_output nor last_hidden_state")
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)   # L2 normalize
        chunks.append(emb.detach().float().cpu().numpy())
        print(f"embedded {min(start + BATCH, len(paths))}/{len(paths)}", flush=True)

    np.save(args.out, np.concatenate(chunks, axis=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
