"""One-shot worker for DINOv3 crop embeddings.

**Runs with the worker venv python (`services.paths.VLLM_VENV`)**, which must have
torch and transformers. The package proper has no torch, so in-process loading
is impossible; exiting also releases the GPU entirely.

Hence this file **imports only stdlib + torch + transformers + numpy + PIL** — it
must not import agentic_configuration_search modules.

Encoding: pooler_output, else the CLS token → L2 normalization, in mini-batches.

One run takes **several manifests (jobs)** — the model is loaded once. DataLoader
worker processes decode and preprocess crops ahead of time (overlapping the GPU
forward), and batches come in plan order. Batches are cut every BATCH from 0
within each manifest — they never mix jobs, so a manifest gets the same batches
and the same forward whatever jobs run with it (the output is the same).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Feeding all crops at once OOMs on activations — split into mini-batches.
BATCH = 64
# batches each DataLoader worker prepares ahead (the torch default)
PREFETCH = 2


def plan_batches(sizes: list[int], batch: int) -> list[tuple[int, int, int]]:
    """(job, start, end) list — job order, then order within the job. Never
    crosses a job boundary."""
    return [(j, start, min(start + batch, n))
            for j, n in enumerate(sizes) for start in range(0, n, batch)]


def read_jobs(path: str) -> list[tuple[str, str]]:
    """jobs JSON `[{"manifest": abs, "out": abs}, ...]` → [(manifest, out)].
    Only absolute paths cross the boundary — the worker's cwd may differ from
    the caller's."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    jobs = []
    for job in raw:
        pair = (job["manifest"], job["out"])
        for p in pair:
            if not os.path.isabs(p):
                raise ValueError(f"jobs paths must be absolute: {p!r}")
        jobs.append(pair)
    return jobs


class _CropBatches:
    """map-style dataset: index i = the i-th batch of the plan. Opens crops,
    converts them to RGB and calls the processor per batch. (It does not
    subclass torch Dataset — DataLoader only uses __getitem__/__len__.)"""

    def __init__(self, jobs_paths, processor, batch, return_tensors="pt"):
        self.jobs_paths = jobs_paths
        self.processor = processor
        self.return_tensors = return_tensors
        self.plan = plan_batches([len(p) for p in jobs_paths], batch)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, i):
        from PIL import Image
        j, start, end = self.plan[i]
        images = [Image.open(p).convert("RGB") for p in self.jobs_paths[j][start:end]]
        inputs = self.processor(images=images, return_tensors=self.return_tensors)
        return {"job": j, "end": end, "inputs": dict(inputs.items())}


def embed_jobs(jobs_paths, processor, model_fn, device, *, loaders, batch=BATCH,
               pin_memory=False, on_job_done) -> None:
    """Embed the crops of jobs_paths[j], calling on_job_done(j, (N, D)) as each
    job finishes (an empty job gets (0, 1)). model_fn() is called **after** the
    loader workers start — they fork before CUDA initializes, and the model load
    overlaps preprocessing of the first batches."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    for j, paths in enumerate(jobs_paths):
        if not paths:
            on_job_done(j, np.zeros((0, 1), dtype=np.float32))
    ds = _CropBatches(jobs_paths, processor, batch)
    if not len(ds):
        return
    loaders = max(0, min(loaders, len(ds)))
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=loaders,
                        pin_memory=pin_memory,
                        prefetch_factor=PREFETCH if loaders else None)
    batches = iter(loader)                 # loader workers start here (fork)
    model = model_fn()
    chunks: dict[int, list] = {}
    for item in batches:                   # in order — the plan order
        j = item["job"]
        inputs = {k: v.to(device, non_blocking=pin_memory)
                  for k, v in item["inputs"].items()}
        with torch.inference_mode():
            out = model(**inputs)
        if getattr(out, "pooler_output", None) is not None:
            emb = out.pooler_output
        elif getattr(out, "last_hidden_state", None) is not None:
            emb = out.last_hidden_state[:, 0]          # CLS token
        else:
            raise RuntimeError("DINOv3 returned neither pooler_output nor last_hidden_state")
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)   # L2 normalize
        chunks.setdefault(j, []).append(emb.detach().float().cpu().numpy())
        n = len(jobs_paths[j])
        print(f"embedded {item['end']}/{n} (job {j + 1}/{len(jobs_paths)})", flush=True)
        if item["end"] == n:
            on_job_done(j, np.concatenate(chunks.pop(j), axis=0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True,
                        help='JSON [{"manifest": crop path list JSON, "out": embeddings '
                             '.npy}, ...] (absolute paths)')
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads of this process")
    parser.add_argument("--loaders", type=int, default=0,
                        help="DataLoader workers for crop decoding and preprocessing "
                             "(0 = in the main process)")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(max(1, args.threads))
    jobs = read_jobs(args.jobs)
    jobs_paths = []
    for manifest, _ in jobs:
        with open(manifest, encoding="utf-8") as fh:
            jobs_paths.append(json.load(fh))

    def save(j, arr):
        np.save(jobs[j][1], arr)

    if not any(jobs_paths):
        for j in range(len(jobs)):
            save(j, np.zeros((0, 1), dtype=np.float32))
        return 0

    processor = AutoImageProcessor.from_pretrained(args.model)
    embed_jobs(jobs_paths, processor,
               lambda: AutoModel.from_pretrained(args.model).eval().to(args.device),
               args.device, loaders=args.loaders,
               pin_memory=str(args.device).startswith("cuda"), on_job_done=save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
