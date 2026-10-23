"""workers/dinov3 — multi-manifest batch plan, order preservation, path contract.

The batch plan and crop loading (`_CropBatches`) are pure Python, checked
without torch. The DataLoader prefetch path runs only in a venv with torch
(skipped otherwise) — no GPU (device="cpu", a fake model).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agentic_configuration_search.workers import dinov3


def test_plan_batches_matches_per_manifest_chunks():
    """Batches never cross a manifest boundary and are cut every BATCH from 0
    within each manifest — the same batches as the old worker running one
    manifest (same input → same output). Order is job order, then order within
    the job."""
    plan = dinov3.plan_batches([130, 0, 64, 1], 64)
    assert plan == [(0, 0, 64), (0, 64, 128), (0, 128, 130),
                    (2, 0, 64),
                    (3, 0, 1)]
    assert dinov3.BATCH == 64            # a bigger batch grows activation memory


def test_read_jobs_requires_absolute_paths(tmp_path):
    good = tmp_path / "jobs.json"
    good.write_text(json.dumps([{"manifest": str(tmp_path / "m.json"),
                                 "out": str(tmp_path / "o.npy")}]))
    assert dinov3.read_jobs(str(good)) == [(str(tmp_path / "m.json"),
                                            str(tmp_path / "o.npy"))]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"manifest": "m.json", "out": str(tmp_path / "o.npy")}]))
    with pytest.raises(ValueError, match="absolute"):
        dinov3.read_jobs(str(bad))


def _write_crops(tmp_path, n, tag):
    paths = []
    for i in range(n):
        p = tmp_path / f"{tag}{i}.png"
        Image.new("L", (3 + i % 4, 5), color=i).save(p)     # grayscale → converted to RGB
        paths.append(str(p))
    return paths


class _FakeProcessor:
    """Per image [first pixel R, width, is RGB] — exposes order and conversion."""

    def __init__(self):
        self.calls = []

    def __call__(self, images, return_tensors):
        self.calls.append(len(images))
        rows = [[float(im.getpixel((0, 0))[0]), float(im.size[0]),
                 float(im.mode == "RGB")] for im in images]
        if return_tensors == "pt":
            import torch
            return {"pixel_values": torch.tensor(rows)}
        return {"pixel_values": np.array(rows)}


def test_crop_batches_loads_in_plan_order(tmp_path):
    a = _write_crops(tmp_path, 5, "a")
    b = _write_crops(tmp_path, 2, "b")
    proc = _FakeProcessor()
    ds = dinov3._CropBatches([a, b], proc, 2, return_tensors="np")
    assert len(ds) == 4
    items = [ds[i] for i in range(4)]
    assert [it["job"] for it in items] == [0, 0, 0, 1]
    assert [it["end"] for it in items] == [2, 4, 5, 2]
    first = items[0]["inputs"]["pixel_values"]
    assert first[:, 0].tolist() == [0.0, 1.0] and first[:, 2].tolist() == [1.0, 1.0]
    assert proc.calls == [2, 2, 1, 2]    # the processor is called per batch (the same call as before)


@pytest.mark.parametrize("loaders", [0, 3])
def test_embed_jobs_preserves_order_with_prefetch(tmp_path, loaders):
    torch = pytest.importorskip("torch")
    jobs = [_write_crops(tmp_path, 7, "a"), [], _write_crops(tmp_path, 3, "b")]

    class FakeOut:
        def __init__(self, x):
            self.pooler_output = x

    class FakeModel:
        def __call__(self, pixel_values):
            return FakeOut(pixel_values)

    done = {}
    dinov3.embed_jobs(jobs, _FakeProcessor(), lambda: FakeModel(), "cpu",
                      loaders=loaders, batch=2, pin_memory=False,
                      on_job_done=lambda j, arr: done.__setitem__(j, arr))
    assert sorted(done) == [0, 1, 2]
    assert done[1].shape == (0, 1)
    for j, n in [(0, 7), (2, 3)]:
        raw = np.array([[float(i), float(3 + i % 4), 1.0] for i in range(n)])
        want = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        np.testing.assert_allclose(done[j], want, rtol=1e-6)
    del torch
