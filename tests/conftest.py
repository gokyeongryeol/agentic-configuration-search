"""Shared test fixtures: a synthetic Roboflow dataset + a fake JSON HTTP server.

Tests must be hermetic — never assume real data exists.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from PIL import Image

# sklearn (OpenMP) spawns one thread per core by default. Models in this suite have
# a few hundred rows, so thread creation/sync dwarfs the compute. pytest imports
# conftest before test modules, so pinning here takes effect before OMP initializes.
for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")


# Includes a Roboflow-style dummy supercategory (mirrors real data layout)
CATS = [
    {"id": 0, "name": "stuff", "supercategory": "none"},
    {"id": 1, "name": "cat a", "supercategory": "stuff"},
    {"id": 2, "name": "cat b", "supercategory": "stuff"},
]


def make_split(root: Path, images: list[dict]) -> None:
    """images: [{"name", "w", "h", "anns": [(category_id, x, y, w, h), ...]}]"""
    root.mkdir(parents=True, exist_ok=True)
    coco = {"info": {}, "licenses": [], "categories": CATS, "images": [], "annotations": []}
    ann_id = 1
    for i, spec in enumerate(images, start=1):
        Image.new("RGB", (spec["w"], spec["h"]), (120, 120, 120)).save(root / spec["name"])
        coco["images"].append({"id": i, "file_name": spec["name"],
                               "width": spec["w"], "height": spec["h"]})
        for cat, x, y, w, h in spec.get("anns", []):
            coco["annotations"].append({"id": ann_id, "image_id": i, "category_id": cat,
                                        "bbox": [x, y, w, h], "area": w * h,
                                        "segmentation": [], "iscrowd": 0})
            ann_id += 1
    (root / "_annotations.coco.json").write_text(json.dumps(coco))


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Synthetic train/valid/test 3-split dataset (Roboflow format).

    train has 10 images mixing two classes so the 80/20 stratified split is
    non-trivial per class. valid is the loop's holdout; **test exists but no loop
    path may read it** (report-only, measured once on the final champion).
    """
    ds = tmp_path / "ds"
    train = []
    for i in range(1, 11):
        cat = 1 if i % 2 else 2
        train.append({"name": f"tr{i}.png", "w": 100, "h": 80,
                      "anns": [(cat, 10, 10, 20, 20)]})
    make_split(ds / "train", train)
    make_split(ds / "valid", [
        {"name": "v1.png", "w": 100, "h": 80, "anns": [(1, 10, 10, 20, 20)]},
        {"name": "v2.png", "w": 100, "h": 80, "anns": [(2, 40, 20, 20, 20)]},
    ])
    make_split(ds / "test", [
        {"name": "t1.png", "w": 100, "h": 80, "anns": [(1, 10, 10, 20, 20)]},
    ])
    return ds


@pytest.fixture
def json_server():
    """Fake JSON HTTP server. Takes respond(path, payload) -> (status, body_dict)
    and returns (base_url, records); received requests accumulate in records."""
    servers = []

    def _start(respond):
        records: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence test output
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                records.append({"path": self.path, "payload": payload})
                status, body = respond(self.path, payload)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}", records

    yield _start
    for srv in servers:
        srv.shutdown()
