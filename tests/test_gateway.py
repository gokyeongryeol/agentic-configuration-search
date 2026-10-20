"""gateway contract — batching, text-based class resolution, neg kwarg passthrough, alias dedup, tiling.

Pins the gateway contract directly. All inference goes through a stub client.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.coco import Ontology
from agentic_configuration_search.infer import gateway
from tests.conftest import CATS

ONTO = Ontology.from_categories(CATS)
CFG_OFF = {"visual_negatives": False, "neg_eta": None, "tiling": None}


def _ctx(**overrides):
    return SimpleNamespace(cfg=config_from_dict({"dataset": "x", **overrides}))


def _pred(names_scores):
    return {"boxes": [[10.0, 10.0, 30.0, 30.0]] * len(names_scores),
            "scores": [s for _, s in names_scores],
            "text": [n for n, _ in names_scores]}


class StubClient:
    """Fixed response per image; records every call."""

    def __init__(self, per_image=None):
        self.calls: list[dict] = []
        self.per_image = per_image or _pred([("cat a", 0.9)])

    def predict_images(self, images, queries, **kw):
        self.calls.append({"n": len(images), "kw": kw})
        return [self.per_image for _ in images]


def test_batching_respects_batch_size(tmp_path):
    paths = []
    for i in range(gateway.BATCH + 2):
        p = tmp_path / f"i{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)
    client = StubClient()
    out = gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
    assert [c["n"] for c in client.calls] == [gateway.BATCH, 2]
    assert len(out) == gateway.BATCH + 2
    d = out[0][0]
    assert d["category_id"] == 1            # text "cat a" → id 1
    assert d["sources"] == ["full"]


def test_negative_kwargs_from_config(tmp_path):
    p = tmp_path / "i.png"
    Image.new("RGB", (8, 8)).save(p)
    client = StubClient()
    cfg_on = {"visual_negatives": True, "neg_eta": 0.5, "tiling": None}
    gateway.infer_config(_ctx(), client, cfg_on, [], ONTO, [p])
    kw = client.calls[0]["kw"]
    assert kw["neg_eta"] == 0.5
    # with negatives off the neg kwarg is absent entirely (positive-only path preserved)
    client2 = StubClient()
    gateway.infer_config(_ctx(), client2, CFG_OFF, [], ONTO, [p])
    assert "neg_eta" not in client2.calls[0]["kw"]


def test_alias_multiplicity_triggers_dedup(tmp_path):
    p = tmp_path / "i.png"
    Image.new("RGB", (8, 8)).save(p)
    # same object detected twice via name and alias phrase — near-duplicates must fold after merging
    per_image = _pred([("cat a", 0.9), ("feline", 0.8)])
    alias = {"cat a": "cat a", "feline": "cat a"}
    client = StubClient(per_image)
    out = gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, [p],
                               alias_to_name=alias)
    assert len(out[0]) == 1 and out[0][0]["score"] == 0.9


def test_no_multiplicity_no_dedup(tmp_path):
    p = tmp_path / "i.png"
    Image.new("RGB", (8, 8)).save(p)
    # with a 1:1 alias map (no multiplicity) same-class near-duplicates are not folded —
    # the detector's own NMS output is respected
    per_image = _pred([("cat a", 0.9), ("cat a", 0.8)])
    client = StubClient(per_image)
    out = gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, [p],
                               alias_to_name={"cat a": "cat a"})
    assert len(out[0]) == 2


def test_unknown_text_dropped(tmp_path):
    p = tmp_path / "i.png"
    Image.new("RGB", (8, 8)).save(p)
    client = StubClient(_pred([("cat a", 0.9), ("ghost", 0.8)]))
    out = gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, [p])
    assert [d["category_id"] for d in out[0]] == [1]


def test_tiling_grid_single_batched_call(tmp_path):
    """grid g → full + g² tiles sent in one batched call."""
    p = tmp_path / "i.png"
    Image.new("RGB", (100, 80)).save(p)
    cfg = {"visual_negatives": False, "neg_eta": None,
           "tiling": {"grid": 2, "overlap": 0.5, "merge": "nmm"}}
    client = StubClient()
    out = gateway.infer_config(_ctx(), client, cfg, [], ONTO, [p])
    assert [c["n"] for c in client.calls] == [1 + 4]
    assert out[0] and all(d["category_id"] == 1 for d in out[0])


TILE_CFG = {"visual_negatives": False, "neg_eta": None,
            "tiling": {"grid": 2, "overlap": 0.25, "merge": "nmm"}}


def _content_pred(im):
    """A response that depends on the pixels — mixing pieces across images changes it."""
    im = (im if isinstance(im, Image.Image) else Image.open(im)).convert("RGB")
    r = im.getpixel((0, 0))[0]
    w, h = im.size
    return _pred([("cat a", round(0.3 + r / 512, 4))]) | {
        "boxes": [[2.0, 2.0, min(w, 20.0 + r % 7), min(h, 18.0)]]}


class ContentClient(StubClient):
    def predict_images(self, images, queries, **kw):
        self.calls.append({"n": len(images), "kw": kw})
        return [_content_pred(im) for im in images]


def _colored_images(tmp_path, n, size=(100, 80)):
    paths = []
    for i in range(n):
        p = tmp_path / f"c{i}.png"
        Image.new("RGB", size, (40 * i + 10, 0, 0)).save(p)
        paths.append(p)
    return paths


def _reference_tiled(paths):
    """Per-image run_tiled — the unbatched behavior (reference)."""
    from agentic_configuration_search.clients.wire import parse_prediction
    from agentic_configuration_search.core import tiling
    out = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")

            def predict_fn(images):
                return [[{"box": d["box"], "score": d["score"], "label": d["category_id"]}
                         for d in parse_prediction(_content_pred(x), ONTO.name_to_id)]
                        for x in images]

            merged = tiling.run_tiled(predict_fn, im, TILE_CFG["tiling"])
        out.append([{"box": d["box"], "score": d["score"], "category_id": d["label"],
                     "sources": d["sources"]} for d in merged])
    return out


def test_tiled_pieces_fill_batches_across_images(tmp_path):
    """Pieces (full + tiles) are grouped BATCH at a time across image boundaries —
    results equal per-image run_tiled. The detector runs each image of a call on
    its own, so grouping changes only the number of calls."""
    paths = _colored_images(tmp_path, 3)          # grid 2 → 5 pieces per image (full + 4 tiles)
    client = ContentClient()
    out = gateway.infer_config(_ctx(), client, TILE_CFG, [], ONTO, paths)
    assert [c["n"] for c in client.calls] == [gateway.BATCH, 15 - gateway.BATCH]
    assert out == _reference_tiled(paths)


def test_image_prep_overlaps_with_inference(tmp_path, monkeypatch):
    """The next chunk is prepared (decode, tile crops) while the current one runs
    on the detector. Calls themselves stay serial and in order."""
    import time
    paths = _colored_images(tmp_path, 3, size=(8, 8))
    real_pieces = gateway._tile_pieces

    def slow_pieces(*a, **kw):
        for im in real_pieces(*a, **kw):
            time.sleep(0.05)
            yield im

    monkeypatch.setattr(gateway, "_tile_pieces", slow_pieces)

    class SlowClient(StubClient):
        def predict_images(self, images, queries, **kw):
            time.sleep(0.5)
            return super().predict_images(images, queries, **kw)

    cfg = {**TILE_CFG, "tiling": {"grid": 2, "overlap": 0.0, "merge": "nmm"}}
    client = SlowClient()
    t0 = time.perf_counter()
    out = gateway.infer_config(_ctx(), client, cfg, [], ONTO, paths)
    elapsed = time.perf_counter() - t0
    assert len(out) == 3 and [c["n"] for c in client.calls] == [gateway.BATCH, 7]
    serial = 2 * 0.5 + 15 * 0.05                  # no overlap = 1.75s, overlapped ≈ 1.4s
    assert elapsed < serial - 0.2, f"{elapsed:.2f}s — prep does not overlap inference"


def test_dedup_is_deterministic():
    dets = [{"box": [10, 10, 30, 30], "score": 0.8, "category_id": 1},
            {"box": [11, 11, 31, 31], "score": 0.9, "category_id": 1},
            {"box": [60, 60, 80, 80], "score": 0.7, "category_id": 1}]
    out1 = gateway.dedup_class_duplicates(list(dets), 0.5)
    out2 = gateway.dedup_class_duplicates(list(reversed(dets)), 0.5)
    assert out1 == out2                       # independent of input order
    assert [d["score"] for d in out1] == [0.9, 0.7]


def test_seam_is_module_qualified(tmp_path, monkeypatch):
    """Premise of the consumer convention — one monkeypatch point blocks all inference."""
    called = []
    monkeypatch.setattr(gateway, "infer_config", lambda *a, **k: called.append(1) or [[]])
    gateway.infer_config(None, None, CFG_OFF, [], ONTO, [])
    assert called == [1]
