"""gateway contract — batching, text-based class resolution, neg kwarg passthrough, alias dedup, tiling.

Pins the gateway contract directly. All inference goes through a stub client.
"""

from types import SimpleNamespace

import pytest
import numpy as np
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


# ------------------------------------------- reuse of identical inference (pass cache)

def test_tiled_full_piece_is_sent_like_the_untiled_pass(tmp_path):
    """The tiled path sends the full image as the file path, like the untiled
    path — the detector must get the same input for the full-image inference
    to be reused. OpenCV (path input) and Pillow (image input) decode all 6,549
    demo images to identical pixels."""
    p = _colored_images(tmp_path, 1)[0]
    sent = []

    class Rec(ContentClient):
        def predict_images(self, images, queries, **kw):
            sent.extend(images)
            return super().predict_images(images, queries, **kw)

    gateway.infer_config(_ctx(), Rec(), TILE_CFG, [], ONTO, [p])
    assert sent[0] == str(p)


def _noise_images(tmp_path, n, size=(100, 80)):
    """Images whose tiles differ in pixels — with a solid color, an image's tiles
    would be the same input."""
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = tmp_path / f"n{i}.png"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_pass_cache_reuses_identical_inference(tmp_path):
    """With the cache on, pieces with the same input and request conditions do not
    call the detector again, and results equal those without the cache. A tiled
    pass after the untiled one reuses the full-image pieces."""
    paths = _noise_images(tmp_path, 3)
    ref_full = gateway.infer_config(_ctx(), ContentClient(), CFG_OFF, [], ONTO, paths)
    ref_tile = gateway.infer_config(_ctx(), ContentClient(), TILE_CFG, [], ONTO, paths)
    client = ContentClient()
    with gateway.pass_cache():
        full = gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
        tile = gateway.infer_config(_ctx(), client, TILE_CFG, [], ONTO, paths)
        again = gateway.infer_config(_ctx(), client, TILE_CFG, [], ONTO, paths)
    assert full == ref_full and tile == ref_tile and again == ref_tile
    assert sum(c["n"] for c in client.calls) == 3 + 3 * 4     # 3 full + 12 tiles, no repeats


def test_pass_cache_keys_on_request_conditions(tmp_path):
    """Different queries or negative settings are different inferences.
    Embedding requests are not cached."""
    paths = _colored_images(tmp_path, 2)
    client = ContentClient()
    neg = {**CFG_OFF, "visual_negatives": True, "neg_eta": 0.5}
    q = [{"text": "cat a"}]
    with gateway.pass_cache():
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
        gateway.infer_config(_ctx(), client, CFG_OFF, q, ONTO, paths)        # other queries
        gateway.infer_config(_ctx(), client, neg, [], ONTO, paths)           # other negatives
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths,
                             return_embeddings=True)
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths,
                             return_embeddings=True)
    assert sum(c["n"] for c in client.calls) == 2 * 5


def test_pass_cache_keys_prompt_images_by_content(tmp_path):
    """Exemplar prompt images count by content, not path — rewriting the file at
    the same path is a different request."""
    paths = _noise_images(tmp_path, 1)
    ex = tmp_path / "ex.png"
    Image.new("RGB", (40, 40), (10, 20, 30)).save(ex)
    q = [{"prompt_image": str(ex), "prompts": [{"bbox": [0, 0, 20, 20], "label": "cat a"}]}]
    client = ContentClient()
    with gateway.pass_cache():
        gateway.infer_config(_ctx(), client, CFG_OFF, q, ONTO, paths)
        gateway.infer_config(_ctx(), client, CFG_OFF, q, ONTO, paths)       # hit
        Image.new("RGB", (40, 40), (200, 20, 30)).save(ex)
        gateway.infer_config(_ctx(), client, CFG_OFF, q, ONTO, paths)       # new content
    assert sum(c["n"] for c in client.calls) == 2


def test_pass_cache_is_cleared_when_detector_state_changes(tmp_path):
    """Once the detector state (LoRA weights, classifier artifact) changes, the
    same input is a different inference."""
    paths = _colored_images(tmp_path, 2)
    client = ContentClient()
    with gateway.pass_cache():
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
        gateway.invalidate_pass_cache()
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
    assert sum(c["n"] for c in client.calls) == 4
    # without the cache on, the detector is always called (previous behavior)
    gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
    assert sum(c["n"] for c in client.calls) == 6


def test_detector_state_hooks_invalidate_the_pass_cache():
    """The runner wraps the detector_hooks controls that change detector state so
    they empty the cache."""
    from agentic_configuration_search.loop.runner import _invalidating_hooks
    calls = []
    hooks = _invalidating_hooks({"weights_ctl": lambda w: calls.append(("w", w)),
                                 "adapt_ctl": lambda a: calls.append(("a", a)),
                                 "repo_root": "r"})
    cleared = []
    orig = gateway.invalidate_pass_cache
    gateway.invalidate_pass_cache = lambda: cleared.append(1)
    try:
        hooks["weights_ctl"]("w.pt")
        hooks["adapt_ctl"](None)
    finally:
        gateway.invalidate_pass_cache = orig
    assert calls == [("w", "w.pt"), ("a", None)] and len(cleared) == 2
    assert hooks["repo_root"] == "r"


def test_pass_cache_eviction_never_drops_a_hit_of_the_same_batch(tmp_path):
    """With the cache full and a hit of the same batch being the oldest entry,
    storing new responses does not evict it — evicting it and then reading
    raised KeyError and killed the run."""
    a, b, c, d, e = _noise_images(tmp_path, 5)
    client = ContentClient()
    with gateway.pass_cache(max_entries=4):
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, [a])       # the oldest hit
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, [b, c, d])  # full
        out = gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, [a, e])
    ref = gateway.infer_config(_ctx(), ContentClient(), CFG_OFF, [], ONTO, [a, e])
    assert out == ref


def test_pass_cache_full_then_tiled_pass(tmp_path):
    """The order that actually breaks — an untiled pass fills the cache, another
    pass fills it to the cap, and a tiled pass under the same conditions sends
    the full-image pieces (oldest hits) together with tiles (misses)."""
    (tmp_path / "f").mkdir()
    (tmp_path / "o").mkdir()
    first = _noise_images(tmp_path / "f", 4)
    other = _noise_images(tmp_path / "o", 4)
    client = ContentClient()
    q = [{"text": "cat a"}]
    with gateway.pass_cache(max_entries=8):
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, first)
        gateway.infer_config(_ctx(), client, CFG_OFF, q, ONTO, other)
        out = gateway.infer_config(_ctx(), client, TILE_CFG, [], ONTO, first)
    assert out == gateway.infer_config(_ctx(), ContentClient(), TILE_CFG, [], ONTO, first)


def test_pass_cache_key_covers_every_request_argument(tmp_path):
    """The key is built from the whole argument bundle sent to the detector —
    building key and arguments separately would leave a new argument out of the
    key. A different argument (topk here) asks again for the same input."""
    paths = _noise_images(tmp_path, 2)
    client = ContentClient()
    with gateway.pass_cache():
        gateway.infer_config(_ctx(), client, CFG_OFF, [], ONTO, paths)
        gateway.infer_config(_ctx(detector={"apply_topk": 7}), client, CFG_OFF, [], ONTO, paths)
    assert sum(c["n"] for c in client.calls) == 4
    assert len({c["kw"]["apply_topk"] for c in client.calls}) == 2


def test_pass_cache_rejects_negative_capacity():
    """A negative cap is rejected up front — left alone, evicting from an empty
    cache raised KeyError."""
    with pytest.raises(ValueError):
        with gateway.pass_cache(max_entries=-1):
            pass
    assert gateway._pass_cache is None                    # a failure leaves no cache on


def test_tiles_share_the_detector_frame_of_exif_oriented_images(tmp_path):
    """The detector opens a path with Pillow and applies its EXIF orientation;
    tiles and merge geometry are cut from the same oriented image, so the full
    piece sent as a path and the tiles agree (and match the untiled pass)."""
    from ultralytics.data.build import autocast_list
    p = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6                                  # rotate 90° on display
    Image.new("RGB", (100, 80), (90, 30, 10)).save(p, exif=exif)
    plans: list = []
    pieces = list(gateway._tile_pieces([p], TILE_CFG["tiling"], plans))
    seen = autocast_list([pieces[0]])[0]              # what the detector gets for the path
    assert plans[0][2] == seen.size == (80, 100)
    assert all(piece.size[0] <= 80 and piece.size[1] <= 100 for piece in pieces[1:])
