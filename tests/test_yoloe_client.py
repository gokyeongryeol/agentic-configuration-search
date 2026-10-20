"""clients/yoloe — predict_images contract logic (fake backend, CPU).

Without GPU/ultralytics, pins query parsing, negative emulation (neg_eta reinterpreted as IoU),
and empty embeddings (emb_dim=0).
"""

import pytest

from agentic_configuration_search.clients.yoloe import DEFAULT_SUP_IOU, YoloeClient, parse_queries
from agentic_configuration_search.clients.wire import parse_prediction
from agentic_configuration_search.core.prompts import PromptBox, build_queries, build_text_queries


def text_bundle(negatives=None, aliases=None):
    return build_text_queries(["cat a", "cat b"], dict(aliases or {}),
                              box_threshold=0.001, multimodal_threshold=0.001,
                              visual_negatives=negatives)


def semantic_bundle(negatives=None):
    ex = {"cat a": [PromptBox("cat a", "img1.png", [0, 0, 10, 10]),
                    PromptBox("cat a", "img2.png", [5, 5, 15, 15])],
          "cat b": [PromptBox("cat b", "img1.png", [20, 20, 30, 30])]}
    return build_queries(ex, negatives, box_threshold=0.001,
                         multimodal_threshold=0.001)


# ------------------------------------------------------------ parse_queries

def test_parse_text_queries_with_alias_and_negatives():
    queries, _order, _alias = text_bundle(
        negatives={"cat a": ["shadow", "shadow"], "cat b": ["shadow"]},
        aliases={"cat a": "feline"})
    channels, floor = parse_queries(queries)
    assert floor == 0.001
    pos = [c["name"] for c in channels if c["kind"] == "pos"]
    neg = [c["name"] for c in channels if c["kind"] == "neg"]
    assert pos == ["cat a", "feline", "cat b"]   # aliases are a parallel channel
    assert neg == ["shadow"]                     # target duplication and repeats fold into phrases


def test_parse_semantic_queries_groups_crops_per_class():
    queries, _order, _exc = semantic_bundle(
        negatives={"cat a": [PromptBox("cat a", "img3.png", [1, 1, 5, 5])]})
    channels, _floor = parse_queries(queries)
    by = {c["name"]: c for c in channels}
    assert len(by["cat a"]["crops"]) == 2 and len(by["cat b"]["crops"]) == 1
    assert by["~cat a"]["kind"] == "neg" and len(by["~cat a"]["crops"]) == 1


def test_parse_rejects_mixed_modalities():
    queries = [{"prompts": [
        {"text": "a", "box": [], "box_threshold": 0.1, "multimodal_threshold": 0.1},
        {"text": "b", "box": [1, 1, 2, 2], "box_threshold": 0.1,
         "multimodal_threshold": 0.1}]}]
    with pytest.raises(ValueError, match="mixed text/semantic queries"):
        parse_queries(queries)


# --------------------------------------------------------------- contract behavior

class FakeBackend:
    """Canned detections by channel index — the set_vocab order is the channel space."""

    def __init__(self, dets_by_image):
        self.dets_by_image = dets_by_image      # image index -> [(box, s, name)]
        self.channels = None
        self.set_vocab_calls = 0

    def set_vocab(self, channels):
        self.channels = channels
        self.set_vocab_calls += 1

    def infer(self, images, *, conf, iou, max_det, return_embeddings=False):
        name_to_idx = {c["name"]: i for i, c in enumerate(self.channels)}
        out = []
        for i, _img in enumerate(images):
            dets = []
            for box, s, name in self.dets_by_image[i]:
                if s < conf:
                    continue
                row = (box, s, name_to_idx[name])
                if return_embeddings and getattr(self, "emb_dim", 0):
                    # deterministic vector derived from box coordinates — for alignment checks
                    row = (*row, [float(box[0])] * self.emb_dim)
                dets.append(row)
            out.append(dets)
        return out


NAME_TO_ID = {"cat a": 1, "cat b": 2}


def test_positive_only_pass_returns_zero_wire_format():
    queries, _o, alias = text_bundle()
    client = YoloeClient(backend=FakeBackend([
        [([0, 0, 10, 10], 0.9, "cat a"), ([1, 0, 11, 10], 0.4, "cat a"),
         ([50, 50, 60, 60], 0.8, "cat b")]]))
    preds = client.predict_images(["im0.png"], queries, iou_threshold=0.5)
    dets = parse_prediction(preds[0], NAME_TO_ID, alias_to_name=alias)
    # same-class duplicates fold via batched NMS (end2end models have no NMS)
    assert [(d["category_id"], d["score"]) for d in dets] == [(1, 0.9), (2, 0.8)]


def test_negative_emulation_sweeps_neg_eta_as_iou():
    queries, _o, alias = text_bundle(negatives={"cat a": ["shadow"]})
    dets = [[([0, 0, 10, 10], 0.9, "cat a"),          # IoU 1/3 with the negative
             ([0, 5, 10, 15], 0.7, "shadow"),
             ([50, 50, 60, 60], 0.8, "cat b")]]
    client = YoloeClient(backend=FakeBackend(dets))
    # suppression threshold 0.25 ≤ 1/3 — cat a removed, the negative itself also dropped from the response
    low = client.predict_images(["im"], queries, neg_eta=0.25, iou_threshold=0.5)
    got = parse_prediction(low[0], NAME_TO_ID, alias_to_name=alias)
    assert [d["category_id"] for d in got] == [2]
    # suppression threshold 0.75 > 1/3 — cat a survives
    high = client.predict_images(["im"], queries, neg_eta=0.75, iou_threshold=0.5)
    got = parse_prediction(high[0], NAME_TO_ID, alias_to_name=alias)
    assert sorted(d["category_id"] for d in got) == [1, 2]
    assert "shadow" not in high[0]["text"]


def test_negative_default_iou_when_eta_missing():
    """Default text-pass call (no neg kwargs) — suppresses with DEFAULT_SUP_IOU."""
    assert DEFAULT_SUP_IOU == 0.5
    queries, _o, alias = text_bundle(negatives={"cat a": ["shadow"]})
    dets = [[([0, 0, 10, 10], 0.9, "cat a"),
             ([0, 2, 10, 12], 0.7, "shadow")]]       # IoU = 8/12 ≈ 0.67 ≥ 0.5
    client = YoloeClient(backend=FakeBackend(dets))
    preds = client.predict_images(["im"], queries)
    assert parse_prediction(preds[0], NAME_TO_ID, alias_to_name=alias) == []


def test_no_negative_channels_means_no_suppression():
    """Query without negatives + neg_eta passed — nothing to suppress, unchanged."""
    queries, _o, alias = text_bundle()
    client = YoloeClient(backend=FakeBackend(
        [[([0, 0, 10, 10], 0.9, "cat a")]]))
    preds = client.predict_images(["im"], queries, neg_eta=0.25)
    got = parse_prediction(preds[0], NAME_TO_ID, alias_to_name=alias)
    assert [d["category_id"] for d in got] == [1]


def test_return_embeddings_falls_back_to_empty_vectors():
    """A backend that returns no embeddings (older test double) — degrades to empty vectors (emb_dim=0)."""
    queries, _o, alias = text_bundle()
    client = YoloeClient(backend=FakeBackend(
        [[([0, 0, 10, 10], 0.9, "cat a")]]))
    preds = client.predict_images(["im"], queries, return_embeddings=True)
    assert preds[0]["embeddings"] == [[]]
    det = parse_prediction(preds[0], NAME_TO_ID, alias_to_name=alias)[0]
    assert det["embedding"] == []


def test_return_embeddings_consistent_dim_and_alignment():
    """Per-detection embeddings: constant dimension, and alignment with their own detection is
    kept through suppression/NMS (verified via a fingerprint derived from box coordinates)."""
    queries, _o, alias = text_bundle(negatives={"cat a": ["shadow"]})
    dets = [[([0, 0, 10, 10], 0.9, "cat a"),        # IoU 1/3 with the negative — survives
             ([50, 50, 60, 60], 0.8, "cat b"),
             ([50, 51, 60, 61], 0.3, "cat b"),      # folded by NMS
             ([0, 5, 10, 15], 0.7, "shadow")]]
    backend = FakeBackend(dets)
    backend.emb_dim = 8
    client = YoloeClient(backend=backend)
    preds = client.predict_images(["im"], queries, neg_eta=0.75,
                                  return_embeddings=True)
    got = parse_prediction(preds[0], NAME_TO_ID, alias_to_name=alias)
    assert [d["category_id"] for d in got] == [1, 2]
    dims = {len(d["embedding"]) for d in got}
    assert dims == {8}                              # constant dimension
    # fingerprint: emb == [x1]*8 — each detection keeps its own embedding
    for d in got:
        assert d["embedding"] == [d["box"][0]] * 8


def test_vocab_cached_across_batches():
    queries, _o, _a = text_bundle()
    backend = FakeBackend([[([0, 0, 10, 10], 0.9, "cat a")]] * 3)
    client = YoloeClient(backend=backend)
    client.predict_images(["a"], queries)
    client.predict_images(["b"], queries)        # same queries — vocab reused
    assert backend.set_vocab_calls == 1


def test_cuda_visibility_restored_after_backend_calls():
    """ultralytics select_device's global mutation of CUDA_VISIBLE_DEVICES does not leak into the
    worker subprocess — it is restored at the call boundary."""
    import os

    from agentic_configuration_search.clients.yoloe import _preserve_cuda_visibility

    @_preserve_cuda_visibility()
    def call():
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"    # mimic select_device

    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    call()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


