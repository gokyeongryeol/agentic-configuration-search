"""Client-side crop reclassification in clients.yoloe (CPU, stubbed).

Order contract: NMS -> negative suppression -> **reclassification** -> response (rescorer and τ are downstream).
Pins category replacement, background (0) discard, and artifact-load validation.
"""

import json

import pytest
from PIL import Image

from agentic_configuration_search.clients.yoloe import YoloeClient, _reclassify, read_arm  # noqa: F401
from agentic_configuration_search.clients.wire import parse_prediction
from agentic_configuration_search.core.prompts import build_text_queries

NAME_TO_ID = {"cat a": 1, "cat b": 2}


def queries():
    return build_text_queries(["cat a", "cat b"], {}, box_threshold=0.01,
                              multimodal_threshold=0.01)[0]


class StubReclassifier:
    """Stub that assigns labels by box x1 — {x1: label}."""

    class_names = ["cat a", "cat b"]

    def __init__(self, rule):
        self.rule = rule
        self.batches = []

    def classify(self, crops):
        self.batches.append(len(crops))
        return self.rule[: len(crops)]


class Backend:
    def __init__(self, dets):
        self.dets = dets

    def set_vocab(self, channels):
        self.channels = channels

    def infer(self, images, *, conf, iou, max_det, return_embeddings=False):
        name_to_idx = {c["name"]: i for i, c in enumerate(self.channels)}
        rows = [(box, s, name_to_idx[n]) for box, s, n in self.dets]
        if return_embeddings:
            rows = [(*r, [1.0, 2.0]) for r in rows]
        return [rows for _ in images]


def client_with(dets, labels):
    stub = StubReclassifier(labels)
    c = YoloeClient(backend=Backend(dets),
                    reclassifier_factory=lambda root, device: stub)
    c.set_adapt_root("artifact_dir")
    return c, stub


def test_reclassify_remaps_and_rejects(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    dets = [([0, 0, 20, 20], 0.9, "cat a"),      # -> label 2 = replaced with cat b
            ([40, 0, 60, 20], 0.8, "cat a"),     # -> label 0 = background, discarded
            ([0, 40, 20, 60], 0.7, "cat b")]     # -> label 2 = kept (cat b)
    client, stub = client_with(dets, [2, 0, 2])
    preds = client.predict_images([str(img)], queries(),
                                  return_embeddings=True)
    got = parse_prediction(preds[0], NAME_TO_ID)
    assert [(d["category_id"], d["score"]) for d in got] == [(2, 0.9), (2, 0.7)]
    assert preds[0]["text"] == ["cat b", "cat b"]
    # score and embedding are kept (reclassification only corrects labels)
    assert all(d["embedding"] == [1.0, 2.0] for d in got)
    assert stub.batches == [3]                   # one batch per image


def test_reclassify_unmount_restores_raw_labels(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    dets = [([0, 0, 20, 20], 0.9, "cat a")]
    client, stub = client_with(dets, [2])
    client.set_adapt_root(None)                  # cleared — original labels as-is
    preds = client.predict_images([str(img)], queries())
    assert preds[0]["text"] == ["cat a"]
    assert stub.batches == []


def test_reclassify_keeps_degenerate_boxes(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    dets = [([0, 0, 1, 1], 0.9, "cat a"),        # below promptable — original label kept
            ([0, 40, 30, 70], 0.8, "cat a")]
    client, _ = client_with(dets, [0])           # the only crop -> background
    preds = client.predict_images([str(img)], queries())
    assert preds[0]["text"] == ["cat a"]         # only the degenerate box remains
    assert preds[0]["boxes"] == [[0.0, 0.0, 1.0, 1.0]]


def test_reclassify_keeps_thin_boundary_crops(tmp_path):
    """Even a promptable box can become a thin sliver crop after clamping to the image bounds —
    if one side shrinks to the channel count (3), the SigLIP slow processor's channel-axis
    inference flips and the classifier dies (observed: a (1,1,23) uint8 crop -> PIL fromarray
    TypeError, killing the whole run). Thin crops skip the classifier and keep their original label."""
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    dets = [([98, 10, 104, 63], 0.9, "cat a"),   # out of bounds — crop is 2×53
            ([0, 40, 30, 70], 0.8, "cat a")]     # normal
    client, stub = client_with(dets, [2])        # only the one normal crop is classified
    preds = client.predict_images([str(img)], queries())
    assert stub.batches == [1]                   # thin crop is not in the batch
    assert sorted(preds[0]["text"]) == ["cat a", "cat b"]


def test_reclassify_runs_after_negative_suppression(tmp_path):
    """Detections removed by suppression are not reclassified — order contract."""
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    q = build_text_queries(["cat a", "cat b"], {}, box_threshold=0.01,
                           multimodal_threshold=0.01,
                           visual_negatives={"cat a": ["shadow"]})[0]
    dets = [([0, 0, 20, 20], 0.9, "cat a"),      # IoU 1 with the negative — suppressed
            ([0, 0, 20, 20], 0.95, "shadow"),
            ([40, 40, 60, 60], 0.8, "cat b")]    # survives — 1 reclassification
    stub = StubReclassifier([1])
    client = YoloeClient(backend=Backend(dets),
                         reclassifier_factory=lambda root, device: stub)
    client.set_adapt_root("artifact")
    preds = client.predict_images([str(img)], q, neg_eta=0.5)
    assert stub.batches == [1]                   # only 1 classified after suppression
    assert preds[0]["text"] == ["cat a"]         # label 1 -> replaced with cat a


def test_pil_source_reclassified_without_reopen():
    im = Image.new("RGB", (100, 80))
    dets = [([0, 0, 20, 20], 0.9, "cat a")]
    client, stub = client_with(dets, [1])
    preds = client.predict_images([im], queries())   # tiling path = PIL input
    assert preds[0]["text"] == ["cat a"] and stub.batches == [1]


def test_reclassifier_label_out_of_range_is_error(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    client, _ = client_with([([0, 0, 20, 20], 0.9, "cat a")], [7])
    with pytest.raises(RuntimeError, match="out of range"):
        client.predict_images([str(img)], queries())


def test_artifact_config_mismatch_is_error(tmp_path):
    """Format validation in the real loader — num_classes / class_names mismatch is an explicit error.

    (Only the validation path before model load — the real SigLIP load is covered by the GPU smoke test.)"""
    from agentic_configuration_search.clients.yoloe import _CropReclassifier
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "classifier_config.json").write_text(json.dumps({
        "classifier_model": "siglip",
        "classifier_model_name": "google/siglip2-base-patch16-224",
        "classifier_lora_r": 16, "classifier_lora_alpha": 32,
        "classifier_lora_target": "qkvo", "classifier_image_size": 224,
        "class_names": ["cat a", "cat b"]}))
    import torch
    torch.save({"state_dict": {}, "num_classes": 5,
                "classifier_model": "siglip"}, root / "classifier.pt")
    with pytest.raises(ValueError, match="num_classes"):
        _CropReclassifier(root, "cpu")
