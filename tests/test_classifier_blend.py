"""crop classifier probability blend — the client returns per-box class
probabilities and the gateway blends them with λ.

class = foreground argmax c*, score = s^(1-λ)·p(c*)^λ. λ=0 is relabel-only
(scores unchanged); a box whose argmax is background is not dropped — its score
goes down and the per-class thresholds decide. None (no classifier) and 0.0
(relabel-only) are different states.
"""

import pytest
from PIL import Image

from agentic_configuration_search.clients.wire import parse_prediction
from agentic_configuration_search.clients.yoloe import YoloeClient
from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.infer import gateway
from tests.test_reclassify import NAME_TO_ID, Backend, queries


@pytest.fixture(autouse=True)
def _blend_off():
    gateway.set_classifier_blend(None)
    yield
    gateway.set_classifier_blend(None)


def _pred():
    # three boxes: confused (cat a → cat b), background-ish, uncroppable (None)
    return {"boxes": [[0, 0, 20, 20], [40, 0, 60, 20], [0, 40, 20, 60]],
            "scores": [0.9, 0.8, 0.5], "text": ["cat a", "cat a", "cat b"],
            "embeddings": [[1.0], [2.0], [3.0]],
            "classifier_probs": [[0.1, 0.2, 0.7], [0.9, 0.06, 0.04], None],
            "classifier_class_names": ["background", "cat a", "cat b"]}


def test_blend_relabels_and_mixes_scores():
    out = gateway.blend_classifier(_pred(), 0.5)
    assert out["text"] == ["cat b", "cat a", "cat b"]          # foreground argmax
    assert out["scores"][0] == pytest.approx(0.9 ** 0.5 * 0.7 ** 0.5)
    # background argmax is not dropped — the score goes down instead
    assert out["scores"][1] == pytest.approx(0.8 ** 0.5 * 0.06 ** 0.5)
    assert out["scores"][2] == 0.5                              # no probabilities — unchanged
    assert out["embeddings"] == [[1.0], [2.0], [3.0]] and len(out["boxes"]) == 3
    assert "classifier_probs" not in out


def test_lambda_zero_is_relabel_only_and_one_is_the_probability():
    zero = gateway.blend_classifier(_pred(), 0.0)
    assert zero["scores"] == [0.9, 0.8, 0.5] and zero["text"][0] == "cat b"
    one = gateway.blend_classifier(_pred(), 1.0)
    assert one["scores"][:2] == [pytest.approx(0.7), pytest.approx(0.06)]


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.5", True])
def test_lambda_must_be_in_unit_interval(bad):
    with pytest.raises(ValueError):
        gateway.set_classifier_blend(bad)
    with pytest.raises(ValueError):
        gateway.blend_classifier(_pred(), bad)


def test_none_and_zero_are_different_states():
    gateway.set_classifier_blend(0.0)
    assert gateway.classifier_blend() == 0.0
    assert gateway._probs_kwargs() == {"return_classifier_probs": True}
    gateway.set_classifier_blend(None)
    assert gateway.classifier_blend() is None and gateway._probs_kwargs() == {}


def test_contract_violations_stop():
    p = _pred()
    del p["classifier_probs"]
    with pytest.raises(RuntimeError, match="classifier_probs"):
        gateway.blend_classifier(p, 0.5)
    p = _pred()
    p["classifier_probs"] = p["classifier_probs"][:2]
    with pytest.raises(RuntimeError, match="contract"):
        gateway.blend_classifier(p, 0.5)
    p = _pred()
    p["classifier_class_names"] = ["background"]
    with pytest.raises(RuntimeError, match="contract"):
        gateway.blend_classifier(p, 0.5)


# ------------------------------------------------------------------ client

class ProbReclassifier:
    class_names = ["cat a", "cat b"]

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def classify(self, crops):
        raise AssertionError("argmax path used while probabilities were requested")

    def probs(self, crops):
        self.calls += 1
        return self.rows[:len(crops)]


def test_client_returns_probabilities_instead_of_reclassifying(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    dets = [([0, 0, 20, 20], 0.9, "cat a"), ([40, 0, 60, 20], 0.8, "cat a")]
    stub = ProbReclassifier([[0.1, 0.2, 0.7], [0.9, 0.06, 0.04]])
    c = YoloeClient(backend=Backend(dets), reclassifier_factory=lambda r, d: stub)
    c.set_adapt_root("artifact_dir")
    item = c.predict_images([str(img)], queries(), return_classifier_probs=True)[0]
    assert item["text"] == ["cat a", "cat a"]                  # not reclassified, nothing dropped
    assert item["classifier_probs"] == [[0.1, 0.2, 0.7], [0.9, 0.06, 0.04]]
    assert item["classifier_class_names"] == ["background", "cat a", "cat b"]
    got = parse_prediction(gateway.blend_classifier(item, 0.0), NAME_TO_ID)
    assert [(d["category_id"], d["score"]) for d in got] == [(2, 0.9), (1, 0.8)]


def test_client_without_classifier_ignores_the_flag(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    c = YoloeClient(backend=Backend([([0, 0, 20, 20], 0.9, "cat a")]))
    item = c.predict_images([str(img)], queries(), return_classifier_probs=True)[0]
    assert "classifier_probs" not in item


def test_uncroppable_boxes_get_no_probabilities(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    dets = [([0, 0, 20, 20], 0.9, "cat a"), ([50, 50, 51, 51], 0.8, "cat b")]
    stub = ProbReclassifier([[0.1, 0.2, 0.7]])
    c = YoloeClient(backend=Backend(dets), reclassifier_factory=lambda r, d: stub)
    c.set_adapt_root("artifact_dir")
    item = c.predict_images([str(img)], queries(), return_classifier_probs=True)[0]
    assert item["classifier_probs"] == [[0.1, 0.2, 0.7], None]


# ------------------------------------------------------- gateway + pass cache

class CountingClient:
    """Raw responses with probabilities — counts detector calls."""

    def __init__(self):
        self.calls = []

    def predict_images(self, images, queries, **kw):
        self.calls.append(kw)
        out = []
        for _ in images:
            p = {"boxes": [[0, 0, 20, 20]], "scores": [0.9], "text": ["cat a"]}
            if kw.get("return_classifier_probs"):
                p |= {"classifier_probs": [[0.1, 0.2, 0.7]],
                      "classifier_class_names": ["background", "cat a", "cat b"]}
            out.append(p)
        return out


class _Ctx:
    cfg = config_from_dict({"dataset": "x"})


class _Onto:
    name_to_id = NAME_TO_ID


def test_lambda_only_changes_reuse_the_pass_cache(tmp_path):
    img = tmp_path / "im.png"
    Image.new("RGB", (100, 80)).save(img)
    client = CountingClient()
    cfg = {"visual_negatives": False, "tiling": None}
    q = queries()
    with gateway.pass_cache():
        got = {}
        for lam in (0.0, 0.5, 1.0):
            gateway.set_classifier_blend(lam)
            got[lam] = gateway.infer_config(_Ctx(), client, cfg, q, _Onto(), [img])[0][0]
        assert len(client.calls) == 1, "a λ-only change re-ran the detector"
        assert client.calls[0]["return_classifier_probs"] is True
        assert [got[lam]["category_id"] for lam in got] == [2, 2, 2]
        assert got[0.0]["score"] == pytest.approx(0.9)
        assert got[1.0]["score"] == pytest.approx(0.7)
        # blend off asks without probabilities — a different request, a different key
        gateway.set_classifier_blend(None)
        plain = gateway.infer_config(_Ctx(), client, cfg, q, _Onto(), [img])[0][0]
        assert len(client.calls) == 2 and "return_classifier_probs" not in client.calls[1]
        assert plain["category_id"] == 1 and plain["score"] == pytest.approx(0.9)
