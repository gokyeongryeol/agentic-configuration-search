"""scripts/predict.py replays the champion's serving state — weights, classifier
artifact and its blend λ (the artifact alone would serve argmax reclassification,
which is not what won adoption)."""

import importlib.util
from pathlib import Path

import pytest

from agentic_configuration_search.infer import gateway

_SPEC = importlib.util.spec_from_file_location(
    "predict_script", Path(__file__).resolve().parents[1] / "scripts" / "predict.py")
predict = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(predict)


@pytest.fixture(autouse=True)
def _blend_off():
    gateway.set_classifier_blend(None)
    yield
    gateway.set_classifier_blend(None)


def _hooks(seen):
    return {"weights_ctl": lambda w: seen.append(("w", w)),
            "adapt_ctl": lambda p: seen.append(("a", p))}


def test_replay_applies_artifact_and_lambda():
    seen = []
    predict.apply_serving_state(_hooks(seen), {"weights": "/w/arm.json",
                                               "adapt_root": "/a", "adapt_lambda": 0.0})
    assert seen == [("w", Path("/w/arm.json")), ("a", Path("/a"))]
    assert gateway.classifier_blend() == 0.0               # relabel-only, not None


def test_replay_without_classifier_turns_the_blend_off():
    seen = []
    gateway.set_classifier_blend(0.5)
    predict.apply_serving_state(_hooks(seen), {"weights": None, "adapt_root": None,
                                               "adapt_lambda": None})
    assert seen == [("w", None), ("a", None)] and gateway.classifier_blend() is None


def test_replay_refuses_an_artifact_without_lambda():
    with pytest.raises(SystemExit, match="adapt_lambda"):
        predict.apply_serving_state(_hooks([]), {"weights": None, "adapt_root": "/a"})
