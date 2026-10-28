"""scripts/ablate.py — the pure parts of the per-component marginal contribution (leave-one-out).

The GPU path (detector, inference) is not covered. This pins that removing an
axis yields a valid combination, and that replaying the detector state handles
the blend λ and the pass cache together.
"""

import importlib.util
from pathlib import Path

import pytest

from agentic_configuration_search.infer import gateway
from agentic_configuration_search.planner.axes import validate_axes

ROOT = Path(__file__).resolve().parents[1]


def _ablate():
    spec = importlib.util.spec_from_file_location("ablate_script",
                                                  ROOT / "scripts" / "ablate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_blend():
    yield
    gateway.set_classifier_blend(None)


def test_dropping_exemplar_also_drops_its_dependents():
    ab = _ablate()
    axes = validate_axes({"exemplar": "fn_arm", "exemplar_base_k": 2, "exemplar_fn_k": 4,
                          "visual_negatives": "gt_crop", "visual_negative_k": 2})
    out, also = ab._axes_without(axes, "exemplar")
    assert out["exemplar"] == "none"
    assert set(also) == {"exemplar_base_k", "exemplar_fn_k", "visual_negatives",
                         "visual_negative_k"}
    validate_axes(out)                     # what is left is a valid combination


def test_active_reads_the_served_state_not_the_label():
    ab = _ablate()
    champ = {"axes": {"tiling": 2, "crop_classifier": False, "rescorer": True},
             "setting": {"configs": [{"config_id": 0, "tiling": None},
                                     {"config_id": 1, "tiling": 2}],
                         "per_class": {"1": {"config_id": 0}}},
             "adapt_root": "/x/artifact"}
    on = ab._active(champ)
    assert "tiling" not in on              # no class is assigned to a tiled config
    assert "crop_classifier" in on         # the label is off but the artifact is loaded
    assert "rescorer" in on


class _Hooks(dict):
    def __init__(self):
        self.calls = []
        super().__init__(weights_ctl=lambda w: self.calls.append(("w", w)),
                         adapt_ctl=lambda a: self.calls.append(("a", a)))


def test_detector_state_sets_blend_and_clears_the_pass_cache(monkeypatch):
    ab = _ablate()
    cleared = []
    monkeypatch.setattr(gateway, "invalidate_pass_cache", lambda: cleared.append(1))
    hooks = _Hooks()
    ab.apply_detector_state(hooks, "/w", "/art", 0.5)
    assert hooks.calls == [("w", Path("/w")), ("a", Path("/art"))]
    assert gateway._classifier_blend == 0.5 and cleared
    ab.apply_detector_state(hooks, "/w", None, None)    # classifier ablated — blend off too
    assert gateway._classifier_blend is None and len(cleared) == 2


def test_detector_state_refuses_an_artifact_without_lambda():
    ab = _ablate()
    with pytest.raises(SystemExit, match="blend λ"):
        ab.apply_detector_state(_Hooks(), None, "/art", None)
