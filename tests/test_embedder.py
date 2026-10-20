"""embedder.py: k-center greedy + worker call argument assembly.

`workers/dinov3.py` itself needs torch, so it is not unit-tested — covered end-to-end.
"""

from pathlib import Path

import numpy as np
import pytest

from agentic_detector_config.infer import embedder


@pytest.fixture(autouse=True)
def _relative_vllm_venv(monkeypatch):
    """Pin a tmp-relative path so the real worker venv (absolute, ADC_VLLM_VENV) is
    never touched — repo_root(tmp_path) / VLLM_VENV stays inside tmp."""
    monkeypatch.setattr(embedder, "VLLM_VENV", Path("external/vllm-venv"))


def test_kcenter_picks_farthest_points():
    """A layout whose correct order can be computed by hand:
    3 points clustered at (0,0), with (10,0) and (0,10) far away.
    Seed = point nearest the mean (≈(2,2)) → one of the cluster.
    Then the farthest points must be picked in order."""
    emb = np.array([[0., 0.], [0.1, 0.], [0., 0.1], [10., 0.], [0., 10.]])
    idx = embedder.kcenter_greedy(emb, 3)
    assert idx[0] in (0, 1, 2)          # seed from the cluster
    assert set(idx[1:]) == {3, 4}       # the two far points are picked


def test_kcenter_k_larger_than_n():
    emb = np.array([[0., 0.], [1., 1.]])
    assert sorted(embedder.kcenter_greedy(emb, 10)) == [0, 1]


def test_kcenter_empty():
    assert embedder.kcenter_greedy(np.zeros((0, 4)), 3) == []


def test_kcenter_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 8))
    assert embedder.kcenter_greedy(emb, 5) == embedder.kcenter_greedy(emb, 5)


def test_kcenter_no_duplicates():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(20, 4))
    idx = embedder.kcenter_greedy(emb, 7)
    assert len(set(idx)) == len(idx) == 7


def test_kcenter_duplicate_embeddings_return_distinct_indices():
    """With duplicate embeddings (identical crops), stop once distinct points run out —
    the same GT box repeated in exemplars skews prompt weight toward one instance."""
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    idx = embedder.kcenter_greedy(emb, 3)
    assert idx == [0]                              # only one distinct point
    emb2 = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    idx2 = embedder.kcenter_greedy(emb2, 3)
    assert len(set(idx2)) == len(idx2) == 2        # 2 without duplicates


def test_zero_k_selects_nothing():
    """k=0 yields 0 under every strategy — if kcenter kept its seed,
    exemplars_per_class=0 would silently produce 1 per class (diverging from mean)."""
    emb = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert embedder.kcenter_greedy(emb, 0) == []
    assert embedder.mean_nearest(emb, 0) == []
    assert embedder.select_indices("kcenter", 0, 2, embeddings=emb) == []
    assert embedder.select_indices("mean", 0, 2, embeddings=emb) == []


# --- outlier trim = distribution-based (median + k·MAD) --------------------------
# Only candidates whose centroid distance stands out from the distribution are
# cut — unlike a fixed-fraction trim, nothing is dropped when there are no outliers.

def test_trim_keeps_everything_when_no_outliers():
    """An even distribution loses nothing."""
    emb = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 6.0, 10)])
    assert list(embedder.trim_outliers(emb, 3.0)) == list(range(10))


def test_trim_drops_only_real_outliers():
    """Only the outliers are cut; returns original indices ascending."""
    rng = np.random.default_rng(0)
    body = rng.normal(scale=1.0, size=(20, 4))
    far = np.full((2, 4), 40.0)
    emb = np.vstack([body[:10], far[:1], body[10:], far[1:]])   # 10 and 21 are outliers
    kept = embedder.trim_outliers(emb, 3.0)
    assert 10 not in kept and 21 not in kept
    assert len(kept) == 20 and list(kept) == sorted(kept)


def test_trim_zero_mad_spread():
    """Even with a zero MAD (most distances equal) an extreme is cut; if all are equal, none are."""
    same = np.zeros((9, 2))
    emb = np.vstack([same, [[50.0, 0.0]]])
    assert list(embedder.trim_outliers(emb, 3.0)) == list(range(9))
    assert list(embedder.trim_outliers(np.zeros((5, 2)), 3.0)) == list(range(5))


def test_trim_off_keeps_everything():
    emb = np.array([[0., 0.], [0., 0.1], [9., 9.]])
    assert list(embedder.trim_outliers(emb, 0.0)) == [0, 1, 2]


def test_trim_never_goes_below_min_keep():
    """Trimming below the pick count (k) is meaningless — the k candidates nearest the
    centroid are always kept."""
    emb = np.vstack([np.zeros((2, 2)), np.full((4, 2), 30.0) + np.arange(4)[:, None]])
    kept = embedder.trim_outliers(emb, 0.5, min_keep=5)
    assert len(kept) == 5


def test_trim_rejects_negative():
    with pytest.raises(ValueError, match=">= 0"):
        embedder.trim_outliers(np.zeros((4, 2)), -1.0)


def test_kcenter_trim_excludes_outliers_but_keeps_spread():
    """Trimmed k-center must avoid outliers while still spreading within the remaining candidates."""
    rng = np.random.default_rng(11)
    dense = rng.normal(scale=1.0, size=(40, 6))          # bulk
    outliers = rng.normal(loc=30.0, scale=0.1, size=(6, 6))  # obvious outliers
    emb = np.vstack([dense, outliers])
    outlier_ids = set(range(40, 46))

    pure = embedder.kcenter_greedy(emb, 5, outlier_mad=0.0)
    trimmed = embedder.kcenter_greedy(emb, 5, outlier_mad=3.0)

    assert outlier_ids & set(pure)                        # plain k-center picks outliers
    assert not (outlier_ids & set(trimmed))               # trimmed does not
    # still differs from centroid-nearest (=mean) — diversity is preserved
    assert set(trimmed) != set(embedder.mean_nearest(emb, 5))


def test_kcenter_trim_returns_original_indices():
    """After picking from the trimmed subset, indices must map back to the original."""
    rng = np.random.default_rng(5)
    emb = np.vstack([rng.normal(size=(20, 4)), np.full((1, 4), 50.0)])
    idx = embedder.kcenter_greedy(emb, 4, outlier_mad=3.0)
    assert len(set(idx)) == 4
    assert all(0 <= i < 20 for i in idx)


def test_mean_nearest_picks_prototypical_not_outliers():
    """Visual prompts need representativeness — pick the centroid-nearest.
    (k-center, by contrast, picks outliers.)"""
    emb = np.array([[0., 0.], [0.1, 0.], [0., 0.1], [10., 0.], [0., 10.]])
    idx = embedder.mean_nearest(emb, 3)
    assert set(idx) == {0, 1, 2}        # the cluster = representative
    assert 3 not in idx and 4 not in idx  # outliers excluded


def test_mean_nearest_is_ordered_by_distance():
    emb = np.array([[0., 0.], [5., 0.], [1., 0.]])
    # centroid = (2,0) → distances 2, 3, 1 → nearest-first [2, 0, 1]
    assert embedder.mean_nearest(emb, 3) == [2, 0, 1]


def test_mean_nearest_k_larger_than_n_and_empty():
    assert sorted(embedder.mean_nearest(np.array([[0., 0.], [1., 1.]]), 10)) == [0, 1]
    assert embedder.mean_nearest(np.zeros((0, 4)), 3) == []


def test_kcenter_and_mean_disagree_on_skewed_data():
    """Contract: the two strategies actually pick differently — otherwise comparing them is meaningless."""
    rng = np.random.default_rng(7)
    emb = np.vstack([rng.normal(scale=0.1, size=(30, 8)),      # dense cluster
                     rng.normal(loc=5.0, scale=0.1, size=(3, 8))])  # far outliers
    assert set(embedder.kcenter_greedy(emb, 4)) != set(embedder.mean_nearest(emb, 4))


def test_encode_crops_empty_skips_worker(tmp_path):
    called = []
    out = embedder.encode_crops(tmp_path, [], "m", "cuda:0", tmp_path / "e.log",
                                popen=lambda *a, **k: called.append(a))
    assert out.shape == (0, 1) and called == []


def test_encode_crops_passes_absolute_paths(tmp_path, monkeypatch):
    """The worker's cwd is repo_root, not the caller's cwd — relative log/crop
    paths from the caller must reach the worker (argv and manifest) as absolute."""
    import json
    import os
    repo = tmp_path / "repo"
    (repo / embedder.VLLM_VENV / "bin").mkdir(parents=True)
    (repo / embedder.VLLM_VENV / "bin" / "python").touch()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)                      # caller cwd != worker cwd (repo)
    captured = {}

    class FakeProc:
        def wait(self):
            np.save(captured["out"], np.zeros((1, 4), dtype=np.float32))
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["out"] = cmd[cmd.index("--out") + 1]
        return FakeProc()

    embedder.encode_crops(repo, [Path("crops/a.jpg")], "m", "cuda:0",
                          Path("logs/e.log"), popen=fake_popen)
    cmd = captured["cmd"]
    for flag in ("--manifest", "--out"):
        assert os.path.isabs(cmd[cmd.index(flag) + 1]), flag
    manifest = json.loads(Path(cmd[cmd.index("--manifest") + 1]).read_text())
    assert manifest == [str(work / "crops" / "a.jpg")]


def test_encode_crops_builds_worker_command(tmp_path, monkeypatch):
    (tmp_path / embedder.VLLM_VENV / "bin").mkdir(parents=True)
    (tmp_path / embedder.VLLM_VENV / "bin" / "python").touch()
    crops = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    captured = {}

    class FakeProc:
        def wait(self):
            # write the .npy the worker would produce
            np.save(captured["out"], np.zeros((2, 4), dtype=np.float32))
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        captured["out"] = cmd[cmd.index("--out") + 1]
        return FakeProc()

    emb = embedder.encode_crops(tmp_path, crops, "facebook/dinov3-vitl16-pretrain-lvd1689m",
                                "cuda:3", tmp_path / "logs" / "e.log", popen=fake_popen)
    cmd = captured["cmd"]
    # runs under the vllm-venv python — agentic_detector_config itself has no torch
    assert cmd[0].endswith("external/vllm-venv/bin/python")
    assert cmd[1:3] == ["-m", "agentic_detector_config.workers.dinov3"]
    assert cmd[cmd.index("--device") + 1] == "cuda:3"
    assert cmd[cmd.index("--model") + 1] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    # the vLLM venv does not have this package installed — PYTHONPATH must be the
    # directory that *contains* the package (src/), not the repo root
    from agentic_detector_config.services.paths import PACKAGE_PARENT
    assert captured["env"]["PYTHONPATH"] == str(PACKAGE_PARENT)
    assert (PACKAGE_PARENT / "agentic_detector_config" / "workers").is_dir()
    assert emb.shape == (2, 4)

    manifest = Path(cmd[cmd.index("--manifest") + 1])
    import json
    assert json.loads(manifest.read_text()) == [str(p) for p in crops]


def test_encode_crops_count_mismatch_raises(tmp_path):
    (tmp_path / embedder.VLLM_VENV / "bin").mkdir(parents=True)
    (tmp_path / embedder.VLLM_VENV / "bin" / "python").touch()
    state = {}

    class FakeProc:
        def wait(self):
            np.save(state["out"], np.zeros((1, 4), dtype=np.float32))  # only 1
            return 0

    def fake_popen(cmd, **kwargs):
        state["out"] = cmd[cmd.index("--out") + 1]
        return FakeProc()

    with pytest.raises(RuntimeError, match="embedding count mismatch"):
        embedder.encode_crops(tmp_path, [tmp_path / "a.jpg", tmp_path / "b.jpg"],
                              "m", "cuda:0", tmp_path / "e.log", popen=fake_popen)


def test_encode_crops_worker_failure_raises(tmp_path):
    (tmp_path / embedder.VLLM_VENV / "bin").mkdir(parents=True)
    (tmp_path / embedder.VLLM_VENV / "bin" / "python").touch()

    class FakeProc:
        def wait(self):
            return 2

    with pytest.raises(RuntimeError, match="exit 2"):
        embedder.encode_crops(tmp_path, [tmp_path / "a.jpg"], "m", "cuda:0",
                              tmp_path / "e.log", popen=lambda *a, **k: FakeProc())


def test_encode_crops_missing_venv_raises(tmp_path):
    with pytest.raises(RuntimeError, match="ADC_VLLM_VENV"):
        embedder.encode_crops(tmp_path, [tmp_path / "a.jpg"], "m", "cuda:0",
                              tmp_path / "e.log")


# ------------------------------------------------- select_indices (shared dispatch)

def test_select_indices_dispatch_matches_primitives():
    """The single exemplar-selection dispatch — must match the primitives."""
    import random

    import numpy as np
    import pytest

    emb = np.array([[0.0], [0.1], [0.2], [5.0], [9.0]])
    assert embedder.select_indices("mean", 2, 5, embeddings=emb) \
        == embedder.mean_nearest(emb, 2)
    assert embedder.select_indices("kcenter", 2, 5, embeddings=emb, outlier_mad=3.0) \
        == embedder.kcenter_greedy(emb, 2, outlier_mad=3.0)
    r1 = embedder.select_indices("random", 3, 5, rng=random.Random(7))
    r2 = embedder.select_indices("random", 3, 5, rng=random.Random(7))
    assert r1 == r2 and len(r1) == 3               # seed determinism + k cap
    with pytest.raises(ValueError, match="exemplar_selection"):
        embedder.select_indices("magic", 2, 5, embeddings=emb)
