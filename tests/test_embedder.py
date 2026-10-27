"""embedder.py: k-center greedy + worker call argument assembly.

The torch path of `workers/dinov3.py` is covered in test_dinov3_worker.py
(skipped without torch); its batch plan (pure Python) is checked there without
torch.
"""

from pathlib import Path

import numpy as np
import pytest

from agentic_configuration_search.infer import embedder


@pytest.fixture(autouse=True)
def _relative_vllm_venv(monkeypatch):
    """Pin a tmp-relative path so the real worker venv (absolute, ACS_VLLM_VENV) is
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
    assert list(embedder.trim_outliers(emb, None)) == [0, 1, 2]


def test_trim_never_goes_below_min_keep():
    """Trimming below the pick count (k) is meaningless — the k candidates nearest the
    centroid are always kept."""
    emb = np.vstack([np.zeros((2, 2)), np.full((4, 2), 30.0) + np.arange(4)[:, None]])
    kept = embedder.trim_outliers(emb, 0.5, min_keep=5)
    assert len(kept) == 5


def test_trim_rejects_negative():
    with pytest.raises(ValueError, match="> 0"):
        embedder.trim_outliers(np.zeros((4, 2)), -1.0)


def test_kcenter_trim_excludes_outliers_but_keeps_spread():
    """Trimmed k-center must avoid outliers while still spreading within the remaining candidates."""
    rng = np.random.default_rng(11)
    dense = rng.normal(scale=1.0, size=(40, 6))          # bulk
    outliers = rng.normal(loc=30.0, scale=0.1, size=(6, 6))  # obvious outliers
    emb = np.vstack([dense, outliers])
    outlier_ids = set(range(40, 46))

    pure = embedder.kcenter_greedy(emb, 5, outlier_mad=None)
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
    paths from the caller must reach the worker (argv, jobs and manifest) as
    absolute."""
    import json
    import os
    repo = tmp_path / "repo"
    _fake_venv(repo)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)                      # caller cwd != worker cwd (repo)
    captured = {}
    embedder.encode_crops(repo, [Path("crops/a.jpg")], "m", "cuda:0",
                          Path("logs/e.log"), popen=_fake_worker(captured))
    assert os.path.isabs(captured["jobs"])
    job = json.loads(Path(captured["jobs"]).read_text())[0]
    assert os.path.isabs(job["manifest"]) and os.path.isabs(job["out"])
    manifest = json.loads(Path(job["manifest"]).read_text())
    assert manifest == [str(work / "crops" / "a.jpg")]


def _fake_venv(tmp_path):
    (tmp_path / embedder.VLLM_VENV / "bin").mkdir(parents=True)
    (tmp_path / embedder.VLLM_VENV / "bin" / "python").touch()


def _fake_worker(captured, rows=None):
    """A popen fake that reads the jobs file and writes each job's .npy instead.
    Without rows, as many rows as the manifest has; row i = [job index, i]."""
    import json

    class FakeProc:
        def wait(self):
            jobs = json.loads(Path(captured["jobs"]).read_text())
            for j, job in enumerate(jobs):
                n = len(json.loads(Path(job["manifest"]).read_text()))
                n = n if rows is None else rows[j]
                np.save(job["out"], np.array([[float(j), float(i)] for i in range(n)],
                                             dtype=np.float32).reshape(n, 2))
            return 0

    def fake_popen(cmd, **kwargs):
        captured.setdefault("calls", []).append(cmd)
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        captured["stdout"] = kwargs.get("stdout")
        captured["jobs"] = cmd[cmd.index("--jobs") + 1]
        return FakeProc()

    return fake_popen


def test_encode_crops_builds_worker_command(tmp_path):
    _fake_venv(tmp_path)
    crops = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    captured = {}
    emb = embedder.encode_crops(tmp_path, crops, "facebook/dinov3-vitl16-pretrain-lvd1689m",
                                "cuda:3", tmp_path / "logs" / "e.log",
                                popen=_fake_worker(captured))
    cmd = captured["cmd"]
    # runs under the vllm-venv python — agentic_configuration_search itself has no torch
    assert cmd[0].endswith("external/vllm-venv/bin/python")
    assert cmd[1:3] == ["-m", "agentic_configuration_search.workers.dinov3"]
    assert cmd[cmd.index("--device") + 1] == "cuda:3"
    assert cmd[cmd.index("--model") + 1] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    # the vLLM venv does not have this package installed — PYTHONPATH must be the
    # directory that *contains* the package (src/), not the repo root
    from agentic_configuration_search.services.paths import PACKAGE_PARENT
    assert captured["env"]["PYTHONPATH"] == str(PACKAGE_PARENT)
    assert (PACKAGE_PARENT / "agentic_configuration_search" / "workers").is_dir()
    assert emb.shape == (2, 2)

    import json
    jobs = json.loads(Path(captured["jobs"]).read_text())
    assert len(jobs) == 1
    assert json.loads(Path(jobs[0]["manifest"]).read_text()) == [str(p) for p in crops]
    assert Path(jobs[0]["out"]).name == "e.npy"


def test_encode_crops_many_one_worker_for_all_jobs(tmp_path):
    """A round's c1..cN embeddings finish in one worker run (one model load).
    Per-job output names (log stems) and the return order follow the input
    order."""
    _fake_venv(tmp_path)
    logs = tmp_path / "logs"
    jobs_in = [([tmp_path / "a.jpg", tmp_path / "b.jpg"], logs / "embed_c1.log"),
               ([], logs / "embed_c2.log"),                    # empty job — not sent to the worker
               ([tmp_path / "c.jpg"], logs / "embed_c3.log")]
    captured = {}
    out = embedder.encode_crops_many(tmp_path, jobs_in, "m", "cuda:0",
                                     popen=_fake_worker(captured))
    assert len(captured["calls"]) == 1
    import json
    jobs = json.loads(Path(captured["jobs"]).read_text())
    assert [Path(j["out"]).name for j in jobs] == ["embed_c1.npy", "embed_c3.npy"]
    assert [len(o) for o in out] == [2, 0, 1]
    assert out[0][:, 0].tolist() == [0.0, 0.0] and out[2][:, 0].tolist() == [1.0]
    assert out[1].shape == (0, 1)
    # the worker log goes to the first job's log; the other job logs point to it
    assert "embed_c1.log" in (logs / "embed_c3.log").read_text()


def test_encode_crops_many_passes_absolute_paths(tmp_path, monkeypatch):
    """Only absolute paths cross the boundary — the worker's cwd may differ from
    the caller's."""
    _fake_venv(tmp_path)
    monkeypatch.chdir(tmp_path)
    captured = {}
    embedder.encode_crops_many(tmp_path, [([Path("a.jpg")], Path("logs/e.log"))],
                               "m", "cuda:0", popen=_fake_worker(captured))
    import json
    assert Path(captured["jobs"]).is_absolute()
    job = json.loads(Path(captured["jobs"]).read_text())[0]
    assert Path(job["manifest"]).is_absolute() and Path(job["out"]).is_absolute()
    assert all(Path(p).is_absolute()
               for p in json.loads(Path(job["manifest"]).read_text()))


def test_encode_crops_many_all_empty_skips_worker(tmp_path):
    called = []
    out = embedder.encode_crops_many(tmp_path, [([], tmp_path / "a.log"), ([], tmp_path / "b.log")],
                                     "m", "cuda:0", popen=lambda *a, **k: called.append(a))
    assert called == [] and [o.shape for o in out] == [(0, 1), (0, 1)]


def test_worker_env_raises_thread_cap_only_for_worker(tmp_path, monkeypatch):
    """The orchestrator pins OMP/BLAS to 1 (cli), but the worker gets more threads
    for preprocessing — the inherited "1" is overridden only in the worker env.
    Tunable with ACS_EMBED_THREADS."""
    _fake_venv(tmp_path)
    for var in embedder.THREAD_ENV_VARS:
        monkeypatch.setenv(var, "1")
    monkeypatch.delenv("ACS_EMBED_THREADS", raising=False)
    monkeypatch.delenv("ACS_EMBED_LOADERS", raising=False)
    captured = {}
    embedder.encode_crops(tmp_path, [tmp_path / "a.jpg"], "m", "cuda:0",
                          tmp_path / "e.log", popen=_fake_worker(captured))
    want = embedder.worker_threads()
    assert want == min(8, embedder.available_cores()) and want >= 1
    env, cmd = captured["env"], captured["cmd"]
    assert {v: env[v] for v in embedder.THREAD_ENV_VARS} == \
        {v: str(want) for v in embedder.THREAD_ENV_VARS}
    assert cmd[cmd.index("--threads") + 1] == str(want)
    assert cmd[cmd.index("--loaders") + 1] == str(embedder.worker_loaders())
    import os
    assert os.environ["OMP_NUM_THREADS"] == "1"        # the parent env is unchanged

    monkeypatch.setenv("ACS_EMBED_THREADS", "3")
    monkeypatch.setenv("ACS_EMBED_LOADERS", "2")
    embedder.encode_crops(tmp_path, [tmp_path / "a.jpg"], "m", "cuda:0",
                          tmp_path / "e.log", popen=_fake_worker(captured))
    env, cmd = captured["env"], captured["cmd"]
    assert env["OMP_NUM_THREADS"] == "3" and cmd[cmd.index("--threads") + 1] == "3"
    assert cmd[cmd.index("--loaders") + 1] == "2"


def test_encode_crops_count_mismatch_raises(tmp_path):
    _fake_venv(tmp_path)
    with pytest.raises(RuntimeError, match="embedding count mismatch"):
        embedder.encode_crops(tmp_path, [tmp_path / "a.jpg", tmp_path / "b.jpg"],
                              "m", "cuda:0", tmp_path / "e.log",
                              popen=_fake_worker({}, rows=[1]))   # only 1


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
    with pytest.raises(RuntimeError, match="ACS_VLLM_VENV"):
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


# --- per-run embedding cache — the same crop is not re-embedded every round and candidate

def test_cached_encoder_reuses_by_content(tmp_path):
    """The key is crop content, not path — the same crop is rewritten into a
    different folder for every candidate. Only misses go to the real encoder,
    and the order follows the input."""
    calls = []

    def encode(paths):
        calls.append([p.name for p in paths])
        return np.array([[float(p.read_bytes()[0]), 1.0] for p in paths])

    (tmp_path / "r1").mkdir(); (tmp_path / "r2").mkdir()
    a1, b1 = tmp_path / "r1" / "a.jpg", tmp_path / "r1" / "b.jpg"
    a1.write_bytes(b"\x01aa"); b1.write_bytes(b"\x02bb")
    a2, c2 = tmp_path / "r2" / "a.jpg", tmp_path / "r2" / "c.jpg"
    a2.write_bytes(b"\x01aa"); c2.write_bytes(b"\x03cc")

    enc = embedder.cached_encoder(encode, tmp_path / "cache.npz")
    first = enc([a1, b1])
    second = enc([c2, a2])                       # a has the same content — cache hit
    assert calls == [["a.jpg", "b.jpg"], ["c.jpg"]]
    assert second[1].tolist() == first[0].tolist()
    assert second[0].tolist() == [3.0, 1.0]


def test_cached_encoder_persists_across_instances(tmp_path):
    """A new process after resume reads the cache back."""
    p = tmp_path / "x.jpg"; p.write_bytes(b"\x07xx")
    calls = []

    def encode(paths):
        calls.append(len(paths))
        return np.ones((len(paths), 3))

    embedder.cached_encoder(encode, tmp_path / "c.npz")([p])
    out = embedder.cached_encoder(encode, tmp_path / "c.npz")([p])
    assert calls == [1] and out.shape == (1, 3)
    assert embedder.cached_encoder(encode, tmp_path / "c.npz")([]).shape[0] == 0


def test_cached_encoder_passes_extra_args_on_miss(tmp_path):
    """Extra args (the per-call log path) are passed through when misses are
    sent — worker outputs stay split per class for later analysis."""
    p = tmp_path / "x.jpg"; p.write_bytes(b"\x01")
    seen = []

    def encode(paths, log_path):
        seen.append(log_path)
        return np.ones((len(paths), 2))

    enc = embedder.cached_encoder(encode, tmp_path / "c.npz")
    enc([p], "embed_c1.log")
    enc([p], "embed_c2.log")                      # hit — the encoder is not called
    assert seen == ["embed_c1.log"]


def test_cached_encoder_many_one_call_same_misses_as_sequential(tmp_path):
    """Sending several classes at once (many) gives each group the same miss list
    as sequential calls — what an earlier group already sent is left out of later
    groups. The worker must get the same manifests for the embeddings to be
    bit-identical."""
    (tmp_path / "x").mkdir()
    files = {}
    for name, body in [("a", b"\x01"), ("b", b"\x02"), ("a2", b"\x01"), ("c", b"\x03")]:
        files[name] = tmp_path / "x" / f"{name}.jpg"
        files[name].write_bytes(body)
    groups = [([files["a"], files["b"]], "c1.log"), ([files["a2"], files["c"]], "c2.log"),
              ([], "c3.log")]

    seq_calls = []

    def encode(paths, log):
        seq_calls.append(([p.name for p in paths], log))
        return np.array([[float(p.read_bytes()[0]), 0.0] for p in paths])

    seq = embedder.cached_encoder(encode, tmp_path / "seq.npz")
    want = [seq(p, log) for p, log in groups[:2]]

    many_calls = []

    def encode_many(gs):
        many_calls.append([([p.name for p in ps], log) for ps, log in gs])
        return [encode(ps, log) for ps, log in gs]

    seq_calls.clear()
    enc = embedder.cached_encoder(encode, tmp_path / "many.npz", encode_many=encode_many)
    got = enc.many(groups)
    assert many_calls == [[(["a.jpg", "b.jpg"], "c1.log"), (["c.jpg"], "c2.log")]]
    assert [g.tolist() for g in got[:2]] == [w.tolist() for w in want]
    assert got[2].shape[0] == 0
    # later single calls all hit — the encoder is not called again
    seq_calls.clear()
    many_calls.clear()
    assert enc(groups[1][0], "c2.log").tolist() == want[1].tolist()
    assert seq_calls == [] and many_calls == []
    # the disk cache stays too (resume)
    again = embedder.cached_encoder(encode, tmp_path / "many.npz")
    assert again(groups[0][0], "c1.log").tolist() == want[0].tolist()
    assert seq_calls == []


def test_cached_encoder_single_call_goes_through_encode_many(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\x05")
    calls = []

    def encode_many(gs):
        calls.append(len(gs))
        return [np.ones((len(ps), 2)) for ps, *_ in gs]

    enc = embedder.cached_encoder(None, tmp_path / "c.npz", encode_many=encode_many)
    assert enc([p], "l").shape == (1, 2) and calls == [1]


def test_hooks_encode_fn_prefetch_batches_classes_into_one_worker(tmp_path, monkeypatch):
    """The real wiring: encode_fn.prefetch({cid: paths}) sends the missed crops in
    one encode_crops_many call (job log = embed_c{cid}.log), and later
    encode_fn(cid, paths) reads the run cache (embed_*.npz)."""
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop import hooks as hooks_mod

    calls = []

    def fake_many(repo_root, jobs, model, device, popen=None):
        calls.append([(len(ps), Path(lp).name) for ps, lp in jobs])
        return [np.full((len(ps), 2), float(i)) for i, (ps, _) in enumerate(jobs)]

    monkeypatch.setattr(embedder, "encode_crops_many", fake_many)
    cfg = config_from_dict({"dataset": str(tmp_path)})
    out_dir = tmp_path / "run"
    h = hooks_mod.build_hooks(cfg, tmp_path, out_dir)
    crops = {}
    for cid, n in [(1, 2), (2, 3)]:
        crops[cid] = []
        for i in range(n):
            p = tmp_path / f"c{cid}_{i}.jpg"
            p.write_bytes(bytes([cid, i]))
            crops[cid].append(p)

    h["encode_fn"].prefetch(crops)
    assert calls == [[(2, "embed_c1.log"), (3, "embed_c2.log")]]
    assert h["encode_fn"](2, crops[2]).tolist() == [[1.0, 1.0]] * 3
    assert len(calls) == 1
    assert list((out_dir / "_cache").glob("embed_*.npz"))


def test_trim_outliers_none_is_off_and_zero_is_rejected():
    emb = np.random.default_rng(0).normal(size=(10, 4))
    assert list(embedder.trim_outliers(emb, None)) == list(range(10))
    with pytest.raises(ValueError, match="None"):
        embedder.trim_outliers(emb, 0.0)
