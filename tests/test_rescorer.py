"""rescorer — identity preservation, logit blending, serialization, α selection.

Includes train/dev discipline tests (fit_rescorer)."""

import numpy as np
import pytest

from agentic_detector_config.fitting.rescorer import (Rescorer, blend, dataset_fingerprint,
                                     fit_discriminator, logit, oof_phat, sigmoid)

FEATS = ["score", "overlap_margin", "area_z", "tile_agree"]


class _Ont:
    """Minimal ontology required by dataset_fingerprint (reads only id/name)."""

    def __init__(self, pairs):
        from types import SimpleNamespace
        self.classes = [SimpleNamespace(id=i, name=n) for i, n in pairs]


def _sep(n=300, dim=16, seed=0):
    """Data whose only signal is in the embedding — also checks that the discriminator actually uses emb."""
    rng = np.random.default_rng(seed)
    y = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
    hand = rng.normal(0, 1, size=(n, 4))
    hand[:, 0] = rng.uniform(0.05, 0.95, size=n)          # score column
    E = rng.normal(0, 0.5, size=(n, dim))
    E[:, 0] += np.where(y > 0, 2.0, -2.0)
    return hand, E, y


def test_blend_identity_at_zero():
    s = np.array([0.03, 0.2, 0.5, 0.99])
    p = np.array([0.9, 0.1, 0.5, 0.01])
    assert np.array_equal(blend(s, p, 0.0), s)            # bit-exact identity


def test_blend_is_logit_sum_and_bounded():
    s, p, a = np.array([0.2]), np.array([0.8]), 0.5
    out = blend(s, p, a)
    assert out[0] == pytest.approx(float(sigmoid(logit(s) + a * logit(p))[0]))
    assert 0.0 < out[0] < 1.0
    # high P̂ raises the score, low P̂ lowers it (both directions)
    assert blend(np.array([0.3]), np.array([0.99]), 1.0)[0] > 0.3
    assert blend(np.array([0.3]), np.array([0.01]), 1.0)[0] < 0.3


def test_blend_preserves_order_within_equal_phat():
    """Within the same P̂ the original score order is preserved — the key property distinguishing it from pure reranking."""
    s = np.array([0.31, 0.32, 0.33])
    p = np.full(3, 0.7)
    out = blend(s, p, 1.0)
    assert list(np.argsort(out)) == list(np.argsort(s))


def test_rescorer_rescore_and_roundtrip(tmp_path):
    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    fp = dataset_fingerprint("demo", _Ont([(1, "a"), (2, "b")]))
    ad = Rescorer(disc, {1: 0.5, 2: 0.5}, meta={"n": len(y)}, fingerprint=fp)
    feats = [dict(zip(disc.features, row)) for row in hand[:5]]
    cids = [1, 2, 1, 2, 1]
    new = ad.rescore(feats, [E[i] for i in range(5)], list(hand[:5, 0]), cids)
    assert new.shape == (5,) and np.all((new > 0) & (new < 1))

    path = tmp_path / "rescorer.json"
    ad.save(path)
    assert (tmp_path / "rescorer.pkl").exists()
    ad2 = Rescorer.load(path, expect=fp)
    assert ad2.alphas == {1: 0.5, 2: 0.5}
    assert np.allclose(
        ad2.rescore(feats, [E[i] for i in range(5)], list(hand[:5, 0]), cids), new)


def test_per_class_alpha_zero_is_bit_exact():
    """Classes with α_c=0 do not even compute P̂ and pass inputs through unchanged (condition 6)."""
    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    ad = Rescorer(disc, {1: 0.0, 2: 1.0})
    feats = [dict(zip(disc.features, row)) for row in hand[:4]]
    scores = list(hand[:4, 0])
    out = ad.rescore(feats, [E[i] for i in range(4)], scores, [1, 2, 1, 2])
    assert out[0] == scores[0] and out[2] == scores[2]      # α=0 classes — bit-exact
    assert out[1] != scores[1] and out[3] != scores[3]      # α>0 classes — changed


def test_all_alpha_zero_is_noop():
    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    ad = Rescorer(disc, {1: 0.0})
    assert not ad.active
    feats = [dict(zip(disc.features, row)) for row in hand[:4]]
    scores = list(hand[:4, 0])
    assert np.array_equal(
        ad.rescore(feats, [E[i] for i in range(4)], scores, [1] * 4),
        np.array(scores))


def test_unseen_class_falls_back_to_identity():
    """Classes unseen in training are left as identity rather than guessed."""
    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    ad = Rescorer(disc, {1: 1.0})
    feats = [dict(zip(disc.features, hand[0]))]
    assert ad.alpha_for(999) == 0.0
    assert ad.rescore(feats, [E[0]], [0.4], [999])[0] == 0.4


def test_fingerprint_mismatch_is_explicit_error(tmp_path):
    """A rescorer from another dataset is not silently used — it is an explicit error."""
    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    ont = _Ont([(1, "a")])
    fp = dataset_fingerprint("ds-a", ont)
    path = tmp_path / "rescorer.json"
    Rescorer(disc, {1: 0.5}, fingerprint=fp).save(path)
    Rescorer.load(path, expect=fp)                       # match — passes
    with pytest.raises(ValueError, match="different dataset fingerprint"):
        Rescorer.load(path, expect=dataset_fingerprint("ds-b", ont))
    # renaming or adding classes makes α's keys meaningless, so that is a mismatch too
    with pytest.raises(ValueError, match="different dataset fingerprint"):
        Rescorer.load(path, expect=dataset_fingerprint(
            "ds-a", _Ont([(1, "a"), (2, "b")])))


def test_pickle_from_old_package_name_loads(tmp_path):
    """rescorer.pkl files from runs before the package rename (zero_agent -> agentic_detector_config)
    still load — pickle records classes by module path, so failing to read them would kill resume
    and predict replay of old runs with ModuleNotFoundError."""
    import pickle

    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    ad = Rescorer(disc, {1: 0.5})
    path = tmp_path / "rescorer.json"
    ad.save(path)
    # reproduce a pickle written under the old package name (protocol 0 stores the module path as text)
    legacy = pickle.dumps(disc, protocol=0).replace(
        b"agentic_detector_config.", b"zero_agent.")
    assert b"zero_agent.fitting.verifier" in legacy
    (tmp_path / "rescorer.pkl").write_bytes(legacy)

    ad2 = Rescorer.load(path)
    feats = [dict(zip(disc.features, row)) for row in hand[:5]]
    assert np.allclose(
        ad2.rescore(feats, [E[i] for i in range(5)], list(hand[:5, 0]), [1] * 5),
        ad.rescore(feats, [E[i] for i in range(5)], list(hand[:5, 0]), [1] * 5))


def test_legacy_scalar_alpha_artifact_rejected(tmp_path):
    import json
    p = tmp_path / "rescorer.json"
    p.write_text(json.dumps({"alpha": 0.5, "model_file": "rescorer.pkl"}))
    with pytest.raises(ValueError, match="is a legacy"):
        Rescorer.load(p)


def test_category_ids_length_mismatch_raises():
    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    ad = Rescorer(disc, {1: 1.0})
    feats = [dict(zip(disc.features, hand[0]))]
    with pytest.raises(ValueError, match="category_ids length mismatch"):
        ad.rescore(feats, [E[0]], [0.5], [1, 1])


def test_emb_dim_mismatch_raises():
    hand, E, y = _sep()
    disc = fit_discriminator(FEATS, hand, E, y, members=2)
    ad = Rescorer(disc, {1: 1.0})
    feats = [dict(zip(disc.features, hand[0]))]
    with pytest.raises(ValueError, match="embedding dimension mismatch"):
        ad.rescore(feats, [np.zeros(3)], [0.5], [1])


def test_oof_phat_separates_and_is_out_of_fold():
    from agentic_detector_config.core.metrics import rank_auc
    hand, E, y = _sep()
    p = oof_phat(FEATS, hand, E, y, members=2)
    assert p.shape == y.shape
    assert rank_auc(p, y.astype(bool)) > 0.9      # recovers the emb signal
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_ensemble_members_are_distinct_and_averaged():
    """There really are multiple members, and the ensemble P̂ differs from a single model."""
    hand, E, y = _sep()
    one = fit_discriminator(FEATS, hand, E, y, members=1)
    five = fit_discriminator(FEATS, hand, E, y, members=5)
    assert one.members == 1 and five.members == 5
    X = np.hstack([hand, E])
    assert not np.allclose(one.predict_proba(X), five.predict_proba(X))


def test_ensemble_roundtrip_preserves_members():
    import pickle

    hand, E, y = _sep()
    v = fit_discriminator(FEATS, hand, E, y, members=3)
    back = pickle.loads(pickle.dumps(v))
    assert back.members == 3
    X = np.hstack([hand, E])
    assert np.allclose(back.predict_proba(X), v.predict_proba(X))


def test_config_alphas_must_include_identity():
    """Adoption is structurally safe only if the search space includes the baseline —
    validation at config load time enforces this."""
    from agentic_detector_config.config import ConfigError, config_from_dict

    assert 0.0 in config_from_dict({"dataset": "x"}).rescorer.alphas
    with pytest.raises(ConfigError, match=r"rescorer\.alphas must include 0\.0"):
        config_from_dict({"dataset": "x", "rescorer": {"alphas": [0.5, 1.0]}})


def test_config_rejects_negative_margin_and_zero_members():
    from agentic_detector_config.config import ConfigError, config_from_dict

    with pytest.raises(ConfigError, match=r"rescorer\.margin cannot be negative"):
        config_from_dict({"dataset": "x", "rescorer": {"margin": -0.01}})
    with pytest.raises(ConfigError, match=r"rescorer\.ensemble_members must be >= 1"):
        config_from_dict({"dataset": "x", "rescorer": {"ensemble_members": 0}})


# ------------------------------------------- fit_rescorer (fit_train/fit_dev discipline)

def _mk_train(root, prefix, n_images):
    from agentic_detector_config.core.coco import Split
    from tests.conftest import make_split

    make_split(root, [
        {"name": f"{prefix}{i}.png", "w": 200, "h": 160,
         "anns": [(1, 10, 10, 30, 30), (2, 100, 80, 30, 30)]}
        for i in range(n_images)])
    return Split.load(root)


def _mk_preds(train, prefix, seed=0):
    """Class 1 has a TP signal in the embedding; class 2 is pure noise.
    Score alone mixes TP/FP, so the discriminator only improves by using the embedding."""
    rng = np.random.default_rng(seed)
    preds, k = {}, 0
    for entry in train.images:
        for a in entry.annotations:
            cid = a.category_id
            x1, y1, x2, y2 = a.bbox_xyxy
            for is_tp in (True, False):
                box = ([x1 + 1, y1 + 1, x2 - 1, y2 - 1] if is_tp
                       else [x1 + 60, y1 + 40, x2 + 60, y2 + 40])
                emb = rng.normal(0, 0.5, size=8)
                if cid == 1:
                    emb[0] += 2.0 if is_tp else -2.0      # signal
                preds[f"{prefix}{k}"] = {
                    "det_id": f"{prefix}{k}", "image": entry.file_name, "box": box,
                    "category_id": cid, "sources": ["full"],
                    "score": float(rng.uniform(0.3, 0.7)),   # score is uninformative
                    "embedding": emb.tolist()}
                k += 1
    return preds


def _rs(**over):
    from agentic_detector_config.config import config_from_dict
    return config_from_dict({"dataset": "x", "rescorer": over}).rescorer


def _fit(tmp_path, rs, n_train=24, n_dev=12):
    from agentic_detector_config.fitting.rescorer import fit_rescorer

    train = _mk_train(tmp_path / "tr", "t", n_train)
    dev = _mk_train(tmp_path / "dv", "d", n_dev)
    return fit_rescorer(
        rs, train_preds=_mk_preds(train, "t"), train_split=train,
        dev_preds=_mk_preds(dev, "d", seed=1), dev_split=dev,
        dev_image_paths=[dev.image_path(e) for e in dev.images],
        ontology=train.ontology, dataset_name="demo")


def test_fit_rescorer_adopts_signal_and_rejects_noise(tmp_path):
    rs = _rs(min_samples=50, min_samples_per_class=1, margin=0.005,
             ensemble_members=2)
    rescorer, info = _fit(tmp_path, rs)
    assert rescorer is not None, info
    # the class with signal is accepted; the noise-only class is held by the margin -> identity
    assert rescorer.alpha_for(1) > 0.0, info["sel_f_beta_by_class_alpha"]
    assert rescorer.alpha_for(2) == 0.0, info["sel_f_beta_by_class_alpha"]
    assert info["alpha_active_classes"] == 1 and info["n_classes"] == 2
    assert info["auc_dev"] > info["auc_score_only_dev"]   # the discriminator also recovers the signal on dev
    assert rescorer.fingerprint == dataset_fingerprint("demo", rescorer_ont())


def rescorer_ont():
    from tests.conftest import CATS
    from agentic_detector_config.core.coco import Ontology
    return Ontology.from_categories(CATS)


def test_fit_rescorer_high_margin_falls_back_to_identity(tmp_path):
    rs = _rs(min_samples=50, min_samples_per_class=1, margin=1.0,
             ensemble_members=2)
    rescorer, info = _fit(tmp_path, rs)
    assert rescorer is not None and not rescorer.active   # identity for every class


def test_thin_class_forced_to_identity_without_skipping_dataset(tmp_path):
    rs = _rs(min_samples=50, min_samples_per_class=10_000, margin=0.005,
             ensemble_members=2)
    rescorer, info = _fit(tmp_path, rs)
    assert rescorer is not None                            # does not give up on the dataset
    assert not rescorer.active                             # α=0 for every class
    assert set(info["thin_classes"]) == {"1", "2"}


def test_total_guard_skips_when_train_too_small(tmp_path):
    rs = _rs(min_samples=64, min_samples_per_class=1, ensemble_members=2)
    rescorer, info = _fit(tmp_path, rs, n_train=2)         # 8 rows < 64
    assert rescorer is None and info["skipped"] is True


def test_apply_rescorer_preserves_raw_and_identity(tmp_path):
    from agentic_detector_config.fitting.rescorer import _apply_rescorer

    rs = _rs(min_samples=50, min_samples_per_class=1, margin=0.005,
             ensemble_members=2)
    rescorer, _ = _fit(tmp_path, rs)
    assert rescorer is not None and rescorer.active
    target = _mk_train(tmp_path / "ap", "a", 6)
    preds = _mk_preds(target, "a", seed=2)
    before = {k: v["score"] for k, v in preds.items()}
    n = _apply_rescorer(rescorer, preds, target, set())
    assert n > 0
    for k, v in preds.items():
        if v["category_id"] == 2:                          # α=0 class — unchanged
            assert v["score"] == before[k] and "score_raw" not in v
        elif v["score"] != before[k]:
            assert v["score_raw"] == before[k]


def test_fits_without_embeddings_when_server_omits_them():
    """The discriminator is fit from hand-crafted features alone even with no embeddings at all.

    Detections may come back without query embeddings (not every detector path provides them),
    leaving every training row without one. Requiring embeddings would make rescorer refit
    impossible in principle for such rounds — as long as detections exist, the hand-crafted
    features (score, overlap_margin, area_z, tile_agree) are enough to fit.
    """
    import numpy as np

    from agentic_detector_config.fitting.rescorer import _rescorer_rows as _rows

    preds = {}
    for i in range(4):
        preds[f"d{i}"] = {"det_id": f"d{i}", "image": "a.png", "score": 0.5,
                          "box": [0, 0, 10, 10], "category_id": 1,
                          "embedding": None}
    rows = _rows(preds, _FakeSplit(), tiled_ids=set(), iou_thr=0.5,
                 area_stats={1: {"mean": 100.0, "std": 10.0, "count": 4}})
    assert len(rows) == 4, "rows must not be dropped for lacking an embedding"
    assert all(r["emb"] is None for r in rows)


class _FakeSplit:
    class _E:
        file_name = "a.png"
        annotations = []
    images = [_E()]
