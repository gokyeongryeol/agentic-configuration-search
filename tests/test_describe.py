"""diagnose/describe — VLM crop descriptions: selection rule, artifact replay, error isolation.

Crops come only from train details (holdout diagnosis is anonymized, so it is
structurally impossible), and results flow only as diagnostic signal — runner wiring
keeps them out of the decision path.
"""

import json

from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.diagnose import describe


def _fp(kind, image, cid, score, box=(0, 0, 10, 10)):
    return {"kind": kind, "image": image, "category_id": cid,
            "score": score, "box": list(box)}


def test_select_crops_cover_every_prompt_sample():
    """Describe **every** sample that goes into the prompt. Describing only some makes
    the call budget, not significance, decide which get a note, and the planner
    misreads 'no note' as 'less important'."""
    from agentic_configuration_search.diagnose.sample import pick_detail
    from agentic_configuration_search.planner.llm import _compact_diagnosis, _det_key

    fp = ([_fp("background", f"b{i}.png", 1, 0.5 + i * 0.01) for i in range(5)]
          + [_fp("class_confusion", "c0.png", 2, 0.9)]
          + [_fp("duplicate", "d0.png", 1, 0.99)]
          + [_fp("localization", "l0.png", 1, 0.99)])
    fn = ([{"kind": "missed", "image": f"m{i}.png", "category_id": 1,
            "box": [0, 0, 10 * (i + 1), 10]} for i in range(4)]
          + [{"kind": "below_threshold", "image": "t0.png", "category_id": 1,
              "box": [0, 0, 99, 99]}])
    diag = {"fp": fp, "fn": fn}
    crops = describe.select_crops(diag)

    # samples and crops are **exactly the same set** — no kind dropped, no count truncated
    sample_keys = {_det_key(r) for side in ("fp", "fn")
                   for r in pick_detail(diag[side])}
    assert {_det_key(c) for c in crops} == sample_keys
    assert {c["kind"] for c in crops} == {"background", "class_confusion",
                                          "duplicate", "localization",
                                          "missed", "below_threshold"}

    # so every note is folded into the samples — no separate crop_notes entry remains
    notes = [{**c, "note": "x"} for c in crops]
    compact = _compact_diagnosis({"train": {**diag, "crop_notes": notes}})
    assert "crop_notes" not in compact["train"]
    described = [r for side in ("fp_sample", "fn_sample")
                 for r in compact["train"][side] if "note" in r]
    assert len(described) == len(crops)

    bg = [c for c in crops if c["kind"] == "background"]
    assert bg[0]["score"] == 0.54                          # top significance
    missed = [c for c in crops if c["kind"] == "missed"]
    assert missed[0]["image"] == "m3.png"                  # largest area


def test_crop_notes_batches_and_isolates_failure(dataset_root, tmp_path):
    """Calls are split into CROPS_PER_CALL crops each; if one batch fails the other
    notes survive. Crop indices start at 0 within each batch."""
    from agentic_configuration_search.core.data import load_dataset

    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    names = [e.file_name for e in ds.train.images]
    # KIND_CAP (4) per kind means one kind cannot span batches — use two kinds
    fp = [_fp(kind, names[i], 1 + k, 0.9 - i * 0.01, box=(5 + i, 5, 40 + i, 40))
          for k, kind in enumerate(("background", "class_confusion"))
          for i in range(5)]
    n = 8                                    # pick_detail keeps 4 per kind
    diag = {"fp": fp, "fn": []}
    seen = []

    def chat(messages, **kw):
        idx = [p["text"] for p in messages[0]["content"]
               if p["type"] == "text" and p["text"].startswith("crop ")]
        seen.append(len(idx))
        if len(seen) == 1:
            raise RuntimeError("first batch down")     # fail only the first batch
        return {"notes": [{"crop": i, "note": f"n{i}"} for i in range(len(idx))]}

    notes = describe.crop_notes(ds.train, diag, chat, tmp_path / "c.json")
    assert len(seen) == 2 and seen[0] == describe.CROPS_PER_CALL
    assert notes and len(notes) == n - describe.CROPS_PER_CALL   # only the 2nd batch
    stored = json.loads((tmp_path / "c.json").read_text())
    assert len(stored["calls"]) == 2 and stored["calls"][0]["error"]
    assert stored["calls"][1]["error"] is None


def test_crop_notes_artifact_and_replay(dataset_root, tmp_path):
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    img = ds.train.images[0].file_name
    diag = {"fp": [_fp("background", img, 1, 0.9, box=(5, 5, 40, 40))]}
    calls = []

    def chat(messages, **kw):
        calls.append(messages)
        # Non-ASCII (Korean) note on purpose: exercises the ensure_ascii=False artifact round-trip.
        return {"notes": [{"crop": 0, "note": "반사광으로 보인다"}]}

    out = tmp_path / "crop_notes.json"
    notes = describe.crop_notes(ds.train, diag, chat, out)
    assert notes and notes[0]["note"] == "반사광으로 보인다"
    assert notes[0]["kind"] == "background" and notes[0]["image"] == img
    # multimodal message: one image part + data URI
    parts = calls[0][0]["content"]
    uris = [p for p in parts if p.get("type") == "image_url"]
    assert len(uris) == 1
    assert uris[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    stored = json.loads(out.read_text())
    assert stored["notes"] == notes and stored["crops"]
    # replay — zero re-calls
    again = describe.crop_notes(ds.train, diag, chat, out)
    assert again == notes and len(calls) == 1


def test_crop_notes_error_isolated(dataset_root, tmp_path):
    """A chat failure or invalid reply degrades to no description (None) — the round
    continues. The failure is recorded as an artifact so resume does not retry it."""
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    img = ds.train.images[0].file_name
    diag = {"fp": [_fp("background", img, 1, 0.9, box=(5, 5, 40, 40))]}
    calls = []

    def boom(messages, **kw):
        calls.append(1)
        raise RuntimeError("server down")

    out = tmp_path / "crop_notes.json"
    assert describe.crop_notes(ds.train, diag, boom, out) is None
    assert json.loads(out.read_text())["calls"][0]["error"]
    assert describe.crop_notes(ds.train, diag, boom, out) is None
    assert len(calls) == 1                                 # replay — no retry


def test_crop_notes_empty_detail_skips_chat(dataset_root, tmp_path):
    ds = load_dataset(dataset_root, seed=0, fit_train_ratio=0.8)
    calls = []
    notes = describe.crop_notes(ds.train, {"fp": [], "fn": []},
                                lambda m, **k: calls.append(1),
                                tmp_path / "crop_notes.json")
    assert notes is None and calls == []
