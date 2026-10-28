"""diagnose/story — the run report (report.md). No GPU or LLM needed."""

import json
import re
from pathlib import Path

from agentic_configuration_search.core import metrics
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.diagnose import story
from agentic_configuration_search.planner.axes import AXES
from tests.test_report import finished_run  # noqa: F401 — fixture reuse

PKG = Path(story.__file__).parents[1]
# technical terms that must not reach a non-technical reader (only ones that do not clash
# with class or file names)
JARGON = ("micro", "F(β)", "holdout", "fit_dev", "fit_train", "LoRA", "lora",
          "rescorer", "threshold", "gate", "exemplar", "tiling", "baseline",
          "champion", "FP", "FN", "λ")


def _jargon(text: str) -> list[str]:
    return [w for w in JARGON if re.search(rf"(?<![\w/]){re.escape(w)}(?![\w.])", text)]


def test_vocab_covers_every_axis_error_kind_and_skip_reason():
    """A new axis, error cause or skip reason must go into the plain-word tables too —
    otherwise the technical term reaches the report as is."""
    assert set(AXES) - story._QUANTITY_KEYS <= set(story.STRATEGY)
    assert set(metrics.FP_KINDS) | set(metrics.FN_KINDS) <= set(story.CAUSE)
    from agentic_configuration_search.loop import runner
    from agentic_configuration_search.planner import llm
    reasons = set()
    for p in PKG.rglob("*.py"):
        text = p.read_text()
        reasons |= {a or b for a, b in re.findall(
            r'"skipped":\s*"(\w+)"|skipped="(\w+)"', text)}
        reasons |= set(re.findall(r'_drop\(\w+, \w+, "(\w+)"', text))   # planner.axes
    assert {"no_change", "max_candidates", "invalid_axes", "same_setting"} <= reasons, \
        "the skip-reason regex no longer matches the code"
    reasons |= set(llm._UNMEASURED) | set(runner.NOOP_SKIPS) | set(runner.FIT_DEV_SKIPS)
    reasons |= {"gate_failed", "gate_passed_but_lost", "screen_top_k", "adopted"}
    missing = reasons - set(story._OUTCOME_GROUPS)
    assert not missing, f"skip reasons missing from the plain-word table: {missing}"


def test_strategy_label_is_plain():
    assert story.strategy_label({"lora": True}) == "Extra training on this data"
    assert story.strategy_label({"tiling": 3}) == "Zoom into parts of the photo (3×3)"
    assert story.strategy_label({"rescorer": False, "lora": True}) == \
        "Extra training on this data + drop second review"
    assert story.strategy_label({"exemplar_base_k": 4}) == "Show example photos (different number)"


def test_run_end_writes_the_story(finished_run):  # noqa: F811
    """When run_rounds ends, report.md and its figures are left — without technical terms."""
    md = (finished_run / story.REPORT_FILE).read_text()
    for head in ("## At a glance", "## 1. What is the data?", "## 2. How did it improve?",
                 "## 3. What problems are left?", "## 4. What to do next"):
        assert head in md
    assert "from 66.7 to 100.0" in md
    assert "Show example photos" in md                 # the adopted strategy's plain name
    for name in ("rounds.png", "dataset_classes.png", "dataset_sizes.png",
                 "dataset_samples.jpg"):
        assert (finished_run / story.ASSET_DIR / name).exists(), name
        assert f"{story.ASSET_DIR}/{name}" in md
    assert "could not be generated" not in md          # not passing on fallback headings alone
    assert "a little better than on photos never seen before" in md   # holdout reuse is stated
    assert not _jargon(md)
    summary = json.loads((finished_run / "run_summary.json").read_text())
    assert Path(summary["dataset"]).is_absolute()


def _inject_residual(run: Path) -> None:
    """Plant remaining errors in the final diagnosis — misses (not seen) and wrong finds (background)."""
    fin = run / "rounds" / "final"
    fn = [{"kind": "missed", "box": [10, 10, 30, 30], "category": "cat a",
           "category_id": 1, "image": f"tr{i:02d}.png", "size": "large"}
          for i in (1, 3, 5)]
    fp = [{"kind": "background", "box": [50, 40, 70, 60], "category": "cat b",
           "category_id": 2, "image": f"tr{i:02d}.png", "score": 0.9}
          for i in (0, 2)]
    (fin / "diagnosis_train.json").write_text(json.dumps({
        "n_images": 12, "summary": {"fn": {"missed": 3}, "fp": {"background": 2}},
        "per_class": {}, "fp": fp, "fn": fn}))
    (fin / "diagnosis_test.json").write_text(json.dumps({
        "n_images": 8,
        "summary": {"fn": {"missed": 3, "below_threshold": 0},
                    "fp": {"background": 2}},
        "per_class": {"cat a": {"gt": 4, "fn": {"missed": 3}, "fp": {},
                                "miss_rate": 0.75},
                      "cat b": {"gt": 4, "fn": {}, "fp": {"background": 2},
                                "miss_rate": 0.0}}}))
    (fin / "crop_notes.json").write_text(json.dumps({"notes": [
        {"image": "tr01.png", "box": [10, 10, 30, 30], "kind": "missed",
         "category": "cat a", "note": "The cat a is dark and blurry."}]}))
    (fin / "residual.json").unlink(missing_ok=True)


def _ds(run: Path):
    return load_dataset(Path(json.loads((run / "run_summary.json").read_text())["dataset"]))


def test_examples_use_notes_then_fixed_sentences(finished_run):  # noqa: F811
    _inject_residual(finished_run)
    md = story.render_report(finished_run, ds=_ds(finished_run))
    assert "1. The cat a is dark and blurry." in md     # the note as written
    assert "Objects not seen at all (missed)" in md
    assert "Background taken for an object (wrong find)" in md
    assert "2. A case of 'Objects not seen at all' (large cat a)." in md
    assert "A case of 'Background taken for an object' (name the program gave: cat b)." in md
    assert (finished_run / story.ASSET_DIR / "pattern_1_missed.jpg").exists()
    assert "photos where the object is blurry, dark" in md      # note terms → plain words
    assert not _jargon(md)


def test_examples_never_open_holdout_images(finished_run):  # noqa: F811
    """Example photos come from train only — the report draws even without holdout image files."""
    _inject_residual(finished_run)
    ds = _ds(finished_run)
    for e in ds.holdout.images:
        ds.holdout.image_path(e).unlink()
    md = story.render_report(finished_run, ds=ds)
    assert (finished_run / story.ASSET_DIR / "pattern_1_missed.jpg").exists()
    assert "## 3. What problems are left?" in md


def test_without_dataset_skips_dataset_section(finished_run):  # noqa: F811
    _inject_residual(finished_run)
    md = story.render_report(finished_run, ds=None)
    assert "The dataset path is unknown" in md
    assert "## 4. What to do next" in md               # aggregate sections stay
    assert not list((finished_run / story.ASSET_DIR).glob("pattern_*"))


def test_cli_report_regenerates_story(finished_run):  # noqa: F811
    from agentic_configuration_search import cli
    (finished_run / story.REPORT_FILE).unlink()
    assert cli.main(["report", str(finished_run)]) == 0
    assert (finished_run / story.REPORT_FILE).exists()


def test_examples_and_samples_open_train_images_only(finished_run, monkeypatch):  # noqa: F811
    """Only train files are opened — diagnosis rows with a holdout name are not drawn."""
    from PIL import Image
    _inject_residual(finished_run)
    ds = _ds(finished_run)
    holdout_name = ds.holdout.images[0].file_name
    fin = finished_run / "rounds" / "final" / "diagnosis_train.json"
    d = json.loads(fin.read_text())
    d["fn"].append({"kind": "missed", "box": [10, 10, 30, 30], "category": "cat a",
                    "category_id": 1, "image": holdout_name, "size": "large"})
    fin.write_text(json.dumps(d))
    opened, real_open = [], Image.open

    def spy(fp, *a, **k):
        opened.append(Path(fp))
        return real_open(fp, *a, **k)
    monkeypatch.setattr(Image, "open", spy)
    md = story.render_report(finished_run, ds=ds)
    assert "could not be generated" not in md
    assert opened and all(p.parent == ds.train.root for p in opened), opened
    # photos drawn = captions (holdout rows are not picked)
    sec = md.split("### 3-1.")[1].split("###")[0]
    assert "3 examples" in sec and re.findall(r"^\d\. ", sec, re.M) == ["1. ", "2. ", "3. "]
    assert holdout_name not in md


def test_no_adoption_does_not_say_score_rose(finished_run):  # noqa: F811
    rows = [json.loads(x) for x in
            (finished_run / "history.jsonl").read_text().splitlines()][:1]
    (finished_run / "history.jsonl").write_text(json.dumps(rows[0]) + "\n")
    md = story.render_report(finished_run, ds=None)
    assert "rose" not in md and "the same as at the start" in md


def test_stale_final_diagnosis_is_flagged(finished_run):  # noqa: F811
    """When the last round's diagnosis stands in for a missing final one, it is not called "final"."""
    import shutil
    _inject_residual(finished_run)
    last = max(int(p.name[-3:]) for p in (finished_run / "rounds").glob("round_*"))
    rd = finished_run / "rounds" / f"round_{last:03d}"
    for f in ("diagnosis_train.json", "diagnosis_test.json"):
        shutil.copy(finished_run / "rounds" / "final" / f, rd / f)
    shutil.rmtree(finished_run / "rounds" / "final")
    rows = [json.loads(x) for x in
            (finished_run / "history.jsonl").read_text().splitlines()]
    rows[-1]["decision"]["adopted"] = True          # a run whose last round adopted
    (finished_run / "history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    md = story.render_report(finished_run, ds=None)
    assert md.count("no diagnosis record of the final result") == 2     # sections 3 and 4
    assert "the diagnosis at the start of the last round" in md         # at a glance
    assert "errors_before_after.png" not in md


def test_class_confusion_is_attributed_to_the_predicted_name(finished_run):  # noqa: F811
    _inject_residual(finished_run)
    p = finished_run / "rounds" / "final" / "diagnosis_test.json"
    d = json.loads(p.read_text())
    d["summary"]["fp"]["class_confusion"] = 4
    d["per_class"]["cat a"]["fp"] = {"class_confusion": 4}
    p.write_text(json.dumps(d))
    md = story.render_report(finished_run, ds=None)
    assert "Wrong name given: cat a 4" in md              # section 3 pattern
    d["summary"]["fp"]["class_confusion"] = 6
    d["per_class"]["cat a"]["fp"] = {"class_confusion": 6}
    p.write_text(json.dumps(d))
    md = story.render_report(finished_run, ds=None)
    assert "The wrong name given was cat a in 6 cases." in md   # section 4, CONFUSION_MIN or more


def test_sample_images_never_repeat_on_small_pools():
    from types import SimpleNamespace as NS
    imgs = [NS(file_name=f"{i}.png", annotations=[NS(category_id=1)]) for i in range(3)]
    for n in (1, 2, 3):
        ds = NS(train=NS(images=imgs[:n]))
        picks = story._sample_images(ds, 4)
        assert len(picks) == n == len({e.file_name for e in picks})


def test_size_wish_ignores_low_confidence_misses():
    """A size bucket with mostly low-confidence misses does not become "collect more of these"."""
    facets = {"gt": 100, "fn": 30, "size": {"small": [10, 10], "large": [90, 20]},
              "kind_size": {"below_threshold": {"small": 10, "large": 10},
                            "missed": {"large": 10}},
              "crowd": {"1": [10, 9]}, "border": [0, 0], "elongated": [0, 0]}
    wish = story._class_wishes("cat a", facets, [], {"small": "s", "medium": "m",
                                                    "large": "l"})
    assert not any(w.startswith("small") for w in wish)
    assert not any("scene" in w for w in wish)         # most misses are low confidence


def test_fallback_to_non_adopting_last_round_is_not_stale(finished_run):  # noqa: F811
    """If the last round adopted nothing, its diagnosis is the final result's."""
    import shutil
    _inject_residual(finished_run)
    last = max(int(p.name[-3:]) for p in (finished_run / "rounds").glob("round_*"))
    rd = finished_run / "rounds" / f"round_{last:03d}"
    for f in ("diagnosis_train.json", "diagnosis_test.json"):
        shutil.copy(finished_run / "rounds" / "final" / f, rd / f)
    shutil.rmtree(finished_run / "rounds" / "final")
    md = story.render_report(finished_run, ds=None)
    assert "no diagnosis record" not in md and "with the final result left" in md


def test_leftover_final_of_an_old_champion_is_not_current(finished_run):  # noqa: F811
    """After a resume, the new champion's final diagnosis failed and the old rounds/final is left."""
    _inject_residual(finished_run)
    key = finished_run / "rounds" / "final" / "diagnosis_key.json"
    k = json.loads(key.read_text())
    k["champion"]["id"] = "r0_old"
    key.write_text(json.dumps(k))
    rows = [json.loads(x) for x in
            (finished_run / "history.jsonl").read_text().splitlines()]
    rows[-1]["decision"]["adopted"] = True
    (finished_run / "history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    from agentic_configuration_search.diagnose.report import _final_diag, trajectory
    fd = _final_diag(finished_run, trajectory(finished_run))
    assert fd is None or fd["source"] == "round"


def test_grey_points_include_gate_passed_but_lost():
    hist = [{"round": 1, "decision": {"adopted": True, "id": "a", "evaluated": {
        "a": {"micro_f_beta": 0.9}, "b": {"micro_f_beta": 0.91, "gate": {"adopted": True}},
        "c": {"micro_f_beta": 0.5}}}},
            {"round": 2, "decision": {"adopted": False, "evaluated": {
                "d": {"micro_f_beta": 0.8}}}}]
    assert story.grey_points(hist) == [(1, 0.91), (1, 0.5), (2, 0.8)]


def test_every_stop_reason_has_plain_words():
    src = (PKG / "loop" / "runner.py").read_text()
    stops = set(re.findall(r'result\.stopped = "(\w+)"', src)) | {"saturation",
                                                                   "empty_rounds",
                                                                   "rounds"}
    assert stops <= set(story.STOP_PLAIN)


def test_class_wishes_name_skewed_buckets_with_counts():
    facets = {"gt": 200, "fn": 30,
              "size": {"small": [40, 20], "medium": [60, 5], "large": [100, 5]},
              "kind_size": {"missed_small": {"small": 20}, "missed": {"medium": 5,
                                                                     "large": 5}},
              "crowd": {"21+": [50, 15], "1": [150, 15]}, "border": [30, 12],
              "elongated": [0, 0]}
    notes = [{"category": "cat a", "kind": "missed", "note": "A dark, blurry object."}]
    wish = story._class_wishes("cat a", facets, notes, {"small": "s", "medium": "m",
                                                       "large": "l"})
    assert any(w.startswith("small cat a") and "20 of 40" in w for w in wish)
    assert any("crowded scenes" in w for w in wish)
    assert any("photo edge" in w for w in wish)
    assert wish[-1] == "photos where the object is blurry, dark"


def test_regenerate_with_config_survives_bad_dataset(finished_run):  # noqa: F811
    """With a broken dataset in --config, the report is still drawn."""
    from agentic_configuration_search.config import config_from_dict
    from agentic_configuration_search.loop.hooks import regenerate_report
    cfg = config_from_dict({"dataset": str(finished_run / "nope")})
    md = regenerate_report(finished_run, cfg).read_text()
    assert "The dataset path is unknown" in md and "could not be generated" not in md


def test_tied_kinds_are_not_named_twice():
    facts = {"classes": ["cat a", "cat b"], "size_ranges": {"small": "s"},
             "train": {"per_class": {"cat a": {"objects": 6}, "cat b": {"objects": 6}},
                       "images": 12, "empty": 0, "objects": 12, "size": {}, "border": 0}}
    line = story._dataset_bullets(facts)[0]
    assert "**cat a**" in line and "**cat b**" in line
