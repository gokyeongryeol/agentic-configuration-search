"""Round orchestration —
propose → materialize → screen(fit_dev) → evaluate(holdout) → decide → record.

Diagnosis is computed at the **start** of each round from the current
champion — equivalent to "end-of-round diagnosis feeds the next propose", and
it keeps resume simple.

rescorer candidates are wired by loop.rescore, lora candidates (YOLOE tail
fine-tune) by loop.tail_round, crop classifier candidates by loop.adapt_round.
The latter two are **detector-state candidates**: they swap the
in-process detector client's weights (weights_ctl) or reclassification
artifact (adapt_ctl), so they run only when `detector_hooks` is injected;
otherwise they are just recorded as `skipped`. They are handled **after all
other candidates are evaluated**, and the current state is restored if they
are not adopted.
"""

from __future__ import annotations

import dataclasses
import hashlib
from contextlib import contextmanager
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ..config import Config, config_from_dict
from ..core.data import Dataset, load_dataset, prescreen_subset, subset_split
from ..diagnose import describe, residual
from ..diagnose.errors import diagnose, size_profile
from ..eval.evaluate import evaluate_on_holdout, gate
from ..eval.runner import chosen_groups, infer_for_groups, setting_records
from ..eval.screen import fit_dev_gap, screen_candidate
from ..fitting import subsample
from ..fitting.materials import Materials
from ..fitting.tail import split_identity
from ..infer import gateway
from ..planner import deterministic
from ..planner.deterministic import tried_combos
from ..planner.gates import tiling_gate, tiling_shortfall
from ..planner.axes import AXES, BASELINE, DatasetFacts, sanitize_candidates
from ..planner.llm import CONTEXT_MARGIN, OUTPUT_MAX_TOKENS, LLMPlanner
from ..planner.ledger import champion_requests, lost_values, variant_of
from . import adapt_round, rescore, tail_round
from .materialize import materialize

log = logging.getLogger(__name__)

# Round diagnosis artifacts — file names as before (diagnosis_test is the legacy
# name of the holdout diagnosis)
DIAG_FILES = ("diagnosis_train.json", "diagnosis_test.json")
DIAG_KEY = "diagnosis_key.json"       # fingerprint of the diagnosis inputs — the next round reuses it if equal
FINAL_DIR = "final"                   # the final champion's diagnosis at the end of the run — rounds/final/
RESIDUAL = "residual.json"            # structure of residual misses (report only, diagnose.residual)
# run-level lora training cache (loop.tail_round._train_or_reuse) — never deleted while the run is alive
LORA_CACHE = "lora_cache"
# run-level crop classifier training cache (loop.adapt_round._train_or_reuse)
CLASSIFIER_CACHE = "classifier_cache"

@dataclass
class RunResult:
    out_dir: Path
    decisions: list = field(default_factory=list)
    history: list = field(default_factory=list)
    champion: dict | None = None
    stopped: str = "rounds"           # rounds | saturation | empty_rounds | planner_stop


@dataclass
class _RoundAcc:
    """State accumulated across a round's attempts (the first proposal + re-asks)."""
    served_w: object = None              # weights the detector has loaded now
    served_a: object = None              # reclassification artifact the detector has loaded now
    served_l: object = None              # that artifact's blend λ (None without an artifact)
    skipped: list = field(default_factory=list)
    screened: list = field(default_factory=list)
    evaluated: list = field(default_factory=list)
    n_runnable: int = 0                  # candidates that reached wiring (filter/wiring drops excluded)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "cand"


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# Per-image fit_dev (tp, fp, fn) saved with champion.json — the baseline for the
# next round's fit_dev comparison (fit_dev_cmp) and the lost-value prescreen. They
# must be saved with the champion for resume to reach the same verdict
_COUNT_KEYS = ("fit_dev_counts", "fit_dev_counts_raw")


def _setting_lite(setting: dict) -> dict:
    return {"configs": setting["configs"],
            "per_class": {str(k): v for k, v in setting["per_class"].items()}} \
        | {k: setting[k] for k in _COUNT_KEYS if setting.get(k) is not None}


def _setting_load(lite: dict) -> dict:
    return {"configs": lite["configs"],
            "per_class": {int(k): v for k, v in lite["per_class"].items()}} \
        | {k: lite[k] for k in _COUNT_KEYS if lite.get(k) is not None}


def _load_candidates_file(path, disabled_axes=(), disabled_values=None, *,
                          base: dict | None = None,
                          facts: DatasetFacts | None = None
                          ) -> tuple[list[dict], list[dict]]:
    """Human-injected candidates — (kept, dropped). Drops are recorded by the same
    rules as planner candidates (no silent drops — sanitize_candidates)."""
    if not path:
        return [], []
    p = Path(path)
    if not p.exists():
        return [], []
    raw = yaml.safe_load(p.read_text()) or []
    dropped: list[dict] = []
    cands = sanitize_candidates(raw, max_candidates=64, dropped=dropped,
                                disabled_axes=disabled_axes,
                                disabled_values=disabled_values,
                                base=base, facts=facts)
    for c in cands + dropped:
        c["id"] = f"human_{c['id']}"
        c["source"] = "human"
    return cands, dropped


class Loop:
    def __init__(self, cfg: Config, dataset_root, out_dir, *, client=None,
                 llm_chat=None, encode_fn=None, caption_fn=None,
                 vlm_caption_fn=None, refine_alias_fn=None,
                 refine_negatives_fn=None,
                 candidates_file=None, detector_hooks: dict | None = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.ds: Dataset = self._subsample_fit_train(
            load_dataset(dataset_root, seed=cfg.split.seed,
                         fit_train_ratio=cfg.split.fit_train_ratio), encode_fn)
        self.client = client
        self.llm_chat = llm_chat
        self.candidates_file = candidates_file
        self._mat_kw = dict(client=client, encode_fn=encode_fn,
                            caption_fn=caption_fn,
                            vlm_caption_fn=vlm_caption_fn,
                            refine_alias_fn=refine_alias_fn,
                            refine_negatives_fn=refine_negatives_fn)
        self.cat_ids = [c.id for c in self.ds.ontology.classes]
        self.iou = cfg.detector.iou_threshold
        # detector_hooks: {"weights_ctl": fn(weights|None), "adapt_ctl": fn(root|None),
        #             "repo_root": Path, "base_weights": Path}
        #   — without it, lora/crop_classifier candidates are skipped
        # controls that change the detector state empty the pass cache — the
        # same input becomes a different inference
        self.detector_hooks = (_invalidating_hooks(detector_hooks) if detector_hooks
                               else detector_hooks)
        # loaded classifier artifact and its blend λ — λ is the loop-side half of
        # the detector state (gateway holds it). A new process's client starts
        # without a classifier, so blending starts off too
        self._adapt_path = None
        self._adapt_lambda = None
        gateway.set_classifier_blend(None)
        # loaded tail weights (None = base) — unknown until the first swap; only
        # _weights_ctl changes them
        self._weights_path = None
        self._weights_synced = False
        # validated here because config is a leaf — no silent absorption of typos (see config.py)
        bad = sorted(set(cfg.search.disabled_axes) - set(AXES))
        if bad:
            raise ValueError(f"search.disabled_axes: unknown axes {bad} — "
                             f"allowed: {sorted(AXES)}")
        self.disabled_axes = tuple(cfg.search.disabled_axes)
        for axis, values in (cfg.search.disabled_values or {}).items():
            if axis not in AXES:
                raise ValueError(f"search.disabled_values: unknown axis {axis!r} "
                                 f"— allowed: {sorted(AXES)}")
            bad_v = [v for v in values if v not in AXES[axis]]
            if bad_v:
                raise ValueError(f"search.disabled_values.{axis}: values {bad_v} are "
                                 f"not in {AXES[axis]}")
        self.disabled_values = {k: tuple(v) for k, v in
                                (cfg.search.disabled_values or {}).items() if v}

    # ------------------------------------------------------------- state

    def _round_dir(self, r: int) -> Path:
        return self.out_dir / "rounds" / f"round_{r:03d}"

    def _save_champion(self, champion: dict) -> None:
        _write(self.out_dir / "champion.json",
               {k: champion[k] for k in ("round", "id", "axes", "materials_dir")}
               | {"setting": _setting_lite(champion["setting"]),
                  "micro_f_beta": champion["eval"]["micro_f_beta"],
                  "weights": champion.get("weights"),
                  "adapt_root": champion.get("adapt_root"),
                  "adapt_lambda": champion.get("adapt_lambda"),
                  "rescorer_path": champion.get("rescorer_path"),
                  "forwards_per_image": setting_forwards(champion["setting"]),
                  "fit_dev_micro_f_beta":
                      champion["setting"].get("fit_dev_micro_f_beta")})
        _write(self.out_dir / "champion_eval.json", champion["eval"])

    def _load_champion(self) -> dict:
        meta = json.loads((self.out_dir / "champion.json").read_text())
        ev = json.loads((self.out_dir / "champion_eval.json").read_text())
        ev["tau_test"] = {int(k): v for k, v in ev["tau_test"].items()}
        setting = _setting_load(meta["setting"])
        setting["fit_dev_micro_f_beta"] = meta["fit_dev_micro_f_beta"]
        return {"round": meta["round"], "id": meta["id"], "axes": meta["axes"],
                "setting": setting,
                "materials_dir": meta["materials_dir"], "eval": ev,
                "weights": meta.get("weights"),
                "adapt_root": meta.get("adapt_root"),
                "adapt_lambda": meta.get("adapt_lambda"),
                "rescorer_path": meta.get("rescorer_path")}

    def _append_history(self, entry: dict) -> None:
        # timestamps for analysis — ts is write time, dur_s is round wall-clock duration
        entry.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        if getattr(self, "_round_t0", None) is not None:
            entry.setdefault("dur_s", round(time.monotonic() - self._round_t0, 1))
        pc = _pass_cache_delta(getattr(self, "_pass_t0", None),
                               gateway.pass_cache_stats())
        if pc is not None:
            entry.setdefault("pass_cache", pc)
            log.info("round %s pass cache: hit %d · miss %d (hit rate %.0f%%) · "
                     "invalidate %d · %d pieces held", entry.get("round"), pc["hits"],
                     pc["misses"], 100 * pc["hit_rate"], pc["invalidations"],
                     pc["entries"])
        with open(self.out_dir / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _read_history(self) -> list[dict]:
        p = self.out_dir / "history.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------- stages

    def _evaluate(self, setting: dict, materials: Materials) -> dict:
        return evaluate_on_holdout(self.cfg, self.client, self.ds.holdout,
                                self.ds.ontology, setting, materials)

    def _evaluate_candidate(self, entry: dict, setting: dict,
                            materials: Materials) -> dict:
        """Holdout evaluation for the gate — measured on **exactly the score
        channel the candidate will serve**.

        The rescorer is a client-side calibration layer, so skipping it in
        evaluation changes the score channel and the paired Δ becomes
        "candidate ± calibration layer" instead of the candidate's increment.
        If the candidate re-chose the classifier λ (_choose_lambda), it is
        measured with that λ. The branch lives in this one place so call sites
        cannot drift apart.
        """
        with self._lambda_for(entry):
            if entry.get("_rescorer") is not None:
                return rescore.rescored_evaluate(self.cfg, self.client, self.ds,
                                                 setting, materials,
                                                 entry["_rescorer"])
            return self._evaluate(setting, materials)

    @contextmanager
    def _lambda_for(self, entry: dict):
        """Apply the classifier λ the candidate re-chose for this block — the
        artifact stays loaded, only the blend changes, and afterwards it returns
        to the detector state's λ (self._adapt_lambda)."""
        lam = entry.get("_adapt_lambda")
        if lam is None or self._adapt_path is None or lam == self._adapt_lambda:
            yield
            return
        gateway.set_classifier_blend(lam)
        try:
            yield
        finally:
            gateway.set_classifier_blend(self._adapt_lambda)

    def _choose_lambda(self, materials, axes: dict, entry: dict, setting0: dict) -> dict:
        """If the champion has a classifier, re-choose the blend λ on this
        candidate's setting.

        The blend mixes into detection scores, so when something upstream
        changes (exemplar, negatives, tiling, …) the best λ can change too — it
        is refit per candidate, like the rescorer. Order: λ first, then the
        rescorer refit on top (the rescorer takes the blended scores and
        relabelled classes as input). The artifact stays loaded, so detector
        responses are the same and the pass cache serves the re-inference. The
        champion λ's setting (setting0) is the reference, and λ changes only if
        the best beats it by more than crop_classifier.margin — so λ does not
        drift with noise on the same fit_dev. The chosen λ stays in
        entry["_adapt_lambda"] for the rescorer refit, holdout, served
        fingerprint and adoption. Called for ordinary candidates (after screen)
        and lora candidates (on the winning arm). The lora candidate's own λ
        (weight interpolation) is not re-chosen after adoption."""
        cc = self.cfg.crop_classifier
        base = self._adapt_lambda
        if not cc.retune_lambda or self._adapt_path is None or base is None:
            return setting0
        f0 = setting0["fit_dev_micro_f_beta"]
        grid = [{"lam": base, "f_beta": f0}]
        settings = {base: setting0}
        for lam in sorted({float(x) for x in cc.lambdas} - {base}):
            gateway.set_classifier_blend(lam)
            try:
                s = self._screen(materials, axes)
            finally:
                gateway.set_classifier_blend(base)
            grid.append({"lam": lam, "f_beta": s["fit_dev_micro_f_beta"]})
            settings[lam] = s
        best = max(grid, key=lambda g: (g["f_beta"], g["lam"] == base, -g["lam"]))
        if best["f_beta"] - f0 <= cc.margin:
            best = grid[0]
        entry["_adapt_lambda"] = best["lam"]
        entry["classifier_lambda_grid"] = grid
        if best["lam"] != base:
            log.info("candidate %s: classifier λ %g → %g (fit_dev %.4f → %.4f)",
                     entry.get("id"), base, best["lam"], f0, best["f_beta"])
        return settings[best["lam"]]

    def _subsample_fit_train(self, ds: Dataset, encode_fn) -> Dataset:
        """If fit_train is larger than subsample.max_images, swap the training
        material for a subset.

        Only ds.fit_train changes — what lora, the rescorer discriminator and
        the crop classifier read. train (mining), fit_dev (verdicts) and holdout
        (adoption) stay as they are. The picked images are recorded in the run
        directory and resume reads them back (key = fit_train fingerprint and
        settings — a different key picks again). The training reuse keys hold
        the fit_train fingerprint, so a changed subset also breaks reuse."""
        sc = self.cfg.subsample
        n = len(ds.fit_train.images)
        self._fit_train_before, self._fit_train_subset = n, None
        if sc.max_images <= 0 or n <= sc.max_images:
            return ds
        key = {"fit_train": split_identity(ds.fit_train), "max_images": sc.max_images,
               "center_ratio": sc.center_ratio}
        path = self.out_dir / subsample.SUBSET_FILE
        rec = json.loads(path.read_text()) if path.exists() else {}
        if rec.get("key") != key:
            if encode_fn is None:
                raise ValueError(
                    f"fit_train has {n} images > subsample.max_images {sc.max_images} — "
                    "picking the subset needs GT embeddings (encode_fn) "
                    "(subsample.max_images: 0 turns it off)")
            emb = subsample.fit_train_embeddings(
                ds.fit_train, ds.ontology, encode_fn, self.out_dir / "_subsample" / "gt_crops")
            idx, report = subsample.select_fit_train(
                ds.fit_train, emb, max_images=sc.max_images, center_ratio=sc.center_ratio)
            rec = {"key": key, "report": report,
                   "images": [ds.fit_train.images[i].file_name for i in idx]}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        self._fit_train_subset = rec["report"]
        names = set(rec["images"])
        keep = {i for i, e in enumerate(ds.fit_train.images) if e.file_name in names}
        log.info("fit_train subset: %d/%d images (%d background; the first pass's %d "
                 "images cover %.0f%% of centers) — %s",
                 len(keep), n, rec["report"]["n_background"],
                 rec["report"]["first_pass_images"],
                 100 * rec["report"]["first_pass_coverage"], path)
        return dataclasses.replace(ds, fit_train=subset_split(ds.fit_train, keep))

    def fit_train_summary(self) -> dict:
        """run_summary's fit_train record — images used for training and the
        subset selection record (None without one)."""
        return {"images": len(self.ds.fit_train.images),
                "images_before_subset": self._fit_train_before,
                "subset": self._fit_train_subset}

    def _adapt_ctl(self, path, lam: float | None = None) -> None:
        """Load/unload the classifier artifact and set its blend λ — recording
        the detector state.

        The artifact is reloaded only when it changes (the hook also empties the
        pass cache). λ is the loop-side blend (gateway.set_classifier_blend), so
        detector responses stay the same and an arm grid that only changes λ
        runs without reloading (re-inference is served by the pass cache). λ=0
        is relabel-only and differs from None (no classifier)."""
        if path is not None and lam is None:
            raise ValueError("loading a classifier needs a blend λ")
        if not _same_state(path, self._adapt_path):
            self.detector_hooks["adapt_ctl"](path)
        self._adapt_path = path
        self._adapt_lambda = lam if path is not None else None
        gateway.set_classifier_blend(self._adapt_lambda)

    def _weights_ctl(self, weights) -> None:
        """Swap the detector weights (None = base) — skipped when that path is
        already loaded.

        The hook empties the pass cache on every call (_invalidating_hooks). A
        lora candidate starts on the champion state (_serve_champion) and then
        run_tail_candidate asks for the same weights again as its baseline —
        that would throw away the cache filled on the champion state before the
        baseline screen. Arm descriptors are written fresh per candidate
        directory, so one path never holds different contents. At first and
        after a failed swap the loaded state is unknown, so the same path is
        swapped in again."""
        if self._weights_synced and _same_state(weights, self._weights_path):
            return
        self._weights_synced = False
        self.detector_hooks["weights_ctl"](weights)
        self._weights_path = weights
        self._weights_synced = True

    def _restore_detector_state(self, champion: dict) -> None:
        """On resume, re-apply the champion's detector state (weights and
        reclassification artifact) to the in-process client.

        A new process's client starts with base weights and no reclassifier.
        Without re-applying them, candidates are scored with a different
        detector than the champion, and the gate's Δ against the champion's
        recorded records is silently wrong.
        """
        if not self.detector_hooks:
            return                        # run without client state controls
        weights = champion.get("weights")
        adapt_root = champion.get("adapt_root")
        if weights and self.detector_hooks.get("weights_ctl"):
            log.info("resume: restored champion weights %s", weights)
            self._weights_ctl(Path(weights))
        if adapt_root and self.detector_hooks.get("adapt_ctl"):
            if champion.get("adapt_lambda") is None:
                raise RuntimeError(
                    f"champion.json has adapt_root ({adapt_root}) but no adapt_lambda — "
                    "the run predates the classifier blend λ. The client would serve "
                    "argmax reclassification and different predictions, so it cannot "
                    "be continued — start a new run")
            log.info("resume: restored champion reclassification artifact %s (λ=%s)",
                     adapt_root, champion.get("adapt_lambda"))
            self._adapt_ctl(Path(adapt_root), champion.get("adapt_lambda"))

    def _screen(self, materials: Materials, axes: dict, split=None) -> dict:
        """Fix the candidate setting on fit_dev (or a subset of it — prescreen)."""
        s = self.cfg.search
        # text-only candidate: the text pass sweeps the text-negative suppression
        # IoU (neg_eta) over the grid (text-only branch of screen_candidate). With
        # exemplars, semantic picks the grid point and text follows that config.
        text_neg_sweep = (axes.get("text_negatives")
                          and axes.get("exemplar") == "none")
        return screen_candidate(
            self.cfg, self.client, self.ds.fit_dev if split is None else split,
            self.ds.ontology, materials,
            neg_eta_grid=(s.neg_eta_grid
                          if (axes.get("visual_negatives") or text_neg_sweep)
                          else ()),
            tiling_variants=_tiling_variants(s, axes))

    def _diagnosis_key(self, champion: dict) -> dict:
        """Fingerprint of the diagnosis inputs — the champion (its whole served
        state and holdout thresholds) and the config.

        The config is hashed whole except rounds and out_dir, which do not reach
        the diagnosis — picking which keys reach it (inference, cause
        classification, crop notes) and missing one would reuse a stale
        diagnosis, so the key errs wide and recomputes when in doubt."""
        cfg = dataclasses.asdict(self.cfg)
        for k in ("rounds", "out_dir"):
            cfg.pop(k, None)
        ev = champion["eval"]
        key = {"champion": {k: champion.get(k) for k in
                            ("round", "id", "materials_dir", "weights",
                             "adapt_root", "adapt_lambda", "rescorer_path")}
               | {"micro_f_beta": ev["micro_f_beta"], "tau_test": ev["tau_test"]},
               "config": hashlib.sha256(json.dumps(
                   cfg, sort_keys=True, default=str).encode()).hexdigest()}
        # normalize through a JSON round trip — compare in the stored key's shape
        # (string keys, paths)
        return json.loads(json.dumps(key, sort_keys=True, default=str))

    def _reuse_diagnosis(self, key: dict, round_dir: Path,
                         prev: Path | None) -> dict | None:
        """Use an existing diagnosis of the same inputs — this round's directory
        first (resuming an interrupted round), then the previous round's. From
        the previous round, the diagnosis, crop notes and key are copied into
        this round's directory (_crop_notes replays the notes). Rounds under an
        unchanged champion would otherwise repeat the full train inference and
        the VLM notes — the diagnosis is a function of the champion and config
        alone. When a finished run continues with more rounds, the end-of-run
        final diagnosis (rounds/final) is a reuse candidate too."""
        final = self.out_dir / "rounds" / FINAL_DIR
        for src in (round_dir, prev, final if final != round_dir else None):
            if src is None or not (src / DIAG_KEY).exists() or not all(
                    (src / name).exists() for name in DIAG_FILES):
                continue
            if json.loads((src / DIAG_KEY).read_text()) != key:
                continue
            if src != round_dir:
                # the key goes last — dying mid-copy leaves no key, so it is recomputed
                for name in DIAG_FILES + (describe.ARTIFACT, DIAG_KEY):
                    if (src / name).exists():
                        shutil.copyfile(src / name, round_dir / name)
                    elif name == describe.ARTIFACT:
                        (round_dir / name).unlink(missing_ok=True)
            log.info("diagnosis reused: %s — same champion %s (round %s) and config",
                     src.name, key["champion"]["id"], key["champion"]["round"])
            return {"train": json.loads((round_dir / DIAG_FILES[0]).read_text()),
                    "test": json.loads((round_dir / DIAG_FILES[1]).read_text())}
        return None

    def _diagnosis(self, champion: dict, round_dir: Path,
                   prev: Path | None = None) -> dict:
        """Diagnosis on train (detailed) + holdout (anonymized aggregates) — the
        planner's input. (Artifact key/file name "test" is legacy.)

        If a diagnosis with the same inputs (_diagnosis_key) exists in this round
        or the previous one (prev), it is not recomputed (_reuse_diagnosis)."""
        key = self._diagnosis_key(champion)
        reused = self._reuse_diagnosis(key, round_dir, prev)
        self._diag_reused = reused is not None
        if reused is not None:
            return reused
        # with no key, or notes from another diagnosis left over, delete them so
        # they are not replayed
        (round_dir / describe.ARTIFACT).unlink(missing_ok=True)
        (round_dir / DIAG_KEY).unlink(missing_ok=True)
        tau = champion["eval"]["tau_test"]
        train = self.ds.train
        paths = [train.image_path(e) for e in train.images]
        grid = {c["config_id"]: c for c in champion["setting"]["configs"]}
        groups = chosen_groups(champion["setting"]["per_class"])
        materials = Materials.load(champion["materials_dir"])
        if champion.get("rescorer_path"):
            # the champion's score channel is the rescored one — diagnosis must use
            # the same channel for below_threshold classification to match tau_test
            from ..fitting.rescorer import Rescorer, dataset_fingerprint
            rescorer = Rescorer.load(
                Path(champion["rescorer_path"]),
                expect=dataset_fingerprint(self.ds.root.name, self.ds.ontology))
            setting = champion["setting"]
            preds, _ = rescore.split_preds(self.cfg, self.client, self.ds,
                                           setting, materials, train, "lb")
            preds = rescore.apply_to_preds(self.cfg, self.ds, setting,
                                           rescorer, preds)
            floor = rescore.dets_by_image(train, preds)
            from ..core.metrics import per_image_records
            floor_recs = per_image_records(train, paths, floor)
            kept = [[d for d in dets
                     if d["category_id"] not in tau
                     or d["score"] >= tau[d["category_id"]]] for dets in floor]
            recs = per_image_records(train, paths, kept)
        else:
            passes = infer_for_groups(self.cfg, self.client, groups, grid,
                                      self.ds.ontology, paths, materials)
            floor_recs, recs = setting_records(train, paths, passes, groups, tau)
        small = dict(small_ratio=self.cfg.diagnose.small_object_ratio)
        d_train = diagnose(recs, floor_recs, tau, self.ds.ontology,
                           iou_thr=self.iou, anonymize=False, **small)
        # material for s* and m of the tiling near-miss — the floor is recounted at
        # the champion's fit_dev screen thresholds, not tau_test (holdout-
        # estimated). s* feeds the fold gate's exception (a screen decision), so
        # nothing derived from holdout may enter it.
        screen_thr = {int(cid): e["threshold"]
                      for cid, e in champion["setting"]["per_class"].items()}
        d_train["tiling_sizes"] = size_profile(floor_recs, screen_thr,
                                               self.ds.ontology,
                                               iou_thr=self.iou, **small)
        d_test = diagnose(champion["eval"]["records"],
                          champion["eval"]["floor_records"], tau,
                          self.ds.ontology, iou_thr=self.iou, anonymize=True,
                          **small)
        _write(round_dir / DIAG_FILES[0], d_train)
        _write(round_dir / DIAG_FILES[1], d_test)
        _write(round_dir / DIAG_KEY, key)          # last — only once the diagnosis is fully written
        return {"train": d_train, "test": d_test}

    def final_diagnosis(self, champion: dict, history: list[dict]) -> dict:
        """The final champion's diagnosis at the end of the run — rounds/final/
        (diagnosis, crop notes, residual).

        A round's diagnosis is of the champion at the **start** of the round, so
        when the last round adopts, the final champion's residual errors are
        recorded nowhere. This goes through the same _diagnosis path, so if the
        last round adopted nothing it only copies that diagnosis (no inference);
        if it adopted, it is one train inference plus notes. Resume reuses this
        directory itself, so it is idempotent. residual is computed from GT
        annotations and the diagnosis alone (no inference).

        Report-only, so a failure does not kill the run — it is recorded as an
        error in run_summary. Returns run_summary's final_diagnosis key."""
        t0 = time.monotonic()
        rd = self.out_dir / "rounds" / FINAL_DIR
        last = history[-1]["round"] if history else 0
        info = {"dir": f"rounds/{FINAL_DIR}", "champion_round": champion.get("round"),
                "after_round": last}
        try:
            rd.mkdir(parents=True, exist_ok=True)
            diagnosis = self._diagnosis(
                champion, rd, prev=self._round_dir(last) if last >= 1 else None)
            info["reused"] = self._diag_reused
            self._crop_notes(diagnosis, rd)
            ev = champion["eval"]
            _write(rd / RESIDUAL, residual.residual(
                residual.image_rows(self.ds.train), diagnosis["train"],
                ev["records"], ev.get("floor_records"), ev["tau_test"],
                self.ds.ontology, iou_thr=self.iou,
                small_ratio=self.cfg.diagnose.small_object_ratio))
        except Exception as exc:  # noqa: BLE001 — a report-only diagnosis must not kill the run
            log.exception("final champion diagnosis failed — the report falls back to the "
                          "last round's diagnosis")
            info["error"] = f"{type(exc).__name__}: {exc}"
        info["dur_s"] = round(time.monotonic() - t0, 1)
        return info

    def _planner_chat(self):
        """Chat callable shared by the llm planner and crop notes — the injected
        one first, else an LLMClient built from base_url, else None."""
        if self.llm_chat is not None:
            return self.llm_chat
        if not self.cfg.planner.base_url:
            return None
        from ..clients.llm import LLMClient
        client = LLMClient(self.cfg.planner.base_url, self.cfg.planner.model)
        return lambda messages, **kw: client.chat_json(messages, **kw)

    def _crop_notes(self, diagnosis: dict, round_dir: Path) -> None:
        """Merge VLM crop notes into diagnosis["train"] — diagnostic signal only.
        Failures degrade to no notes and the round continues."""
        if self.cfg.planner.kind != "llm" or not self.cfg.planner.crop_notes:
            return
        chat = self._planner_chat()
        if chat is None or not isinstance(diagnosis.get("train"), dict):
            return
        try:
            notes = describe.crop_notes(self.ds.train, diagnosis["train"],
                                        chat, round_dir / describe.ARTIFACT)
        except Exception:  # noqa: BLE001 — a notes failure must not kill the round
            log.exception("crop description failed — continuing the round without descriptions")
            return
        if notes:
            diagnosis["train"]["crop_notes"] = notes

    def _llm_planner(self, round_dir: Path) -> LLMPlanner:
        chat = self._planner_chat()
        if chat is None:
            raise ValueError("planner.kind=llm requires planner.base_url or "
                             "an injected llm_chat")
        return LLMPlanner(chat=chat, artifact_dir=round_dir,
                          max_candidates=self.cfg.planner.max_candidates,
                          notes=self.cfg.dataset_notes,
                          token_budget=_token_budget(self.cfg),
                          disabled_axes=self.disabled_axes,
                          disabled_values=self.disabled_values,
                          facts=self._facts(),
                          tiling_overlap=self.cfg.search.tiling_overlap,
                          tiling_tp_quantile=self.cfg.search.tiling_tp_quantile,
                          tiling_tp_min=self.cfg.search.tiling_tp_min,
                          budget=getattr(self, "_budget", None),
                          prescreen=self._prescreen_info())

    def _prescreen_info(self) -> dict:
        """The lost-value prescreen settings the planner prompt's allowed moves
        carry — the subset size (without one — small, or max_share or more — the
        full fit_dev)."""
        s = self.cfg.screen
        n_all = len(self.ds.fit_dev.images)
        sub = prescreen_subset(self.ds.fit_dev, s.prescreen_fraction,
                               s.prescreen_min_images,
                               max_share=s.prescreen_max_share)
        return {"on": bool(s.lost_value_prescreen), "of": n_all,
                "n_images": len(sub.images) if sub is not None else n_all}

    def _propose(self, history: list[dict], diagnosis: dict,
                 round_dir: Path) -> dict:
        if self.cfg.planner.kind == "llm":
            proposal = self._llm_planner(round_dir).propose(history, diagnosis)
        else:
            proposal = deterministic.propose(history, diagnosis,
                                             max_candidates=self.cfg.planner.max_candidates,
                                             disabled_axes=self.disabled_axes,
                                             disabled_values=self.disabled_values,
                                             facts=self._facts())
            _write(round_dir / "proposal.json", proposal)
        # merge human-injected candidates (hybrid loop)
        humans, human_dropped = _load_candidates_file(
            self.candidates_file, self.disabled_axes, self.disabled_values,
            base=(history[-1].get("champion", {}).get("axes") if history else None),
            facts=self._facts())
        # candidates that passed the gate last round but lost to the winner —
        # carried onto the new champion whatever the planner (the shared filter
        # below still applies)
        carried = deterministic.carryover(history, self.disabled_axes,
                                          self.disabled_values, facts=self._facts())
        fresh, dropped = self._filter(proposal["candidates"] + humans + carried,
                                      history)
        # merge with planner-stage drops (invalid_axes etc.) — overwriting would
        # erase candidates discarded by sanitization from history.
        return {**proposal, "candidates": fresh,
                "dropped": (list(proposal.get("dropped") or [])
                            + human_dropped + dropped),
                "stop": proposal["stop"] and not fresh}

    def _reask(self, k: int, r: int, history: list[dict], diagnosis: dict,
               round_dir: Path, champion: dict, acc: "_RoundAcc") -> dict | None:
        """The k-th re-ask of the round — when every candidate so far was rejected
        before holdout. Diagnosis and champion are this round's.

        Re-asked candidates go through the shared filter too. The history it
        sees gets this round's rows so far appended — a combination proposed
        earlier in the round is combo_tried, and a grid at or above one just
        left unused is tiling_blocked. Human and carried candidates already ran
        with the first proposal and are not added again. Candidate and drop rows
        get reask=k.

        The deterministic planner is not re-asked (None) — it is the fixed path
        and the A/B control. Neither is the LLM when its first call failed and
        there is no conversation to continue."""
        if self.cfg.planner.kind != "llm":
            return None
        rows = acc.skipped + [{k_: c[k_] for k_ in ("id", "axes")} | _requested(c)
                              | _unused_record(c) | _tiling_record(c)
                              | _ledger_record(c) | _skip_record(c)
                              for c in acc.screened]
        # screen rows dropped by the pre-holdout fit_dev verdict are reported as rejections too
        outcomes = acc.skipped + [r for r in rows[len(acc.skipped):] if r.get("skipped")]
        proposal = self._llm_planner(round_dir).reask(
            history, diagnosis, outcomes, k=k, taken={c["id"] for c in rows})
        if proposal is None:              # no conversation to continue (first call failed)
            return None
        for c in proposal["candidates"] + proposal["dropped"]:
            c["reask"] = k
        so_far = history + [{"round": r, "champion": _champion_summary(champion),
                             "candidates": rows}]
        fresh, dropped = self._filter(proposal["candidates"], so_far)
        log.info("round %d: no candidate reached holdout, re-asked (%d) — %d new candidates",
                 r, k, len(fresh))
        return _with_round_ids({**proposal, "candidates": fresh,
                                "dropped": proposal["dropped"] + dropped,
                                "stop": proposal["stop"] and not fresh},
                               r, {c["id"] for c in rows})

    def _filter(self, merged: list[dict],
                history: list[dict]) -> tuple[list[dict], list[dict]]:
        """(kept, dropped) — the source-independent shared filter
        (tiling_blocked, champion_combo, combo_tried)."""
        # Common combo filter: regardless of source (LLM, deterministic, human),
        # block retrying **an already-tried full axes combination**. It works per
        # combination, not per (axis, value), so retrying the same axis after the
        # champion changes is allowed. The LLM can forget failures outside the
        # prompt window (MAX_HISTORY), so this filter is the only permanent barrier.
        tried = tried_combos(history)
        gate = tiling_gate(history)
        champ_axes = (history[-1].get("champion", {}).get("axes")
                      if history else None) or BASELINE
        # the current champion itself — its label (axes) and the request that
        # adopted it (requested_axes)
        itself = {tuple(sorted(a.items()))
                  for a in [champ_axes, *champion_requests(history)] if a}
        # Values that clearly lost on fit_dev under the current champion
        # (planner.ledger) — candidates containing one are not blocked, only marked
        # (lost_values). Screen measures them on a subset first
        lost = lost_values(history) if self.cfg.screen.lost_value_prescreen else {}
        fresh, dropped = [], []
        for c in merged:
            # tiling fold gate (planner.gates) — screening a grid at or above an
            # unused one costs 1+g² forwards per image. Only planner candidates are
            # blocked: a person decides for human candidates, and carried ones
            # already passed the holdout gate.
            why = (gate.block(c["axes"])
                   if c.get("source") not in ("human", "carryover") else None)
            if why:
                dropped.append({**c, "skipped": "tiling_blocked", "note": why})
                continue
            key = tuple(sorted(c["axes"].items()))
            if key in itself:
                # re-proposes the champion — changes nothing. Kept apart from the
                # plain combo dedup so the round gets one more re-ask without
                # counting it (run_round)
                dropped.append({**c, "skipped": "champion_combo",
                                "note": "the current champion itself (its axes or the "
                                        "request that adopted it) — changes nothing"})
                tried.add(key)
                continue
            if key in tried:
                # no silent drops — record as combo_tried in skipped so it flows via
                # history into the next round's prompt (learning about out-of-window re-proposals).
                dropped.append({**c, "skipped": "combo_tried",
                                "note": "axes combination already tried — combination-level dedup "
                                        "blocks it permanently. Vary it or try another axis."})
                continue
            tried.add(key)                # also collapses duplicates within one proposal
            # Planner candidates only — human and carried candidates are left alone
            # for the same reason as the tiling gate. A candidate newly turning on a
            # detector-state axis trains before screening, so a subset screen has
            # nothing to save
            if (lost and c.get("source") not in ("human", "carryover")
                    and _route_detector_state(c["axes"], champ_axes) == "wired"):
                names = variant_of(c["axes"], champ_axes, lost)
                if names:
                    c = {**c, "lost_values": names}
            fresh.append(c)
        if dropped:
            log.info("filtered %d already-tried combinations: %s", len(dropped),
                     [c["id"] for c in dropped])
        return fresh, dropped

    def _facts(self) -> DatasetFacts:
        """Facts that let sanitization block candidates whose outcome the data already decides."""
        return DatasetFacts(n_classes=len(self.cat_ids),
                            n_fit_train=len(self.ds.fit_train.images),
                            lora_min_images=self.cfg.lora.min_images)

    def _guard_material(self, c: dict, materials, skipped: list) -> bool:
        """If materials are empty, record in skipped and return True — the caller drops the candidate."""
        axis = _empty_material(c["axes"], materials, self.cfg.mine.text_aliases)
        if axis is None:
            return False
        log.info("candidate %s: %s material is empty, same as baseline — rejected", c["id"], axis)
        # empty_axis is what the planner uses to collect "axes whose materials are
        # empty on this dataset" regardless of the history window
        # (planner.llm._dead_axis_lines) — so it never has to parse the note text.
        skipped.append({**c, "skipped": "no_material", "empty_axis": axis,
                        "note": f"{axis} mining came back empty — this combination is identical "
                                f"to the current one, so it is not evaluated"})
        return True

    def _parent_kw(self) -> dict:
        """materialize kwargs for inheriting inherited axes' materials from the
        champion. The champion's score was achieved with those materials, so
        they are not re-mined."""
        ch = getattr(self, "_champion", None)
        if not ch or not ch.get("materials_dir"):
            return {}
        return {"parent": ch["materials_dir"], "parent_axes": ch.get("axes")}

    def _arm_rescorer(self, entry: dict, setting: dict, materials: Materials,
                      mat_dir) -> dict:
        """Refit the rescorer **on the new distribution** produced by an upstream arm.

        The rescorer is calibrated to the detector's score distribution —
        detection score is a feature and labels come from category agreement —
        so changing upstream changes the fit target. Measuring with it removed
        would make the gate's Δ a "+upstream, −rescorer" composite, so the
        upstream axis would fight carrying a handicap equal to the rescorer's gain.

        If the refit ends up identity (α=0 for every class), axes.rescorer is
        corrected to False so the label matches the actually measured state.
        """
        if not entry["axes"].get("rescorer"):
            return setting
        rescorer, _info, dev_preds = rescore.fit_for_setting(
            self.cfg, self.client, self.ds, setting, materials, mat_dir)
        if rescorer is None or not rescorer.active:
            log.info("candidate %s: rescorer re-fit on the arm is identity — label corrected",
                     entry["id"])
            # once axes is turned off, "requested but identity" and "never
            # requested" are indistinguishable — record the attempt separately (read by the decision)
            entry["rescorer_refit"] = "identity"
            entry["axes"] = _measured_axes(entry["axes"], rescorer=False)
            return setting
        entry["_rescorer"] = rescorer
        entry["rescorer_path"] = str(Path(mat_dir) / "rescorer.json")
        return rescore.rescored_screen(self.cfg, self.ds, setting,
                                       rescorer, dev_preds)

    def _shortfall(self, c: dict, entry: dict) -> None:
        """Attach the magnification-shortfall verdict (tiling_shortfall) to a
        candidate asking for a grid above the champion's — so the fold gate's
        near-miss exception reads history only, it is computed at screen time
        from this round's champion train size profile (`tiling_sizes` — TP and
        missed_small counted at the fit_dev screen thresholds, holdout-free)."""
        g = (c.get("requested_axes") or c["axes"]).get("tiling")
        champ = (getattr(self, "_champion", None) or {}).get("axes") or {}
        if not g or int(g) <= int(champ.get("tiling") or 0):
            return
        s = self.cfg.search
        verdict = tiling_shortfall(getattr(self, "_diag_train", None), int(g),
                                   overlap=s.tiling_overlap,
                                   quantile=s.tiling_tp_quantile,
                                   tp_min=s.tiling_tp_min)
        if verdict is not None:
            entry["tiling_shortfall"] = verdict

    def _fit_dev_cmp(self, setting: dict | None, *, raw: bool = False,
                     split=None) -> dict | None:
        """Delta and one-sided paired-bootstrap upper bound of (candidate −
        champion) fit_dev micro F(β) (eval.screen.fit_dev_gap) — the history
        row's `fit_dev_cmp`.

        With raw, compare against the champion's pre-rescorer channel (the
        prescreen of a candidate inheriting the rescorer runs before the refit).
        If split is a subset of fit_dev, the champion rows are cut down to its
        images. Reads fit_dev only — the champion's rows were also counted by
        screen at fit_dev thresholds. None if there are no rows to compare. If
        every per-image row equals the champion's, `identical: True` is set (the
        pre-holdout verdict — _fit_dev_skip)."""
        champ = (getattr(self, "_champion", None) or {}).get("setting") or {}
        base = ((champ.get("fit_dev_counts_raw") or champ.get("fit_dev_counts"))
                if raw else champ.get("fit_dev_counts"))
        rows = (setting or {}).get("fit_dev_counts")
        if not base or not rows:
            return None
        if split is not None:
            index = {e.file_name: i for i, e in enumerate(self.ds.fit_dev.images)}
            base = [base[index[e.file_name]] for e in split.images]
        if len(base) != len(rows):
            return None
        g = self.cfg.gate
        out = fit_dev_gap(base, rows, beta=self.cfg.beta, n_boot=g.bootstrap,
                          seed=self.cfg.run_seed, alpha=1.0 - g.confidence)
        # Identity is only an exact match of every per-image (tp, fp, fn) row — a
        # rounded delta and ub of 0 cannot tell "the same" from "very close"
        if _as_rows(base) == _as_rows(rows):
            out["identical"] = True
        return out

    def _prescreen(self, c: dict, axes: dict, materials: Materials, entry: dict,
                   skipped: list) -> bool:
        """Screen on a fit_dev subset first — True if rejected. Two cases share
        the same subset screen.

        (1) Large-grid tiling: a grid g screen costs 1+g² forwards, so from g ≥
        screen.prescreen_min_grid the full fit_dev is expensive. If no class uses
        a tile, reject as axis_unused. Only for single increments that change
        tiling alone and raise the grid — a candidate that also changes other
        axes, or lowers the champion's grid, is not the champion even if tiles
        go unused. Not done without a subset (fit_dev is
        small, or the subset is screen.prescreen_max_share of fit_dev or more).

        (2) Lost value: a candidate containing a value that clearly lost on
        fit_dev under the current champion (lost_values, set by _filter). If the
        bootstrap upper bound of (candidate − champion) micro F(β) is below 0,
        reject as lost_value_prescreen — no point-estimate win is required (the
        same bar as a near miss). Without a subset (fit_dev is small, or the
        subset is screen.prescreen_max_share or more) it is measured on the full
        fit_dev — that still saves the rescorer refit and holdout, and if it
        passes that screen is used as is (_full_screen). A rescorer candidate is
        compared **before** the refit, against the champion's pre-rescorer
        channel.

        If it passes on the subset, the full fit_dev is screened as before — the
        final numbers and adoption verdict equal those without a prescreen (the
        pass cache reuses the subset's passes)."""
        s = self.cfg.screen
        champ = (getattr(self, "_champion", None) or {}).get("axes") or {}
        g = axes.get("tiling")
        # only increments that **raise** the grid — a grid below the champion's
        # is cheaper, and unused tiles there do not make it the champion (the
        # champion uses tiles)
        tiling = bool(g and g >= s.prescreen_min_grid and g > int(champ.get("tiling") or 0)
                      and {k for k, v in c["axes"].items()
                           if champ.get(k) != v} == {"tiling"})
        lost = list(c.get("lost_values") or ()) if s.lost_value_prescreen else []
        if not tiling and not lost:
            return False
        sub = prescreen_subset(self.ds.fit_dev, s.prescreen_fraction,
                               s.prescreen_min_images,
                               max_share=s.prescreen_max_share)
        if sub is None and not lost:
            return False
        setting = self._screen(materials, axes, split=sub)
        info = {"n_images": len(sub.images) if sub is not None
                else len(self.ds.fit_dev.images),
                "of": len(self.ds.fit_dev.images),
                "fit_dev_micro_f_beta": setting["fit_dev_micro_f_beta"]}
        if tiling and sub is not None:
            info["passed"] = _uses_tiling(setting)
            if setting.get("tiling_f"):
                info["tiling_f"] = setting["tiling_f"]
        cmp = None
        if lost:
            cmp = self._fit_dev_cmp(setting, raw=True, split=sub)
            info["lost_values"] = lost
            if cmp is not None:
                info["fit_dev_cmp"] = cmp
        entry["prescreen"] = info
        if tiling and sub is not None and not info["passed"]:
            rec = {**c, "skipped": "axis_unused", "unused_axes": ["tiling"],
                   "tiling_used": False, "prescreen": info,
                   "note": f"prescreen: no class picked the grid {g} tile config on "
                           f"fit_dev {info['n_images']}/{info['of']} images — the full "
                           "screen is skipped"}
            if setting.get("tiling_f"):
                rec["tiling_f"] = setting["tiling_f"]
            if entry.get("tiling_shortfall"):
                rec["tiling_shortfall"] = entry["tiling_shortfall"]
        elif cmp is not None and cmp["ub"] < 0:
            rec = {**c, "skipped": "lost_value_prescreen", "prescreen": info,
                   "fit_dev_cmp": cmp,
                   "note": f"prescreen: {', '.join(lost)} clearly lost on fit_dev under "
                           f"the current champion — on fit_dev {info['n_images']}/"
                           f"{info['of']} images (candidate − champion) micro F(β) "
                           f"{cmp['delta']:+.4f}, upper bound {cmp['ub']:+.4f} < 0, so "
                           "the full screen is skipped"}
            # If tiling was requested, that grid reached the (subset) screen — record
            # whether tiles were picked there so the fold gate spends the near-miss
            # exception (the same convention as the axis prescreen's axis_unused row)
            rec |= _tiling_record({**c, "setting": setting,
                                   "tiling_shortfall": entry.get("tiling_shortfall")})
        else:
            if lost:
                info["passed"] = True
            if sub is None:
                # Without a subset (fit_dev is small, or the subset is max_share or
                # more) the full fit_dev was already screened — hand it over so the
                # caller does not run the same screen again
                entry["_full_screen"] = setting
            return False
        log.info("candidate %s: %s", c["id"], rec["note"])
        skipped.append(rec)
        return True

    def _materialize_timed(self, axes: dict, mat_dir, wall: dict) -> Materials:
        """materialize — records the time taken (seconds) in wall["materialize"]
        (even on failure)."""
        t0 = time.monotonic()
        try:
            return materialize(self.cfg, self.ds, axes, mat_dir,
                               seed=self.cfg.run_seed,
                               **self._parent_kw(), **self._mat_kw)
        finally:
            wall["materialize"] = time.monotonic() - t0

    def _screen_one(self, c: dict, mat_dir, entry: dict,
                    screened: list, skipped: list) -> None:
        """materialize + screen for one candidate — the caller records failures as rejections."""
        self._shortfall(c, entry)
        if c["axes"].get("rescorer"):
            # the rescorer is a post-processing layer on the setting fixed by the
            # other axes — discriminator on fit_train, α grid on fit_dev (loop.rescore)
            base_axes = {**c["axes"], "rescorer": False}
            materials = self._materialize_timed(base_axes, mat_dir,
                                                entry.setdefault("_wall", {}))
            if self._guard_material(c, materials, skipped):
                return
            if self._prescreen(c, base_axes, materials, entry, skipped):
                return
            setting0 = (entry.pop("_full_screen", None)
                        or self._screen(materials, base_axes))
            # choose the classifier λ first (the rescorer is fit on top — downstream on upstream)
            setting0 = self._choose_lambda(materials, base_axes, entry, setting0)
            # judge sameness **before** the refit — if the served state equals the
            # champion's, a sibling's or an already-evaluated candidate's, skip
            # the discriminator refit (train re-inference) and holdout altogether
            if self._skip_same(c, entry, setting0, materials, skipped):
                return
            with self._lambda_for(entry):
                rescorer, info, dev_preds = rescore.fit_for_setting(
                    self.cfg, self.client, self.ds, setting0, materials, mat_dir)
            if rescorer is None or not rescorer.active:
                # the identity verdict is deterministic — a sibling in the same
                # state gets the same result, so block it
                self._seen_fps.add(entry["setting_fp"])
                skipped.append({**c, "skipped": "rescorer_identity",
                                "note": "rescorer skipped or α=0 for all classes — "
                                        "identical to champion, evaluation skipped"})
                return
            setting = rescore.rescored_screen(self.cfg, self.ds, setting0,
                                              rescorer, dev_preds)
            entry["_rescorer"] = rescorer
            entry["rescorer_path"] = str(Path(mat_dir) / "rescorer.json")
        else:
            materials = self._materialize_timed(c["axes"], mat_dir,
                                                entry.setdefault("_wall", {}))
            if self._guard_material(c, materials, skipped):
                return
            if self._prescreen(c, c["axes"], materials, entry, skipped):
                return
            setting = (entry.pop("_full_screen", None)
                       or self._screen(materials, c["axes"]))
            setting = self._choose_lambda(materials, c["axes"], entry, setting)
            if self._skip_same(c, entry, setting, materials, skipped):
                return
        entry.update({"setting": setting,
                      "fit_dev_micro_f_beta": setting["fit_dev_micro_f_beta"],
                      "_materials": materials})
        cmp = self._fit_dev_cmp(setting)   # served channel — the ledger's near-miss/loss verdict
        if cmp is not None:
            entry["fit_dev_cmp"] = cmp
        _relabel_unused(entry)
        # only once screen has fully passed is this a served state "seen this
        # round" — a candidate that failed earlier (e.g. a transient error during
        # the refit) blocking a sibling in the same state would make tried_combos
        # block a combination for good with nothing measured
        self._seen_fps.add(entry["setting_fp"])
        screened.append(entry)

    def _skip_same(self, c: dict, entry: dict, setting: dict, materials: Materials,
                   skipped: list) -> bool:
        """Reject without holdout evaluation (True) if the served state
        (_served_fp) equals the champion's, a sibling's in the same round, or a
        candidate evaluated on holdout before — the same state gives the same
        holdout predictions. Axis labels are not compared because materials can
        differ under the same label (a candidate with exemplar on re-mines text
        negatives from the exemplar pass's FPs).

        Same as the champion with an unused axis → axis_unused (which axes in
        unused_axes); otherwise same_setting. It still counts as tried: the
        skipped row keeps the requested axes, so tried_combos blocks it."""
        fp = _served_fp(setting, materials, entry["axes"], self._served_state(entry))
        entry["setting_fp"] = fp
        unused = _unused_axes(entry["axes"], setting)
        if fp == self._champion_fp():
            if unused:
                rec = {**c, "skipped": "axis_unused", "unused_axes": unused,
                       "note": f"{unused}: the screened setting does not use this axis — "
                               "same setting as champion, evaluation skipped"}
            else:
                rec = {**c, "skipped": "same_setting",
                       "note": "served state equals the champion's — evaluation skipped"}
        elif fp in self._seen_fps:
            rec = {**c, "skipped": "same_setting",
                   "note": "served state equals another candidate's in this round — "
                           "evaluation skipped"}
        elif fp in self._tried_fps:
            rec = {**c, "skipped": "same_setting",
                   "note": "served state equals a setting evaluated on holdout in an "
                           "earlier round — evaluation skipped"}
        else:
            return False
        if rec["skipped"] == "same_setting":
            # what was the same — the planner sees whether another k refined to
            # the same material (e.g. text_negative_k 2 and 4 both to one phrase)
            # and folds variations of that axis
            mat = _material_summary(setting, materials)
            if mat:
                rec["material"] = mat
                rec["note"] += f" — served material: {mat}"
        rec |= _modality_record(setting, self.ds.ontology)
        if entry.get("prescreen"):
            rec["prescreen"] = entry["prescreen"]
        if unused:
            rec.setdefault("unused_axes", unused)
        # a candidate that asked for tiling records that it reached screen, used
        # or not — the fold gate's near-miss exception counts "g+1 was already
        # screened" from this
        if (c.get("requested_axes") or c["axes"]).get("tiling"):
            rec["tiling_used"] = "tiling" not in unused
            if setting.get("tiling_f"):
                rec["tiling_f"] = setting["tiling_f"]   # how far it fell short
            if entry.get("tiling_shortfall"):
                rec["tiling_shortfall"] = entry["tiling_shortfall"]
        log.info("candidate %s: %s", c["id"], rec["note"])
        skipped.append(rec)
        return True

    def _served_state(self, entry: dict) -> dict:
        """Detector state a wired candidate is served with — the champion's
        weights and artifact, and the classifier blend λ (the candidate's
        re-chosen one, if any). λ is left
        out without a classifier so fingerprints of classifier-free states stay
        as they were."""
        ch = getattr(self, "_champion", None) or {}
        state = {"weights": ch.get("weights"),
                 "adapt_root": entry.get("_adapt_root", ch.get("adapt_root"))}
        lam = entry.get("_adapt_lambda", ch.get("adapt_lambda"))
        return state | ({"adapt_lambda": lam} if lam is not None else {})

    def _champion_fp(self) -> str | None:
        ch = getattr(self, "_champion", None) or {}
        if "_fp" not in ch:
            fp = None
            if ch.get("setting") and ch.get("materials_dir") \
                    and Path(ch["materials_dir"]).exists():
                fp = _served_fp(ch["setting"], Materials.load(Path(ch["materials_dir"])),
                                ch.get("axes") or {}, self._served_state(ch))
            ch["_fp"] = fp
        return ch["_fp"]

    # ------------------------------------------------------------- rounds

    def baseline(self) -> dict:
        self._round_t0 = time.monotonic()
        self._pass_t0 = gateway.pass_cache_stats()
        rd = self._round_dir(0)
        mat_dir = rd / "settings" / "baseline"
        materials = materialize(self.cfg, self.ds, BASELINE, mat_dir,
                                seed=self.cfg.run_seed, **self._mat_kw)
        setting = self._screen(materials, BASELINE)
        ev = self._evaluate(setting, materials)
        champion = {"round": 0, "id": "baseline", "axes": dict(BASELINE),
                    "setting": setting, "materials_dir": str(mat_dir),
                    "eval": ev, "weights": None, "adapt_root": None,
                    "adapt_lambda": None, "rescorer_path": None}
        decision = {"round": 0, "kind": "baseline", "adopted": True,
                    "id": "baseline", "micro_f_beta": ev["micro_f_beta"],
                    "tau_test": ev["tau_test"]}
        _write(rd / "eval.json", ev)
        _write(rd / "decision.json", decision)
        self._save_champion(champion)
        self._append_history({
            "round": 0,
            "champion": _champion_summary(champion),
            "candidates": [], "decision": decision})
        return champion

    def run_round(self, r: int, champion: dict, history: list[dict]) -> tuple[dict, dict]:
        """One round — returns (decision, champion).

        If every candidate of the first proposal is rejected before holdout
        (filters, screen), the planner is asked again within the same round (up
        to planner.max_reasks times — _reask). Diagnosis and champion are reused,
        and re-asked candidates go through the same path and are recorded in the
        same round."""
        self._round_t0 = time.monotonic()
        self._pass_t0 = gateway.pass_cache_stats()
        self._round_no = r
        self._champion = champion          # parent for material inheritance
        self._champion.pop("_fp", None)
        self._seen_fps: set[str] = set()     # served states that passed screen this round
        self._tried_fps = _evaluated_fps(history)
        sat = self.cfg.saturation
        # The planner prompt's budget card — rounds left and stop counters (counted
        # from the tail of history the same way run_rounds does)
        self._budget = {"round": r, "rounds": getattr(self, "_rounds_total", None),
                        "stagnant": _trailing_stagnant(history, sat.epsilon),
                        "saturation_rounds": sat.rounds,
                        "empty": _trailing_empty(history),
                        "empty_rounds": sat.empty_rounds}
        rd = self._round_dir(r)
        rd.mkdir(parents=True, exist_ok=True)
        diagnosis = self._diagnosis(champion, rd,
                                    prev=self._round_dir(r - 1) if r > 1 else None)
        # material for the near-miss exception's shortfall verdict — reads only
        # the train diagnosis's tiling_sizes (sizes counted at fit_dev screen
        # thresholds)
        self._diag_train = diagnosis.get("train")
        self._crop_notes(diagnosis, rd)      # merge VLM notes (diagnostic signal only)
        proposal = _with_round_ids(self._propose(history, diagnosis, rd), r, set())
        # served_* is what the detector has loaded right now — detector-state
        # candidates swap it in turn
        acc = _RoundAcc(served_w=champion.get("weights"),
                        served_a=champion.get("adapt_root"),
                        served_l=champion.get("adapt_lambda"))
        self._attempt(rd, champion, proposal, acc)
        # re-asks — k (the artifact number) runs in order whether or not the
        # re-ask is free. If a proposal only re-proposed the champion
        # (_noop_only), FREE_REASKS per round do not count toward
        # planner.max_reasks. The verdict reads only the proposal (a replayed
        # artifact) and history, so resume decides the same. max_reasks=0 means
        # no free re-ask either (re-asks off)
        reasks = free = 0
        max_reasks = self.cfg.planner.max_reasks
        while not acc.evaluated and not proposal["stop"] and max_reasks > 0:
            bonus = free < FREE_REASKS and _noop_only(proposal)
            if not bonus and reasks - free >= max_reasks:
                break
            nxt = self._reask(reasks + 1, r, history, diagnosis, rd, champion, acc)
            if nxt is None:
                break
            reasks += 1
            free += bonus
            if bonus:
                log.info("round %d: the proposal was only the champion itself — "
                         "re-ask %d does not count toward planner.max_reasks", r, reasks)
            proposal = nxt
            self._attempt(rd, champion, proposal, acc)
        skipped, screened, evaluated = acc.skipped, acc.screened, acc.evaluated
        reask_mark = ({"reasks": reasks} if reasks else {}) \
            | ({"free_reasks": free} if free else {})

        if not acc.n_runnable:
            decision = {"round": r, "adopted": False,
                        "kind": ("planner_stop" if proposal["stop"] else "no_candidates"),
                        "skipped": [c["id"] for c in skipped]} | reask_mark
            _write(rd / "decision.json", decision)
            self._append_history({
                "round": r,
                "champion": _champion_summary(champion),
                "candidates": skipped, "decision": decision})
            return decision, champion

        _write(rd / "screen.json",
               [{k: c[k] for k in ("id", "axes", "fit_dev_micro_f_beta")}
                for c in screened])
        # keep records too — re-judging past runs after changing gate criteria
        # (one/two-sided, alpha, margin) needs more than the champion's records.
        _write(rd / "eval.json",
               [{"id": c["id"], "axes": c["axes"],
                 "micro_f_beta": c["eval"]["micro_f_beta"],
                 "tau_test": c["eval"]["tau_test"], "gate": c["gate"],
                 "records": c["eval"]["records"]}
                for c in evaluated])

        winners = [c for c in evaluated if c["gate"]["adopted"]]
        winner = max(winners, key=lambda c: c["gate"]["delta"]) if winners else None
        # passed the gate but lost to a winner with a larger Δ — carried to the next round
        lost = {c["id"] for c in winners if c is not winner}
        decision = {
            "round": r, "adopted": winner is not None,
            "id": winner["id"] if winner else None,
            "screened": [c["id"] for c in screened],
            "evaluated": {c["id"]: {"micro_f_beta": c["eval"]["micro_f_beta"],
                                    "gate": {k: c["gate"][k] for k in
                                             ("delta", "lo", "hi", "adopted")}}
                          for c in evaluated},
            "skipped": [c["id"] for c in skipped]
                       + [c["id"] for c in screened if c.get("skipped")],
        } | reask_mark
        # Candidates dropped only by the pre-holdout fit_dev verdict — with any,
        # the round is not empty (_empty_round: the skip saves holdout time but
        # does not change when the run stops)
        fd_skipped = list(dict.fromkeys(
            c["id"] for c in [*skipped, *screened]
            if c.get("skipped") in FIT_DEV_SKIPS))
        if fd_skipped:
            decision["fit_dev_skipped"] = fd_skipped
        if proposal["stop"]:
            # the planner declared "nothing to try" while re-asked — this round
            # measured nothing (which is why it was re-asked) and the run stops here
            decision["kind"] = "planner_stop"
        if winner:
            w_axes, _ = _winner_axes(winner)     # defensive rescorer label reconciliation
            # downstream inheritance: a classifier-arm winner keeps its own artifact;
            # any other winner (lora included) inherits the champion's as-is.
            adapt_root = winner.get("_adapt_root", champion.get("adapt_root"))
            # the blend λ follows the same rule — a classifier-arm winner keeps its
            # own λ, any other winner its re-chosen λ or the champion's
            adapt_lambda = (winner.get("_adapt_lambda", champion.get("adapt_lambda"))
                            if adapt_root else None)
            if not adapt_root:
                w_axes["crop_classifier"] = False
            # The champion can lose the rescorer in two ways that mean different things:
            # (1) the candidate never requested it — the planner's choice;
            # (2) it was requested but the refit on the arm was identity — a label correction.
            # Without distinguishing them, the ledger would record (1) as a correction.
            if champion["axes"].get("rescorer") and not w_axes.get("rescorer"):
                asked = winner.get("rescorer_refit") == "identity"
                decision["rescorer_dropped_by"] = "refit_identity" if asked \
                    else "proposal"
                decision["note"] = (
                    "rescorer re-fit on the winning arm is identity (α=0 for all classes) — "
                    "champion label corrected" if asked else
                    "winner did not request the rescorer — champion loses that layer")
            champion = {"round": r, "id": winner["id"], "axes": w_axes,
                        "setting": winner["setting"],
                        "materials_dir": winner["materials_dir"],
                        "eval": winner["eval"],
                        # a detector-state winner keeps its own; otherwise inherit current
                        "weights": winner.get("_weights", champion.get("weights")),
                        "adapt_root": adapt_root,
                        "adapt_lambda": adapt_lambda,
                        "rescorer_path": winner.get("rescorer_path")}
            self._save_champion(champion)
            decision["micro_f_beta"] = winner["eval"]["micro_f_beta"]
        # Bring the detector to the final champion's state. Detector-state
        # candidates swap it in turn, so the last candidate's state is not
        # necessarily the winner's — a lora winner may have been unloaded before
        # the classifier candidates.
        self._serve_champion(champion, acc)
        _write(rd / "decision.json", decision)
        self._append_history({
            "round": r,
            "champion": _champion_summary(champion),
            "candidates": ([{k: c[k] for k in ("id", "axes")}
                            | {"adopted": bool(winner and c["id"] == winner["id"]),
                               "fit_dev_micro_f_beta": c["fit_dev_micro_f_beta"]}
                            | _tiling_record(c) | _neg_eta_record(c) | _requested(c)
                            | ({"prescreen": c["prescreen"]} if c.get("prescreen") else {})
                            | _ledger_record(c)
                            | _unused_record(c) | _reask_record(c)
                            | ({"setting_fp": c["setting_fp"]} if c.get("setting_fp") else {})
                            | _modality_record(c["setting"], self.ds.ontology)
                            | {"forwards": setting_forwards(c["setting"])}
                            | ({"wall_min": _wall_min(c["_wall"])}
                               if c.get("_wall") else {})
                            | ({"gate_passed_but_lost": True} if c["id"] in lost else {})
                            | _source_record(c) | _reuse_record(c)
                            | _classifier_lambda_record(c) | _skip_record(c)
                            for c in screened] + skipped),
            "decision": decision})
        return decision, champion

    def _gate(self, champion: dict, ev: dict) -> dict:
        return gate(champion["eval"]["records"], ev["records"],
                    cat_ids=self.cat_ids, iou_thr=self.iou,
                    margin=self.cfg.gate.margin,
                    min_delta=self.cfg.gate.min_delta,
                    n_boot=self.cfg.gate.bootstrap, seed=self.cfg.run_seed,
                    alpha=1.0 - self.cfg.gate.confidence,
                    beta=self.cfg.beta)

    def _arm_skip(self, row: dict) -> bool:
        """The pre-holdout fit_dev verdict for the lora and classifier arms — the
        same rule as plain candidates (_holdout_pick). The arm's own fit_dev
        margin gate (arm vs the same candidate's base setting, point estimate)
        asks "did training gain anything"; this asks whether the final setting,
        rescorer refit included, clearly lost to or equals **the champion** —
        different questions, so both are kept. If filtered, the reason goes on
        the row and True is returned."""
        if not self.cfg.screen.skip_fit_dev_losers:
            return False
        why = _fit_dev_skip(row)
        if why is None:
            return False
        row.update(why)
        log.info("candidate %s: %s", row["id"], why["note"])
        return True

    def _serve_champion(self, champion: dict, acc: "_RoundAcc") -> None:
        """Bring the detector back to the champion state (weights, artifact, blend λ).

        A detector-state arm that clears its margin leaves the detector in its
        own state on the way to holdout, and stays there if the pre-holdout
        fit_dev verdict (_arm_skip) drops it. If the round continues with a
        re-ask, the next attempt's ordinary candidates would be measured on a
        non-champion state — and a winner would then be served (inheriting the
        champion state) differently from how it was measured. A lora arm
        restores only the weights (tail_round), so this is needed before lora
        candidates too, so they are not measured on a leftover classifier arm —
        hence it runs at the start of every attempt. It also runs at the start
        of every lora and classifier candidate, so the next candidate's
        materials are not mined with an earlier candidate's adopted arm still
        loaded. Nothing happens when the detector is already in the champion
        state. Also used for the end-of-round restore.

        The artifact and λ are compared against what is actually loaded
        (self._adapt_path / self._adapt_lambda, which only _adapt_ctl changes),
        not the round record — so a swap the record missed is still undone.
        The weights are compared against the round record (acc.served_w): only
        detector-state candidates change them and the record follows each
        change. Same-path requests and the retry after a failed swap are
        _weights_ctl's job."""
        if not self.detector_hooks:
            return                        # run without state controls — the state never changed
        if not _same_state(acc.served_w, champion.get("weights")):
            self._weights_ctl(champion.get("weights"))
            acc.served_w = champion.get("weights")
        if not (_same_state(self._adapt_path, champion.get("adapt_root"))
                and self._adapt_lambda == champion.get("adapt_lambda")):
            self._adapt_ctl(champion.get("adapt_root"), champion.get("adapt_lambda"))
        acc.served_a = champion.get("adapt_root")
        acc.served_l = champion.get("adapt_lambda")

    def _attempt(self, rd: Path, champion: dict, proposal: dict,
                 acc: "_RoundAcc") -> None:
        """Run one proposal's candidates (the first one or a re-ask) through
        wiring, screen and holdout, accumulating into acc. The decision is made
        once at the end of the round."""

        skipped = acc.skipped
        skipped.extend(proposal.get("dropped", []))   # include combo_tried records
        wired, lora_cands, classifier_cands = [], [], []
        n_filtered = len(skipped)                     # up to here: filter drops, not wiring
        champ_axes = champion.get("axes") or {}
        for c in proposal["candidates"]:
            miss = _missing_hook(c["axes"], self._mat_kw)
            if miss:
                skipped.append({**c, "skipped": "missing_hook",
                                "note": f"{miss} not injected — cannot produce material "
                                        f"for this axis (needs LLM/worker wiring)"})
                continue
            route = _route_detector_state(c["axes"], champ_axes)
            if route == "off":
                off = [k for k in ("lora", "crop_classifier")
                       if champ_axes.get(k) and not c["axes"].get(k)]
                skipped.append({**c, "skipped": "detector_state_off",
                                "note": f"{off}: the champion's detector-state axes cannot "
                                        "be measured turned off (the state stays loaded "
                                        "while measuring) — keep them on"})
            elif route == "compound":
                skipped.append({**c, "skipped": "compound_server_state",
                                "note": "turning on both lora and crop_classifier relative to "
                                        "champion is unsupported (gain cannot be attributed) — "
                                        "split it into single increments"})
            elif route == "lora":
                if self.detector_hooks is None:
                    skipped.append({**c, "skipped": "lora_deps",
                                    "note": "weights_ctl/base_weights not injected — "
                                            "needs GPU wiring (detector_hooks)"})
                else:
                    lora_cands.append(c)
            elif route == "classifier":
                if self.detector_hooks is None or not self.detector_hooks.get("adapt_ctl"):
                    skipped.append({**c, "skipped": "adapt_ctl",
                                    "note": "adapt_ctl not injected — needs detector client "
                                            "wiring (detector_hooks)"})
                else:
                    classifier_cands.append(c)
            else:
                wired.append(c)
        if len(skipped) > n_filtered:
            log.warning("skipped %d candidates that cannot be wired: %s",
                        len(skipped) - n_filtered,
                        [c["id"] for c in skipped[n_filtered:]])

        if not wired and not lora_cands and not classifier_cands:
            return
        acc.n_runnable += len(wired) + len(lora_cands) + len(classifier_cands)

        self._serve_champion(champion, acc)
        screened = []
        for c in wired:
            mat_dir = rd / "settings" / _slug(c["id"])
            entry = {**c, "materials_dir": str(mat_dir), "_wall": {}}
            t0, n_skip = time.monotonic(), len(skipped)
            try:
                self._screen_one(c, mat_dir, entry, screened, skipped)
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("candidate %s failed — rejected, round continues", c["id"])
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
            _close_wall(entry["_wall"], t0, skipped[n_skip:])

        top = _holdout_pick(screened, self.cfg.screen.top_k,
                            skip=self.cfg.screen.skip_fit_dev_losers)
        evaluated = acc.evaluated
        _gate = lambda ev: self._gate(champion, ev)  # noqa: E731

        for c in top:
            t0 = time.monotonic()
            ev = self._evaluate_candidate(c, c["setting"], c["_materials"])
            c["_wall"]["holdout"] = time.monotonic() - t0
            evaluated.append({**c, "eval": ev, "gate": _gate(ev)})

        # --- lora candidates — they swap the client's weights, so always last ---
        # Evaluate with the champion's classifier left loaded — a crop -> class
        # function is not tied to the detector's score distribution, and removing
        # it would make lora fight carrying a handicap equal to its gain.
        # served_* is what the detector has loaded right now (updated per candidate).
        served_w, served_a, served_l = acc.served_w, acc.served_a, acc.served_l
        for c in lora_cands:
            # Mine the materials (FP-mining train pass) on the champion state — an
            # earlier candidate's adopted arm may still be loaded from its holdout,
            # and run_tail_candidate restores the baseline only after the materials are built
            acc.served_w, acc.served_a, acc.served_l = served_w, served_a, served_l
            self._serve_champion(champion, acc)
            served_w, served_a, served_l = acc.served_w, acc.served_a, acc.served_l
            mat_dir = rd / "settings" / _slug(c["id"])
            base_axes = {**c["axes"], "lora": False}
            wall, t0, n_skip = {}, time.monotonic(), len(skipped)
            try:
                materials = self._materialize_timed(base_axes, mat_dir, wall)
            except Exception as exc:  # noqa: BLE001
                log.exception("lora candidate %s material failed — rejected", c["id"])
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                _close_wall(wall, t0, skipped[n_skip:])
                continue
            if self._guard_material(c, materials, skipped):
                _close_wall(wall, t0, skipped[n_skip:])
                continue
            try:
                # lora axis = tail fine-tune + λ-interpolation arms —
                # contract {"decision","weights","grid","setting"}
                res = tail_round.run_tail_candidate(
                    self.cfg, self.ds, base_axes, materials, mat_dir,
                    screen_fn=self._screen,
                    weights_ctl=self._weights_ctl,
                    base_weights=self.detector_hooks["base_weights"],
                    restore_weights=champion.get("weights"),
                    seed=self.cfg.run_seed,
                    # training reuse — an equal key reuses the subset (per run, survives resume)
                    cache_dir=self.out_dir / LORA_CACHE,
                    origin={"round": self._round_no, "id": c["id"]})
                # adopted: the winning arm stays loaded; otherwise restore_weights (current)
                served_w = res["weights"] or champion.get("weights")
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("lora candidate %s failed — restored current weights and rejected", c["id"])
                self._weights_ctl(champion.get("weights"))
                served_w = champion.get("weights")
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                _close_wall(wall, t0, skipped[n_skip:])
                continue
            _close_wall(wall, t0, ())
            entry = {**c, "materials_dir": str(mat_dir),
                     "setting": res["setting"],
                     "fit_dev_micro_f_beta": max((g["f_beta"] for g in res["grid"]),
                                             default=0.0),
                     "lora": res["decision"]}
            _relabel_unused(entry)
            row = ({k: entry[k] for k in
                    ("id", "axes", "materials_dir", "setting",
                     "fit_dev_micro_f_beta")} | {"lora": res["decision"]}
                   | _requested(entry) | _unused_record(entry)
                   | _reask_record(entry) | _ledger_record(entry)
                   | _source_record(entry) | {"_wall": wall})
            screened.append(row)
            if not res["decision"].get("adopted"):
                skipped.append({**c, "skipped": "lora_gate",
                                "note": res["decision"].get("reason", ""),
                                "wall_min": _wall_min(wall)} | _reuse_record(row))
                continue
            # lora changes the detections — on the winning arm (the weights
            # loaded now) re-choose the classifier λ by the same rule as ordinary
            # candidates, and refit the rescorer on that λ
            setting = self._choose_lambda(materials, base_axes, entry, res["setting"])
            row.update(_classifier_lambda_record(entry))
            with self._lambda_for(entry):
                setting = self._arm_rescorer(entry, setting, materials, mat_dir)
            entry["setting"] = setting
            row.update(_cmp_of(self._fit_dev_cmp(setting)))
            if self._arm_skip(row):
                continue
            t1 = time.monotonic()
            ev = self._evaluate_candidate(entry, setting, materials)  # winning arm's setting
            wall["holdout"] = time.monotonic() - t1
            evaluated.append({**entry, "_materials": materials,
                              "_weights": str(res["weights"]),
                              "eval": ev, "gate": _gate(ev)})

        # --- crop classifier candidates — detector-state candidates, same discipline as lora ---
        # Measure them on the champion's weights. If a winning lora arm stayed
        # loaded and the classifier then won, the champion would be served without
        # that arm while its τ and holdout F came from on top of it — they would
        # not match the serving state (adapt_round restores only the artifact).
        # Each candidate starts on the champion state (weights, artifact, λ).
        for c in classifier_cands:
            # Mine the materials (FP-mining train pass) on the champion state — an
            # earlier candidate's adopted arm may still be loaded from its holdout,
            # and run_classifier_candidate restores the baseline only after the materials are built
            acc.served_w, acc.served_a, acc.served_l = served_w, served_a, served_l
            self._serve_champion(champion, acc)
            served_w, served_a, served_l = acc.served_w, acc.served_a, acc.served_l
            mat_dir = rd / "settings" / _slug(c["id"])
            base_axes = {**c["axes"], "crop_classifier": False}
            wall, t0, n_skip = {}, time.monotonic(), len(skipped)
            try:
                materials = self._materialize_timed(base_axes, mat_dir, wall)
                if self._guard_material(c, materials, skipped):
                    _close_wall(wall, t0, skipped[n_skip:])
                    continue
                res = adapt_round.run_classifier_candidate(
                    self.cfg, self.ds, base_axes, materials, mat_dir,
                    screen_fn=self._screen,
                    adapt_ctl=self._adapt_ctl,
                    repo_root=self.detector_hooks["repo_root"],
                    restore_root=champion.get("adapt_root"),
                    restore_lambda=champion.get("adapt_lambda"),
                    seed=self.cfg.run_seed, client=self.client,
                    # training reuse — trained once per run, each proposal re-scores the λ grid
                    cache_dir=(self.out_dir / CLASSIFIER_CACHE
                               if self.cfg.crop_classifier.reuse else None),
                    origin={"round": self._round_no, "id": c["id"]})
                # adopted: its own artifact and λ stay loaded; otherwise the current state
                if res["adapt_root"]:
                    served_a, served_l = res["adapt_root"], res["adapt_lambda"]
                else:
                    served_a = champion.get("adapt_root")
                    served_l = champion.get("adapt_lambda")
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("classifier candidate %s failed — restored current state and rejected", c["id"])
                self._adapt_ctl(champion.get("adapt_root"), champion.get("adapt_lambda"))
                served_a = champion.get("adapt_root")
                served_l = champion.get("adapt_lambda")
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                _close_wall(wall, t0, skipped[n_skip:])
                continue
            _close_wall(wall, t0, ())
            entry = {**c, "materials_dir": str(mat_dir),
                     "setting": res["setting"],
                     "fit_dev_micro_f_beta":
                         res["setting"]["fit_dev_micro_f_beta"],
                     "crop_classifier": res["decision"]}
            _relabel_unused(entry)
            row = ({k: entry[k] for k in
                    ("id", "axes", "materials_dir", "setting",
                     "fit_dev_micro_f_beta")}
                   | {"crop_classifier": res["decision"]}
                   | _requested(entry) | _unused_record(entry)
                   | _reask_record(entry) | _ledger_record(entry)
                   | _source_record(entry) | {"_wall": wall})
            screened.append(row)
            if not res["decision"].get("adopted"):
                # The tripwire (too few training crops) keeps the same material
                # until the champion changes — record the axis as empty so it reaches
                # the planner's dead-axis list. It cannot be blocked up front because
                # the background FP count is an inference result.
                dead = ({"empty_axis": "crop_classifier"}
                        if res["decision"].get("tripwire") else {})
                skipped.append({**c, "skipped": "classifier_gate", **dead,
                                "note": res["decision"].get("reason", ""),
                                "wall_min": _wall_min(wall)} | _reuse_record(row)
                               | _classifier_lambda_record(row))
                continue
            setting = self._arm_rescorer(entry, res["setting"], materials, mat_dir)
            entry["setting"] = setting
            row.update(_cmp_of(self._fit_dev_cmp(setting)))
            if self._arm_skip(row):
                continue
            t1 = time.monotonic()
            ev = self._evaluate_candidate(entry, setting, materials)
            wall["holdout"] = time.monotonic() - t1
            evaluated.append({**entry, "_materials": materials,
                              "_adapt_root": str(res["adapt_root"]),
                              "_adapt_lambda": res["adapt_lambda"],
                              "eval": ev, "gate": _gate(ev)})
        acc.screened.extend(screened)
        acc.served_w, acc.served_a, acc.served_l = served_w, served_a, served_l


def run_rounds(dataset_root, *, out_dir, planner: str = "deterministic",
               rounds: int = 1, cfg: Config | None = None, client=None,
               llm_chat=None, encode_fn=None, caption_fn=None,
               vlm_caption_fn=None, refine_alias_fn=None,
               refine_negatives_fn=None,
               candidates_file=None, resume: bool = False,
               detector_hooks: dict | None = None) -> RunResult:
    """baseline (round 0) + `rounds` planner rounds — saturation stops earlier if it triggers.

    resume=True continues from the round after history.jsonl. In LLM mode, if
    an interrupted round's proposal.json remains, LLMPlanner replays it (zero calls).
    """
    if cfg is None:
        cfg = config_from_dict({"dataset": str(dataset_root),
                                "planner": {"kind": planner},
                                "rounds": max(1, rounds)})
    loop = Loop(cfg, dataset_root, out_dir, client=client, llm_chat=llm_chat,
                encode_fn=encode_fn, caption_fn=caption_fn,
                vlm_caption_fn=vlm_caption_fn,
                refine_alias_fn=refine_alias_fn,
                refine_negatives_fn=refine_negatives_fn,
                candidates_file=candidates_file, detector_hooks=detector_hooks)
    result = RunResult(out_dir=Path(out_dir))
    loop._rounds_total = rounds             # rounds left for the planner budget card

    # inference for the same input under the same request conditions is reused
    # for the run (gateway.pass_cache)
    with gateway.pass_cache():
        history = loop._read_history() if resume else []
        if history:
            champion = loop._load_champion()
            loop._restore_detector_state(champion)
            start = history[-1]["round"] + 1
            result.decisions = [e["decision"] for e in history]
        else:
            loop.out_dir.mkdir(parents=True, exist_ok=True)
            (loop.out_dir / "history.jsonl").write_text("")
            champion = loop.baseline()
            history = loop._read_history()
            result.decisions = [history[-1]["decision"]]
            start = 1

        # if resume reset the counter to 0, an already-saturated run would spend more
        # rounds — restore the trailing stagnant count from history and continue it.
        # the count of empty rounds (nothing measured on holdout) is restored the same way
        stagnant_rounds = _trailing_stagnant(history, cfg.saturation.epsilon)
        empty_rounds = _trailing_empty(history)
        result.stopped = _saturated(cfg.saturation, stagnant_rounds,
                                    empty_rounds) or result.stopped
        if result.stopped == "rounds":
            for r in range(start, rounds + 1):
                decision, champion = loop.run_round(r, champion, history)
                history = loop._read_history()
                result.decisions.append(decision)
                if decision.get("kind") == "planner_stop":
                    result.stopped = "planner_stop"
                    break
                stagnant_rounds = (stagnant_rounds + 1
                                   if _stagnant(decision, cfg.saturation.epsilon)
                                   else 0)
                empty_rounds = empty_rounds + 1 if _empty_round(decision) else 0
                stop = _saturated(cfg.saturation, stagnant_rounds, empty_rounds)
                if stop:
                    result.stopped = stop
                    break
        # the final champion's diagnosis — material for the report's residual-miss
        # analysis (rounds/final)
        final_diag = loop.final_diagnosis(champion, history)

    _write(loop.out_dir / "run_summary.json", {
        "stopped": result.stopped,
        # whether holdout was valid or test as a stand-in — in the latter case
        # there is no sealed split and this run's micro_f_beta is optimistically
        # biased. Reporters must check this to avoid mixing both in one table (see core.data).
        "holdout_split": loop.ds.holdout_split,
        # whether the training material (lora, rescorer, classifier) was a
        # fit_train subset — look at it when comparing training time and scores
        # across datasets (fitting.subsample)
        "fit_train": loop.fit_train_summary(),
        "last_round": history[-1]["round"] if history else None,
        "adopted_rounds": sum(1 for e in history if e["round"] > 0
                              and (e.get("decision") or {}).get("adopted")),
        "champion": {"id": champion.get("id"), "axes": champion.get("axes"),
                     "micro_f_beta": (champion.get("eval") or {}).get("micro_f_beta")},
        "final_diagnosis": final_diag,
        "finished_at": datetime.now().isoformat(timespec="seconds")})
    result.history = history
    result.champion = champion
    return result


def _invalidating_hooks(hooks: dict) -> dict:
    """Wrap the detector-state controls (weights_ctl, adapt_ctl) to empty the pass
    cache on every call. Once weights or the artifact change, the same input is
    a different inference — every state change goes through these controls
    (_restore_detector_state, the lora arm, the classifier arm), so one place
    covers them."""
    out = dict(hooks)
    for key in ("weights_ctl", "adapt_ctl"):
        fn = hooks.get(key)
        if fn is not None:
            def wrapped(*a, _fn=fn, **kw):
                try:
                    return _fn(*a, **kw)
                finally:
                    gateway.invalidate_pass_cache()
            out[key] = wrapped
    return out


def _pass_cache_delta(before: dict | None, now: dict | None) -> dict | None:
    """Pass cache counts during the round (per piece) — history's pass_cache key.
    entries is the number of pieces held at the end of the round. None outside
    the cache block."""
    if now is None:
        return None
    before = before or {}
    out = {k: now[k] - before.get(k, 0) for k in ("hits", "misses", "invalidations")}
    seen = out["hits"] + out["misses"]
    out["hit_rate"] = round(out["hits"] / seen, 4) if seen else 0.0
    out["entries"] = now["entries"]
    return out


def _wall_min(wall: dict) -> dict:
    """One candidate's wall clock (seconds) → history's wall_min (minutes), per
    stage and total.

    materialize (material mining) · screen (fit_dev scoring — including the
    prescreen, the rescorer refit and lora/classifier training) · holdout
    (evaluation). The planner reads per-axis cost from it."""
    out = {k: round(v / 60, 2) for k, v in wall.items()}
    out["total"] = round(sum(wall.values()) / 60, 2)
    return out


def _close_wall(wall: dict, t0: float, new_skips) -> None:
    """Close screen as the time since t0 minus materialize. Candidate rows
    rejected in between (new_skips) get wall_min right away — they never reach
    holdout, so this is their whole cost."""
    wall["screen"] = max(time.monotonic() - t0 - wall.get("materialize", 0.0), 0.0)
    for rec in new_skips:
        rec.setdefault("wall_min", _wall_min(wall))


def _tiling_variants(search, axes: dict) -> tuple:
    """The one grid the candidate chose — screen compares only it against tiling off.

    Why not the whole grid list: the zoom is a planner-chosen axis value (when it
    falls short the next round raises it a step), and one candidate's screen cost
    should be a predictable 1 + (1+g²) passes."""
    g = axes.get("tiling")
    if not g:
        return ()
    return ({"grid": int(g), "overlap": search.tiling_overlap,
             "merge": search.tiling_merge, "min_score": search.tiling_merge_min_score},)


def setting_forwards(setting: dict) -> int:
    """How many detector forwards the setting runs per image — 1 + g² per
    (config, modality) pass (grid g tiles, 1 without tiling). The grid tile
    count does not depend on image size."""
    configs = {cf["config_id"]: cf for cf in setting.get("configs", [])}
    groups = {(e["config_id"], e.get("modality", "semantic"))
              for e in (setting.get("per_class") or {}).values()}
    total = 0
    for config_id, _modality in groups:
        tile = (configs.get(config_id) or {}).get("tiling")
        total += 1 + (int(tile["grid"]) ** 2 if tile else 0)
    return total or 1


def _champion_summary(champion: dict) -> dict:
    """The champion summary in a history row — with its serving cost (forwards_per_image)."""
    return {"axes": champion["axes"], "round": champion["round"],
            "id": champion["id"], "micro_f_beta": champion["eval"]["micro_f_beta"],
            "fit_dev_micro_f_beta": champion["setting"]["fit_dev_micro_f_beta"],
            "forwards_per_image": setting_forwards(champion["setting"])} \
        | ({"classifier_lambda": champion["adapt_lambda"]}
           if champion.get("adapt_lambda") is not None else {})


def _uses_tiling(setting: dict) -> bool:
    """Whether any pass the setting actually runs uses a tile config."""
    configs = {cf["config_id"]: cf for cf in setting.get("configs", [])}
    return any(configs.get(e["config_id"], {}).get("tiling")
               for e in (setting.get("per_class") or {}).values())


def _tiling_record(c: dict) -> dict:
    """For a candidate that requested tiling, whether screen actually assigned a
    tile config — the signal the planner reads to raise or drop the zoom. False
    means that grid helped no class on fit_dev. The request survives relabelling
    in requested_axes, so read that first."""
    requested = c.get("requested_axes") or c["axes"]
    if not requested.get("tiling") or not c.get("setting"):
        return {}
    out = {"tiling_used": _uses_tiling(c["setting"])}
    if c["setting"].get("tiling_f"):
        # per-class-mean fit_dev F(β) of the tiled and untiled configs and the
        # bootstrap upper bound of their difference (gap_ub) — by how much it won
        # or fell short, and whether within noise
        out["tiling_f"] = c["setting"]["tiling_f"]
    if c.get("tiling_shortfall"):
        out["tiling_shortfall"] = c["tiling_shortfall"]   # magnification shortfall verdict (gates)
    return out


def _neg_eta_record(c: dict) -> dict:
    """For a candidate that requested visual negatives, the neg_eta values screen
    picked for semantic classes (neg_eta_used) and the grid it searched
    (neg_eta_grid) — a sizing signal like tiling_used. neg_eta is the
    suppression IoU threshold, so lower means stronger suppression: picks at the
    low end of the grid point the planner toward more negatives
    (visual_negative_k), picks at the high end toward folding the axis. An empty
    list means no class picked a negative config."""
    requested = c.get("requested_axes") or c["axes"]
    if not requested.get("visual_negatives") or not c.get("setting"):
        return {}
    configs = {cf["config_id"]: cf for cf in c["setting"].get("configs", [])}
    used = {configs[e["config_id"]]["neg_eta"]
            for e in (c["setting"].get("per_class") or {}).values()
            if e.get("modality", "semantic") == "semantic"
            and configs.get(e["config_id"], {}).get("visual_negatives")}
    grid = {cf["neg_eta"] for cf in configs.values() if cf.get("visual_negatives")}
    return {"neg_eta_used": sorted(used), "neg_eta_grid": sorted(grid)}


# Measured state with an unused axis turned off — when exemplar is unused, the
# semantic-only axis tied to it (visual negatives) and the count axes go back to
# their safe point too
_UNUSED_OFF = {
    "tiling": {"tiling": False},
    "exemplar": {"exemplar": "none", "exemplar_k": None, "fn_k": None,
                 "visual_negatives": False, "visual_negative_k": None},
    "visual_negatives": {"visual_negatives": False, "visual_negative_k": None},
}


def _unused_axes(axes: dict, setting: dict) -> list[str]:
    """Axes the candidate turned on that the setting does not actually use.

    exemplar is unused when no class picks the semantic modality (text wins for
    every class), visual_negatives when no semantic class's config has
    negatives, and tiling when no class uses a tile config."""
    configs = {cf["config_id"]: cf for cf in setting.get("configs", [])}
    semantic = [e for e in (setting.get("per_class") or {}).values()
                if e.get("modality", "semantic") == "semantic"]
    out = []
    if axes.get("tiling") and not _uses_tiling(setting):
        out.append("tiling")
    if axes.get("exemplar") not in (None, "none") and not semantic:
        out.append("exemplar")
    if axes.get("visual_negatives") and not any(
            configs.get(e["config_id"], {}).get("visual_negatives") for e in semantic):
        out.append("visual_negatives")
    return out


def _served_fp(setting: dict, materials: Materials, axes: dict, server: dict) -> str:
    """Fingerprint of the served state — equal fingerprints give equal holdout
    predictions.

    Per class (config content, modality) — config ids differ across candidates
    because each has its own grid, so content is compared —, the materials of
    the modalities the setting uses (text: aliases and suppression phrases;
    semantic: exemplars, plus visual negatives if a negative config is used),
    whether the rescorer is on, and the detector state (weights, artifact).
    Thresholds are left out — they are re-estimated on holdout."""
    configs = {cf["config_id"]: {k: v for k, v in cf.items() if k != "config_id"}
               for cf in setting.get("configs", [])}
    per = {str(cid): [configs.get(e["config_id"]), e.get("modality", "semantic")]
           for cid, e in (setting.get("per_class") or {}).items()}
    mods = {m for _, m in per.values()}
    mat = materials.fingerprint()
    served = {}
    if "text" in mods:
        served |= {"aliases": mat["aliases"], "text_negatives": mat["text_negatives"]}
    if "semantic" in mods:
        served["exemplars"] = mat["exemplars"]
        if any(cf and cf.get("visual_negatives") for cf, m in per.values()
               if m == "semantic"):
            served["visual_negatives"] = mat["visual_negatives"]
    blob = json.dumps({"per_class": per, "materials": served,
                       "rescorer": bool(axes.get("rescorer")), "server": server},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


MATERIAL_SUMMARY_MAX = 160    # cap on the material summary in same_setting rows


def _material_summary(setting: dict, materials: Materials) -> str:
    """A short summary of the material in the served state (_served_fp) — carried
    in same_setting rows.

    For text-modality classes, per-class suppression phrases and aliases
    verbatim (what refinement converged to is the point); for semantic ones,
    only exemplar and visual negative counts. Truncated at MATERIAL_SUMMARY_MAX."""
    mods = {e.get("modality", "semantic")
            for e in (setting.get("per_class") or {}).values()}
    parts = []
    if "text" in mods:
        neg = {n: v for n, v in sorted(materials.text_negatives.items()) if v}
        parts.append("text_negatives " + ("; ".join(
            f"{n}=[{', '.join(sorted(v))}]" for n, v in neg.items()) or "none"))
        if materials.aliases:
            parts.append("aliases " + "; ".join(
                f"{k}={v}" for k, v in sorted(materials.aliases.items())))
    if "semantic" in mods:
        n_ex = sum(len(v) for v in materials.exemplars.values())
        n_vn = sum(len(v) for v in materials.visual_negatives.values())
        parts.append(f"exemplars {n_ex}" + (f", visual_negatives {n_vn}"
                                             if n_vn else ""))
    out = " | ".join(parts)
    return out if len(out) <= MATERIAL_SUMMARY_MAX \
        else out[:MATERIAL_SUMMARY_MAX - 1] + "…"


def _modality_record(setting: dict, ontology) -> dict:
    """fit_dev F(β) of the semantic and text passes screen measured per class,
    keyed by class name.

    Whether exemplar lost narrowly or badly is the evidence for trying another
    k or fn_arm. A candidate filtered before holdout has no saved setting, so
    this record is the only evidence left. Candidates with no semantic pass
    (text only) record nothing."""
    diag = (setting or {}).get("modality_diagnostic") or {}
    # classes sharing a name get their id appended — if name keys overwrote
    # each other the counts would be wrong
    counts: dict[str, int] = {}
    for c in ontology.classes:
        counts[c.name] = counts.get(c.name, 0) + 1
    names = {str(c.id): (c.name if counts[c.name] == 1 else f"{c.name}#{c.id}")
             for c in ontology.classes}
    # chosen is screen's actual pick — recounting wins from rounded F would
    # count a narrow text win as semantic
    out = {names.get(cid, cid): {"semantic": round(d["semantic"]["f_beta"], 4),
                                 "text": round(d["text"]["f_beta"], 4),
                                 "chosen": d.get("chosen")}
           for cid, d in diag.items()}
    return {"modality_f": out} if out else {}


def _evaluated_fps(history: list[dict]) -> set[str]:
    """Served states of candidates earlier rounds evaluated on holdout — the
    champion only improves, so the same state evaluated again cannot pass the
    gate."""
    out = set()
    for e in history:
        evaluated = (e.get("decision") or {}).get("evaluated") or {}
        out |= {c["setting_fp"] for c in e.get("candidates", [])
                if c.get("setting_fp") and c.get("id") in evaluated}
    return out


def _measured_unused(axes: dict, unused: list[str]) -> dict:
    out = dict(axes)
    for axis in unused:
        out.update(_UNUSED_OFF[axis])
    return out


def _relabel_unused(entry: dict) -> None:
    """If the setting does not use a requested axis, correct axes to the measured
    state and keep the request in requested_axes. A champion labelled with an
    axis it never uses would credit other axes' gains to that axis. Same
    principle as the rescorer identity-refit relabel (_measured_axes)."""
    unused = _unused_axes(entry["axes"], entry["setting"])
    if unused:
        entry["requested_axes"] = dict(entry["axes"])
        entry["axes"] = _measured_unused(entry["axes"], unused)
        entry["unused_axes"] = unused


def _requested(c: dict) -> dict:
    return {"requested_axes": c["requested_axes"]} if c.get("requested_axes") else {}


def _cmp_of(cmp: dict | None) -> dict:
    return {"fit_dev_cmp": cmp} if cmp is not None else {}


def _as_rows(rows) -> list:
    """Normal form of per-image fit_dev rows for comparison — makes a JSON round
    trip (lists) and computed values (tuples, numpy) the same shape."""
    return json.loads(json.dumps(rows, default=lambda x: x.tolist()))


# drops that change nothing — the champion itself (runner._filter), no increment (planner hygiene)
NOOP_SKIPS = ("champion_combo", "no_change")
# re-asks per round that do not count toward planner.max_reasks when a proposal
# is only such drops (keeps a planner repeating the same proposal from never
# ending the round)
FREE_REASKS = 1


def _noop_only(proposal: dict) -> bool:
    """Whether no candidate of the proposal is left and every drop changes nothing (NOOP_SKIPS)."""
    dropped = proposal.get("dropped") or []
    return (not proposal.get("candidates") and bool(dropped)
            and all(d.get("skipped") in NOOP_SKIPS for d in dropped))


# rejection reasons of the pre-holdout fit_dev verdict (_fit_dev_skip)
FIT_DEV_SKIPS = ("fit_dev_lost", "fit_dev_identical")


def _fit_dev_skip(c: dict) -> dict | None:
    """The pre-holdout fit_dev verdict — {skipped, note} for a candidate not to
    send, otherwise None.

    Reads only fit_dev_cmp (per-image fit_dev rows counted at screen thresholds
    — holdout plays no part).
    (1) every per-image row equals the champion's (identical) — fit_dev_identical.
    (2) the bootstrap upper bound of (candidate − champion) micro F(β) < 0 —
    fit_dev_lost.
    Without comparison rows (no champion rows etc.) nothing is filtered."""
    cmp = c.get("fit_dev_cmp")
    if not cmp or cmp.get("ub") is None:
        return None
    if cmp.get("identical"):
        return {"skipped": "fit_dev_identical",
                "note": "every per-image fit_dev (tp, fp, fn) equals the champion's — "
                        "holdout evaluation is skipped"}
    if cmp["ub"] < 0:
        return {"skipped": "fit_dev_lost",
                "note": f"clearly lost to the champion on fit_dev — (candidate − "
                        f"champion) micro F(β) {cmp['delta']:+.4f}, upper bound "
                        f"{cmp['ub']:+.4f} < 0, so holdout evaluation is skipped"}
    return None


def _holdout_pick(screened: list[dict], top_k: int, *, skip: bool) -> list[dict]:
    """Candidates to send to the holdout gate — the top top_k by fit_dev score.

    With skip, candidates that clearly lost on fit_dev or equal the champion
    (_fit_dev_skip) get the reason and are left out. top_k counts **only
    sendable candidates** — top_k is the budget of holdout reuse, so a slot a
    dropped candidate did not use goes to the next one. Every screened candidate
    gets the same verdict regardless of rank (so the record does not depend on
    rank)."""
    ranked = sorted(screened, key=lambda c: -c["fit_dev_micro_f_beta"])
    if not skip:
        return ranked[:top_k]
    out = []
    for c in ranked:
        why = _fit_dev_skip(c)
        if why is not None:
            c.update(why)
            log.info("candidate %s: %s", c["id"], why["note"])
            continue
        out.append(c)
    return out[:top_k]


def _skip_record(c: dict) -> dict:
    """The reason of a candidate that passed screen but dropped out at the
    pre-holdout fit_dev verdict."""
    return {k: c[k] for k in ("skipped", "note") if c.get(k)} if c.get("skipped") \
        else {}


def _ledger_record(c: dict) -> dict:
    """Row fields the ledger (planner.ledger) reads — the fit_dev comparison, the
    clearly-lost variant mark (lost_values), and the planner's expected gain and
    cost (plan)."""
    return (_cmp_of(c.get("fit_dev_cmp"))
            | ({"lost_values": c["lost_values"]} if c.get("lost_values") else {})
            | ({"plan": c["plan"]} if c.get("plan") else {}))


def _reask_record(c: dict) -> dict:
    """For a candidate from a re-ask in the same round, which re-ask it was (reask=k)."""
    return {"reask": c["reask"]} if c.get("reask") else {}


def _source_record(c: dict) -> dict:
    """The source of carried and human candidates — shared by plain candidates and
    the lora and classifier arms (the report's carry verdict reads only this)."""
    return {"source": c["source"]} if c.get("source") else {}


def _reuse_record(c: dict) -> dict:
    """The origin of a run cache a detector-state arm used instead of training —
    lora: delta_reused ({"tail": {round, id}}, loop.tail_round._train_or_reuse),
    classifier: classifier_reused ({round, id}, loop.adapt_round._train_or_reuse).
    It explains why wall_min is short."""
    out = {}
    if reused := (c.get("lora") or {}).get("delta_reused"):
        out["delta_reused"] = reused
    if reused := (c.get("crop_classifier") or {}).get("classifier_reused"):
        out["classifier_reused"] = reused
    return out


def _classifier_lambda_record(c: dict) -> dict:
    """The classifier blend λ the candidate chose — a classifier arm (the grid's
    best when not adopted; Δ is in classifier_decision.json's grid), or an
    ordinary candidate that re-chose λ while the champion had a classifier
    (_choose_lambda — its grid is classifier_lambda_grid)."""
    lam = (c.get("crop_classifier") or {}).get("lam")
    if lam is None:
        lam = c.get("_adapt_lambda")
    out = {"classifier_lambda": lam} if lam is not None else {}
    if c.get("classifier_lambda_grid"):
        out["classifier_lambda_grid"] = c["classifier_lambda_grid"]
    return out


def _unused_record(c: dict) -> dict:
    """Unused axes to keep on the history row — shared by plain candidates and
    the lora and classifier arms."""
    return {"unused_axes": c["unused_axes"]} if c.get("unused_axes") else {}


def _stagnant(decision: dict, epsilon: float) -> bool:
    """Saturation counting rule — a round is stagnant if nothing was adopted, or
    if the winner's Δ CI lower bound ≤ ε. With the default ε=0.0 an adoption
    (the gate requires lo > margin ≥ 0) never counts, so this equals counting
    non-adoptions; only when ε > margin do marginal adoptions count as stagnant."""
    if not decision.get("adopted"):
        return True
    lo = ((decision.get("evaluated") or {}).get(decision.get("id"), {})
          .get("gate", {}).get("lo"))
    return lo is not None and lo <= epsilon


def _empty_material(axes: dict, materials, manual_aliases: dict) -> str | None:
    """Name of an enabled axis whose materials are empty — meaning the candidate
    equals baseline.

    Common for alias and text_negatives — caption pools get exhausted by the
    safety rules. Going through screen and holdout eval while empty just spends
    budget re-confirming Δ≈0. Rejecting here leaves a history entry that also
    feeds back to the planner."""
    if axes.get("exemplar", "none") != "none" and not any(
            (materials.exemplars or {}).values()):
        return "exemplar"
    if axes.get("visual_negatives") and not any(
            (materials.visual_negatives or {}).values()):
        return "visual_negatives"
    # generated aliases must be non-empty — manual aliases alone do not count as mined material
    if axes.get("alias"):
        generated = {k: v for k, v in (materials.aliases or {}).items()
                     if (manual_aliases or {}).get(k) != v}
        if not generated:
            return "alias"
    if axes.get("text_negatives") and not any(
            (materials.text_negatives or {}).values()):
        return "text_negatives"
    return None


_ROUND_PREFIX = re.compile(r"^(?:carry_)*(?:r\d+_)?")


def _round_ids(items: list[dict], r: int, taken: set) -> list[dict]:
    """Make candidate ids unique within the run — an r<round>_ prefix (if the
    planner imitated an earlier round's prefix, it is stripped and re-added).
    Carried candidates made by the loop (source=carryover, carry_<id>) keep
    their id, since the original id already has a prefix. On a clash with taken
    (ids already used this round) _2, _3… is appended. The input dicts are not
    modified — replay artifacts (proposal*.json) keep the planner's ids, and
    resume rebuilds the same ids by the same rule."""
    out = []
    for c in items:
        cid = str(c["id"])
        # Only real carried candidates are exempt — if the planner imitates a
        # carry_ id, carry_ and the round prefix are stripped and this round's
        # prefix is added (so ids do not clash across rounds)
        if c.get("source") != "carryover":
            cid = f"r{r}_" + _ROUND_PREFIX.sub("", cid)
        base, n = cid, 2
        while cid in taken:
            cid, n = f"{base}_{n}", n + 1
        taken.add(cid)
        out.append({**c, "id": cid})
    return out


def _with_round_ids(proposal: dict, r: int, taken: set) -> dict:
    """Put round ids on a proposal (passing candidates + drop rows) — once, before
    recording."""
    cands = _round_ids(proposal["candidates"], r, taken)
    return {**proposal, "candidates": cands,
            "dropped": _round_ids(list(proposal.get("dropped") or []), r, taken)}


def _token_budget(cfg) -> int:
    """Prompt token budget (by the estimate) — the vLLM server context minus
    the planner output cap and template slack. Even when attached to an
    external server via base_url (actual context unknown), the launch value is
    the reference."""
    ctx = getattr(cfg.planner.launch, "max_model_len", 8192)
    return ctx - OUTPUT_MAX_TOKENS - CONTEXT_MARGIN


def _winner_axes(winner: dict) -> tuple[dict, bool]:
    """Champion label reconciliation — axes.rescorer must match whether
    rescorer_path exists.

    The backstop after _arm_rescorer's own correction: any label that arrives
    enabled without a rescorer_path is turned off. If labels drift from
    serving, dedup, diagnosis and champion.json record a layer that does not exist."""
    axes = dict(winner["axes"])
    if axes.get("rescorer") and not winner.get("rescorer_path"):
        axes["rescorer"] = False
        return axes, True
    return axes, False


def _measured_axes(axes: dict, *, rescorer: bool) -> dict:
    """Reconcile candidate axes with **the actually measured state**.

    Upstream arms (lora, classifier) also refit the rescorer and measure with
    it on. If the refit is identity (α=0 for every class), axes reflects that."""
    return {**axes, "rescorer": bool(rescorer)}


# axis -> injected hook required to build that axis's materials. Without it
# materialize raises in _require (no silent axis disabling), but by then mining
# has started and round budget is spent — so filter unwireable ones up front.
_AXIS_HOOKS: tuple = (
    ("alias", lambda v: "vlm_caption_fn"),
    ("alias", lambda v: "refine_alias_fn"),
    ("text_negatives", lambda v: "caption_fn"),
    ("text_negatives", lambda v: "refine_negatives_fn"),
)


def _missing_hook(axes: dict, hooks: dict) -> str | None:
    """First hook required by an enabled axis that was not injected."""
    for axis, pick in _AXIS_HOOKS:
        value = axes.get(axis)
        if not value:
            continue
        name = pick(value)
        if hooks.get(name) is None:
            return name
    return None


def _same_state(a, b) -> bool:
    """Compare detector-state paths (weights, artifact) — str and Path mix, None is the base state."""
    return (a is None) == (b is None) and (a is None or Path(a) == Path(b))


def _route_detector_state(axes: dict, champion_axes: dict) -> str:
    """Routing for detector-state axes (lora, crop_classifier) — based on axes
    **newly enabled** relative to the champion. Axes inherited from the champion
    are already loaded in the client (2D state: weights × adapt_root), so normal
    wiring suffices and the tail arm grid is not needlessly re-run. Two newly
    enabled axes in one candidate cannot be attributed, so it is rejected.
    Sequential combinations (a single increment of one axis on top of a
    champion that holds the other) are allowed via the corresponding arm: the
    classifier arm runs on the current weights, the lora arm preserves adapt_root.

    A candidate that turns off an axis the champion holds is "off" — the
    detector state is not unloaded per candidate, so measuring it through
    normal wiring would put an off label on a result measured with the state
    still loaded (and, if adopted, split the label from what is served)."""
    if any(champion_axes.get(k) and not axes.get(k)
           for k in ("lora", "crop_classifier")):
        return "off"
    new_lora = bool(axes.get("lora")) and not champion_axes.get("lora")
    new_cls = (bool(axes.get("crop_classifier"))
               and not champion_axes.get("crop_classifier"))
    if new_lora and new_cls:
        return "compound"
    if new_lora:
        return "lora"
    if new_cls:
        return "classifier"
    return "wired"


def _empty_round(decision: dict) -> bool:
    """Did the round measure nothing on holdout — evaluated still empty after the
    re-asks (no_candidates and planner_stop included)? Not the baseline.

    If some candidate dropped out at the pre-holdout fit_dev verdict
    (fit_dev_lost, fit_dev_identical — the decision's `fit_dev_skipped`), the
    round is not empty — without the skip it would have measured on holdout.
    The skip only saves holdout time; it does not change when the run stops.
    With no adoption it still counts as stagnant toward saturation.rounds
    (_stagnant)."""
    return (decision.get("kind") != "baseline" and not decision.get("evaluated")
            and not decision.get("fit_dev_skipped"))


def _trailing_empty(history: list[dict]) -> int:
    """Consecutive empty rounds at the tail of history — restores the
    empty_rounds counter on resume. The baseline (round 0) is always a
    boundary."""
    n = 0
    for e in reversed(history):
        if e.get("round", 0) == 0 or not _empty_round(e.get("decision") or {}):
            break
        n += 1
    return n


def _saturated(sat, stagnant: int, empty: int) -> str | None:
    """The stop reason, or None. An empty round is also stagnant and counts
    toward saturation.rounds, so when both are reached in the same round the
    wider one, saturation, is recorded."""
    if stagnant >= sat.rounds:
        return "saturation"
    if sat.empty_rounds and empty >= sat.empty_rounds:
        return "empty_rounds"
    return None


def _trailing_stagnant(history: list[dict], epsilon: float) -> int:
    """Number of consecutive stagnant rounds at the tail of history — restores
    the saturation counter on resume. baseline (round 0) is always a boundary."""
    n = 0
    for e in reversed(history):
        if e.get("round", 0) == 0 or not _stagnant(e.get("decision") or {}, epsilon):
            break
        n += 1
    return n
