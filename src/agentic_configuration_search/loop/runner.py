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

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ..config import Config, config_from_dict
from ..core.data import Dataset, load_dataset
from ..diagnose import describe
from ..diagnose.errors import diagnose
from ..eval.evaluate import evaluate_on_holdout, gate
from ..eval.runner import chosen_groups, infer_for_groups, setting_records
from ..eval.screen import screen_candidate
from ..fitting.materials import Materials
from ..planner import deterministic
from ..planner.deterministic import tried_combos
from ..planner.axes import AXES, BASELINE, DatasetFacts, sanitize_candidates
from ..planner.llm import CONTEXT_MARGIN, OUTPUT_MAX_TOKENS, LLMPlanner
from . import adapt_round, rescore, tail_round
from .materialize import materialize

log = logging.getLogger(__name__)

@dataclass
class RunResult:
    out_dir: Path
    decisions: list = field(default_factory=list)
    history: list = field(default_factory=list)
    champion: dict | None = None
    stopped: str = "rounds"           # rounds | saturation | planner_stop


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "cand"


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _setting_lite(setting: dict) -> dict:
    return {"configs": setting["configs"],
            "per_class": {str(k): v for k, v in setting["per_class"].items()}}


def _setting_load(lite: dict) -> dict:
    return {"configs": lite["configs"],
            "per_class": {int(k): v for k, v in lite["per_class"].items()}}


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
        self.ds: Dataset = load_dataset(dataset_root, seed=cfg.split.seed,
                                        fit_train_ratio=cfg.split.fit_train_ratio)
        self.out_dir = Path(out_dir)
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
        self.detector_hooks = detector_hooks
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
                  "rescorer_path": champion.get("rescorer_path"),
                  "forwards_per_image": setting_forwards(champion["setting"])})
        _write(self.out_dir / "champion_eval.json", champion["eval"])

    def _load_champion(self) -> dict:
        meta = json.loads((self.out_dir / "champion.json").read_text())
        ev = json.loads((self.out_dir / "champion_eval.json").read_text())
        ev["tau_test"] = {int(k): v for k, v in ev["tau_test"].items()}
        return {"round": meta["round"], "id": meta["id"], "axes": meta["axes"],
                "setting": _setting_load(meta["setting"]),
                "materials_dir": meta["materials_dir"], "eval": ev,
                "weights": meta.get("weights"),
                "adapt_root": meta.get("adapt_root"),
                "rescorer_path": meta.get("rescorer_path")}

    def _append_history(self, entry: dict) -> None:
        # timestamps for analysis — ts is write time, dur_s is round wall-clock duration
        entry.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        if getattr(self, "_round_t0", None) is not None:
            entry.setdefault("dur_s", round(time.monotonic() - self._round_t0, 1))
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
        The branch lives in this one place so call sites cannot drift apart.
        """
        if entry.get("_rescorer") is not None:
            return rescore.rescored_evaluate(self.cfg, self.client, self.ds,
                                             setting, materials,
                                             entry["_rescorer"])
        return self._evaluate(setting, materials)

    def _adapt_ctl(self, path) -> None:
        """Load/unload the classifier artifact."""
        self.detector_hooks["adapt_ctl"](path)

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
            self.detector_hooks["weights_ctl"](Path(weights))
        if adapt_root and self.detector_hooks.get("adapt_ctl"):
            log.info("resume: restored champion reclassification artifact %s", adapt_root)
            self._adapt_ctl(Path(adapt_root))

    def _screen(self, materials: Materials, axes: dict) -> dict:
        s = self.cfg.search
        # text-only candidate: the text pass sweeps the text-negative suppression
        # IoU (neg_eta) over the grid (text-only branch of screen_candidate). With
        # exemplars, semantic picks the grid point and text follows that config.
        text_neg_sweep = (axes.get("text_negatives")
                          and axes.get("exemplar") == "none")
        return screen_candidate(
            self.cfg, self.client, self.ds.fit_dev, self.ds.ontology, materials,
            neg_eta_grid=(s.neg_eta_grid
                          if (axes.get("visual_negatives") or text_neg_sweep)
                          else ()),
            tiling_variants=_tiling_variants(s, axes))

    def _diagnosis(self, champion: dict, round_dir: Path) -> dict:
        """Diagnosis on train (detailed) + holdout (anonymized aggregates) — the
        planner's input. (Artifact key/file name "test" is legacy.)"""
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
        d_test = diagnose(champion["eval"]["records"],
                          champion["eval"]["floor_records"], tau,
                          self.ds.ontology, iou_thr=self.iou, anonymize=True,
                          **small)
        _write(round_dir / "diagnosis_train.json", d_train)
        _write(round_dir / "diagnosis_test.json", d_test)
        return {"train": d_train, "test": d_test}

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

    def _propose(self, history: list[dict], diagnosis: dict,
                 round_dir: Path) -> dict:
        if self.cfg.planner.kind == "llm":
            chat = self._planner_chat()
            if chat is None:
                raise ValueError("planner.kind=llm requires planner.base_url or "
                                 "an injected llm_chat")
            proposal = LLMPlanner(chat=chat, artifact_dir=round_dir,
                                  notes=self.cfg.dataset_notes,
                                  token_budget=_token_budget(self.cfg),
                                  disabled_axes=self.disabled_axes,
                                  disabled_values=self.disabled_values,
                                  facts=self._facts()).propose(
                history, diagnosis)
        else:
            proposal = deterministic.propose(history, diagnosis,
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
        merged = proposal["candidates"] + humans + carried

        # Common combo filter: regardless of source (LLM, deterministic, human),
        # block retrying **an already-tried full axes combination**. It works per
        # combination, not per (axis, value), so retrying the same axis after the
        # champion changes is allowed. The LLM can forget failures outside the
        # prompt window (MAX_HISTORY), so this filter is the only permanent barrier.
        tried = tried_combos(history)
        fresh, dropped = [], []
        for c in merged:
            key = tuple(sorted(c["axes"].items()))
            if key in tried:
                # no silent drops — record as combo_tried in skipped so it flows via
                # history into the next round's prompt (learning about out-of-window re-proposals).
                dropped.append({**c, "skipped": "combo_tried",
                                "note": "axes combination already tried — combination-level dedup "
                                        "blocks it permanently. Vary it or try another axis."})
                continue
            tried.add(key)                # also collapses duplicates within one proposal
            fresh.append(c)
        if dropped:
            log.info("filtered %d already-tried combinations: %s", len(dropped),
                     [c["id"] for c in dropped])
        # merge with planner-stage drops (invalid_axes etc.) — overwriting would
        # erase candidates discarded by sanitization from history.
        return {**proposal, "candidates": fresh,
                "dropped": (list(proposal.get("dropped") or [])
                            + human_dropped + dropped),
                "stop": proposal["stop"] and not fresh}

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

    def _screen_one(self, c: dict, mat_dir, entry: dict,
                    screened: list, skipped: list) -> None:
        """materialize + screen for one candidate — the caller records failures as rejections."""
        if c["axes"].get("rescorer"):
            # the rescorer is a post-processing layer on the setting fixed by the
            # other axes — discriminator on fit_train, α grid on fit_dev (loop.rescore)
            base_axes = {**c["axes"], "rescorer": False}
            materials = materialize(self.cfg, self.ds, base_axes, mat_dir,
                                    seed=self.cfg.run_seed,
                                    **self._parent_kw(), **self._mat_kw)
            if self._guard_material(c, materials, skipped):
                return
            setting0 = self._screen(materials, base_axes)
            # judge sameness **before** the refit — if the served state equals the
            # champion's, a sibling's or an already-evaluated candidate's, skip
            # the discriminator refit (train re-inference) and holdout altogether
            if self._skip_same(c, entry, setting0, materials, skipped):
                return
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
            materials = materialize(self.cfg, self.ds, c["axes"], mat_dir,
                                    seed=self.cfg.run_seed,
                                    **self._parent_kw(), **self._mat_kw)
            if self._guard_material(c, materials, skipped):
                return
            setting = self._screen(materials, c["axes"])
            if self._skip_same(c, entry, setting, materials, skipped):
                return
        entry.update({"setting": setting,
                      "fit_dev_micro_f_beta": setting["fit_dev_micro_f_beta"],
                      "_materials": materials})
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
        rec |= _modality_record(setting, self.ds.ontology)
        if unused:
            rec.setdefault("unused_axes", unused)
            if "tiling" in unused:
                rec["tiling_used"] = False
        log.info("candidate %s: %s", c["id"], rec["note"])
        skipped.append(rec)
        return True

    def _served_state(self, entry: dict) -> dict:
        """Server state a wired candidate is served with — the champion's weights
        and artifact (or the prompt dictionary alias widened, if any)."""
        ch = getattr(self, "_champion", None) or {}
        return {"weights": ch.get("weights"),
                "adapt_root": entry.get("_adapt_root", ch.get("adapt_root"))}

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
        rd = self._round_dir(0)
        mat_dir = rd / "settings" / "baseline"
        materials = materialize(self.cfg, self.ds, BASELINE, mat_dir,
                                seed=self.cfg.run_seed, **self._mat_kw)
        setting = self._screen(materials, BASELINE)
        ev = self._evaluate(setting, materials)
        champion = {"round": 0, "id": "baseline", "axes": dict(BASELINE),
                    "setting": setting, "materials_dir": str(mat_dir),
                    "eval": ev, "weights": None, "adapt_root": None,
                    "rescorer_path": None}
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
        """One round — returns (decision, champion)."""
        self._round_t0 = time.monotonic()
        self._champion = champion          # parent for material inheritance
        self._champion.pop("_fp", None)
        self._seen_fps: set[str] = set()     # served states that passed screen this round
        self._tried_fps = _evaluated_fps(history)
        rd = self._round_dir(r)
        rd.mkdir(parents=True, exist_ok=True)
        diagnosis = self._diagnosis(champion, rd)
        self._crop_notes(diagnosis, rd)      # merge VLM notes (diagnostic signal only)
        proposal = self._propose(history, diagnosis, rd)

        wired, lora_cands, classifier_cands = [], [], []
        skipped = list(proposal.get("dropped", []))   # include combo_tried records
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
            decision = {"round": r, "adopted": False,
                        "kind": ("planner_stop" if proposal["stop"] else "no_candidates"),
                        "skipped": [c["id"] for c in skipped]}
            _write(rd / "decision.json", decision)
            self._append_history({
                "round": r,
                "champion": _champion_summary(champion),
                "candidates": skipped, "decision": decision})
            return decision, champion

        screened = []
        for c in wired:
            mat_dir = rd / "settings" / _slug(c["id"])
            entry = {**c, "materials_dir": str(mat_dir)}
            try:
                self._screen_one(c, mat_dir, entry, screened, skipped)
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("candidate %s failed — rejected, round continues", c["id"])
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
        _write(rd / "screen.json",
               [{k: c[k] for k in ("id", "axes", "fit_dev_micro_f_beta")}
                for c in screened])

        top = sorted(screened, key=lambda c: -c["fit_dev_micro_f_beta"])
        top = top[: self.cfg.screen.top_k]
        evaluated = []

        def _gate(ev):
            return gate(champion["eval"]["records"], ev["records"],
                        cat_ids=self.cat_ids, iou_thr=self.iou,
                        margin=self.cfg.gate.margin,
                        n_boot=self.cfg.gate.bootstrap, seed=self.cfg.run_seed,
                        alpha=1.0 - self.cfg.gate.confidence,
                        beta=self.cfg.beta)

        for c in top:
            ev = self._evaluate_candidate(c, c["setting"], c["_materials"])
            evaluated.append({**c, "eval": ev, "gate": _gate(ev)})

        # --- lora candidates — they swap the client's weights, so always last ---
        # Evaluate with the champion's classifier left loaded — a crop -> class
        # function is not tied to the detector's score distribution, and removing
        # it would make lora fight carrying a handicap equal to its gain.
        # served_* is what the detector has loaded right now (updated per candidate).
        served_w = champion.get("weights")
        served_a = champion.get("adapt_root")
        for c in lora_cands:
            mat_dir = rd / "settings" / _slug(c["id"])
            base_axes = {**c["axes"], "lora": False}
            try:
                materials = materialize(self.cfg, self.ds, base_axes, mat_dir,
                                        seed=self.cfg.run_seed,
                                        **self._parent_kw(), **self._mat_kw)
            except Exception as exc:  # noqa: BLE001
                log.exception("lora candidate %s material failed — rejected", c["id"])
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                continue
            if self._guard_material(c, materials, skipped):
                continue
            try:
                # lora axis = tail fine-tune + λ-interpolation arms —
                # contract {"decision","weights","grid","setting"}
                res = tail_round.run_tail_candidate(
                    self.cfg, self.ds, base_axes, materials, mat_dir,
                    screen_fn=self._screen,
                    weights_ctl=self.detector_hooks["weights_ctl"],
                    base_weights=self.detector_hooks["base_weights"],
                    restore_weights=champion.get("weights"),
                    seed=self.cfg.run_seed)
                # adopted: the winning arm stays loaded; otherwise restore_weights (current)
                served_w = res["weights"] or champion.get("weights")
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("lora candidate %s failed — restored current weights and rejected", c["id"])
                self.detector_hooks["weights_ctl"](champion.get("weights"))
                served_w = champion.get("weights")
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                continue
            entry = {**c, "materials_dir": str(mat_dir),
                     "setting": res["setting"],
                     "fit_dev_micro_f_beta": max((g["f_beta"] for g in res["grid"]),
                                             default=0.0),
                     "lora": res["decision"]}
            _relabel_unused(entry)
            screened.append({k: entry[k] for k in
                             ("id", "axes", "materials_dir", "setting",
                              "fit_dev_micro_f_beta")} | {"lora": res["decision"]}
                            | _requested(entry) | _unused_record(entry))
            if not res["decision"].get("adopted"):
                skipped.append({**c, "skipped": "lora_gate",
                                "note": res["decision"].get("reason", "")})
                continue
            setting = self._arm_rescorer(entry, res["setting"], materials, mat_dir)
            entry["setting"] = setting
            ev = self._evaluate_candidate(entry, setting, materials)  # winning arm's setting
            evaluated.append({**entry, "_materials": materials,
                              "_weights": str(res["weights"]),
                              "eval": ev, "gate": _gate(ev)})
        # keep records too — re-judging past runs after changing gate criteria
        # (one/two-sided, alpha, margin) needs more than the champion's records.
        _write(rd / "eval.json",
               [{"id": c["id"], "axes": c["axes"],
                 "micro_f_beta": c["eval"]["micro_f_beta"],
                 "tau_test": c["eval"]["tau_test"], "gate": c["gate"],
                 "records": c["eval"]["records"]}
                for c in evaluated])

        # --- crop classifier candidates — detector-state candidates, same discipline as lora ---
        # Measure them on the champion's weights. If a winning lora arm stayed
        # loaded and the classifier then won, the champion would be served without
        # that arm while its τ and holdout F came from on top of it — they would
        # not match the serving state (adapt_round restores only the artifact).
        if classifier_cands and not _same_state(served_w, champion.get("weights")):
            self.detector_hooks["weights_ctl"](champion.get("weights"))
            served_w = champion.get("weights")
        for c in classifier_cands:
            mat_dir = rd / "settings" / _slug(c["id"])
            base_axes = {**c["axes"], "crop_classifier": False}
            try:
                materials = materialize(self.cfg, self.ds, base_axes, mat_dir,
                                        seed=self.cfg.run_seed,
                                        **self._parent_kw(), **self._mat_kw)
                if self._guard_material(c, materials, skipped):
                    continue
                res = adapt_round.run_classifier_candidate(
                    self.cfg, self.ds, base_axes, materials, mat_dir,
                    screen_fn=self._screen,
                    adapt_ctl=self._adapt_ctl,
                    repo_root=self.detector_hooks["repo_root"],
                    restore_root=champion.get("adapt_root"),
                    seed=self.cfg.run_seed, client=self.client)
                # adopted: its own artifact stays loaded; otherwise restore_root (current)
                served_a = res["adapt_root"] or champion.get("adapt_root")
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("classifier candidate %s failed — restored current state and rejected", c["id"])
                self._adapt_ctl(champion.get("adapt_root"))
                served_a = champion.get("adapt_root")
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                continue
            entry = {**c, "materials_dir": str(mat_dir),
                     "setting": res["setting"],
                     "fit_dev_micro_f_beta":
                         res["setting"]["fit_dev_micro_f_beta"],
                     "crop_classifier": res["decision"]}
            _relabel_unused(entry)
            screened.append({k: entry[k] for k in
                             ("id", "axes", "materials_dir", "setting",
                              "fit_dev_micro_f_beta")}
                            | {"crop_classifier": res["decision"]}
                            | _requested(entry) | _unused_record(entry))
            if not res["decision"].get("adopted"):
                # The tripwire (too few training crops) keeps the same material
                # until the champion changes — record the axis as empty so it reaches
                # the planner's dead-axis list. It cannot be blocked up front because
                # the background FP count is an inference result.
                dead = ({"empty_axis": "crop_classifier"}
                        if res["decision"].get("tripwire") else {})
                skipped.append({**c, "skipped": "classifier_gate", **dead,
                                "note": res["decision"].get("reason", "")})
                continue
            setting = self._arm_rescorer(entry, res["setting"], materials, mat_dir)
            entry["setting"] = setting
            ev = self._evaluate_candidate(entry, setting, materials)
            evaluated.append({**entry, "_materials": materials,
                              "_adapt_root": str(res["adapt_root"]),
                              "eval": ev, "gate": _gate(ev)})

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
            "skipped": [c["id"] for c in skipped],
        }
        if winner:
            w_axes, _ = _winner_axes(winner)     # defensive rescorer label reconciliation
            # downstream inheritance: a classifier-arm winner keeps its own artifact;
            # any other winner (lora included) inherits the champion's as-is.
            adapt_root = winner.get("_adapt_root", champion.get("adapt_root"))
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
                        "rescorer_path": winner.get("rescorer_path")}
            self._save_champion(champion)
            decision["micro_f_beta"] = winner["eval"]["micro_f_beta"]
        # Bring the detector to the final champion's state. Detector-state
        # candidates swap it in turn, so the last candidate's state is not
        # necessarily the winner's — a lora winner may have been unloaded before
        # the classifier candidates.
        if not _same_state(served_w, champion.get("weights")):
            self.detector_hooks["weights_ctl"](champion.get("weights"))
        if not _same_state(served_a, champion.get("adapt_root")):
            self._adapt_ctl(champion.get("adapt_root"))
        _write(rd / "decision.json", decision)
        self._append_history({
            "round": r,
            "champion": _champion_summary(champion),
            "candidates": ([{k: c[k] for k in ("id", "axes")}
                            | {"adopted": bool(winner and c["id"] == winner["id"]),
                               "fit_dev_micro_f_beta": c["fit_dev_micro_f_beta"]}
                            | _tiling_record(c) | _neg_eta_record(c) | _requested(c)
                            | _unused_record(c)
                            | ({"setting_fp": c["setting_fp"]} if c.get("setting_fp") else {})
                            | _modality_record(c["setting"], self.ds.ontology)
                            | {"forwards": setting_forwards(c["setting"])}
                            | ({"gate_passed_but_lost": True} if c["id"] in lost else {})
                            | ({"source": c["source"]} if c.get("source") else {})
                            for c in screened] + skipped),
            "decision": decision})
        return decision, champion


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
    stagnant_rounds = _trailing_stagnant(history, cfg.saturation.epsilon)
    if stagnant_rounds >= cfg.saturation.rounds:
        result.stopped = "saturation"
    else:
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
            if stagnant_rounds >= cfg.saturation.rounds:
                result.stopped = "saturation"
                break

    _write(loop.out_dir / "run_summary.json", {
        "stopped": result.stopped,
        # whether holdout was valid or test as a stand-in — in the latter case
        # there is no sealed split and this run's micro_f_beta is optimistically
        # biased. Reporters must check this to avoid mixing both in one table (see core.data).
        "holdout_split": loop.ds.holdout_split,
        "last_round": history[-1]["round"] if history else None,
        "adopted_rounds": sum(1 for e in history if e["round"] > 0
                              and (e.get("decision") or {}).get("adopted")),
        "champion": {"id": champion.get("id"), "axes": champion.get("axes"),
                     "micro_f_beta": (champion.get("eval") or {}).get("micro_f_beta")},
        "finished_at": datetime.now().isoformat(timespec="seconds")})
    result.history = history
    result.champion = champion
    return result


def _tiling_variants(search, axes: dict) -> tuple:
    """The one grid the candidate chose — screen compares only it against tiling off.

    Why not the whole grid list: the zoom is a planner-chosen axis value (when it
    falls short the next round raises it a step), and one candidate's screen cost
    should be a predictable 1 + (1+g²) passes."""
    g = axes.get("tiling")
    if not g:
        return ()
    return ({"grid": int(g), "overlap": search.tiling_overlap,
             "merge": search.tiling_merge},)


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
            "forwards_per_image": setting_forwards(champion["setting"])}


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
    return {"tiling_used": _uses_tiling(c["setting"])}


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
    whether the rescorer is on, and the server state (weights, artifact).
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


def _trailing_stagnant(history: list[dict], epsilon: float) -> int:
    """Number of consecutive stagnant rounds at the tail of history — restores
    the saturation counter on resume. baseline (round 0) is always a boundary."""
    n = 0
    for e in reversed(history):
        if e.get("round", 0) == 0 or not _stagnant(e.get("decision") or {}, epsilon):
            break
        n += 1
    return n
