"""Round orchestration —
propose → materialize → screen(fit_dev) → evaluate(holdout) → decide → record.

Diagnosis is computed at the **start** of each round from the current
champion — equivalent to "end-of-round diagnosis feeds the next propose", and
it keeps resume simple.

rescorer candidates are wired by loop.rescore, lora candidates (YOLOE tail
fine-tune) by loop.tail_round, crop classifier candidates by loop.adapt_round.
The latter two are **detector-state candidates**: they swap the
in-process detector client's weights (weights_ctl) or reclassification
artifact (adapt_ctl), so they run only when `detector_hooks` is injected; otherwise
they are just recorded as `skipped` (being recorded, they join the tried set
and are not re-proposed). Because they mutate client state they are handled
**after all other candidates are evaluated**, and the current state is
restored if they are not adopted.
"""

from __future__ import annotations

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
from ..planner.axes import AXES, BASELINE, sanitize_candidates
from ..planner.llm import LLMPlanner
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


def _load_candidates_file(path, disabled_axes=(), disabled_values=None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or []
    cands = sanitize_candidates(raw, max_candidates=64,
                                disabled_axes=disabled_axes,
                                disabled_values=disabled_values)
    for c in cands:
        c["id"] = f"human_{c['id']}"
        c["source"] = "human"
    return cands


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
                  "rescorer_path": champion.get("rescorer_path")})
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
        evaluation changes the score channel. If champion and candidate are on
        different channels, the paired Δ becomes "candidate ± calibration
        layer" instead of the candidate's increment, and if those records become
        the champion the contamination spreads to later rounds. The branch lives
        in this one place so call sites cannot drift apart.
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

        champion.json only **records** the paths; a new process's client starts
        with base weights and no reclassifier. Without re-applying them, the
        round's candidates are scored with a **different detector** than the
        champion, while the gate compares paired against the champion's recorded
        records — so Δ becomes "candidate − champion's state gain" rather than
        the candidate's increment. No error or warning is raised and the axis
        labels, inherited from the champion, still look enabled — a silent
        failure.
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
            tiling_variants=(s.tiling_configs if axes.get("tiling") else ()))

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
        d_train = diagnose(recs, floor_recs, tau, self.ds.ontology,
                             iou_thr=self.iou, anonymize=False)
        d_test = diagnose(champion["eval"]["records"],
                          champion["eval"]["floor_records"], tau,
                          self.ds.ontology, iou_thr=self.iou, anonymize=True)
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
                                  char_budget=_char_budget(self.cfg),
                                  disabled_axes=self.disabled_axes,
                                  disabled_values=self.disabled_values).propose(
                history, diagnosis)
        else:
            proposal = deterministic.propose(history, diagnosis,
                                             disabled_axes=self.disabled_axes,
                                             disabled_values=self.disabled_values)
            _write(round_dir / "proposal.json", proposal)
        # merge human-injected candidates (hybrid loop)
        humans = _load_candidates_file(self.candidates_file, self.disabled_axes,
                                       self.disabled_values)
        merged = proposal["candidates"] + humans

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
                "dropped": list(proposal.get("dropped") or []) + dropped,
                "stop": proposal["stop"] and not fresh}

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
            rescorer, info, dev_preds = rescore.fit_for_setting(
                self.cfg, self.client, self.ds, setting0, materials, mat_dir)
            if rescorer is None or not rescorer.active:
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
        entry.update({"setting": setting,
                      "fit_dev_micro_f_beta": setting["fit_dev_micro_f_beta"],
                      "_materials": materials})
        screened.append(entry)

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
            "champion": {"axes": champion["axes"], "round": 0, "id": "baseline",
                         "micro_f_beta": ev["micro_f_beta"]},
            "candidates": [], "decision": decision})
        return champion

    def run_round(self, r: int, champion: dict, history: list[dict]) -> tuple[dict, dict]:
        """One round — returns (decision, champion)."""
        self._round_t0 = time.monotonic()
        self._champion = champion          # parent for material inheritance
        rd = self._round_dir(r)
        rd.mkdir(parents=True, exist_ok=True)
        diagnosis = self._diagnosis(champion, rd)
        self._crop_notes(diagnosis, rd)      # merge VLM notes (diagnostic signal only)
        proposal = self._propose(history, diagnosis, rd)

        wired, lora_cands, classifier_cands = [], [], []
        skipped = list(proposal.get("dropped", []))   # include combo_tried records
        champ_axes = champion.get("axes") or {}
        for c in proposal["candidates"]:
            miss = _missing_hook(c["axes"], self._mat_kw)
            if miss:
                skipped.append({**c, "skipped": "missing_hook",
                                "note": f"{miss} not injected — cannot produce material "
                                        f"for this axis (needs LLM/worker wiring)"})
                continue
            route = _route_detector_state(c["axes"], champ_axes)
            if route == "compound":
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
        if skipped:
            log.warning("skipped %d candidates that cannot be wired: %s", len(skipped),
                        [c["id"] for c in skipped])

        if not wired and not lora_cands and not classifier_cands:
            decision = {"round": r, "adopted": False,
                        "kind": ("planner_stop" if proposal["stop"] else "no_candidates"),
                        "skipped": [c["id"] for c in skipped]}
            _write(rd / "decision.json", decision)
            self._append_history({
                "round": r,
                "champion": {"axes": champion["axes"], "round": champion["round"],
                             "id": champion["id"],
                             "micro_f_beta": champion["eval"]["micro_f_beta"]},
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
        # Evaluate with the champion's classifier **left loaded**. The classifier
        # is a crop -> class function, not calibrated to the detector's score
        # distribution (its only detector dependence is that background crops
        # were mined from old FPs). Removing it would make lora fight carrying a
        # handicap equal to the classifier's gain.
        adapt_dirty = False
        lora_dirty = False
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
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("lora candidate %s failed — restored current weights and rejected", c["id"])
                self.detector_hooks["weights_ctl"](champion.get("weights"))
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                continue
            entry = {**c, "materials_dir": str(mat_dir),
                     "setting": res["setting"],
                     "fit_dev_micro_f_beta": max((g["f_beta"] for g in res["grid"]),
                                             default=0.0),
                     "lora": res["decision"]}
            screened.append({k: entry[k] for k in
                             ("id", "axes", "materials_dir", "setting",
                              "fit_dev_micro_f_beta")} | {"lora": res["decision"]})
            if not res["decision"].get("adopted"):
                skipped.append({**c, "skipped": "lora_gate",
                                "note": res["decision"].get("reason", "")})
                continue
            lora_dirty = True            # client is on the winning arm
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
            except Exception as exc:  # noqa: BLE001 — a candidate failure is that candidate's rejection
                log.exception("classifier candidate %s failed — restored current state and rejected", c["id"])
                self._adapt_ctl(champion.get("adapt_root"))
                skipped.append({**c, "skipped": "error",
                                "note": f"{type(exc).__name__}: {exc}"})
                continue
            entry = {**c, "materials_dir": str(mat_dir),
                     "setting": res["setting"],
                     "fit_dev_micro_f_beta":
                         res["setting"]["fit_dev_micro_f_beta"],
                     "crop_classifier": res["decision"]}
            screened.append({k: entry[k] for k in
                             ("id", "axes", "materials_dir", "setting",
                              "fit_dev_micro_f_beta")}
                            | {"crop_classifier": res["decision"]})
            if not res["decision"].get("adopted"):
                skipped.append({**c, "skipped": "classifier_gate",
                                "note": res["decision"].get("reason", "")})
                continue
            adapt_dirty = True           # client has the artifact loaded
            setting = self._arm_rescorer(entry, res["setting"], materials, mat_dir)
            entry["setting"] = setting
            ev = self._evaluate_candidate(entry, setting, materials)
            evaluated.append({**entry, "_materials": materials,
                              "_adapt_root": str(res["adapt_root"]),
                              "eval": ev, "gate": _gate(ev)})

        winners = [c for c in evaluated if c["gate"]["adopted"]]
        winner = max(winners, key=lambda c: c["gate"]["delta"]) if winners else None
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
        if lora_dirty and not (winner and "_weights" in winner):
            # the lora arm failed the gate (or another candidate won) — restore current
            self.detector_hooks["weights_ctl"](champion.get("weights"))
        if adapt_dirty and not (winner and "_adapt_root" in winner):
            self._adapt_ctl(champion.get("adapt_root"))
        _write(rd / "decision.json", decision)
        self._append_history({
            "round": r,
            "champion": {"axes": champion["axes"], "round": champion["round"],
                         "id": champion["id"],
                         "micro_f_beta": champion["eval"]["micro_f_beta"]},
            "candidates": ([{k: c[k] for k in ("id", "axes")}
                            | {"adopted": bool(winner and c["id"] == winner["id"]),
                               "fit_dev_micro_f_beta": c["fit_dev_micro_f_beta"]}
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

    Especially common for alias and text_negatives — caption pools get
    exhausted by safety rules (own parts, kin classes, category names). Going
    through screen and holdout eval while empty just spends round budget
    re-confirming Δ≈0. Rejecting here leaves a history entry that also feeds
    back to the planner."""
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


def _char_budget(cfg) -> int:
    """Prompt character budget — derived from the vLLM server context.
    (max_model_len − output reserve) × chars/token (conservative 2.0) × 0.9 slack.
    Even when attached to an external server via base_url (actual context
    unknown), the launch value is the reference."""
    ctx = getattr(cfg.planner.launch, "max_model_len", 8192)
    return max(4000, int((ctx - 1024) * 2.0 * 0.9))


def _winner_axes(winner: dict) -> tuple[dict, bool]:
    """Champion label reconciliation — axes.rescorer must match whether
    rescorer_path exists.

    Labels must reflect **the actual serving state**. Arms refit the rescorer
    and measure with it on (_arm_rescorer), which already corrects axes if the
    refit is identity — this function is the backstop after that: any label
    that arrives enabled without a rescorer_path is turned off. If labels drift
    from serving, the next round's dedup, diagnosis and champion.json all
    record a layer that does not exist."""
    axes = dict(winner["axes"])
    if axes.get("rescorer") and not winner.get("rescorer_path"):
        axes["rescorer"] = False
        return axes, True
    return axes, False


def _measured_axes(axes: dict, *, rescorer: bool) -> dict:
    """Reconcile candidate axes with **the actually measured state**.

    Upstream arms (lora, classifier) also refit the rescorer and measure with
    it on — removing it would make the gate's Δ a "+upstream, −rescorer"
    composite, handicapping the upstream axis by the rescorer's gain. If the
    refit is identity (α=0 for every class), axes reflects that: labels must
    match the measured state (_winner_axes checks the same thing)."""
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


def _route_detector_state(axes: dict, champion_axes: dict) -> str:
    """Routing for detector-state axes (lora, crop_classifier) — based on axes
    **newly enabled** relative to the champion. Axes inherited from the champion
    are already loaded in the client (2D state: weights × adapt_root), so normal
    wiring suffices and the tail arm grid is not needlessly re-run. Two newly
    enabled axes in one candidate cannot be attributed, so it is rejected.
    Sequential combinations (a single increment of one axis on top of a
    champion that holds the other) are allowed via the corresponding arm: the
    classifier arm runs on the current weights, the lora arm preserves adapt_root."""
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
