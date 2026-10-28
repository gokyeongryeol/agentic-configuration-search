"""Run configuration — explicit schema. Unknown keys are errors (no silent
absorption of typos).

Each section is one flat dataclass; nesting is only one level under Config.
Every new key must be declared in a dataclass here; an undeclared key fails
immediately with a ConfigError naming its path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar, Optional

import yaml


class ConfigError(ValueError):
    pass


PLANNER_KINDS = ("deterministic", "llm")
EXEMPLAR_SELECTIONS = ("kcenter", "mean", "random")


@dataclass(frozen=True)
class VllmLaunchCfg:
    """Launch parameters for the vLLM (planner) server."""
    gpu: int = 3
    host: str = "127.0.0.1"
    port: int = 8104
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    startup_timeout: int = 1800
    enforce_eager: bool = False
    # Persistent mode: attach to a healthy server on the port and leave it up when
    # the run ends. The planner makes few calls per round, so a restart (minutes)
    # is wasteful — on by default.
    persist: bool = True


# Cap on planner.max_candidates — the most candidates for which the longest
# response the schema allows (about 1.1k tokens per candidate) fits in
# planner.llm.OUTPUT_MAX_TOKENS (4096). Raising it means re-sizing the output
# cap (and the prompt budget it takes away) too (the truncation guard in
# tests/test_planner.py is pinned to this value).
PLANNER_MAX_CANDIDATES = 3


@dataclass(frozen=True)
class PlannerCfg:
    kind: str = "llm"                    # one of PLANNER_KINDS (deterministic is the A/B control and for environments without an LLM)
    model: str = "Qwen/Qwen3.8-27B-FP8"
    base_url: Optional[str] = None       # vLLM OpenAI-compatible endpoint; None -> services launches one
    # Show **all** prompt-sample crops to the VLM and merge its descriptions into
    # the diagnosis (kind=llm only; diagnostic signal only — never on the decision
    # path). Images per call are batched; a failed batch degrades to no notes for
    # that batch only.
    crop_notes: bool = True
    # When every candidate of a round is rejected before holdout (filters,
    # screen), how many times to ask again in the same conversation within the
    # same round — diagnosis and champion are reused. 0 turns it off. kind=llm
    # only (the deterministic planner is not re-asked)
    max_reasks: int = 1
    # Planner candidates per round — the prompt's "Propose up to N", the response
    # schema's maxItems and the re-ask cap follow it, as do the deterministic
    # planner (and the LLM's fallback). 1..PLANNER_MAX_CANDIDATES — the response
    # must fit in the output cap
    max_candidates: int = 3
    launch: VllmLaunchCfg = field(default_factory=VllmLaunchCfg)

    _NESTED: ClassVar[dict] = {"launch": VllmLaunchCfg}


@dataclass(frozen=True)
class GateCfg:
    margin: float = 0.0                  # adopt iff holdout Δ micro F(β) CI lower bound ≥ margin and Δ > 0
    min_delta: float = 0.005             # …and point estimate Δ ≥ min_delta (blocks noise-level adoptions, 0 turns it off)
    bootstrap: int = 2000                # number of paired bootstrap resamples
    confidence: float = 0.95


@dataclass(frozen=True)
class SaturationCfg:
    rounds: int = 3                      # stop after this many consecutive stagnant rounds
    epsilon: float = 0.0                 # a round is stagnant if the Δ CI lower bound <= this
    # Stop (stopped="empty_rounds") after this many consecutive rounds that
    # measured nothing on holdout (even after re-asking). 0 turns it off — such
    # rounds still count toward rounds. A round with candidates that skipped
    # holdout for losing on fit_dev (fit_dev_lost, identical) is not counted
    empty_rounds: int = 2


@dataclass(frozen=True)
class ScreenCfg:
    top_k: int = 2                       # candidates sent to holdout eval after fit_dev scoring
    # tiling prescreen — a candidate turning on grid ≥ prescreen_min_grid is first
    # scored on a deterministic fit_dev subset (max(min_images, ⌈fraction·N⌉)
    # images). If no class uses a tile there it is rejected without the full
    # screen; otherwise the full fit_dev is screened as before. A null fraction
    # turns it off. If the subset would be max_share of fit_dev or more, the full
    # screen runs directly — a passing candidate is re-screened in full, so a
    # large subset saves nothing.
    prescreen_fraction: float | None = 0.25
    prescreen_min_images: int = 20
    prescreen_min_grid: int = 3
    prescreen_max_share: float = 0.5
    # lost-value prescreen — a planner candidate containing a value that clearly
    # lost on fit_dev under the current champion (planner.ledger) is first measured
    # on the same subset (the full fit_dev without one); if the bootstrap upper
    # bound of (candidate − champion) micro F(β) is below 0 it is rejected without
    # the rescorer refit, full screen or holdout. false turns it off
    lost_value_prescreen: bool = True
    # Pre-holdout fit_dev verdict — a screened candidate whose (candidate −
    # champion) fit_dev micro F(β) bootstrap upper bound is below 0
    # (fit_dev_lost), or whose per-image fit_dev (tp, fp, fn) all equal the
    # champion's (fit_dev_identical), is not sent to holdout. top_k counts only
    # sendable candidates (the next one fills a freed slot). Applies to plain,
    # carried and lora/classifier arm candidates. false turns it off
    skip_fit_dev_losers: bool = True


@dataclass(frozen=True)
class SplitCfg:
    fit_train_ratio: float = 0.8         # fit_train share of train (image-level stratified)
    seed: int = 0                        # split seed — fixed across rounds and seed waves;
                                         # run stochasticity is held separately by Config.run_seed.


@dataclass(frozen=True)
class SubsampleCfg:
    """fit_train subset — an image cap on the training material (lora, rescorer,
    crop classifier). When fit_train is larger than max_images, images are
    picked in the order that covers the GT embedding clusters
    (fitting.subsample). Mining, fit_dev and holdout are unchanged."""
    max_images: int | None = 500   # null = no cap (all of fit_train) — 0 is refused, it reads as "0 images"
    center_ratio: float = 0.1    # k-centers per class = GT objects × ratio (at least 1)


@dataclass(frozen=True)
class LoraCfg:
    """lora axis = YOLOE tail fine-tune + λ-interpolation arms — **off by
    default**; the planner enables it on a structural-miss diagnosis. Training
    uses fit_train, λ scoring uses fit_dev. λ=0 is in the grid, so without a
    proven gain the result is exactly base. (The axis name `lora` is kept for
    history-artifact compatibility.)"""
    min_images: int = 20             # block lora candidates if fit_train has fewer images (DatasetFacts)
    lambdas: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    margin: float = 0.01             # adopt the arm only if fit_dev ΔF(β) exceeds this
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3                 # AdamW lr0 (the worker pins optimizer=AdamW)
    device: str = "cuda:0"           # keep off the shared vLLM planner GPU (planner.launch.gpu)


@dataclass(frozen=True)
class RescorerCfg:
    """rescorer axis — discriminator fit on fit_train, α grid on fit_dev.
    The α grid includes 0, so the worst case is bit-exact identical to current.
    """
    min_samples: int = 64             # tripwire for fitting the discriminator (fit_train rows)
    min_samples_per_class: int = 50   # per-class sample floor for choosing α (fit_dev rows)
    # If the top end is picked often, the range caps the search — at 4 P̂ nearly
    # overrides the raw score. Ties pick the smaller α, so a larger α wins only
    # when it is clearly better on fit_dev.
    alphas: list = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0,
                                                  3.0, 4.0])
    margin: float = 0.005             # adopt α>0 only if the fit_dev F(β) gain exceeds this
    ensemble_members: int = 5         # discriminator ensemble size
    inner_folds: int = 5              # for the oof_phat diagnostic
    # discriminator training row cap — TPs first (up to half of the cap, plus the
    # slots FPs cannot fill), FPs thinned in (class × score quartile) strata with
    # inverse-probability weights restoring the original distribution
    # (fitting.rescorer.cap_train_rows). null = no cap — 0 is refused, it reads as "0 rows"
    max_train_rows: int | None = 100_000
    # OpenMP threads of the discriminator (GBDT) fit — the entry point pins threads
    # to 1, so they are released only for the fit. The GBDT result does not depend on
    # the thread count (the same random_state gives the same model)
    fit_threads: int = 8


@dataclass(frozen=True)
class CropClassifierCfg:
    """crop classifier axis — the client gives per-box class probabilities and
    the loop blends them with λ (class = foreground argmax, score = s^(1-λ)·p^λ).
    The levers are correcting class_confusion and suppressing background FPs.
    Training runs once per run (loop.adapt_round) and each proposal only
    re-scores the λ grid. Training hyperparameters are the worker's
    universal-best constants (not knobs); only tripwires, the λ grid and the
    gate live here."""
    min_samples: int = 32            # total-crop tripwire — skip training below this
    min_background: int = 4          # minimum background crops — reclassification is pointless without them
    # blend λ grid — [0, 1]. λ=0 is relabel-only (scores unchanged, fixes only
    # class_confusion). No classifier is not a λ — it is measured separately as
    # the baseline (lam=None on the grid)
    lambdas: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    margin: float = 0.01             # adopt the arm only if fit_dev ΔF(β) exceeds this
    # train once per run and let later proposals reuse it (<run>/classifier_cache/).
    # false re-mines and retrains on each proposal's champion
    reuse: bool = True
    # while the champion has a classifier, ordinary candidates (on their own
    # setting) and lora candidates (on the winning arm) re-choose λ too (λ first,
    # then the rescorer refit on top). λ changes only when it beats the
    # champion's λ by more than margin
    retune_lambda: bool = True


@dataclass(frozen=True)
class SearchCfg:
    """Fixed parameters of the screen (fit_dev) grid evaluation. Which axes are
    on is decided by the candidate setting (planner proposal); this only holds
    the grid's value space."""
    max_threshold_candidates: int = 50
    # Tie tolerance for τ_c. Among points whose F(β) is within this band, pick the
    # one with the largest margin above the next observed score below
    # (core.metrics). 0 uses the plain F(β)-max candidate.
    tau_tolerance: float = 0.005
    # Grid of suppression IoU thresholds for emulated negatives (neg_eta)
    neg_eta_grid: list = field(default_factory=lambda: [0.25, 0.35, 0.5, 0.65, 0.75])
    # Parameters shared by every tiling grid — the grid g itself is the value of
    # the planner-chosen tiling axis (planner.axes.TILING_GRIDS). Tiles are an
    # image-relative g×g, so tile count and zoom (g-1)(1-overlap)+1 do not depend
    # on resolution. merge=nmm unions only fragments touching a seam (containment
    # union would swallow instances).
    tiling_overlap: float = 0.5
    # Merge rule — sahi (default): SAHI GreedyNMM (IOS 0.5, class-wise); nmm:
    # union only fragments at a seam; nms. min_score is the score at which
    # detections **take part in the merge** (not a drop threshold) — only those
    # at or above it are merged (SAHI filters by confidence before merging
    # because a large low-score box absorbs high-score boxes and grows); those
    # below it are kept, minus the ones overlapping a merged result, so τ_c can
    # still be picked from the floor up.
    tiling_merge: str = "sahi"
    tiling_merge_min_score: float = 0.3
    # near-miss exception of the tiling fold gate (planner.gates) — from the train
    # sizes counted at the champion's fit_dev screen thresholds
    # (diagnose.errors.size_profile), s* ("objects this small are already
    # detected") is this quantile of the TP sizes of the class with the most
    # missed_small. With fewer TPs than tiling_tp_min in that class the exception
    # is not judged. The first-grid hint (the prompt's missed-small block) shows
    # the same s*.
    tiling_tp_quantile: float = 0.1
    tiling_tp_min: int = 20
    modalities: list = field(default_factory=lambda: ["semantic", "text"])
    # Axes that may not be turned on in this run — unwireable axes are blocked at
    # proposal time instead of failing later. Honored by the planners
    # (deterministic and LLM prompt) and by candidate sanitization; a value the
    # champion already has is inherited, not blocked.
    disabled_axes: list = field(default_factory=list)
    # Forbid specific values of an axis — {axis: [value, ...]} (e.g. a material
    # source the backend does not support). Honored by the same three places
    # (deterministic, LLM prompt, sanitize).
    disabled_values: dict = field(default_factory=dict)
    # Axes turned off automatically during the run (planner.axis_status) — an axis
    # with this many rounds in a row without effect is turned off, and reopened
    # once when the champion changes. Applies to the LLM planner only (the
    # deterministic planner is the A/B control and stays as is). null = off
    # (the record is still carried in the prompt)
    axis_retire_after: int | None = 2


@dataclass(frozen=True)
class MineCfg:
    """Mining parameters — all hygiene rules and quotas. The adoption gate
    belongs to the loop.
    """
    exemplar_base_per_class: int = 4      # default quotas 4/2/1/1
    exemplar_selection: str = "kcenter"   # one of EXEMPLAR_SELECTIONS
    # Outlier trim before kcenter — drop only candidates whose centroid distance
    # exceeds median + k·MAD (null = no trim). 3.0 is about 3σ under a normal distribution.
    kcenter_outlier_mad: float | None = 3.0
    # The FN arm trims less than base (a larger multiple) — filter the noise but
    # keep the diversity that broadly covers missed appearances.
    fn_kcenter_outlier_mad: float | None = 3.5
    exemplar_fn_per_class: int = 2
    fn_min_candidates: int = 5            # headroom pre-filter (0 = off)
    fn_min_miss_rate: float = 0.02
    fn_min_area_ratio: float = 0.3        # cut tiny candidates relative to median GT area
    visual_negatives_per_class: int = 1
    visual_negative_min_score: float = 0.05
    visual_negative_exclude_gt_iom: float = 0.7
    embedder_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # gated — HF_TOKEN, or cache + HF_HUB_OFFLINE=1
    device: str = "cuda:0"                # device for the embedding and crop-classifier workers
    text_negatives_max_per_class: int = 1
    text_aliases: dict = field(default_factory=dict)  # manual aliases — always override generated ones


@dataclass(frozen=True)
class DetectorCfg:
    """Detector (YOLOE, ultralytics in-process) — no server; clients.yoloe loads it."""
    model: str = "weights_yoloe/yoloe-26x-seg.pt"   # checkpoint path (relative paths resolve against repo_root)
    device: str = "cuda:0"
    apply_topk: int = 300
    iou_threshold: float = 0.5           # IoU criterion shared by matching, τ and diagnosis


@dataclass(frozen=True)
class DiagnoseCfg:
    """Classification criteria for the FP/FN diagnosis."""
    # missed_small cut — an object is small if its size relative to the image,
    # √(box area / image area), is below this. The detector sees a resized image,
    # so the ratio, not the original px, sets how large an object looks. 0.04
    # (4% of a side) is about 26px on a square image at YOLOE's 640px input,
    # within COCO's small.
    small_object_ratio: float = 0.04


@dataclass(frozen=True)
class QueryCfg:
    """Fixed query-construction parameters — search axes (negatives, neg_eta,
    ...) travel with the candidate setting; this only holds the detector call
    contract that is fixed regardless of setting."""
    box_threshold: float = 0.03
    threshold_floor: float = 0.03        # multimodal_threshold — lower bound (floor) of τ_c


@dataclass(frozen=True)
class Config:
    dataset: str                         # Roboflow-format dataset root — train/ + holdout (valid/, else test/)
    out_dir: str = "runs"
    rounds: int = 10                     # upper bound — saturation stops earlier if it triggers
    # β of the decision metric — 1.0 is F1, β>1 weights recall, β<1 precision.
    # Threshold estimation (τ), screen scoring, the adoption gate and the
    # rescorer α curve all judge by F-beta with this β (measured = served — one
    # metric must run through the whole path).
    beta: float = 1.0
    # Optional operator-supplied description of the dataset — inserted verbatim
    # into the LLM planner prompt only (the deterministic planner ignores it).
    # With "" the prompt has no such block. The prompt is English, so write it in
    # English (the planner logs a warning on non-English text).
    dataset_notes: str = ""
    # Seed for run stochasticity — used by material-mining sampling, lora and
    # crop-classifier training, and gate bootstrap resampling. Separate from the
    # split seed (split.seed), so search variance can be measured by varying only
    # this with the fit_train/fit_dev boundary fixed.
    run_seed: int = 0
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    saturation: SaturationCfg = field(default_factory=SaturationCfg)
    screen: ScreenCfg = field(default_factory=ScreenCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    subsample: SubsampleCfg = field(default_factory=SubsampleCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    query: QueryCfg = field(default_factory=QueryCfg)
    mine: MineCfg = field(default_factory=MineCfg)
    search: SearchCfg = field(default_factory=SearchCfg)
    rescorer: RescorerCfg = field(default_factory=RescorerCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    crop_classifier: CropClassifierCfg = field(default_factory=CropClassifierCfg)
    diagnose: DiagnoseCfg = field(default_factory=DiagnoseCfg)


_SECTIONS = {"planner": PlannerCfg, "gate": GateCfg, "saturation": SaturationCfg,
             "screen": ScreenCfg, "split": SplitCfg, "subsample": SubsampleCfg,
             "detector": DetectorCfg,
             "query": QueryCfg, "mine": MineCfg, "search": SearchCfg,
             "rescorer": RescorerCfg, "lora": LoraCfg,
             "crop_classifier": CropClassifierCfg, "diagnose": DiagnoseCfg}


def _build(cls, d: dict, where: str):
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: must be a mapping (got {type(d).__name__})")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ConfigError(f"{where}: unknown key {unknown} — allowed keys: {sorted(names)}")
    nested = getattr(cls, "_NESTED", {})
    kwargs = {}
    for k, v in d.items():
        kwargs[k] = _build(nested[k], v, f"{where}.{k}") if k in nested else v
    return cls(**kwargs)


def config_from_dict(d: dict) -> Config:
    if not isinstance(d, dict):
        raise ConfigError(f"config root must be a mapping (got {type(d).__name__})")
    top = {f.name for f in fields(Config)}
    unknown = sorted(set(d) - top)
    if unknown:
        raise ConfigError(f"config: unknown key {unknown} — allowed keys: {sorted(top)}")
    if "dataset" not in d:
        raise ConfigError("config: 'dataset' is required")
    kwargs: dict = {k: v for k, v in d.items() if k not in _SECTIONS}
    for name, cls in _SECTIONS.items():
        if name in d:
            kwargs[name] = _build(cls, d[name], name)
    cfg = Config(**kwargs)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.planner.kind not in PLANNER_KINDS:
        raise ConfigError(f"planner.kind: {cfg.planner.kind!r} must be one of {PLANNER_KINDS}")
    # Axis names in search.disabled_axes are validated by Loop — config is a
    # leaf and cannot import planner.axes (AXES) (layering invariant).
    if not (0.0 < cfg.split.fit_train_ratio < 1.0):
        raise ConfigError(f"split.fit_train_ratio: outside (0,1) ({cfg.split.fit_train_ratio})")
    for key, v in (("subsample.max_images", cfg.subsample.max_images),
                   ("rescorer.max_train_rows", cfg.rescorer.max_train_rows)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            raise ConfigError(f"{key}: must be an integer or null (got {v!r}) — to remove "
                              "the cap, use an unquoted null")
    v = cfg.search.axis_retire_after
    if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
        raise ConfigError("search.axis_retire_after: must be an integer >= 1 or null "
                          f"(off) (got {v!r})")
    if cfg.subsample.max_images is not None and cfg.subsample.max_images < 1:
        raise ConfigError("subsample.max_images: must be >= 1 — to turn subsampling off, "
                          f"use null (no cap) ({cfg.subsample.max_images})")
    if not (0.0 < cfg.subsample.center_ratio <= 1.0):
        raise ConfigError(f"subsample.center_ratio: outside (0,1] ({cfg.subsample.center_ratio})")
    if cfg.rescorer.fit_threads < 1:
        raise ConfigError(f"rescorer.fit_threads: must be >= 1 ({cfg.rescorer.fit_threads})")
    if cfg.rescorer.max_train_rows is not None and cfg.rescorer.max_train_rows < 1:
        raise ConfigError("rescorer.max_train_rows: must be >= 1 — to remove the cap, use "
                          f"null ({cfg.rescorer.max_train_rows})")
    if (cfg.rescorer.max_train_rows is not None
            and cfg.rescorer.max_train_rows < cfg.rescorer.min_samples):
        raise ConfigError(
            f"rescorer.max_train_rows ({cfg.rescorer.max_train_rows}) is below min_samples "
            f"({cfg.rescorer.min_samples}) — the discriminator fit would always be skipped")
    if cfg.rounds < 1:
        raise ConfigError(f"rounds: must be >= 1 ({cfg.rounds})")
    if cfg.beta <= 0:
        raise ConfigError(f"beta: must be > 0 ({cfg.beta})")
    if not isinstance(cfg.dataset_notes, str):
        raise ConfigError("dataset_notes: must be a string "
                          f"(got {type(cfg.dataset_notes).__name__})")
    if cfg.screen.top_k < 1:
        raise ConfigError(f"screen.top_k: must be >= 1 ({cfg.screen.top_k})")
    pf = cfg.screen.prescreen_fraction
    if pf is not None and (isinstance(pf, bool) or not isinstance(pf, (int, float))
                           or not 0.0 < pf <= 1.0):
        raise ConfigError("screen.prescreen_fraction: must be in (0,1] — to turn it off, "
                          f"use null ({pf!r})")
    if cfg.screen.prescreen_min_images < 1:
        raise ConfigError("screen.prescreen_min_images: must be >= 1 "
                          f"({cfg.screen.prescreen_min_images})")
    if cfg.screen.prescreen_min_grid < 2:
        raise ConfigError("screen.prescreen_min_grid: must be >= 2 — the smallest "
                          f"grid is 2 ({cfg.screen.prescreen_min_grid})")
    if not (0.0 < cfg.screen.prescreen_max_share <= 1.0):
        raise ConfigError("screen.prescreen_max_share: outside (0,1] "
                          f"({cfg.screen.prescreen_max_share})")
    if cfg.gate.bootstrap < 1:
        raise ConfigError(f"gate.bootstrap: must be >= 1 ({cfg.gate.bootstrap})")
    if not (0.0 < cfg.gate.confidence < 1.0):
        raise ConfigError(f"gate.confidence: outside (0,1) ({cfg.gate.confidence})")
    if cfg.saturation.empty_rounds < 0:
        raise ConfigError("saturation.empty_rounds: must be >= 0 — 0 is off "
                          f"({cfg.saturation.empty_rounds})")
    if not 1 <= cfg.planner.max_candidates <= PLANNER_MAX_CANDIDATES:
        raise ConfigError(
            f"planner.max_candidates: must be in 1..{PLANNER_MAX_CANDIDATES} "
            f"({cfg.planner.max_candidates}) — larger, the planner response runs "
            "past the output cap (planner.llm.OUTPUT_MAX_TOKENS) and is cut, and a "
            "cut response silently falls back to deterministic")
    if cfg.planner.max_reasks < 0:
        raise ConfigError("planner.max_reasks: must be >= 0 — 0 is off "
                          f"({cfg.planner.max_reasks})")
    if cfg.saturation.rounds < 1:
        raise ConfigError(f"saturation.rounds: must be >= 1 ({cfg.saturation.rounds})")
    if cfg.mine.exemplar_selection not in EXEMPLAR_SELECTIONS:
        raise ConfigError(f"mine.exemplar_selection: {cfg.mine.exemplar_selection!r} must be "
                          f"one of {EXEMPLAR_SELECTIONS}")
    for key in ("kcenter_outlier_mad", "fn_kcenter_outlier_mad"):
        v = getattr(cfg.mine, key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or v <= 0):
            raise ConfigError(f"mine.{key}: must be > 0 — to turn the trim off, use null "
                              f"({v!r})")
    if cfg.mine.fn_min_candidates < 0:
        raise ConfigError("mine.fn_min_candidates cannot be negative — 0 means off "
                          f"(got {cfg.mine.fn_min_candidates})")
    if 0.0 not in cfg.rescorer.alphas:
        raise ConfigError("rescorer.alphas must include 0.0 — adoption is structurally safe "
                          "only if the search space includes the baseline (identity)")
    if not (0.0 < cfg.diagnose.small_object_ratio < 1.0):
        raise ConfigError("diagnose.small_object_ratio: outside (0,1) "
                          f"({cfg.diagnose.small_object_ratio})")
    o = cfg.search.tiling_overlap
    if isinstance(o, bool) or not isinstance(o, (int, float)) or not (0.0 <= o < 1.0):
        raise ConfigError(f"search.tiling_overlap: outside [0,1) ({o!r})")
    s = cfg.search.tiling_merge_min_score
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not (0.0 <= s < 1.0):
        raise ConfigError(f"search.tiling_merge_min_score: outside [0,1) ({s!r})")
    q = cfg.search.tiling_tp_quantile
    if not (0.0 < q < 1.0):
        raise ConfigError(f"search.tiling_tp_quantile: outside (0,1) ({q!r})")
    if cfg.search.tiling_tp_min < 1:
        raise ConfigError("search.tiling_tp_min: must be >= 1 "
                          f"({cfg.search.tiling_tp_min})")
    if cfg.search.tiling_merge not in TILING_MERGES:
        raise ConfigError(f"search.tiling_merge: {cfg.search.tiling_merge!r} must be "
                          f"one of {TILING_MERGES}")
    if cfg.gate.min_delta < 0:
        raise ConfigError("gate.min_delta: must not be negative — 0 is off "
                          f"({cfg.gate.min_delta})")
    if cfg.rescorer.margin < 0:
        raise ConfigError(f"rescorer.margin cannot be negative ({cfg.rescorer.margin})")
    if cfg.rescorer.ensemble_members < 1:
        raise ConfigError("rescorer.ensemble_members must be >= 1 "
                          f"({cfg.rescorer.ensemble_members})")
    lams = list(cfg.crop_classifier.lambdas)
    if (not lams
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and 0.0 <= float(x) <= 1.0 for x in lams)
            or len({float(x) for x in lams}) != len(lams)):
        raise ConfigError(f"crop_classifier.lambdas: {lams} — needs one or more distinct "
                          "numbers in [0, 1] (λ=0 is relabel-only; no classifier is "
                          "measured separately as the baseline)")
    for knob in ("reuse", "retune_lambda"):
        if not isinstance(getattr(cfg.crop_classifier, knob), bool):
            raise ConfigError(f"crop_classifier.{knob} must be true/false "
                              f"({getattr(cfg.crop_classifier, knob)!r})")
    if 0.0 not in cfg.lora.lambdas:
        raise ConfigError("lora.lambdas must include 0.0 — λ=0 (=base) must be on the "
                          "grid so that non-adoption is exactly the status quo")
    if cfg.lora.margin < 0:
        raise ConfigError(f"lora.margin cannot be negative ({cfg.lora.margin})")


TILING_MERGES = ("nmm", "nms", "sahi")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raise ConfigError(f"{path}: empty config")
    return config_from_dict(raw)
