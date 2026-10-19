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
    max_model_len: int = 16384
    startup_timeout: int = 1800
    enforce_eager: bool = False
    # Persistent mode: attach to a healthy server on the port and leave it up when
    # the run ends. The planner makes few calls per round, so a restart (minutes)
    # is wasteful — on by default.
    persist: bool = True


@dataclass(frozen=True)
class PlannerCfg:
    kind: str = "deterministic"          # one of PLANNER_KINDS
    model: str = "Qwen/Qwen3.8-27B-FP8"
    base_url: Optional[str] = None       # vLLM OpenAI-compatible endpoint; None -> services launches one
    # Show **all** prompt-sample crops to the VLM and merge its descriptions into
    # the diagnosis (kind=llm only; diagnostic signal only — never on the decision
    # path). Images per call are batched; a failed batch degrades to no notes for
    # that batch only.
    crop_notes: bool = True
    launch: VllmLaunchCfg = field(default_factory=VllmLaunchCfg)

    _NESTED: ClassVar[dict] = {"launch": VllmLaunchCfg}


@dataclass(frozen=True)
class GateCfg:
    margin: float = 0.0                  # adopt iff lower CI bound of Δ micro F(β) > margin
    bootstrap: int = 2000                # number of paired bootstrap resamples
    confidence: float = 0.95


@dataclass(frozen=True)
class SaturationCfg:
    rounds: int = 5                      # stop after this many consecutive stagnant rounds
    epsilon: float = 0.0                 # a round is stagnant if the Δ CI lower bound <= this


@dataclass(frozen=True)
class ScreenCfg:
    top_k: int = 2                       # candidates sent to holdout eval after fit_dev scoring


@dataclass(frozen=True)
class SplitCfg:
    fit_train_ratio: float = 0.8         # fit_train share of train (image-level stratified)
    seed: int = 0                        # split seed — fixed across rounds and seed waves;
                                         # run stochasticity is held separately by Config.run_seed.


@dataclass(frozen=True)
class LoraCfg:
    """lora axis = YOLOE tail fine-tune + λ-interpolation arms — **off by
    default**; the planner enables it on a structural-miss diagnosis. Training
    uses fit_train, λ scoring uses fit_dev. λ=0 is in the grid, so without a
    proven gain the result is exactly base. (The axis name `lora` is kept for
    history-artifact compatibility.)"""
    min_images: int = 20             # skip (keep base) if fit_train has fewer images
    lambdas: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    margin: float = 0.01             # adopt only if fit_dev ΔF(β) exceeds this
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
    alphas: list = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    margin: float = 0.005             # adopt α>0 only if the fit_dev F(β) gain exceeds this
    ensemble_members: int = 5         # discriminator ensemble size
    inner_folds: int = 5              # for the oof_phat diagnostic


@dataclass(frozen=True)
class CropClassifierCfg:
    """crop classifier axis — the client reclassifies predicted boxes (category
    swap or background). The levers are correcting class_confusion and removing
    background FPs. Training hyperparameters are the worker's universal-best
    constants (not knobs); only tripwires and the gate live here."""
    min_samples: int = 32            # total-crop tripwire — skip training below this
    min_background: int = 4          # minimum background crops — reclassification is pointless without them
    margin: float = 0.01             # adopt the arm only if fit_dev ΔF(β) exceeds this


@dataclass(frozen=True)
class SearchCfg:
    """Fixed parameters of the screen (fit_dev) grid evaluation. Which axes are
    on is decided by the candidate setting (planner proposal); this only holds
    the grid's value space."""
    max_threshold_candidates: int = 50
    # Tie tolerance for τ_c. Among points whose F(β) is within this band, pick the
    # one with the **largest margin above the next observed score below**
    # (core.metrics). 0 uses the plain F(β)-max candidate — that rule keeps the
    # lowest τ on ties and sits right on top of a noise cluster with no margin.
    tau_tolerance: float = 0.005
    # Grid of suppression IoU thresholds for emulated negatives (neg_eta)
    neg_eta_grid: list = field(default_factory=lambda: [0.25, 0.35, 0.5, 0.65, 0.75])
    # Parameters shared by every tiling grid — the grid g itself is the value of
    # the planner-chosen tiling axis (planner.axes.TILING_GRIDS). Tiles are an
    # image-relative g×g, so tile count (= cost) and zoom (g-1)(1-overlap)+1 do
    # not depend on resolution. At overlap 0.5, grid 2 and 3 match the old 640px
    # slice 416 / 320 in tile count and zoom (1.5x / 2.0x). merge=nmm unions only
    # fragments touching a seam (containment union would swallow instances).
    tiling_overlap: float = 0.5
    tiling_merge: str = "nmm"
    modalities: list = field(default_factory=lambda: ["semantic", "text"])
    # Axes that may not be proposed in this run — unwireable axes are blocked at
    # proposal time instead of failing later. Honored by the planners
    # (deterministic and LLM prompt) and by candidate sanitization.
    disabled_axes: list = field(default_factory=list)
    # Forbid specific values of an axis — {axis: [value, ...]} (e.g. a material
    # source the backend does not support). Honored by the same three places
    # (deterministic, LLM prompt, sanitize).
    disabled_values: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MineCfg:
    """Mining parameters — all hygiene rules and quotas. The adoption gate
    belongs to the loop.
    """
    exemplars_per_class: int = 4          # default quotas 4/2/1/1
    exemplar_selection: str = "kcenter"   # one of EXEMPLAR_SELECTIONS
    # Outlier trim before kcenter — drop only candidates whose centroid distance
    # exceeds median + k·MAD (0 = off). A fixed-fraction trim (top 20%) always
    # dropped candidates even with no outliers, 2 of ~10 per class on few-shot
    # datasets. 3.0 is about 3σ under a normal distribution.
    kcenter_outlier_mad: float = 3.0
    # The FN arm trims less than base (a larger multiple) — the FN set is noisy
    # (truncation, occlusion, label errors cluster there), but its purpose is broad
    # coverage of missed appearances, so over-trimming diversity defeats the axis.
    fn_kcenter_outlier_mad: float = 3.5
    fn_exemplars_per_class: int = 2
    fn_min_candidates: int = 5            # headroom pre-filter (0 = off)
    fn_min_miss_rate: float = 0.02
    fn_min_area_ratio: float = 0.3        # cut tiny candidates relative to median GT area
    visual_negatives_per_class: int = 1
    visual_negative_min_score: float = 0.05
    visual_negative_exclude_gt_iom: float = 0.7
    embedder_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # gated — needs HF_TOKEN
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
    # (4% of a side, 0.16% of the area) is about 26px on a square image at YOLOE's
    # 640px input, within COCO's small. A relative cut (lower quartile of a class's
    # GT areas) produced "small" misses even on datasets of only large objects, so
    # the tiling remedy kept repeating.
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
    dataset: str                         # Roboflow-format dataset root (train/valid/test)
    out_dir: str = "runs"
    rounds: int = 10                     # upper bound — saturation stops earlier if it triggers
    # β of the decision metric — 1.0 is F1, β>1 weights recall, β<1 precision.
    # Threshold estimation (τ), screen scoring, the adoption gate and the
    # rescorer α curve all judge by F-beta with this β (measured = served — one
    # metric must run through the whole path).
    beta: float = 1.0
    # Optional operator-supplied description of the dataset — inserted verbatim
    # into the LLM planner prompt only (the deterministic planner ignores it).
    # With "" the prompt is byte-identical to the no-notes prompt (pinned by
    # tests/test_planner.py).
    dataset_notes: str = ""
    # Seed for run stochasticity — used by material-mining sampling, lora and
    # crop-classifier training, and gate bootstrap resampling. **Separate from
    # the split seed (split.seed)**: seed waves must measure only search variance
    # with the fit_train/fit_dev boundary fixed, so they vary this and keep
    # split.seed fixed.
    run_seed: int = 0
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    saturation: SaturationCfg = field(default_factory=SaturationCfg)
    screen: ScreenCfg = field(default_factory=ScreenCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    query: QueryCfg = field(default_factory=QueryCfg)
    mine: MineCfg = field(default_factory=MineCfg)
    search: SearchCfg = field(default_factory=SearchCfg)
    rescorer: RescorerCfg = field(default_factory=RescorerCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    crop_classifier: CropClassifierCfg = field(default_factory=CropClassifierCfg)
    diagnose: DiagnoseCfg = field(default_factory=DiagnoseCfg)


_SECTIONS = {"planner": PlannerCfg, "gate": GateCfg, "saturation": SaturationCfg,
             "screen": ScreenCfg, "split": SplitCfg, "detector": DetectorCfg,
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
    if cfg.rounds < 1:
        raise ConfigError(f"rounds: must be >= 1 ({cfg.rounds})")
    if cfg.beta <= 0:
        raise ConfigError(f"beta: must be > 0 ({cfg.beta})")
    if not isinstance(cfg.dataset_notes, str):
        raise ConfigError("dataset_notes: must be a string "
                          f"(got {type(cfg.dataset_notes).__name__})")
    if cfg.screen.top_k < 1:
        raise ConfigError(f"screen.top_k: must be >= 1 ({cfg.screen.top_k})")
    if cfg.gate.bootstrap < 1:
        raise ConfigError(f"gate.bootstrap: must be >= 1 ({cfg.gate.bootstrap})")
    if not (0.0 < cfg.gate.confidence < 1.0):
        raise ConfigError(f"gate.confidence: outside (0,1) ({cfg.gate.confidence})")
    if cfg.saturation.rounds < 1:
        raise ConfigError(f"saturation.rounds: must be >= 1 ({cfg.saturation.rounds})")
    if cfg.mine.exemplar_selection not in EXEMPLAR_SELECTIONS:
        raise ConfigError(f"mine.exemplar_selection: {cfg.mine.exemplar_selection!r} must be "
                          f"one of {EXEMPLAR_SELECTIONS}")
    for key in ("kcenter_outlier_mad", "fn_kcenter_outlier_mad"):
        if getattr(cfg.mine, key) < 0:
            raise ConfigError(f"mine.{key}: must be >= 0 — 0 is off "
                              f"({getattr(cfg.mine, key)})")
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
    if cfg.search.tiling_merge not in TILING_MERGES:
        raise ConfigError(f"search.tiling_merge: {cfg.search.tiling_merge!r} must be "
                          f"one of {TILING_MERGES}")
    if cfg.rescorer.margin < 0:
        raise ConfigError(f"rescorer.margin cannot be negative ({cfg.rescorer.margin})")
    if cfg.rescorer.ensemble_members < 1:
        raise ConfigError("rescorer.ensemble_members must be >= 1 "
                          f"({cfg.rescorer.ensemble_members})")
    if 0.0 not in cfg.lora.lambdas:
        raise ConfigError("lora.lambdas must include 0.0 — λ=0 (=base) must be on the "
                          "grid so that non-adoption is exactly the status quo")
    if cfg.lora.margin < 0:
        raise ConfigError(f"lora.margin cannot be negative ({cfg.lora.margin})")


TILING_MERGES = ("nmm", "nms")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raise ConfigError(f"{path}: empty config")
    return config_from_dict(raw)
