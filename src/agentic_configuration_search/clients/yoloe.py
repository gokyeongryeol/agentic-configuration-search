"""YOLOE (ultralytics) detector client — in-process implementation of the
predict_images contract.

Runs ultralytics' open-vocabulary YOLOE in-process. **ultralytics is
AGPL-3.0** — this dependency is confined to the research baseline and must not
be merged into product code.

Contract summary (see core.prompts builders / clients.wire.parse_prediction):
- Input queries use the core.prompts query format; text modality (empty boxes)
  and semantic modality (prompt_image + exemplar boxes) are detected
  automatically.
- Response per image is {"boxes","scores","text"}, matching parse_prediction's
  contract that `text` is the only class-resolution path (alias phrases are
  returned as the phrase itself).
- Negative prompts are **emulated**: negative phrases/crops are inferred jointly
  as extra classes, then core.suppress.emulate_negatives applies (union batched
  NMS → IoU suppression → drop negatives). `neg_eta` is the suppression IoU
  threshold (DEFAULT_SUP_IOU if absent), so the neg_eta grid is an IoU sweep.
- `return_embeddings` returns **per-anchor visual embeddings**: a YOLOE score is
  the similarity between an anchor's visual embedding (one2one_cv3 output,
  D=head.embed) and the set_classes vocabulary embedding, so each final
  detection gets its surviving anchor's embedding as `embedding` (the rescorer
  fits with emb_dim=D). The end2end head maps anchors to detections directly:
  wrap head.get_topk_index to capture the top-k anchor indices, replay the
  NMS-free path (conf filter, order-preserving) to align rows, and turn any
  misalignment into an immediate error via **conf-equality checks**.
- Every config is inferred for real during screen: YOLOE-26 head class scores
  depend on vocabulary composition (negative channels absorb positive score),
  so reconstructing negative-off scores from a shared pass would systematically
  diverge from real inference.

All torch/ultralytics imports are lazy — preserving the import-time torch-free
invariant in tests/test_architecture.py; CPU tests inject a backend to pin the
contract logic without a GPU.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image

from ..core.coco import crop_image
from ..core.io import promptable
from ..core.suppress import batched_nms, emulate_negatives

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _preserve_cuda_visibility():
    """ultralytics select_device rewrites CUDA_VISIBLE_DEVICES **process-wide**
    ("cuda:5" → "5"). Left alone, worker subprocesses spawned later (dinov3
    embedder etc.) inherit the narrowed visibility and device ordinals break
    (observed: invalid device ordinal). Restore it at the ultralytics call
    boundary."""
    key = "CUDA_VISIBLE_DEVICES"
    before = os.environ.get(key)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = before

DEFAULT_SUP_IOU = 0.5   # default suppression IoU when negative channels exist
                        # but neg_eta is absent (e.g. default text-pass call)
DEFAULT_CONF_FLOOR = 0.001
_VOCAB_CACHE_MAX = 8    # cap on the (queries fingerprint → vocabulary) cache

# Tail fine-tune arm descriptor (produced by fitting.tail, consumed here) —
# {"kind": ARM_KIND, "tuned": <absolute path to subset .pt>, "lam": λ}.
ARM_KIND = "yoloe_tail_arm"


def read_arm(path) -> dict:
    """Load/validate an arm descriptor — {"tuned": Path, "lam": float}."""
    obj = json.loads(Path(path).read_text())
    if obj.get("kind") != ARM_KIND:
        raise ValueError(f"{path}: not a yoloe tail arm descriptor "
                         f"(kind={obj.get('kind')!r})")
    tuned = Path(obj["tuned"])
    if not tuned.exists():
        raise FileNotFoundError(f"{path}: tuned subset not found — {tuned}")
    return {"tuned": tuned, "lam": float(obj["lam"])}


def parse_queries(queries: list[dict]) -> tuple[list[dict], float]:
    """core.prompts-format queries → (channels, conf_floor).

    channel: {"name": phrase/class name used as response text, "kind": "pos"|"neg",
              "text": phrase (text modality) | None,
              "crops": [(prompt_image, xyxy), ...] (semantic modality)}.

    Rules (see core.prompts.build_queries / build_text_queries):
    - Text modality if every prompt's box is empty, otherwise semantic.
    - Negatives are identified by the `negative_target` key. Text negatives
      arrive duplicated per target (name + alias), so they are folded per
      phrase — emulated suppression is target-agnostic (it removes any
      positive), so nothing is lost.
    """
    prompts = [(q.get("prompt_image"), p) for q in queries
               for p in q.get("prompts", [])]
    if not prompts:
        raise ValueError("empty queries — cannot build channels")
    boxed = [bool(p.get("box")) for _img, p in prompts]
    if any(boxed) and not all(boxed):
        raise ValueError("mixed text/semantic queries are not supported "
                         "(the gateway contract is one call per modality)")
    thresholds = [float(p[k]) for _img, p in prompts
                  for k in ("box_threshold", "multimodal_threshold") if k in p]
    floor = min(thresholds) if thresholds else DEFAULT_CONF_FLOOR

    channels: list[dict] = []
    index: dict[tuple, int] = {}   # (kind, key) → channel idx

    def _channel(kind: str, key: str, name: str, text: Optional[str]) -> dict:
        k = (kind, key)
        if k not in index:
            index[k] = len(channels)
            channels.append({"name": name, "kind": kind, "text": text,
                             "crops": []})
        return channels[index[k]]

    if not any(boxed):               # ---- text modality
        for _img, p in prompts:
            phrase = str(p["text"])
            if p.get("negative_target"):
                _channel("neg", phrase, phrase, phrase)
            else:
                _channel("pos", phrase, phrase, phrase)
    else:                            # ---- semantic (visual) modality
        for img, p in prompts:
            if img is None:
                raise ValueError("semantic query has no prompt_image")
            box = [float(v) for v in p["box"]]
            if p.get("negative_target"):
                target = str(p["negative_target"])
                ch = _channel("neg", target, f"~{target}", None)
            else:
                ch = _channel("pos", str(p["text"]), str(p["text"]), None)
            ch["crops"].append((img, box))

    names = [c["name"] for c in channels]
    if len(set(names)) != len(names):
        raise ValueError(f"channel name collision: {names}")
    return channels, floor


class YoloeClient:
    """In-process YOLOE client providing the predict_images contract.

    Backend injection (tests): pass as `backend` an object implementing
    set_vocab(channels) / infer(images, conf, iou, max_det) → per image
    [(xyxy, score, channel_idx), ...], and the contract logic runs without a GPU.
    """

    def __init__(self, model_path: str = "weights_yoloe/yoloe-26x-seg.pt",
                 device: str = "cuda:0", imgsz: int = 640, backend=None,
                 reclassifier_factory=None):
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = int(imgsz)
        self._backend = backend
        self._vocab: dict[str, tuple] = {}       # queries fingerprint → channels/floor
        self._applied_key: Optional[str] = None  # vocabulary loaded in backend
        self._weights: Optional[str] = None      # current tail arm descriptor (None=base)
        self._adapt_root: Optional[str] = None   # crop classifier artifact (None=off)
        self._reclassifier = None
        self._reclassifier_factory = (reclassifier_factory
                                      or (lambda root, device:
                                          _CropReclassifier(root, device)))

    # ------------------------------------------------------------- helpers

    def ping(self, timeout: float = 5.0) -> bool:  # noqa: ARG002 — contract symmetry
        return True

    def set_adapt_root(self, root=None) -> None:
        """Mount/unmount the crop classifier artifact (the loop's adapt_ctl). The
        artifact is classifier.pt / classifier_config.json (fitting.crop_classifier)."""
        target = str(root) if root else None
        if target == self._adapt_root:
            return
        self._adapt_root = target
        self._reclassifier = None                # rebuilt lazily

    def _get_reclassifier(self):
        if self._adapt_root is None:
            return None
        if self._reclassifier is None:
            self._reclassifier = self._reclassifier_factory(self._adapt_root,
                                                            self.device)
        return self._reclassifier

    def set_weights(self, weights=None) -> None:
        """Swap serving weights (the loop's weights_ctl).

        weights: tail arm descriptor path (fitting.tail) or None (= base).
        A swap invalidates vocabulary embeddings (get_text_pe/VPE pass through
        the head) — the next predict recomputes them."""
        target = str(weights) if weights else None
        if target == self._weights:
            return
        self._weights = target
        self._applied_key = None                 # force vocab recompute
        if self._backend is not None:
            self._backend.load_weights(target)

    def _get_backend(self):
        if self._backend is None:
            self._backend = _UltralyticsBackend(self.model_path, self.device,
                                                self.imgsz)
            if self._weights:
                self._backend.load_weights(self._weights)
        return self._backend

    def _channels_for(self, queries: list[dict]) -> tuple[str, list[dict], float]:
        key = json.dumps(queries, sort_keys=True, default=str)
        if key not in self._vocab:
            if len(self._vocab) >= _VOCAB_CACHE_MAX:
                self._vocab.pop(next(iter(self._vocab)))
            self._vocab[key] = parse_queries(queries)
        channels, floor = self._vocab[key]
        return key, channels, floor

    # ------------------------------------------------------------- contract surface

    def predict_images(self, search_images: list, queries: list[dict], *,
                       apply_topk: int = 300,
                       iou_threshold: float = 0.5,
                       neg_eta: Optional[float] = None,
                       return_embeddings: bool = False,
                       return_classifier_probs: bool = False) -> list[dict]:
        """Batch inference — per image {"boxes","scores","text"[,"embeddings"]}
        (clients.wire contract). neg_eta = negative-suppression IoU threshold.

        With return_classifier_probs=True and a crop classifier loaded, boxes are
        not reclassified: each item instead carries `classifier_probs` (per box
        the class probabilities, background first, or None for a box that could
        not be cropped) and `classifier_class_names` (["background", *names]) —
        the caller blends them (infer.gateway.blend_classifier).
        """
        key, channels, floor = self._channels_for(queries)
        backend = self._get_backend()
        if key != self._applied_key:
            backend.set_vocab(channels)
            self._applied_key = key
        raw = backend.infer(list(search_images), conf=floor,
                            iou=float(iou_threshold), max_det=int(apply_topk),
                            return_embeddings=return_embeddings)
        if len(raw) != len(search_images):
            raise RuntimeError(f"backend image count mismatch: requested {len(search_images)} "
                               f"vs returned {len(raw)}")

        def _det(row) -> dict:
            b, s, ci = row[0], row[1], row[2]
            d = {"box": [float(v) for v in b], "score": float(s),
                 "category_id": int(ci)}
            if return_embeddings:
                # Per-anchor visual embedding (see module docstring), carried in
                # the row by the backend. Empty vector if absent (emb_dim=0
                # fallback, compatible with test doubles).
                d["embedding"] = ([float(v) for v in row[3]]
                                  if len(row) > 3 else [])
            return d

        sup_iou = float(neg_eta) if neg_eta is not None else DEFAULT_SUP_IOU
        has_neg = any(c["kind"] == "neg" for c in channels)
        reclassifier = self._get_reclassifier()
        out = []
        for image, dets in zip(search_images, raw):
            pos = [_det(r) for r in dets if channels[int(r[2])]["kind"] == "pos"]
            if has_neg:
                neg = [_det(r) for r in dets
                       if channels[int(r[2])]["kind"] == "neg"]
                kept = emulate_negatives(pos, neg, sup_iou=sup_iou,
                                         nms_iou=float(iou_threshold))
            else:
                # end2end models output without NMS — do per-class folding
                # (batched NMS) here (same rule as the negative path)
                kept = batched_nms(pos, float(iou_threshold))
            probs = None
            if reclassifier is not None and return_classifier_probs:
                probs = _classifier_probs(reclassifier, image, kept)
            elif reclassifier is not None and kept:
                # Order contract: NMS → negative suppression → **reclassify** →
                # response. Reclassification happens before the loop sees the
                # detections (i.e. before the rescorer and τ).
                kept = _reclassify(reclassifier, image, kept)
            item = {"boxes": [d["box"] for d in kept],
                    "scores": [d["score"] for d in kept],
                    "text": [d.get("reclass_name")
                             or channels[d["category_id"]]["name"]
                             for d in kept]}
            if probs is not None:
                item["classifier_probs"] = probs
                item["classifier_class_names"] = ["background",
                                                  *reclassifier.class_names]
            if return_embeddings:
                item["embeddings"] = [d["embedding"] for d in kept]
            out.append(item)
        return out


def _croppable(image, dets: list[dict]) -> tuple[list[int], list]:
    """(indices, crops) of the boxes the classifier can read — promptable boxes
    whose crop is larger than 3 px on both sides. Uses the same crop_image
    (default padding) as training crops to match the distribution."""
    idx = [i for i, d in enumerate(dets) if promptable(d["box"])]
    if not idx:
        return [], []
    if isinstance(image, Image.Image):
        im, close = image, False
    else:
        im, close = Image.open(image), True
    try:
        rgb = im.convert("RGB")
        # Also recheck the crop's **actual pixel size**: a promptable box can
        # become a thin sliver after clamping to the image border, and once a
        # side is as small as the channel count (3) the SigLIP slow processor
        # infers the channel axis wrongly (observed: a 2px side read as
        # (1,1,23) uint8 → PIL fromarray TypeError, killing the whole run).
        # Thin crops are not classified and keep their label — a 2px sliver
        # carries no class signal anyway.
        kept_idx, crops = [], []
        for i in idx:
            crop = crop_image(rgb, dets[i]["box"])
            if min(crop.size) > 3:
                kept_idx.append(i)
                crops.append(crop)
        return kept_idx, crops
    finally:
        if close:
            im.close()


def _classifier_probs(reclassifier, image, dets: list[dict]) -> list:
    """Per detection the classifier's class probabilities (background first),
    or None for a box that cannot be cropped — the blend leaves those as they are."""
    out: list = [None] * len(dets)
    idx, crops = _croppable(image, dets)
    if not idx:
        return out
    probs = reclassifier.probs(crops)
    if len(probs) != len(idx):
        raise RuntimeError(f"reclassifier prob count mismatch: {len(probs)} vs {len(idx)}")
    width = len(reclassifier.class_names) + 1
    for i, p in zip(idx, probs):
        if len(p) != width:
            raise RuntimeError(f"reclassifier probs have {len(p)} classes, expected "
                               f"{width} (background included)")
        out[i] = [float(v) for v in p]
    return out


def _reclassify(reclassifier, image, dets: list[dict]) -> list[dict]:
    """Reclassify detection-box crops — replace the category, or drop on
    background (0).

    Scores are kept (reclassification corrects labels; it does not re-estimate
    confidence). Boxes that cannot be cropped (_croppable) keep their original
    label."""
    idx, crops = _croppable(image, dets)
    if not idx:
        return dets
    labels = reclassifier.classify(crops)
    if len(labels) != len(idx):
        raise RuntimeError(f"reclassifier label count mismatch: {len(labels)} vs {len(idx)}")
    names = reclassifier.class_names
    out: list[dict] = []
    relabel = dict(zip(idx, labels))
    for i, d in enumerate(dets):
        label = relabel.get(i)
        if label is None:                        # degenerate box — keep label
            out.append(d)
            continue
        if label == 0:                           # background — drop
            continue
        if not (1 <= label <= len(names)):
            raise RuntimeError(f"reclassifier label {label} out of range (classes {len(names)})")
        out.append({**d, "reclass_name": names[label - 1]})
    return out


def _is_trainable_key(name: str) -> bool:
    """Parameters the classifier worker trains and saves — the LoRA adapter and
    the classification head. Same rule as workers.adapt.is_trainable_key
    (clients may not import workers; tests/test_crop_classifier.py pins them
    together)."""
    return "lora_" in name or "classifier" in name


def _load_classifier_state(model, ckpt: dict, where) -> None:
    """Put a classifier checkpoint onto the rebuilt model.

    "trainable" checkpoints hold only the LoRA adapter and head; the backbone
    stays the pretrained weights the model was rebuilt from. Every trainable
    key the model has must be present and nothing else — a missing one would
    silently leave a freshly initialized layer. Checkpoints without the marker
    (older runs) hold the full state dict and load strictly as before."""
    state = ckpt["state_dict"]
    if ckpt.get("state_dict_format") != "trainable":
        model.load_state_dict(state, strict=True)
        return
    want = {k for k in model.state_dict() if _is_trainable_key(k)}
    missing, extra = sorted(want - set(state)), sorted(set(state) - want)
    if missing or extra:
        raise ValueError(f"{where}: trainable checkpoint does not match the model — "
                         f"missing {missing[:5]}, unexpected {extra[:5]}")
    model.load_state_dict(state, strict=False)


class _CropReclassifier:
    """Serving component that loads the artifact (classifier.pt /
    classifier_config.json) and reclassifies crops — rebuilt with the same
    configuration as build_model in workers/adapt.py (state_dict key contract).
    torch/transformers/peft are imported lazily."""

    BATCH = 64
    _TARGETS = {"qkvo": ["q_proj", "k_proj", "v_proj", "out_proj"]}

    def __init__(self, root, device: str):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoImageProcessor,
                                  AutoModelForImageClassification)
        root = Path(root)
        cfg = json.loads((root / "classifier_config.json").read_text())
        self.class_names = list(cfg["class_names"])
        ckpt = torch.load(root / "classifier.pt", map_location="cpu",
                          weights_only=True)
        num = int(ckpt["num_classes"])
        if num != len(self.class_names) + 1:
            raise ValueError(f"{root}: num_classes {num} != classes "
                             f"{len(self.class_names)}+1(background)")
        model = AutoModelForImageClassification.from_pretrained(
            cfg["classifier_model_name"], num_labels=num,
            ignore_mismatched_sizes=True)
        lora = LoraConfig(r=int(cfg["classifier_lora_r"]),
                          lora_alpha=int(cfg["classifier_lora_alpha"]),
                          lora_dropout=0.05,
                          target_modules=self._TARGETS[cfg["classifier_lora_target"]],
                          bias="none")
        model = get_peft_model(model, lora)
        _load_classifier_state(model, ckpt, root)
        self.model = model.eval().to(device)
        self.processor = AutoImageProcessor.from_pretrained(
            cfg["classifier_model_name"])
        self.device = device
        log.info("loaded crop classifier: %s (classes=%d)", root, num)

    def _logits(self, crops: list):
        import torch
        with torch.inference_mode():
            for i in range(0, len(crops), self.BATCH):
                pv = self.processor(images=crops[i:i + self.BATCH],
                                    return_tensors="pt")["pixel_values"]
                yield self.model(pixel_values=pv.to(self.device)).logits

    def classify(self, crops: list) -> list[int]:
        out: list[int] = []
        for logits in self._logits(crops):
            out += logits.argmax(dim=-1).cpu().tolist()
        return out

    def probs(self, crops: list) -> list[list[float]]:
        """Softmax class probabilities per crop — background (0) first."""
        out: list[list[float]] = []
        for logits in self._logits(crops):
            out += logits.float().softmax(dim=-1).cpu().tolist()
        return out


class _UltralyticsBackend:
    """Real GPU backend — torch/ultralytics are imported only here, and lazily."""

    def __init__(self, model_path: str, device: str, imgsz: int):
        import torch
        from ultralytics import YOLOE  # lazy — import-time torch-free invariant
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = int(imgsz)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            # Initialize the CUDA context **first, under full visibility** — if
            # the first init happens while ultralytics select_device has
            # narrowed CUDA_VISIBLE_DEVICES, this process's torch sees only one
            # GPU for good, and later in-process components (crop reclassifier
            # etc.) die on .to("cuda:N") with invalid ordinal (restoring the env
            # var cannot undo it — the context is already frozen).
            torch.zeros(1, device=device)
        self.model = YOLOE(model_path)
        self._capture = False       # on only during infer(return_embeddings=True)
        self._records: list[dict] = []       # per-forward {"emb","scores"}
        self._pending_maps: list = []        # one2one_cv3 hook outputs (level order)
        self._install_capture()
        log.info("loading YOLOE: %s (device=%s)", model_path, device)

    @_preserve_cuda_visibility()
    def load_weights(self, weights: Optional[str]) -> None:
        """Apply a tail arm descriptor (or None = base) to the serving model.

        For determinism, always **rebuild from the base checkpoint**, then
        overlay the tuned subset as θ_λ = (1−λ)·θ_base + λ·θ_tuned — no drift
        accumulates across swaps, and λ=0/None is bit-exact base. Capture
        instrumentation is reinstalled on the new head."""
        import torch
        from ultralytics import YOLOE
        self.model = YOLOE(self.model_path)          # rebuild base (no predictor)
        if weights is not None:
            arm = read_arm(weights)
            tuned = torch.load(arm["tuned"], map_location="cpu",
                               weights_only=True)
            base = self.model.model.state_dict()
            lam = arm["lam"]
            interp = {}
            for k, v in tuned.items():
                if k not in base:
                    raise RuntimeError(f"tail arm key {k!r} not in base — "
                                       "checkpoint/boundary mismatch")
                if base[k].shape != v.shape:
                    raise RuntimeError(f"tail arm key {k!r} shape mismatch "
                                       f"({tuple(v.shape)} vs "
                                       f"{tuple(base[k].shape)}) — checkpoint did not pass "
                                       "the worker's shape filter")
                b = base[k].float()
                interp[k] = ((1.0 - lam) * b + lam * v.float()).to(base[k].dtype)
            self.model.model.load_state_dict(interp, strict=False)
            log.info("applied yoloe tail arm: λ=%g, keys=%d (%s)",
                     lam, len(interp), weights)
        else:
            log.info("restored yoloe base weights")
        self._pending_maps.clear()
        self._records.clear()
        self._install_capture()

    # ---- per-anchor embedding capture (alignment rules in module docstring) ----

    def _install_capture(self) -> None:
        """Instrument one2one_cv3 (per-anchor visual embeddings) and
        get_topk_index (top-k anchor selection) — the source of per-detection
        embeddings. Must be reinstalled whenever a weight swap rebuilds the model."""
        head = self.model.model.model[-1]

        def _keep(_mod, _inp, out):
            self._pending_maps.append(out)

        for m in head.one2one_cv3:
            m.register_forward_hook(_keep)
        orig = head.get_topk_index

        def wrapped(scores, max_det):
            out = orig(scores, max_det)
            if self._capture and len(self._pending_maps) >= head.nl:
                import torch
                maps = self._pending_maps[-head.nl:]
                flat = torch.cat([m.flatten(2) for m in maps], 2)  # (B, D, A)
                idx = out[2].long()                                # (B, k, 1)
                emb = flat.permute(0, 2, 1).gather(
                    1, idx.expand(-1, -1, flat.shape[1]))          # (B, k, D)
                scores = out[0].reshape(out[0].shape[0], -1)       # (B, k)
                # One entry per batch image — entry order equals image order
                # regardless of predictor batch size (input order within a batch)
                for b in range(emb.shape[0]):
                    self._records.append({
                        "emb": emb[b].float().cpu(),
                        "scores": scores[b].float().cpu()})
            self._pending_maps.clear()
            return out

        head.get_topk_index = wrapped

    # ---- vocabulary ----

    def _text_pe(self, phrases: list[str]):
        return self.model.get_text_pe(phrases)          # (1, n, D)

    def _visual_pe(self, channels: list[dict], idxs: list[int]):
        """VPE for visual channels — one forward per reference image, averaged per
        channel.

        ultralytics get_vpe yields per-channel embeddings (in sorted local cls
        order) from one image and its prompt boxes. Exemplars spread across
        several reference images are combined by averaging per-image VPEs per
        channel (rows are already normalized, so renormalize after averaging).
        """
        import numpy as np
        import torch
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        by_image: dict[str, list[tuple[list, int]]] = {}
        for local, gi in enumerate(idxs):
            for img, box in channels[gi]["crops"]:
                by_image.setdefault(str(img), []).append((box, local))
        dim = None
        acc: dict[int, list] = {i: [] for i in range(len(idxs))}
        for img, entries in sorted(by_image.items()):
            locals_present = sorted({loc for _b, loc in entries})
            remap = {loc: i for i, loc in enumerate(locals_present)}
            bxs = np.array([b for b, _loc in entries], dtype=np.float32)
            cls = np.array([remap[loc] for _b, loc in entries], dtype=np.int32)
            pred = YOLOEVPSegPredictor(overrides={
                "task": self.model.model.task, "mode": "predict", "save": False,
                "verbose": False, "batch": 1, "device": self.device,
                "imgsz": self.imgsz})
            pred.set_prompts({"bboxes": bxs, "cls": cls})
            pred.setup_model(model=self.model.model, verbose=False)
            vpe = pred.get_vpe(img)                     # (1, k, D) — sorted local order
            if vpe.shape[1] != len(locals_present):
                raise RuntimeError(f"VPE row count mismatch: {vpe.shape} vs "
                                   f"{len(locals_present)} (image={img})")
            dim = vpe.shape[-1]
            for loc in locals_present:
                acc[loc].append(vpe[0, remap[loc]])
        rows = []
        for i in range(len(idxs)):
            if not acc[i]:
                raise RuntimeError(f"visual channel {channels[idxs[i]]['name']!r} "
                                   "has no crops")
            rows.append(torch.stack(acc[i]).mean(dim=0))
        out = torch.stack(rows).unsqueeze(0)            # (1, n, D)
        assert dim is not None
        return torch.nn.functional.normalize(out, dim=-1, p=2)

    @_preserve_cuda_visibility()
    def set_vocab(self, channels: list[dict]) -> None:
        import torch
        text_idx = [i for i, c in enumerate(channels) if c["text"] is not None]
        vis_idx = [i for i, c in enumerate(channels) if c["text"] is None]
        parts: dict[int, "torch.Tensor"] = {}
        if text_idx:
            pe = self._text_pe([channels[i]["text"] for i in text_idx])
            for j, i in enumerate(text_idx):
                parts[i] = pe[0, j].float().cpu()
        if vis_idx:
            vpe = self._visual_pe(channels, vis_idx)
            for j, i in enumerate(vis_idx):
                parts[i] = vpe[0, j].float().cpu()
        rows = torch.stack([parts[i] for i in range(len(channels))])
        emb = rows.unsqueeze(0)                         # (1, C, D)
        names = [c["name"] for c in channels]
        if emb.shape[1] != len(names):
            raise RuntimeError(f"vocabulary row count mismatch: {emb.shape} vs {len(names)}")
        # YOLOE.set_classes (model level) skips the update when the name set is
        # unchanged, ignoring new embeddings under the same names (e.g. swapped
        # exemplars), so call the inner contract directly. Reset the predictor
        # since names/nc change.
        self.model.model.set_classes(names, emb)
        self.model.predictor = None

    # ---- inference ----

    @_preserve_cuda_visibility()
    def infer(self, images: list, *, conf: float, iou: float,
              max_det: int, return_embeddings: bool = False) -> list[list[tuple]]:
        import numpy as np
        self._capture = bool(return_embeddings)
        self._records.clear()
        self._pending_maps.clear()
        try:
            results = self.model.predict(images, conf=conf, iou=iou,
                                         max_det=max_det, device=self.device,
                                         verbose=False, agnostic_nms=False)
        finally:
            self._capture = False
        out = []
        if not return_embeddings:
            for r in results:
                b = r.boxes
                out.append([(b.xyxy[i].tolist(), float(b.conf[i]), int(b.cls[i]))
                            for i in range(len(b))])
            return out
        if len(self._records) < len(results):
            raise RuntimeError(f"embedding capture missing: forwards {len(self._records)}"
                               f" < images {len(results)}")
        # Leftover forwards (warmup etc.) come first — the tail entries align
        # with the results. Final result rows are an **order-preserving
        # subsequence** of the head's top-k rows (conf filter + the seg
        # predictor's empty-mask drop). Align the subsequence by exact float32
        # score equality — failure raises immediately (silent misalignment is
        # the worst failure mode).
        for r, rec in zip(results, self._records[-len(results):]):
            b = r.boxes
            conf_np = b.conf.cpu().numpy().astype(np.float32)
            scores_k = rec["scores"].numpy()
            sel: list[int] = []
            j = 0
            for c in conf_np:
                while j < len(scores_k) and scores_k[j] != c:
                    j += 1
                if j >= len(scores_k):
                    raise RuntimeError(
                        "embedding/detection alignment failed — result scores are not a "
                        f"subsequence of the head top-k scores (dets {len(b)})")
                sel.append(j)
                j += 1
            emb = rec["emb"].numpy()
            out.append([(b.xyxy[i].tolist(), float(conf_np[i]), int(b.cls[i]),
                         emb[sel[i]].tolist())
                        for i in range(len(b))])
        return out
