"""Real GPU hooks and server wiring — the loop's composition root (called by cli).

Only what the config supports is wired. Missing hooks stay None, and as soon
as the corresponding axis is enabled loop.materialize raises an explicit error
naming what is missing — silently disabling an axis is the worst failure mode.

The detector (YOLOE) is an in-process client, so there is no detector server.
The only server is vLLM (planner): if planner.base_url is set that endpoint is
used (externally managed), otherwise a VllmServer is launched — persistent by
default.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from ..config import Config

log = logging.getLogger(__name__)


@contextlib.contextmanager
def managed_servers(cfg: Config, repo_root: Path, out_dir: Path):
    """Launch the vLLM server when there is no base_url; yields llm_base_url|None.
    On exit, stops an owned server (a persist vLLM stays up)."""
    from ..services.external import VllmServer
    vllm_server = None
    llm_base = cfg.planner.base_url
    try:
        if not llm_base and cfg.planner.kind == "llm":
            vllm_server = VllmServer(cfg.planner,
                                     out_dir / "_logs" / "vllm_server.log")
            vllm_server.start()
            llm_base = vllm_server.base_url
        yield llm_base
    finally:
        if vllm_server is not None:
            vllm_server.stop()          # no-op for persist (server stays up)


def build_hooks(cfg: Config, repo_root: Path, out_dir: Path, *,
                llm_base_url: str | None = None) -> dict:
    hooks: dict = dict(client=None, llm_chat=None, encode_fn=None,
                       caption_fn=None, vlm_caption_fn=None,
                       refine_alias_fn=None,
                       refine_negatives_fn=None, detector_hooks=None)
    # In-process backend — no server. lora (tail fine-tune) and crop_classifier
    # (client-side reclassification) are wired to the client's state controls:
    # weights_ctl = swap tail-arm weights, adapt_ctl = load artifact.
    from ..clients.yoloe import YoloeClient
    model_path = Path(cfg.detector.model)
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    client = YoloeClient(model_path=str(model_path), device=cfg.detector.device)
    hooks["client"] = client
    hooks["detector_hooks"] = {
        "weights_ctl": client.set_weights,
        "adapt_ctl": client.set_adapt_root,
        "repo_root": repo_root,
        "base_weights": model_path,
    }
    llm_base = llm_base_url or cfg.planner.base_url
    if llm_base:
        from ..clients.llm import LLMClient
        from ..infer import phrases
        llm = LLMClient(llm_base, cfg.planner.model)
        hooks["llm_chat"] = lambda messages, **kw: llm.chat_json(messages, **kw)
        hooks["refine_alias_fn"] = (
            lambda name, cn, pool, others: phrases.refine_alias(
                llm_base, cfg.planner.model, name, cn, pool,
                other_captions=others))
        hooks["refine_negatives_fn"] = (
            lambda name, cn, pool, mk: phrases.refine_negatives(
                llm_base, cfg.planner.model, name, cn, pool,
                max_keep=mk))

    from ..infer import embedder

    # cache per run so the same GT crops are not re-embedded every round and
    # candidate (keyed by crop content hash, one file per model) — reused after
    # resume too
    model_slug = cfg.mine.embedder_model.replace("/", "--")
    # missed groups go out in one encode_crops_many call (one worker run, one model load)
    cached = embedder.cached_encoder(
        None, out_dir / "_cache" / f"embed_{model_slug}.npz",
        encode_many=lambda groups: embedder.encode_crops_many(
            repo_root, [(list(paths), log_path) for paths, log_path in groups],
            cfg.mine.embedder_model, cfg.mine.device))

    def _log(cid):
        return out_dir / "_logs" / f"embed_c{cid}.log"

    def encode_fn(cid, paths):
        return cached(paths, _log(cid))
    # a round's c1..cN up front in one go — later encode_fn(cid) calls hit the cache
    encode_fn.prefetch = lambda groups: cached.many(
        [(paths, _log(cid)) for cid, paths in groups.items()])
    hooks["encode_fn"] = encode_fn

    # Caption mining (including all three text_negatives sources) goes through
    # the VLM (the same vLLM as the planner). Without llm_base it stays None and
    # enabling the axis raises an explicit error.
    if hooks["llm_chat"] is not None:
        from ..infer import phrases as _ph
        _chat = hooks["llm_chat"]
        hooks["caption_fn"] = lambda rows: _ph.vlm_caption(rows, _chat)

    # alias=vlm source — uses the planner's vLLM (a captioner that sees the crop directly)
    chat = hooks.get("llm_chat")
    if chat is not None:
        from ..infer import phrases as _phrases
        hooks["vlm_caption_fn"] = lambda rows: _phrases.vlm_caption(rows, chat)
    return hooks
