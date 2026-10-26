"""vLLM (OpenAI-compatible) chat client — the single LLM path for the planner,
phrase refinement, and FP/FN narration.

We run no server of our own; vLLM serves `/v1/chat/completions`
(default model Qwen/Qwen3.8-27B-FP8, multimodal).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 300.0  # per-request HTTP timeout (seconds)
JSON_ATTEMPTS = 2        # a reply that does not parse as JSON is asked once more (one-off degeneration)
EXCERPT = 300            # characters of the head and tail of an unparseable reply put in the error

RETRY_STATUS = {429, 500, 502, 503, 504}


def post_with_retry(session: requests.Session, url: str, payload: dict,
                    timeout: float, retries: int = 4, backoff: float = 2.0) -> requests.Response:
    """Retry with exponential backoff on 429/5xx/connection errors. Also used by
    phrase refinement (infer.phrases)."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            if resp.status_code in RETRY_STATUS:
                last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            else:
                resp.raise_for_status()
                return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
        if attempt < retries:
            delay = backoff ** attempt
            log.warning("POST %s failed (%s) — retrying in %.1fs (%d/%d)",
                        url, last_exc, delay, attempt + 1, retries)
            time.sleep(delay)
    raise RuntimeError(f"POST {url} — all {retries+1} attempts failed") from last_exc


def _extract_json(text: str) -> dict:
    """Extract a JSON object from model output. Kept defensive even with guided decoding."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class LLMReplyError(ValueError):
    """A reply that does not parse as JSON — carries the finish reason, token
    count and the head and tail of the text."""


def _excerpt(text: str) -> str:
    if len(text) <= 2 * EXCERPT:
        return repr(text)
    return (f"{text[:EXCERPT]!r} … ({len(text) - 2 * EXCERPT} chars omitted) … "
            f"{text[-EXCERPT:]!r}")


def parse_reply(body: dict) -> dict:
    """chat completions response body → JSON object.

    If it does not parse or is not a JSON object, LLMReplyError — the reply text
    is kept nowhere else, so the error message carries the clues to tell the
    cause apart (cut off at the token limit, repetitive degeneration, …)."""
    choice = body["choices"][0]
    text = choice["message"]["content"] or ""

    def fail(why):
        tokens = (body.get("usage") or {}).get("completion_tokens")
        return LLMReplyError(
            f"{why} — finish_reason={choice.get('finish_reason')}, "
            f"completion_tokens={tokens}, reply {len(text)} chars: {_excerpt(text)}")
    try:
        parsed = _extract_json(text)
    except json.JSONDecodeError as exc:
        raise fail(exc) from exc
    if not isinstance(parsed, dict):
        # callers assume a dict — a list or the like is treated as an unreadable
        # reply, so it is asked again and then falls back
        raise fail(f"not a JSON object ({type(parsed).__name__})")
    return parsed


class LLMClient:
    """One chat = one HTTP call (chat_json asks once more if the reply does not
    parse as JSON). Stateless — replay is implemented by the caller persisting
    prompt and response as artifacts."""

    def __init__(self, base_url: str, model: str,
                 session: Optional[requests.Session] = None,
                 timeout: float = REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = session or requests.Session()
        self.timeout = timeout

    def ping(self, timeout: float = 5.0) -> bool:
        try:
            return self.session.get(f"{self.base_url}/health", timeout=timeout).ok
        except requests.RequestException:
            return False

    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float = 0.0, schema: Optional[dict] = None) -> str:
        """`messages` is in OpenAI chat format as-is. Passing `schema` requests
        json_schema guided decoding. Returns the response content text."""
        body = self._complete(messages, max_tokens=max_tokens,
                              temperature=temperature, schema=schema)
        return body["choices"][0]["message"]["content"] or ""

    def _complete(self, messages: list[dict], *, max_tokens: int,
                  temperature: float, schema: Optional[dict]) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        resp = post_with_retry(self.session, f"{self.base_url}/v1/chat/completions",
                               payload, timeout=self.timeout)
        body = resp.json()
        choice = body["choices"][0]
        if choice.get("finish_reason") == "length":
            # a truncated response — logged so a cut-off is visible on the
            # non-JSON path (chat) too
            log.warning("LLM response was cut off at max_tokens (%d)", max_tokens)
        return body

    def chat_json(self, messages: list[dict], *, max_tokens: int = 1024,
                  temperature: float = 0.0, schema: Optional[dict] = None) -> dict:
        """A JSON reply — asked once more if it does not parse. The last failure
        is raised to the caller as LLMReplyError (the caller records it with its
        fallback)."""
        for attempt in range(1, JSON_ATTEMPTS + 1):
            body = self._complete(messages, max_tokens=max_tokens,
                                  temperature=temperature, schema=schema)
            try:
                return parse_reply(body)
            except LLMReplyError as exc:
                if attempt == JSON_ATTEMPTS:
                    raise
                log.warning("LLM reply did not parse as JSON (%s) — asking again", exc)
