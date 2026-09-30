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


class LLMClient:
    """One chat = one HTTP call. Stateless — replay is implemented by the caller
    persisting prompt and response as artifacts."""

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
        return resp.json()["choices"][0]["message"]["content"] or ""

    def chat_json(self, messages: list[dict], *, max_tokens: int = 1024,
                  temperature: float = 0.0, schema: Optional[dict] = None) -> dict:
        return _extract_json(self.chat(messages, max_tokens=max_tokens,
                                       temperature=temperature, schema=schema))
