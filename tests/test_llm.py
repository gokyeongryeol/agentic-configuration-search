"""clients/llm — chat wire format and JSON extraction."""

import json

import pytest

from agentic_detector_config.clients.llm import LLMClient, _extract_json


def test_extract_json_plain_and_wrapped():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('preamble {"a": {"b": 2}} trailer') == {"a": {"b": 2}}
    with pytest.raises(json.JSONDecodeError):
        _extract_json("no json here")


def test_chat_wire_format(json_server):
    def respond(path, payload):
        return 200, {"choices": [{"message": {"content": '{"ok": true}'}}]}

    base, records = json_server(respond)
    client = LLMClient(base, "test-model")
    out = client.chat_json([{"role": "user", "content": "hi"}],
                           schema={"name": "s", "schema": {"type": "object"}})
    assert out == {"ok": True}
    assert records[0]["path"] == "/v1/chat/completions"
    p = records[0]["payload"]
    assert p["model"] == "test-model"
    assert p["temperature"] == 0.0
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["response_format"]["type"] == "json_schema"


def test_chat_without_schema_omits_response_format(json_server):
    base, records = json_server(
        lambda p, b: (200, {"choices": [{"message": {"content": "text"}}]}))
    assert LLMClient(base, "m").chat([{"role": "user", "content": "x"}]) == "text"
    assert "response_format" not in records[0]["payload"]
