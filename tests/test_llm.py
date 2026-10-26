"""clients/llm — chat wire format and JSON extraction."""

import json

import pytest

from agentic_configuration_search.clients.llm import (LLMClient, LLMReplyError, _extract_json,
                                                     parse_reply)


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


def test_truncated_response_is_logged(json_server, caplog):
    """A response cut off at max_tokens is logged — when it then failed to parse
    as JSON and fell back to deterministic, the cause was invisible."""
    import logging
    base, _ = json_server(lambda p, b: (200, {"choices": [
        {"message": {"content": '{"a": "cu'}, "finish_reason": "length"}]}))
    with caplog.at_level(logging.WARNING):
        LLMClient(base, "m").chat([{"role": "user", "content": "x"}], max_tokens=7)
    assert any("max_tokens" in r.message and "7" in r.message for r in caplog.records)


def test_unparseable_reply_names_finish_reason_and_excerpt():
    """A reply that does not parse as JSON carries the finish reason, token count
    and the head and tail of the text in the error — the text is kept nowhere
    else, so without this a token-limit cut-off and repetitive degeneration
    could not be told apart."""
    text = '{"keep": ["' + "lamp " * 2000
    body = {"choices": [{"message": {"content": text}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 1536}}
    with pytest.raises(LLMReplyError) as ei:
        parse_reply(body)
    msg = str(ei.value)
    assert "finish_reason=length" in msg and "1536" in msg
    assert '{"keep": ["lamp' in msg and str(len(text)) in msg
    assert len(msg) < 1000                              # not the whole text
    assert isinstance(ei.value, ValueError)


def _replies(json_server, contents):
    seq = iter(contents)
    return json_server(lambda p, b: (200, {"choices": [
        {"message": {"content": next(seq)}, "finish_reason": "stop"}]}))


def test_chat_json_asks_again_once_after_unparseable_reply(json_server, caplog):
    """An unparseable reply is asked once more — one-off degeneration happens,
    and the same input sent again comes back fine."""
    import logging
    base, records = _replies(json_server, ['{"a": "cu', '{"a": 1}'])
    with caplog.at_level(logging.WARNING):
        out = LLMClient(base, "m").chat_json([{"role": "user", "content": "x"}])
    assert out == {"a": 1} and len(records) == 2
    assert any('{"a": "cu' in r.getMessage() for r in caplog.records)


def test_chat_json_gives_up_after_the_second_unparseable_reply(json_server):
    base, records = _replies(json_server, ['{"a": "cu', '{"b": "again'])
    with pytest.raises(LLMReplyError, match="again"):
        LLMClient(base, "m").chat_json([{"role": "user", "content": "x"}])
    assert len(records) == 2


def test_non_object_json_is_an_unreadable_reply(json_server):
    """Valid JSON that is not an object (a list, …) is an unreadable reply —
    callers assume a dict and call .get, so passing it on would skip the
    fallback and raise AttributeError."""
    with pytest.raises(LLMReplyError, match="object"):
        parse_reply({"choices": [{"message": {"content": "[1, 2]"},
                                  "finish_reason": "stop"}]})
    base, records = _replies(json_server, ["[1, 2]", '{"a": 1}'])
    assert LLMClient(base, "m").chat_json([{"role": "user", "content": "x"}]) == {"a": 1}
    assert len(records) == 2


def test_http_error_is_not_asked_again(json_server):
    """The re-ask is only for unreadable replies — HTTP errors are post_with_retry's
    job (4xx is not retried)."""
    base, records = json_server(lambda p, b: (400, {"error": "bad"}))
    with pytest.raises(Exception):
        LLMClient(base, "m").chat_json([{"role": "user", "content": "x"}])
    assert len(records) == 1


def test_planner_records_the_unreadable_reply(tmp_path):
    """The clues of a reply the planner could not read (finish reason, head and
    tail of the reply) land in the proposal's error."""
    from agentic_configuration_search.planner.llm import LLMPlanner

    def chat(messages, **kw):
        raise LLMReplyError("Unterminated string — finish_reason=length, reply '{\"c'")
    LLMPlanner(chat=chat, artifact_dir=tmp_path).propose(history=[], diagnosis={})
    stored = json.loads((tmp_path / "proposal.json").read_text())
    assert "finish_reason=length" in stored["error"]
