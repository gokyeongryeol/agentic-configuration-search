"""Guard — LLM replay.

Contract pinned here: resume in LLM mode (reusing the same artifact_dir) does not call the LLM
again — it replays `proposal.json` (prompt, response, candidates), and the replayed proposal is
identical to the first run. The resume key is the proposal artifact, not a config hash.
"""

import json


def test_resume_calls_llm_zero_times(tmp_path):
    from agentic_configuration_search.planner.llm import LLMPlanner

    calls = []

    def fake_chat(messages, **kw):
        calls.append(messages)
        return {"candidates": [{"axes": {"alias": "vlm"}, "rationale": "x"}],
                "stop": False}

    planner = LLMPlanner(chat=fake_chat, artifact_dir=tmp_path)
    p1 = planner.propose(history=[], diagnosis={})
    assert len(calls) == 1
    artifact = tmp_path / "proposal.json"
    assert artifact.exists()
    stored = json.loads(artifact.read_text())
    assert stored["prompt"] and stored["response"]        # the full replay material is kept

    # resume: rebuilding with the same artifact_dir must replay without a call
    resumed = LLMPlanner(chat=fake_chat, artifact_dir=tmp_path)
    p2 = resumed.propose(history=[], diagnosis={})
    assert len(calls) == 1, "resume called the LLM again"
    assert p1 == p2


def test_replay_survives_even_if_chat_would_now_fail(tmp_path):
    """Replay does not depend on the current LLM server state — resume works even if chat is dead."""
    from agentic_configuration_search.planner.llm import LLMPlanner

    ok = LLMPlanner(chat=lambda m, **k: {"candidates": [
                        {"id": "a", "axes": {"rescorer": True}, "rationale": "r"}]},
                    artifact_dir=tmp_path)
    p1 = ok.propose(history=[], diagnosis={})
    assert [c["axes"]["rescorer"] for c in p1["candidates"]] == [True]

    def boom(messages, **kw):
        raise RuntimeError("server down")

    resumed = LLMPlanner(chat=boom, artifact_dir=tmp_path)
    assert resumed.propose(history=[], diagnosis={}) == p1
