"""external.py — planner server argument assembly, stop race, persist.

No real server is started — subprocess, ports and clock are all stubbed.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_detector_config.config import config_from_dict
from agentic_detector_config.services import external

CFG = config_from_dict({"dataset": "x"})
# owned (non-persistent) server — the path where stop() actually shuts down
CFG_OWNED = config_from_dict({"dataset": "x",
                              "planner": {"launch": {"persist": False}}})


@pytest.fixture
def repo(tmp_path):
    return tmp_path


def _owned_server(repo):
    return external.VllmServer(CFG_OWNED.planner, repo / "logs" / "vllm.log")


def test_vllm_server_command():
    srv = external.VllmServer(CFG.planner, Path("/tmp/na/vllm.log"))
    cmd = srv.command()
    assert cmd[0] == str(external.VLLM_VENV / "bin" / "vllm")
    assert cmd[1:3] == ["serve", CFG.planner.model]
    assert "--gpu-memory-utilization" in cmd and "0.9" in cmd
    assert "--max-model-len" in cmd and "16384" in cmd
    assert srv.env()["CUDA_VISIBLE_DEVICES"] == str(CFG.planner.launch.gpu)
    assert srv.env()["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert srv.health_url().endswith("/health")
    assert srv.persist is True                     # planner server persists by default


def test_stop_waits_for_port_release(repo, monkeypatch):
    """Restart-race guard: stop() must wait until the port is released."""
    srv = _owned_server(repo)

    class FakeProc:
        def poll(self):
            return 0                    # already exited
    srv.proc = FakeProc()

    calls = {"n": 0}

    def fake_port_free(host, port):
        calls["n"] += 1
        return calls["n"] >= 3          # released only on the 3rd check

    monkeypatch.setattr(external, "port_free", fake_port_free)
    monkeypatch.setattr(external.time, "sleep", lambda s: None)
    srv.stop()
    assert calls["n"] >= 3
    assert srv.proc is None


def test_stop_gives_up_after_timeout(repo, monkeypatch, caplog):
    srv = _owned_server(repo)

    class FakeProc:
        def poll(self):
            return 0
    srv.proc = FakeProc()

    monkeypatch.setattr(external, "port_free", lambda h, p: False)
    monkeypatch.setattr(external.time, "monotonic",
                        iter([0.0, 0.0, 100.0]).__next__)
    monkeypatch.setattr(external.time, "sleep", lambda s: None)
    with caplog.at_level("WARNING"):
        srv.stop()
    assert any("was not released" in r.message for r in caplog.records)


def test_port_free():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert not external.port_free("127.0.0.1", port)
    finally:
        s.close()
    assert external.port_free("127.0.0.1", port)


def test_persist_server_attaches_and_never_stops(repo, monkeypatch):
    """Persist mode: attach to a healthy occupied port; stop is a no-op. An unhealthy occupant is refused."""

    class _Srv(external.ManagedServer):
        name = "t"
        persist = True

        def host_port(self):
            return "127.0.0.1", 59999

        def health_url(self):
            return "http://127.0.0.1:59999/health"

        def startup_timeout(self):
            return 0.01   # time out the unhealthy wait loop immediately

    monkeypatch.setattr(external, "port_free", lambda h, p: False)
    monkeypatch.setattr(external.requests, "get",
                        lambda url, timeout: SimpleNamespace(ok=True))
    s = _Srv(repo / "logs" / "t.log")
    s.start()                      # attach — passes without error
    s.stop()                       # no-op (owns no process)

    monkeypatch.setattr(external.requests, "get",
                        lambda url, timeout: SimpleNamespace(ok=False))
    monkeypatch.setattr(external.time, "sleep", lambda s_: None)
    s2 = _Srv(repo / "logs" / "t.log")
    with pytest.raises(RuntimeError, match="is occupied but not ready"):
        s2.start()


def test_nonpersist_server_rejects_occupied_port(repo, monkeypatch):
    srv = _owned_server(repo)
    monkeypatch.setattr(external, "port_free", lambda h, p: False)
    with pytest.raises(RuntimeError, match="already occupied"):
        srv.start()
