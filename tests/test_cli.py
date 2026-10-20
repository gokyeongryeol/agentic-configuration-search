"""cli — args → config/hook wiring. The loop itself is covered by test_loop."""

import contextlib

import pytest

from agentic_configuration_search import cli


@pytest.fixture(autouse=True)
def _no_real_servers(monkeypatch):
    """Never start a real ManagedServer in unit tests — on a machine with a vLLM venv,
    planner.kind=llm without base_url could launch a real planner server."""
    import agentic_configuration_search.loop.hooks as hooks

    @contextlib.contextmanager
    def fake_servers(cfg, repo_root, out_dir):
        yield cfg.planner.base_url

    monkeypatch.setattr(hooks, "managed_servers", fake_servers)


def test_run_requires_dataset_or_config(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run"])


def test_cli_overrides_reach_runner(tmp_path, monkeypatch, dataset_root):
    captured = {}

    def fake_run_rounds(dataset_root_, **kw):
        captured.update(kw, dataset_root=dataset_root_)

        class R:
            out_dir = tmp_path
            stopped = "rounds"
            champion = {"round": 0, "id": "baseline",
                        "eval": {"micro_f_beta": 0.5}}
            decisions = []
        return R()

    import agentic_configuration_search.loop.runner as runner
    monkeypatch.setattr(runner, "run_rounds", fake_run_rounds)
    rc = cli.main(["run", "--dataset", str(dataset_root), "--planner", "llm",
                   "--rounds", "7", "--out", str(tmp_path / "o"),
                   "--candidates", str(tmp_path / "c.yaml")])
    assert rc == 0
    assert captured["planner"] == "llm" and captured["rounds"] == 7
    assert captured["cfg"].planner.kind == "llm"
    assert str(captured["dataset_root"]) == str(dataset_root)
    assert captured["candidates_file"] == str(tmp_path / "c.yaml")
    # The detector is in-process, so it is always wired (model load is lazy). No
    # planner.base_url — llm_chat/caption_fn are None (materialize raises an explicit error).
    from agentic_configuration_search.clients.yoloe import YoloeClient
    assert isinstance(captured["client"], YoloeClient)
    assert captured["detector_hooks"]["weights_ctl"] == captured["client"].set_weights
    assert captured["llm_chat"] is None and captured["caption_fn"] is None
    assert callable(captured["encode_fn"])


def test_cli_wires_clients_from_config(tmp_path, monkeypatch, dataset_root):
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        f"dataset: {dataset_root}\n"
        "planner:\n  kind: llm\n  base_url: http://127.0.0.1:9998\n")
    captured = {}

    def fake_run_rounds(dataset_root_, **kw):
        captured.update(kw)

        class R:
            out_dir = tmp_path
            stopped = "rounds"
            champion = {"round": 0, "id": "baseline",
                        "eval": {"micro_f_beta": 0.5}}
            decisions = []
        return R()

    import agentic_configuration_search.loop.runner as runner
    monkeypatch.setattr(runner, "run_rounds", fake_run_rounds)
    cli.main(["run", "--config", str(cfg_yaml)])
    from agentic_configuration_search.clients.yoloe import YoloeClient
    assert isinstance(captured["client"], YoloeClient)
    assert callable(captured["llm_chat"])
    assert callable(captured["refine_alias_fn"])
