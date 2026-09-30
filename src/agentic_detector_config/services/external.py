"""Subprocess and server management — the external-venv worker (crop classifier
training) and ManagedServer/VllmServer.

Lifecycle: vLLM may persist across runs. Shutdown is SIGTERM → grace → SIGKILL
per process group; an occupied port fails immediately (persist attaches only to
a healthy server).
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Constants are owned by paths.py — re-exported.
from .paths import (PACKAGE_PARENT, VLLM_VENV,  # noqa: E402,F401
                    TERM_GRACE_SEC, PORT_RELEASE_SEC)


def run_adapt_worker(repo_root: Path, *, manifest: Path, num_classes: int,
                     out_path: Path, device: str, log_path: Path,
                     popen=subprocess.Popen) -> Path:
    """Train the crop classifier (SigLIP2+LoRA) — one-shot subprocess in the vLLM venv.

    Same pattern as the dinov3 worker (`-m agentic_detector_config.workers.adapt`):
    torch and peft live in the vLLM venv and the package stays torch-free.
    Hyperparameters are worker constants (universal-best), so none are taken here.
    """
    python = repo_root / VLLM_VENV / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"vLLM venv python not found: {python} — check ADC_VLLM_VENV")
    manifest, out_path, log_path = (Path(manifest).resolve(),
                                    Path(out_path).resolve(),
                                    Path(log_path).resolve())
    cmd = [str(python), "-m", "agentic_detector_config.workers.adapt",
           "--manifest", str(manifest), "--num-classes", str(int(num_classes)),
           "--out", str(out_path), "--device", device]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("running adapt worker: num_classes=%d device=%s (log: %s)",
             num_classes, device, log_path)
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_PARENT)}   # package not installed in that venv
    with open(log_path, "ab") as fh:
        proc = popen(cmd, cwd=repo_root, env=env, stdout=fh,
                     stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"adapt worker failed (exit {rc}) — log: {log_path}")
    if not out_path.exists():
        raise RuntimeError(f"adapt worker output missing: {out_path}")
    return out_path


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


# ------------------------------------------------------------------ servers
# See the module docstring for lifecycle rules.

import signal      # noqa: E402
import time        # noqa: E402

import requests    # noqa: E402


class ManagedServer:
    """Common subprocess server: launch / health polling / process-group
    shutdown / log tee.

    persist=True is resident mode: attach (without owning) if a healthy server
    already holds the port; otherwise launch one but never take it down in
    stop() — keeps VRAM held between runs and avoids restart cost. An unhealthy
    occupant still fails immediately (silently attaching to a dead server is the
    worst failure mode)."""

    name = "server"
    persist = False

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.proc: Optional[subprocess.Popen] = None
        self._attached = False

    # defined by subclasses
    def command(self) -> list[str]: ...
    def env(self) -> dict: ...
    def cwd(self) -> Optional[Path]: return None
    def health_url(self) -> str: ...
    def startup_timeout(self) -> float: ...
    def host_port(self) -> tuple[str, int]: ...

    def start(self) -> None:
        host, port = self.host_port()
        if not port_free(host, port):
            if self.persist:
                # Another process may be launching it right now — wait until ready,
                # then attach (resolves the concurrent-launch race). Fail if it
                # never becomes healthy.
                deadline = time.monotonic() + self.startup_timeout()
                while time.monotonic() < deadline:
                    try:
                        if requests.get(self.health_url(), timeout=5).ok:
                            log.info("%s: attached to resident server (%s:%d)", self.name, host, port)
                            self._attached = True
                            return
                    except requests.RequestException:
                        pass
                    if port_free(host, port):   # occupant gone — launch ourselves
                        break
                    time.sleep(3)
                if not port_free(host, port):
                    raise RuntimeError(
                        f"{self.name}: {host}:{port} is occupied but not ready within "
                        f"{self.startup_timeout()}s — assuming a dead process; clean it up")
            else:
                raise RuntimeError(
                    f"{self.name}: {host}:{port} already occupied — refusing to silently attach "
                    f"to a leftover process. Kill it or change the port in the config.")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "ab")
        cmd = self.command()
        log.info("starting %s: %s (log: %s)", self.name, " ".join(cmd), self.log_path)
        self.proc = subprocess.Popen(
            cmd, cwd=self.cwd(), env=self.env(),
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True)  # separate process group
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout()
        url = self.health_url()
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                tail = self._log_tail()
                raise RuntimeError(
                    f"{self.name} exited during startup (exit {self.proc.returncode}).\n"
                    f"--- {self.log_path} tail ---\n{tail}")
            try:
                if requests.get(url, timeout=3).ok:
                    log.info("%s ready: %s", self.name, url)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"{self.name}: {url} not ready within {self.startup_timeout()}s. "
                           f"log: {self.log_path}")

    def _log_tail(self, lines: int = 40) -> str:
        try:
            content = self.log_path.read_text(errors="replace").splitlines()
            return "\n".join(content[-lines:])
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        """Kill the process group and **wait until the port is actually released.**

        Without the wait, a restart sees the port it just released as "occupied"
        and fails — a race between process exit and socket release.
        """
        if self._attached or (self.persist and self.proc is not None):
            # resident mode — keep running after the run ends (the next run attaches)
            log.info("%s: resident, left running (stop skipped)", self.name)
            self.proc = None
            self._attached = False
            return
        if not self.proc:
            return
        if self.proc.poll() is None:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=TERM_GRACE_SEC)
            except subprocess.TimeoutExpired:
                log.warning("%s: no response to SIGTERM within %ds — sending SIGKILL", self.name, TERM_GRACE_SEC)
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        self.proc = None
        self._wait_port_free()

    def _wait_port_free(self, timeout: float = PORT_RELEASE_SEC) -> None:
        host, port = self.host_port()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if port_free(host, port):
                return
            time.sleep(0.5)
        log.warning("%s: %.0fs after stop, %s:%d was not released — the next start may fail",
                    self.name, timeout, host, port)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class VllmServer(ManagedServer):
    name = "vllm"

    def __init__(self, planner_cfg, log_path: Path):
        super().__init__(log_path)
        self.cfg = planner_cfg        # Config.planner (model + launch)
        self.persist = bool(planner_cfg.launch.persist)

    def command(self) -> list[str]:
        launch = self.cfg.launch
        cmd = [str(VLLM_VENV / "bin" / "vllm"), "serve", self.cfg.model,
               "--host", launch.host, "--port", str(launch.port),
               "--gpu-memory-utilization", str(launch.gpu_memory_utilization),
               "--max-model-len", str(launch.max_model_len)]
        if launch.enforce_eager:
            cmd.append("--enforce-eager")
        return cmd

    def env(self) -> dict:
        env = dict(os.environ)  # inherit HF_TOKEN / HF_HOME etc. as-is
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.launch.gpu)
        # Work around the flashinfer sampler's runtime JIT dying on an nvcc
        # version mismatch — the native sampler is sufficient
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        return env

    def health_url(self) -> str:
        return f"http://{self.cfg.launch.host}:{self.cfg.launch.port}/health"

    def startup_timeout(self) -> float:
        return self.cfg.launch.startup_timeout

    def host_port(self):
        return self.cfg.launch.host, self.cfg.launch.port

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.launch.host}:{self.cfg.launch.port}"
