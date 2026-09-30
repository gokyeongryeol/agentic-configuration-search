"""One-shot GPU workers — the only place in the package where torch may be imported.

Each module runs as a subprocess (dinov3 and adapt in the vLLM venv via
`-m agentic_detector_config.workers.<name>`; yoloe_tail in the run venv, which has
ultralytics). The vLLM venv imports this __init__ on `-m`, so keep it lightweight.
"""
