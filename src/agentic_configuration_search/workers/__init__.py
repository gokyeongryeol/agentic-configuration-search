"""One-shot GPU workers — the only place in the package where torch may be imported.

Each module runs as a subprocess (dinov3 and adapt in the worker venv via
`-m agentic_configuration_search.workers.<name>`; yoloe_tail in the run venv, which has
ultralytics). This __init__ is imported on `-m`, so keep it lightweight.
"""
