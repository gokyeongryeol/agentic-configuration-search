"""Leaf module for path and lifetime constants — import this when only the
constants are needed outside services.

(Kept separate so the embedder need not depend on the whole process-management
module just for VLLM_VENV.)
"""

import os
from pathlib import Path

# Worker (dinov3, adapt) venv — override with ADC_VLLM_VENV. If absolute,
# `repo_root / VLLM_VENV` resolves independently of repo_root (Path.__truediv__
# returns an absolute right operand unchanged).
VLLM_VENV = Path(os.environ.get("ADC_VLLM_VENV", "/data/krgo/vllm2"))
# Directory that contains the package (src/ in this repo). Workers launched in
# the vLLM venv — where this package is not installed — get it as PYTHONPATH so
# `-m agentic_detector_config.workers.<name>` resolves regardless of cwd.
PACKAGE_PARENT = Path(__file__).resolve().parents[2]
TERM_GRACE_SEC = 15
PORT_RELEASE_SEC = 30   # max wait for port release after stop (avoids restart race)
