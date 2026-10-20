"""Package architecture invariants — layer direction, torch-free.

Pins the README layering in code:

    core ← clients ← infer ← fitting ← eval ← diagnose ← planner ← loop ← cli
    (services orthogonal)

A failure here is an architecture violation, not a refactor — before adding an
exception, question whether the shared logic lives in the right place.
"""

import ast
import subprocess
import sys
from pathlib import Path

import agentic_configuration_search

PKG = Path(agentic_configuration_search.__file__).parent

# In-package zones each zone may import (itself is always allowed).
# "config" means the top-level leaf module config.py.
ALLOWED = {
    "core": set(),
    "clients": {"core"},
    # infer may use only the constant leaf of services (paths) — not process management (external)
    "infer": {"core", "clients", "config", "services.paths"},
    # fitting produces materials — services allowed for the worker-subprocess training path
    "fitting": {"core", "clients", "infer", "services", "config"},
    # eval → diagnose only records the union-modality diagnostic (diagnose.ensemble) —
    # not a decision path
    "eval": {"core", "clients", "infer", "fitting", "diagnose", "config"},
    "diagnose": {"core", "config"},
    "planner": {"core", "clients", "diagnose", "config"},
    "loop": {"core", "clients", "infer", "fitting", "eval", "diagnose",
             "planner", "services", "config"},
    # services → workers only to **look up worker script paths** (run as subprocess)
    "services": {"core", "workers", "config"},
    "workers": set(),                      # no in-package imports (subprocess only)
    # cli → diagnose only for the report subcommand (reads the ledger)
    "cli": {"config", "services", "loop", "diagnose"},
    "config": set(),
    "__init__": set(),
}


def _segment(module: str) -> str:
    """'<pkg>.core.coco' → 'core', '<pkg>.config' → 'config'."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def _internal_imports(path: Path, module: str) -> set[str]:
    """In-package modules this module imports (absolute names)."""
    tree = ast.parse(path.read_text())
    pkg_parts = module.split(".")[:-1] if path.name != "__init__.py" else module.split(".")
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith("agentic_configuration_search")}
        elif isinstance(node, ast.ImportFrom):
            if node.level:                 # relative import → absolute name
                base = pkg_parts[: len(pkg_parts) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
                out.add(mod)
                # `from .core import coco` form — the name may be a submodule
                for a in node.names:
                    out.add(f"{mod}.{a.name}")
            elif node.module and node.module.startswith("agentic_configuration_search"):
                out.add(node.module)
    return {m for m in out if m.startswith("agentic_configuration_search")}


def _all_modules():
    for p in sorted(PKG.rglob("*.py")):
        rel = p.relative_to(PKG).with_suffix("")
        # leftovers in hidden dirs (.ipynb_checkpoints etc.) are not part of the package
        if any(part.startswith(".") for part in rel.parts):
            continue
        parts = ["agentic_configuration_search"] + list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1] or ["agentic_configuration_search"]
        yield ".".join(parts), p


def test_layering():
    violations = []
    for module, path in _all_modules():
        seg = _segment(module) if module != "agentic_configuration_search" else "__init__"
        allowed = ALLOWED.get(seg)
        if allowed is None:
            violations.append(f"{module}: unknown segment {seg!r} — register it in ALLOWED")
            continue
        for target in _internal_imports(path, module):
            tseg = _segment(target)
            if tseg == seg or target == "agentic_configuration_search":
                continue
            # check submodule-level allowances ('services.paths' form) first
            sub = ".".join(target.split(".")[1:3])
            if sub in allowed or tseg in allowed:
                continue
            violations.append(f"{module} → {target} ({seg} segment allows only {sorted(allowed)})")
    assert not violations, "layer violations:\n" + "\n".join(violations)


def test_orchestrator_is_torch_free():
    """No module outside workers/ imports torch (transitively) — every GPU
    dependency lives out of process (vLLM server, worker subprocess) or is a lazy import."""
    mods = [m for m, _ in _all_modules()
            if _segment(m) != "workers" and m != "agentic_configuration_search"]
    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name.split('.')[0] == 'torch' else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('torch-free violation: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        + "".join(f"import {m}\n" for m in mods)
        + "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
