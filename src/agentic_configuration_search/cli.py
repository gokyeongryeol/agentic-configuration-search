"""acs CLI — runs the round loop.

The detector (YOLOE) runs in-process, so there is no detector server. The vLLM
(planner) server is handled by `loop.hooks.managed_servers`: if
`planner.base_url` is set it attaches to that endpoint (externally managed),
otherwise it launches one itself (persistent).

GPU hooks (embedding, captioning) are wired only as far as the config allows;
if an axis is enabled without its hook, loop.materialize raises an explicit
error naming what is missing.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path

# Cap OMP/BLAS threads — must run before numpy/sklearn are imported. Matrices
# here have only hundreds of rows, so thread spawn/sync dominates compute, and
# under parallel GPU runs (many processes x core-count threads) CPU thrashing
# effectively stalls everything (same rationale as the cap in tests/conftest.py).
# setdefault, so a value set externally wins.
for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

from .config import Config, config_from_dict, load_config


def _build_cfg(args) -> Config:
    if args.config:
        cfg = load_config(args.config)
    elif args.dataset:
        cfg = config_from_dict({"dataset": args.dataset})
    else:
        raise SystemExit("acs run: --config or --dataset is required")
    if args.dataset:
        cfg = dataclasses.replace(cfg, dataset=args.dataset)
    if args.planner:
        cfg = dataclasses.replace(
            cfg, planner=dataclasses.replace(cfg.planner, kind=args.planner))
    if args.rounds is not None:
        cfg = dataclasses.replace(cfg, rounds=args.rounds)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acs", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the round loop")
    run.add_argument("--config", help="path to the config yaml")
    run.add_argument("--dataset", help="Roboflow dataset root (overrides dataset in the config)")
    run.add_argument("--out", default=None, help="run directory (default: <out_dir>/<dataset name>/run)")
    run.add_argument("--planner", choices=("deterministic", "llm"))
    run.add_argument("--rounds", type=int)
    run.add_argument("--resume", action="store_true",
                     help="continue from the round after history.jsonl (LLM proposals are replayed)")
    run.add_argument("--candidates", help="human-injected candidates yaml (hybrid loop)")
    rep = sub.add_parser("report", help="run trajectory report — writes report.md")
    rep.add_argument("run_dir")
    ab = sub.add_parser("ab", help="compare two run trajectories (deterministic vs llm A/B)")
    ab.add_argument("run_a")
    ab.add_argument("run_b")
    ab.add_argument("--labels", default="A,B")
    args = parser.parse_args(argv)

    if args.cmd == "report":
        from .diagnose.report import render_report
        print(render_report(Path(args.run_dir)))
        return 0
    if args.cmd == "ab":
        import json as _json

        from .diagnose.report import ab_compare
        la, lb = args.labels.split(",", 1)
        out = ab_compare(Path(args.run_a), Path(args.run_b),
                         label_a=la.strip(), label_b=lb.strip())
        print(_json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = _build_cfg(args)
    # Workers (dinov3, adapt, yoloe_tail) are subprocesses with a different cwd — all paths must be absolute
    dataset_root = Path(cfg.dataset).resolve()
    out_dir = (Path(args.out) if args.out else
               Path(cfg.out_dir) / dataset_root.name / "run").resolve()

    from .loop.hooks import build_hooks, managed_servers
    from .loop.runner import run_rounds
    repo_root = Path.cwd()
    with managed_servers(cfg, repo_root, out_dir) as llm_base:
        result = run_rounds(dataset_root, out_dir=out_dir,
                            planner=cfg.planner.kind, rounds=cfg.rounds,
                            cfg=cfg, resume=args.resume,
                            candidates_file=args.candidates,
                            **build_hooks(cfg, repo_root, out_dir,
                                          llm_base_url=llm_base))
    champ = result.champion
    print(f"stopped: {result.stopped} — champion round {champ['round']} "
          f"({champ['id']}), holdout micro F(β) {champ['eval']['micro_f_beta']:.4f}")
    print(f"artifacts: {result.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
