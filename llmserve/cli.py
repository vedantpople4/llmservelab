from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from llmserve.config.loader import config_hash, load_config


def _validate(paths: list[str]) -> int:
    failed = 0
    for path in paths:
        try:
            cfg = load_config(path)
        except (ValidationError, ValueError, OSError) as e:
            failed += 1
            print(f"FAIL {path}\n{e}\n", file=sys.stderr)
            continue
        load = f"closed c={cfg.load.concurrency}" if cfg.load.mode == "closed" else "open"
        print(
            f"ok   {path}  [{cfg.experiment}] {cfg.server.kind} · {load} · "
            f"{len(cfg.workload.classes)} class(es) · hash {config_hash(cfg)[:12]}"
        )
    return 1 if failed else 0


def _generate_workload(path: str, output: str, rep: int) -> int:
    from llmserve.workload.generator import materialize, save_workload, summarize

    try:
        cfg = load_config(path)
        workload = materialize(cfg, rep=rep)
        save_workload(workload, Path(output))
    except (ValidationError, ValueError, OSError) as e:
        print(f"FAIL {path}\n{e}", file=sys.stderr)
        return 1
    print(f"ok   {output}  {summarize(workload, cfg, rep)}")
    return 0


def _benchmark(path: str, *, out: str, endpoint: str | None, allow_dirty: bool) -> int:
    """Warm-up → N repetitions → parquet + summaries + metadata (plan Phase 2 items 8-10)."""
    import asyncio

    from llmserve.runner.experiment import DirtyTreeError, run_experiment

    try:
        cfg = load_config(path)
    except (ValidationError, ValueError, OSError) as e:
        print(f"FAIL {path}\n{e}", file=sys.stderr)
        return 2
    try:
        run_dir, summary = asyncio.run(
            run_experiment(
                cfg,
                out_root=Path(out),
                endpoint=endpoint,
                allow_dirty=allow_dirty,
                progress=print,
            )
        )
    except DirtyTreeError as e:
        print(f"FAIL {e}", file=sys.stderr)
        return 2
    rate = summary["success_rate"]["mean"]
    shown = "n/a" if rate is None else f"{100 * rate:.1f}%"
    flag = "INVALID" if summary["invalid"] else "valid"
    print(f"ok   {run_dir}  {summary['repetitions']} rep(s) · success {shown} · {flag}")
    return 1 if summary["invalid"] else 0


def _add_benchmark_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--out", default="results", help="results root (default results)")
    parser.add_argument(
        "--endpoint", default=None, help="override server.endpoint (the mock's port is dynamic)"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run even if the git working tree has uncommitted changes",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llmserve")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate experiment configs").add_argument("paths", nargs="+")
    generate = sub.add_parser(
        "generate-workload", help="materialize a frozen workload file (ADR-0004)"
    )
    generate.add_argument("path", help="experiment config")
    generate.add_argument("-o", "--output", default="workload.parquet", help="output path")
    generate.add_argument("--rep", type=int, default=0, help="repetition index (default 0)")
    benchmark = sub.add_parser("benchmark", help="run a complete benchmark from a config")
    benchmark.add_argument("path", help="experiment config")
    _add_benchmark_flags(benchmark)
    run = sub.add_parser("run", help="alias of benchmark (the PRD lists both)")
    run.add_argument("--config", required=True, dest="path", help="experiment config")
    _add_benchmark_flags(run)
    for name in ("analyze", "report"):
        sub.add_parser(name).add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "validate":
        raise SystemExit(_validate(args.paths))
    if args.command == "generate-workload":
        raise SystemExit(_generate_workload(args.path, args.output, args.rep))
    if args.command in ("benchmark", "run"):
        raise SystemExit(
            _benchmark(
                args.path, out=args.out, endpoint=args.endpoint, allow_dirty=args.allow_dirty
            )
        )
    raise SystemExit(f"llmserve {args.command}: not implemented yet")
