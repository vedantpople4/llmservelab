from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from llmserve.config.loader import config_hash, load_config


def _validate(paths: list[str]) -> int:
    from llmserve.runner.suite import validate_suite

    failed = 0
    for path in paths:
        if _is_suite(Path(path)):
            try:
                print(f"ok   {path}  {validate_suite(Path(path))}")
            except (ValidationError, ValueError, OSError) as e:
                failed += 1
                print(f"FAIL {path}\n{e}\n", file=sys.stderr)
            continue
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


def _suite(
    path: str,
    *,
    out: str,
    endpoint: str | None,
    allow_dirty: bool,
    only: list[str] | None,
    sets: list[str],
    plan: bool,
) -> int:
    """Run (or with `plan`, list) a suite's points; resumable, so rerunning skips done points."""
    import asyncio

    from llmserve.runner.experiment import DirtyTreeError
    from llmserve.runner.suite import (
        MissingRuntimeValue,
        estimate_minutes,
        load_suite,
        parse_set,
        plan_suite,
        run_suite,
    )

    try:
        suite = load_suite(Path(path))
        overrides = parse_set(sets)
        points = plan_suite(suite, overrides=overrides)
    except (ValidationError, ValueError, OSError) as e:
        print(f"FAIL {path}\n{e}", file=sys.stderr)
        return 2
    if plan:
        total = 0.0
        for point in points:
            if only and point.label not in only:
                continue
            minutes = estimate_minutes(point)
            total += minutes or 0.0
            est = "n/a" if minutes is None else f"~{minutes:.0f} min"
            print(f"{point.label:<12} {point.experiment:<32} {point.config_hash[:12]}  {est}")
        print(f"total ~{total:.0f} min (rough estimate)")
        return 0
    try:
        manifest = asyncio.run(
            run_suite(
                suite,
                out_root=Path(out),
                overrides=overrides,
                only=only,
                endpoint=endpoint,
                allow_dirty=allow_dirty,
            )
        )
    except (DirtyTreeError, MissingRuntimeValue, ValueError) as e:
        print(f"FAIL {e}", file=sys.stderr)
        return 2
    counts: dict[str, int] = {}
    for entry in manifest.points:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    print(f"ok   suite {suite.name}: " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("failed") else 0


def _analyze(path: str) -> int:
    from llmserve.analysis.report import describe_run
    from llmserve.analysis.runs import Run

    run_dir = Path(path)
    if not (run_dir / "metadata.json").is_file():
        print(f"FAIL {path}: not a run directory (no metadata.json)", file=sys.stderr)
        return 2
    print(describe_run(Run(run_dir)))
    return 0


def _report(path: str, *, out: str, dev: bool) -> int:
    from llmserve.analysis.report import build_report

    root = Path(path)
    if not root.is_dir():
        print(f"FAIL {path}: no results directory", file=sys.stderr)
        return 2
    print(f"ok   {build_report(root, Path(out), dev=dev)}")
    return 0


def _is_suite(path: Path) -> bool:
    import yaml

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and "suite" in data


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
    suite = sub.add_parser("suite", help="run a sweep of experiments (resumable)")
    suite.add_argument("path", help="suite file (configs/suites/*.yaml)")
    _add_benchmark_flags(suite)
    suite.add_argument(
        "--only", nargs="+", default=None, metavar="LABEL", help="run only these point labels"
    )
    suite.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a config field for every point, e.g. load.capacity_rps=3.2",
    )
    suite.add_argument(
        "--plan", action="store_true", help="list points and duration estimates; run nothing"
    )
    analyze = sub.add_parser("analyze", help="describe one run directory")
    analyze.add_argument("path", help="results/<experiment>/<run_id>")
    report = sub.add_parser("report", help="build the Baseline Performance Report")
    report.add_argument("path", nargs="?", default="results", help="results root (default results)")
    report.add_argument(
        "-o", "--out", default="docs/reports", help="output directory (default docs/reports)"
    )
    report.add_argument(
        "--dev",
        action="store_true",
        help="include non-reportable runs (non-vllm-metal, dirty, invalid, drifted) under a banner",
    )
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
    if args.command == "suite":
        raise SystemExit(
            _suite(
                args.path,
                out=args.out,
                endpoint=args.endpoint,
                allow_dirty=args.allow_dirty,
                only=args.only,
                sets=args.sets,
                plan=args.plan,
            )
        )
    if args.command == "analyze":
        raise SystemExit(_analyze(args.path))
    if args.command == "report":
        raise SystemExit(_report(args.path, out=args.out, dev=args.dev))
    raise SystemExit(f"llmserve {args.command}: not implemented yet")
