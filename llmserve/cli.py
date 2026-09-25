from __future__ import annotations

import argparse
import sys

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llmserve")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate experiment configs").add_argument("paths", nargs="+")
    for name in ("run", "generate-workload", "benchmark", "analyze", "report"):
        sub.add_parser(name).add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "validate":
        raise SystemExit(_validate(args.paths))
    raise SystemExit(f"llmserve {args.command}: not implemented yet")
