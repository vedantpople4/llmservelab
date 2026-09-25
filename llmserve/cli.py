from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmserve")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "generate-workload", "benchmark", "analyze", "report"):
        sub.add_parser(name).add_argument("path")
    args = parser.parse_args()
    raise SystemExit(f"llmserve {args.command}: not implemented yet")
