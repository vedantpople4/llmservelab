"""`uv run python -m llmserve.mock --port 8001` — run the mock OpenAI server standalone."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from llmserve.mock.engine import CbParams, ContinuousBatchEngine, DelayModel, MockEngine
from llmserve.mock.server import Engine, create_app


def build_engine(args: argparse.Namespace) -> Engine:
    if args.engine == "cb":
        if args.params is not None:
            params = CbParams.from_json(Path(args.params).read_text())
        else:
            params = CbParams.instant() if args.instant else CbParams()
        return ContinuousBatchEngine(model=args.model, params=params)
    return MockEngine(model=args.model, delay=DelayModel.instant() if args.instant else None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m llmserve.mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="mock")
    parser.add_argument(
        "--engine",
        choices=("delay", "cb"),
        default="delay",
        help="delay = Phase 1 delay model (default); cb = continuous-batching simulator",
    )
    parser.add_argument(
        "--instant",
        action="store_true",
        help="zero delays (harness-overhead measurements and fast tests)",
    )
    parser.add_argument(
        "--params",
        metavar="JSON",
        help="CbParams file for --engine cb, e.g. from scripts/calibrate_mock.py",
    )
    args = parser.parse_args(argv)
    if args.params is not None and args.engine != "cb":
        parser.error("--params needs --engine cb")
    if args.params is not None and args.instant:
        parser.error("--params and --instant are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        engine = build_engine(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"python -m llmserve.mock: --params {args.params}: {exc}") from None
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
