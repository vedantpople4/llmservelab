"""`uv run python -m llmserve.mock --port 8001` — run the mock OpenAI server standalone."""

from __future__ import annotations

import argparse

import uvicorn

from llmserve.mock.engine import DelayModel, MockEngine
from llmserve.mock.server import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m llmserve.mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="mock")
    parser.add_argument(
        "--instant",
        action="store_true",
        help="zero delays (harness-overhead measurements and fast tests)",
    )
    args = parser.parse_args(argv)
    engine = MockEngine(model=args.model, delay=DelayModel.instant() if args.instant else None)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
