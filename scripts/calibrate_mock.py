"""Fit the continuous-batching mock's step costs to measured runs (plan Phase 3 item 6).

    uv run python scripts/calibrate_mock.py results/E02/<run> results/E03/<run> \\
        --kv-capacity-tokens 30000 -o mock_params.json
    uv run python -m llmserve.mock --engine cb --params mock_params.json

The inputs must be closed-loop runs that include a concurrency-1 run with at least two prompt
lengths and span at least two concurrency levels (`llmserve.mock.calibration` has the model).
Calibration is qualitative: it makes laptop experiments point the same direction as real ones,
and the printed table shows how far the calibrated mock is from the measurements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from llmserve.analysis.runs import Run
from llmserve.mock.calibration import COEFFICIENTS, CalibrationError, fit_cb_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibrate_mock.py", description="Fit the cb mock's step costs to measured runs."
    )
    parser.add_argument("runs", nargs="+", type=Path, metavar="RUN_DIR")
    parser.add_argument("--kv-capacity-tokens", type=int, required=True)
    parser.add_argument("--max-batched-tokens", type=int, default=None)
    parser.add_argument("--prefill-chunk", type=int, default=None)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.runs if not (p / "config.resolved.yaml").is_file()]
    if missing:
        print(f"calibrate_mock: not a run directory: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        cal = fit_cb_params(
            [Run(p) for p in args.runs],
            kv_capacity_tokens=args.kv_capacity_tokens,
            max_batched_tokens=args.max_batched_tokens,
            prefill_chunk=args.prefill_chunk,
        )
    except CalibrationError as exc:
        print(f"calibrate_mock: cannot identify the mock's params: {exc}", file=sys.stderr)
        return 2

    args.output.write_text(cal.params.to_json())
    for name in COEFFICIENTS:
        print(f"{name:>20} = {getattr(cal.params, name):.4g}")
    print(f"{'request_overhead_s':>20} = {cal.request_overhead_s:.4g}  (not in the mock)")
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print(cal.table.to_string(index=False))
    for warning in cal.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
