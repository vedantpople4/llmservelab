"""Scheduler decision latency at a fixed queue depth (plan Phase 5 item 3).

    uv run python scripts/bench_scheduler.py            # depth 1,000, budget P99 < 100 µs
    uv run python scripts/bench_scheduler.py --depth 5000

Times one admission decision the way the gateway makes it: copy the waiting set to a list, then
`select_next`. The queue is seeded and mixes prompt lengths, output estimates and classes; each
decision sees a fresh arrival replacing the request admitted before it, so cached keys are
exercised the way a steady queue exercises them. Exits 1 when any policy's P99 is over budget.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from llmserve.scheduler import registry
from llmserve.scheduler.base import Priority, Request, SystemState

POLICIES: list[tuple[str, dict[str, float | str]]] = [
    ("fifo", {}),
    ("spf", {}),
    ("sjf", {}),
    ("priority", {}),
    ("priority", {"aging_s": 30.0}),
]


def request(rng: random.Random, i: int, arrival: float) -> Request:
    return Request(
        request_id=f"r{i:07d}",
        arrival_time=arrival,
        prompt_tokens=rng.randint(16, 4096),
        max_output_tokens=512,
        output_tokens_est=rng.choice([None, rng.randint(16, 512)]),
        priority=rng.choice(list(Priority)),
    )


def bench(name: str, params: dict[str, float | str], depth: int, decisions: int) -> np.ndarray:
    rng = random.Random(42)
    scheduler = registry.create(name, params)
    waiting = {request(rng, i, i * 0.01): None for i in range(depth)}
    times = np.empty(decisions, dtype=np.int64)
    for k in range(decisions):
        state = SystemState(now=depth * 0.01 + k * 0.01, queue_depth=depth, in_flight=16)
        start = time.perf_counter_ns()
        chosen = scheduler.select_next(list(waiting), state)
        times[k] = time.perf_counter_ns() - start
        assert chosen is not None
        del waiting[chosen]
        waiting[request(rng, depth + k, state.now)] = None
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--decisions", type=int, default=5000)
    parser.add_argument("--budget-us", type=float, default=100.0)
    args = parser.parse_args()

    over = []
    print(
        f"queue depth {args.depth}, {args.decisions} decisions, budget P99 < {args.budget_us:g} µs"
    )
    for name, params in POLICIES:
        label = name + "".join(f" {k}={v}" for k, v in params.items())
        times = bench(name, params, args.depth, args.decisions) / 1e3
        p50, p99 = np.percentile(times, [50, 99])
        verdict = "ok" if p99 < args.budget_us else "OVER"
        print(f"  {label:<22} P50 {p50:6.1f} µs  P99 {p99:6.1f} µs  {verdict}")
        if p99 >= args.budget_us:
            over.append(label)
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
