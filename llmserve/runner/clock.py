"""The run's clock (ADR-006).

Durations use `time.perf_counter_ns()` relative to run start; wall-clock time appears only in run
metadata, via the anchor recorded here. One clock per benchmark process, passed to every component
that timestamps anything, so arrival, dispatch and token times are comparable.
"""

from __future__ import annotations

import time
from collections.abc import Callable

Clock = Callable[[], int]
"""Monotonic nanoseconds since run start."""


class RunClock:
    def __init__(self) -> None:
        self._t0_ns = time.perf_counter_ns()
        self.wall_anchor_ns: int = time.time_ns()
        """Wall-clock (unix) nanoseconds captured at the same instant as `_t0_ns`."""

    def __call__(self) -> int:
        return time.perf_counter_ns() - self._t0_ns
