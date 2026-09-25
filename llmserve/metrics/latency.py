from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestTiming:
    """Wall-clock timestamps (seconds) for one streamed request."""

    start: float
    token_times: Sequence[float]

    @property
    def ttft(self) -> float:
        return self.token_times[0] - self.start

    @property
    def e2e(self) -> float:
        return self.token_times[-1] - self.start

    @property
    def tpot(self) -> float | None:
        n = len(self.token_times)
        if n < 2:
            return None
        return (self.e2e - self.ttft) / (n - 1)

    @property
    def inter_token_latencies(self) -> list[float]:
        t = self.token_times
        return [b - a for a, b in zip(t, t[1:], strict=False)]
