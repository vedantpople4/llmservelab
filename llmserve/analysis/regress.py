"""Performance regression check (plan Phase 7 item 6, PRD §39).

A standard short workload runs before and after a harness change. Its headline metrics are
compared with a stored baseline taken with the same config, and any metric that moved by more
than `TOLERANCE` of its baseline value, in either direction, is flagged. A faster number after a
harness change is as suspicious as a slower one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOLERANCE = 0.10

# Cross-rep means of the summary fields a harness change could move.
METRICS: dict[str, tuple[str, ...]] = {
    "ttft p50 (s)": ("latency", "ttft_s", "p50", "mean"),
    "ttft p95 (s)": ("latency", "ttft_s", "p95", "mean"),
    "tpot p50 (s)": ("latency", "tpot_s", "p50", "mean"),
    "tpot p95 (s)": ("latency", "tpot_s", "p95", "mean"),
    "e2e p50 (s)": ("latency", "e2e_s", "p50", "mean"),
    "client lag p99 (s)": ("latency", "client_lag_s", "p99", "mean"),
    "generated tok/s": ("throughput", "gen_toks_s", "mean"),
    "requests/s": ("throughput", "req_s", "mean"),
    "success rate": ("success_rate", "mean"),
}


@dataclass(frozen=True)
class Baseline:
    config_hash: str
    run: str
    metrics: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> Baseline:
        raw = json.loads(path.read_text())
        return cls(raw["config_hash"], raw["run"], {k: float(v) for k, v in raw["metrics"].items()})

    def save(self, path: Path) -> None:
        body = {"config_hash": self.config_hash, "run": self.run, "metrics": self.metrics}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class Change:
    metric: str
    baseline: float
    current: float

    @property
    def relative(self) -> float:
        if self.baseline == 0:
            return 0.0 if self.current == 0 else float("inf")
        return (self.current - self.baseline) / abs(self.baseline)

    @property
    def flagged(self) -> bool:
        return abs(self.relative) > TOLERANCE


def headline(summary: Mapping[str, Any]) -> dict[str, float]:
    """The `METRICS` a run summary has values for; a metric the run could not measure is absent."""
    out: dict[str, float] = {}
    for name, path in METRICS.items():
        value: Any = summary
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, int | float):
            out[name] = float(value)
    return out


def compare(baseline: Mapping[str, float], current: Mapping[str, float]) -> list[Change]:
    """One `Change` per metric present in both, in `METRICS` order."""
    return [Change(m, baseline[m], current[m]) for m in METRICS if m in baseline and m in current]
