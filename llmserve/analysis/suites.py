"""Suite results read back from disk (Phase 3): one swept axis, one `Run` per point.

The suite runner writes `<results_root>/_suites/<suite>.json` naming each point's axis value,
label, status and run directory (relative to the results root). A point is usable when its
status is `done` or `skipped` (an earlier invocation already ran it) and its run directory
exists; every other point is left out here, so analysis never sees a half-written run.

`sweep_table` is the one tidy view of a sweep: per point and metric, the across-rep mean and
its t-based 95% CI (`stats.mean_ci`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from llmserve.analysis.runs import Run
from llmserve.analysis.stats import mean_ci

USABLE_STATUSES = frozenset({"done", "skipped"})
SWEEP_COLUMNS = ("label", "value", "metric", "mean", "lo", "hi", "n")


@dataclass(frozen=True)
class SuitePoint:
    label: str
    value: float
    run: Run


@dataclass(frozen=True)
class SuiteRuns:
    suite: str
    suite_file: str
    axis: str
    points: tuple[SuitePoint, ...]  # sorted by value

    def keep(self, predicate: Callable[[Run], bool]) -> SuiteRuns:
        return replace(self, points=tuple(p for p in self.points if predicate(p.run)))


def suite_file_for(suite: str) -> str:
    """The suite YAML a missing suite would be run from (`concurrency-p128` →
    `configs/suites/concurrency_p128.yaml`)."""
    return f"configs/suites/{suite.replace('-', '_')}.yaml"


def load_suite_runs(root: Path, suite: str) -> SuiteRuns | None:
    """The usable points of `suite`, sorted by axis value; None when it has no manifest."""
    path = root / "_suites" / f"{suite}.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    points = []
    for point in manifest["points"]:
        run_dir = root / point["run_dir"] if point.get("run_dir") else None
        if point["status"] not in USABLE_STATUSES or run_dir is None:
            continue
        if not (run_dir / "summary.json").is_file():
            continue
        points.append(
            SuitePoint(label=point["label"], value=float(point["value"]), run=Run(run_dir))
        )
    return SuiteRuns(
        suite=manifest["suite"],
        suite_file=manifest.get("suite_file") or suite_file_for(suite),
        axis=manifest["axis"],
        points=tuple(sorted(points, key=lambda p: (p.value, p.label))),
    )


def sweep_table(suite: SuiteRuns, metrics: Sequence[str]) -> pd.DataFrame:
    """Long table: one row per (point, metric) with across-rep `mean`, CI `lo`/`hi` and `n`.
    `metrics` are `Run.rep_metric` paths such as `latency.ttft_s.p99`."""
    rows = []
    for point in suite.points:
        for metric in metrics:
            est = mean_ci(point.run.rep_metric(metric))
            rows.append(
                {
                    "label": point.label,
                    "value": point.value,
                    "metric": metric,
                    "mean": est.mean,
                    "lo": est.lo,
                    "hi": est.hi,
                    "n": est.n,
                }
            )
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))
