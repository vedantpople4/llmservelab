"""Read finished runs back from disk (Phase 3): the input to every plot, fit and report.

A `Run` is one run directory written by `runner/experiment.py` (layout in `runner/storage.py`).
Requests are rebuilt as `RequestRecord`s and pushed through `aggregate.window_of`, so analysis
uses exactly the steady window the runner summarized — one definition of "measured".

Reportability is decided here (ADR-009, ADR-0013): only `server.kind == "vllm-metal"` runs
from a clean tree, with a valid summary, no thermal drift, no system sleep and on AC power, may
appear in a report. Everything else is a development result.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from llmserve.analysis.aggregate import window_of
from llmserve.config.loader import load_config
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics import latency
from llmserve.metrics.records import Chunk, RequestRecord, RequestStatus
from llmserve.runner.macos import BATTERY_POWER

MEASUREMENT_KIND = "vllm-metal"
MAX_THERMAL_DRIFT = 0.10
MAX_SLEPT_S = 1.0
_NS = 1e9
_RECORD_FIELDS = {f.name for f in fields(RequestRecord)} - {"status", "chunks", "extra"}


@dataclass(frozen=True)
class Run:
    path: Path

    @cached_property
    def config(self) -> ExperimentConfig:
        return load_config(self.path / "config.resolved.yaml")

    @cached_property
    def metadata(self) -> dict[str, Any]:
        return _read_json(self.path / "metadata.json")

    @cached_property
    def summary(self) -> dict[str, Any]:
        """The cross-rep `summary.json`; `summary["per_rep"]` holds each repetition's block."""
        return _read_json(self.path / "summary.json")

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def experiment(self) -> str:
        return self.config.experiment

    @property
    def unreportable_reasons(self) -> list[str]:
        """Why this run may not appear in a report; empty exactly when it may."""
        reasons = []
        if self.config.server.kind != MEASUREMENT_KIND:
            reasons.append(f"server.kind is {self.config.server.kind}, not {MEASUREMENT_KIND}")
        if self.metadata.get("git_dirty") is not False:
            reasons.append("git tree dirty or unknown")
        if self.summary["invalid"]:
            reasons.append("summary invalid (> 1% failed requests)")
        return reasons + self.machine_reasons

    @property
    def machine_reasons(self) -> list[str]:
        """The unreportable reasons that say the Mac, not the run, was off; a suite reruns such
        a run instead of skipping it."""
        reasons = []
        drift = self.thermal_drift
        if drift is not None and abs(drift) > MAX_THERMAL_DRIFT:
            reasons.append(f"thermal drift {drift:+.0%} (probe after vs before, TPOT p50)")
        # Runs from before these fields existed carry no evidence either way.
        slept = (self.metadata.get("sleep") or {}).get("total_s")
        if slept is not None and slept > MAX_SLEPT_S:
            reasons.append(f"Mac slept {slept:.0f} s during the run")
        power = self.metadata.get("power_source") or {}
        if BATTERY_POWER in (power.get("start"), power.get("end")):
            reasons.append("ran on battery power")
        return reasons

    @property
    def reportable(self) -> bool:
        return not self.unreportable_reasons

    @property
    def thermal_drift(self) -> float | None:
        """The thermal probe's TPOT p50 after the reps over its p50 before them, minus 1. Either
        sign means the machine changed under the run: positive is the fanless Mac slowing as it
        heats, negative is a run that started on a machine already slowed (session 2's h2-mixed
        began at 80 ms against a 53 ms norm). None for runs without probes.

        The probe is a fixed batch; the reps are not, since each materializes its own workload
        sample, so a rep-to-rep TPOT change can be the workload rather than the machine."""
        probe = self.metadata.get("thermal_probe") or {}
        before = (probe.get("before") or {}).get("tpot_p50_s")
        after = (probe.get("after") or {}).get("tpot_p50_s")
        if not before or after is None:
            return None
        return float(after) / float(before) - 1

    def rep_dirs(self) -> list[Path]:
        return sorted(p for p in self.path.glob("rep-*") if p.is_dir())

    def records(self, rep: int) -> list[RequestRecord]:
        """Repetition `rep`'s requests, in stored order, with their chunks."""
        rep_path = self.path / f"rep-{rep:02d}"
        chunks: dict[str, list[Chunk]] = {}
        chunk_file = rep_path / "chunks.parquet"
        if chunk_file.exists():
            rows = pq.read_table(chunk_file).to_pylist()
            for row in sorted(rows, key=lambda r: (r["request_id"], r["idx"])):
                chunks.setdefault(row["request_id"], []).append(
                    Chunk(t_ns=row["t_ns"], n_tokens=row["n_tokens"])
                )
        out: list[RequestRecord] = []
        for row in pq.read_table(rep_path / "requests.parquet").to_pylist():
            kwargs = {k: v for k, v in row.items() if k in _RECORD_FIELDS}
            out.append(
                RequestRecord(
                    **kwargs,
                    status=RequestStatus(row["status"]),
                    chunks=tuple(chunks.get(row["request_id"], ())),
                )
            )
        return out

    def measured(self, rep: int) -> list[RequestRecord]:
        """The steady-window requests of `rep`, as the runner's summary counted them."""
        records = self.records(rep)
        return window_of(records, self.config, t0_ns=self._t0_ns(rep)).measured

    def requests(self) -> pd.DataFrame:
        """Every request of every rep: stored columns plus `rep`, `ok`, `in_window` and the
        latency metrics in seconds (`ttft_s`, `tpot_s`, `e2e_s`, `queue_wait_s`; NaN where the
        metric is undefined, e.g. failed requests)."""
        frames = []
        for rep, rep_path in self._reps():
            records = self.records(rep)
            window = window_of(records, self.config, t0_ns=self._t0_ns(rep))
            measured = {r.request_id for r in window.measured}
            frame = pq.read_table(rep_path / "requests.parquet").to_pandas()
            frame["rep"] = rep
            frame["ok"] = [r.ok for r in records]
            frame["in_window"] = [r.request_id in measured for r in records]
            for name, metric in (
                ("ttft_s", latency.ttft),
                ("tpot_s", latency.tpot),
                ("e2e_s", latency.e2e),
                ("queue_wait_s", latency.queue_wait),
            ):
                frame[name] = [_nan(metric(r)) for r in records]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def server(self) -> pd.DataFrame:
        """`/metrics` samples of every rep with `rep` and `t_s` (seconds on the run clock);
        empty when the server sampler never ran."""
        frames = []
        for rep, rep_path in self._reps():
            path = rep_path / "server.parquet"
            if path.exists():
                frame = pq.read_table(path).to_pandas()
                frame["rep"] = rep
                frame["t_s"] = frame["t_ns"] / _NS
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def rep_metric(self, dotted: str) -> list[float]:
        """One value per valid repetition, e.g. `rep_metric("latency.ttft_s.p95")` or
        `rep_metric("throughput.gen_toks_s")`; reps where the value is missing drop out."""
        values = []
        for rep_summary in self.summary["per_rep"]:
            node: Any = rep_summary
            for key in dotted.split("."):
                node = node.get(key) if isinstance(node, dict) else None
            if node is not None:
                values.append(float(node))
        return values

    def _reps(self) -> Iterator[tuple[int, Path]]:
        for rep_path in self.rep_dirs():
            yield int(rep_path.name.removeprefix("rep-")), rep_path

    def _t0_ns(self, rep: int) -> int:
        """Open loop: the rep's start on the run clock, recovered from its summary window
        (`start = t0 + warmup_s`). Closed-loop windows do not use it."""
        if self.config.load.mode != "open":
            return 0
        window = self.summary["per_rep"][rep]["window"]
        if window["mode"] != "open" or window["start_s"] is None:
            return 0
        start_ns = int(round(window["start_s"] * _NS))
        return start_ns - round(self.config.measurement.warmup_s * _NS)


def find_runs(root: Path, experiment: str | None = None) -> list[Run]:
    """Every run under `results/` (or one experiment's runs), oldest first."""
    pattern = f"{experiment}/*/metadata.json" if experiment else "*/*/metadata.json"
    return [Run(p.parent) for p in sorted(root.glob(pattern), key=lambda p: p.parent.name)]


def latest_run(root: Path, experiment: str, *, valid_only: bool = True) -> Run | None:
    """The newest run of `experiment`; with `valid_only`, the newest whose summary is valid."""
    for run in reversed(find_runs(root, experiment)):
        if not valid_only or not run.summary["invalid"]:
            return run
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _nan(value: float | None) -> float:
    return float("nan") if value is None else value
