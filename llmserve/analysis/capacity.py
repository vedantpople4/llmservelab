"""Capacity: the open-loop arrival rate that defines ρ = 1 for every later phase (plan Phase 3).

Rule, fixed before any data exists (plan §5) and amended once in item 2:

- An offered rate λ is **sustained** iff both hold, with each quantity the across-rep mean:
  1. the completion ratio is ≥ 0.95: the achieved completion rate (completions inside the
     steady window ÷ window length, `throughput.req_s`) over the realized arrival rate (arrivals
     inside the same window ÷ its length). Dividing by realized rather than nominal λ keeps
     Poisson arrival noise from reading as lost capacity; λ itself stays the axis value; and
  2. queueing is bounded: slowdown P50 of the ok requests arriving in the last third of the
     steady window is ≤ 2 × slowdown P50 of those arriving in the first third, where a request's
     slowdown is its TTFT over the unloaded TTFT the H3 prefill fit (`PrefillFit.predict`)
     predicts for its prompt length. A third with no ok request counts as unbounded growth.
     When no prefill fit exists (prompt-length suite not run), raw TTFT stands in for slowdown
     and the capacity note says so.

  *Amended October 2026.* The pre-registered rule compared raw TTFT P50. With a heterogeneous
  prompt mix at low rates, a third holds ~10 requests and its raw median moves with whether an
  8K-token prompt lands in it: session 1's mixed sweep read growth 3.08 at 0.05 req/s while
  completions kept up. Slowdown removes the prompt-length term. For a homogeneous workload
  slowdown growth equals raw TTFT growth, so those verdicts are unchanged.

- **Capacity** (ρ = 1) is the highest sustained rate below the lowest unsustained rate. Points
  sustained *above* an unsustained one are noise and are reported as `monotone = False`, never
  promoted to capacity.
- The sweep **brackets** capacity iff a sustained rate and a higher unsustained rate both exist.
  Otherwise `rate_rps` is None and the note says which side of the sweep capacity lies on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from llmserve.analysis.runs import Run
from llmserve.analysis.suites import SuiteRuns
from llmserve.scheduler.estimators import PrefillFit

MIN_COMPLETION_RATIO = 0.95
MAX_TTFT_GROWTH = 2.0


@dataclass(frozen=True)
class RatePoint:
    label: str
    offered_rps: float  # nominal λ, the suite axis value
    arrived_rps: float  # realized arrival rate inside the steady window
    achieved_rps: float
    ttft_growth: float  # last-third ÷ first-third TTFT P50; inf when unbounded
    slowdown_growth: float | None  # the same on slowdown; None without a prefill fit
    reps: int

    @property
    def growth(self) -> float:
        """The growth the sustained rule judges: slowdown, else raw TTFT."""
        return self.ttft_growth if self.slowdown_growth is None else self.slowdown_growth


@dataclass(frozen=True)
class CapacityEvidence:
    point: RatePoint
    completion_ratio: float
    sustained: bool


@dataclass(frozen=True)
class Capacity:
    rate_rps: float | None
    bracketed: bool
    lower_bound_rps: float | None  # highest sustained rate below the first unsustained one
    upper_bound_rps: float | None  # lowest unsustained rate
    monotone: bool
    evidence: tuple[CapacityEvidence, ...]
    note: str


def is_sustained(point: RatePoint) -> tuple[float, bool]:
    ratio = point.achieved_rps / point.arrived_rps if point.arrived_rps > 0 else math.nan
    ok = ratio >= MIN_COMPLETION_RATIO and point.growth <= MAX_TTFT_GROWTH
    return ratio, bool(ok)


def find_capacity(points: Sequence[RatePoint]) -> Capacity:
    evidence = []
    for point in sorted(points, key=lambda p: p.offered_rps):
        ratio, sustained = is_sustained(point)
        evidence.append(CapacityEvidence(point=point, completion_ratio=ratio, sustained=sustained))
    unsustained = [e.point.offered_rps for e in evidence if not e.sustained]
    upper = min(unsustained) if unsustained else None
    below = [
        e.point.offered_rps
        for e in evidence
        if e.sustained and (upper is None or e.point.offered_rps < upper)
    ]
    lower = max(below) if below else None
    monotone = upper is None or not any(
        e.sustained and e.point.offered_rps > upper for e in evidence
    )
    bracketed = lower is not None and upper is not None
    if not evidence:
        note = "no usable points"
    elif bracketed:
        note = f"capacity {lower:g} req/s (unsustained at {upper:g} req/s)"
    elif upper is None:
        note = f"every rate sustained: capacity ≥ {lower:g} req/s; extend the sweep upward"
    else:
        note = f"no rate sustained: capacity < {upper:g} req/s; extend the sweep downward"
    if any(e.point.slowdown_growth is None for e in evidence):
        note += "; growth on raw TTFT (no prefill fit: prompt-length suite not run)"
    if not monotone:
        note += "; non-monotone (a rate above the first unsustained one was sustained)"
    return Capacity(
        rate_rps=lower if bracketed else None,
        bracketed=bracketed,
        lower_bound_rps=lower,
        upper_bound_rps=upper,
        monotone=monotone,
        evidence=tuple(evidence),
        note=note,
    )


def capacity_points(suite: SuiteRuns, prefill: PrefillFit | None) -> list[RatePoint]:
    """One `RatePoint` per suite point; the axis value is the offered rate. `prefill` is the H3
    fit that predicts unloaded TTFT for slowdown; None falls back to raw TTFT growth."""
    out = []
    for point in suite.points:
        if point.value is None:
            continue
        achieved = point.run.rep_metric("throughput.req_s")
        arrived = [
            s["n_measured"] / s["window"]["length_s"]
            for s in point.run.summary["per_rep"]
            if s["window"]["length_s"]
        ]
        growth = ttft_growth(point.run)
        slowdown = ttft_growth(point.run, prefill) if prefill is not None else None
        out.append(
            RatePoint(
                label=point.label,
                offered_rps=point.value,
                arrived_rps=float(np.mean(arrived)) if arrived else 0.0,
                achieved_rps=float(np.mean(achieved)) if achieved else 0.0,
                ttft_growth=_mean_growth(growth),
                slowdown_growth=_mean_growth(slowdown) if slowdown is not None else None,
                reps=len(achieved),
            )
        )
    return out


def ttft_growth(run: Run, prefill: PrefillFit | None = None) -> list[float]:
    """Per rep: TTFT P50 of the last third of the steady window ÷ that of the first third. With
    `prefill`, each TTFT is first divided by its predicted unloaded TTFT (slowdown)."""
    frame = run.requests()
    out = []
    for rep_summary in run.summary["per_rep"]:
        rep = rep_summary["rep"]
        reqs = frame[(frame["rep"] == rep) & frame["in_window"]]
        if reqs.empty:
            out.append(math.inf)
            continue
        window = rep_summary["window"]
        start = window["start_s"] * 1e9 if window["start_s"] is not None else None
        end = window["end_s"] * 1e9 if window["end_s"] is not None else None
        if start is None or end is None or end <= start:
            start, end = float(reqs["t_arrival"].min()), float(reqs["t_arrival"].max())
        third = (end - start) / 3
        ok = reqs[reqs["ok"]]
        value = ok["ttft_s"]
        if prefill is not None:
            value = value / ok["prompt_tokens_req"].astype(float).map(prefill.predict)
        first = value[ok["t_arrival"] < start + third].dropna()
        last = value[ok["t_arrival"] >= end - third].dropna()
        out.append(_growth(first.to_numpy(np.float64), last.to_numpy(np.float64)))
    return out


def _mean_growth(growth: list[float]) -> float:
    return float(np.mean(growth)) if growth else math.inf


def _growth(first: npt.NDArray[np.float64], last: npt.NDArray[np.float64]) -> float:
    if len(first) == 0 or len(last) == 0:
        return math.inf
    p_first, p_last = float(np.median(first)), float(np.median(last))
    if p_first == 0:
        return 1.0 if p_last == 0 else math.inf
    return p_last / p_first
