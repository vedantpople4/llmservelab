"""Capacity: the open-loop arrival rate that defines ρ = 1 for every later phase (plan Phase 3).

Rule, fixed before any data exists (plan §5):

- An offered rate λ is **sustained** iff both hold, with each quantity the across-rep mean:
  1. the completion ratio is ≥ 0.95: the achieved completion rate (completions inside the
     steady window ÷ window length, `throughput.req_s`) over the realized arrival rate (arrivals
     inside the same window ÷ its length). Dividing by realized rather than nominal λ keeps
     Poisson arrival noise from reading as lost capacity; λ itself stays the axis value; and
  2. queueing is bounded: TTFT P50 of the ok requests arriving in the last third of the steady
     window is ≤ 2 × TTFT P50 of those arriving in the first third. A third with no ok request
     counts as unbounded growth.
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

MIN_COMPLETION_RATIO = 0.95
MAX_TTFT_GROWTH = 2.0


@dataclass(frozen=True)
class RatePoint:
    label: str
    offered_rps: float  # nominal λ, the suite axis value
    arrived_rps: float  # realized arrival rate inside the steady window
    achieved_rps: float
    ttft_growth: float  # last-third ÷ first-third TTFT P50; inf when unbounded
    reps: int


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
    ok = ratio >= MIN_COMPLETION_RATIO and point.ttft_growth <= MAX_TTFT_GROWTH
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


def capacity_points(suite: SuiteRuns) -> list[RatePoint]:
    """One `RatePoint` per suite point; the axis value is the offered rate."""
    out = []
    for point in suite.points:
        achieved = point.run.rep_metric("throughput.req_s")
        arrived = [
            s["n_measured"] / s["window"]["length_s"]
            for s in point.run.summary["per_rep"]
            if s["window"]["length_s"]
        ]
        growth = ttft_growth(point.run)
        out.append(
            RatePoint(
                label=point.label,
                offered_rps=point.value,
                arrived_rps=float(np.mean(arrived)) if arrived else 0.0,
                achieved_rps=float(np.mean(achieved)) if achieved else 0.0,
                ttft_growth=float(np.mean(growth)) if growth else math.inf,
                reps=len(achieved),
            )
        )
    return out


def ttft_growth(run: Run) -> list[float]:
    """Per rep: TTFT P50 of the last third of the steady window ÷ that of the first third."""
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
        first = ok[ok["t_arrival"] < start + third]["ttft_s"].dropna()
        last = ok[ok["t_arrival"] >= end - third]["ttft_s"].dropna()
        out.append(_growth(first.to_numpy(np.float64), last.to_numpy(np.float64)))
    return out


def _growth(first: npt.NDArray[np.float64], last: npt.NDArray[np.float64]) -> float:
    if len(first) == 0 or len(last) == 0:
        return math.inf
    p_first, p_last = float(np.median(first)), float(np.median(last))
    if p_first == 0:
        return 1.0 if p_last == 0 else math.inf
    return p_last / p_first
