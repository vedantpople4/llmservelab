"""Open-loop arrival processes (PRD §9, ADR-003).

`generate_arrivals(load, rng)` returns the arrival offsets (seconds from run start,
non-decreasing) for an open-loop load. A rate may be given directly or as `rho`
(utilization relative to `load.capacity_rps`, measured in Phase 3: rate = rho × capacity).

Process conventions:

- `constant`: the first arrival is at t = 0, gaps are exactly 1/rate.
- `poisson`: homogeneous exponential gaps; the first arrival is after one gap.
- `bursty`: each phase is an independent homogeneous Poisson process on its window; the
  phases must cover `duration_s`, because the run would otherwise idle past the last phase.
- `replay`: offsets read from a text file (one non-decreasing offset per line, `#` comments),
  divided by `time_scale`. `time_scale` is a speed-up factor: 2.0 arrives twice as fast and
  doubles rho.

`load.requests` caps the count and `load.duration_s` bounds the window; whichever cuts first
wins, and a config that cannot deliver `requests` arrivals inside the window is an error, not
an under-filled workload.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from llmserve.config.schema import Bursty, Constant, Load, Poisson, Replay


def _rate(arrival: Constant | Poisson, capacity_rps: float | None) -> float:
    if arrival.rate is not None:
        return arrival.rate
    if capacity_rps is None:  # the schema rejects rho without capacity_rps; belt and braces
        raise ValueError("arrival.rho requires load.capacity_rps (measured in Phase 3)")
    if arrival.rho is None:  # pragma: no cover - schema guarantees exactly one of rate/rho
        raise ValueError("set exactly one of arrival.rate or arrival.rho")
    return arrival.rho * capacity_rps


def _poisson(
    rate: float, rng: Generator, *, want: int | None, horizon: float | None
) -> npt.NDArray[np.float64]:
    """Exponential gaps, stopping at `want` arrivals and/or the `horizon`."""
    if horizon is None:
        if want is None:  # pragma: no cover - the schema requires requests and/or duration
            raise ValueError("open-loop load needs `requests`, `duration_s`, or both")
        return np.cumsum(rng.exponential(1.0 / rate, size=want))
    chunks: list[npt.NDArray[np.float64]] = []
    total, t0 = 0, 0.0
    while True:
        size = max(64, int(horizon * rate * 0.25))
        c = t0 + np.cumsum(rng.exponential(1.0 / rate, size=size))
        keep = c[c < horizon]
        chunks.append(keep)
        total += keep.size
        if keep.size < c.size:  # crossed the horizon; cumsum → ∞ so this happens a.s.
            break
        t0 = float(c[-1])
        if want is not None and total >= want:
            break
    offsets = np.concatenate(chunks)
    return offsets[:want] if want is not None else offsets


def _constant(rate: float, *, want: int | None, horizon: float | None) -> npt.NDArray[np.float64]:
    if want is not None:
        return np.arange(want, dtype=np.float64) / rate
    assert horizon is not None  # schema requires requests and/or duration
    n = int(np.ceil(horizon * rate)) + 1  # the unified horizon cut drops the one at/after it
    return np.arange(n, dtype=np.float64) / rate


def _bursty(arrival: Bursty, rng: Generator) -> npt.NDArray[np.float64]:
    chunks: list[npt.NDArray[np.float64]] = []
    phase_start = 0.0
    for phase in arrival.phases:
        duration, rate = phase.duration_s, phase.rate
        local: list[npt.NDArray[np.float64]] = []
        t_local = 0.0
        while True:
            size = max(16, int(duration * rate * 0.25) + 4)
            c = t_local + np.cumsum(rng.exponential(1.0 / rate, size=size))
            keep = c[c < duration]
            local.append(keep)
            if keep.size < c.size:
                break
            t_local = float(c[-1])
        chunks.append(phase_start + np.concatenate(local))
        phase_start += duration
    return np.concatenate(chunks)


def _replay(arrival: Replay) -> npt.NDArray[np.float64]:
    try:
        raw = np.atleast_1d(np.loadtxt(arrival.trace))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read trace {arrival.trace!r}: {e}") from e
    if raw.size == 0:
        raise ValueError(f"trace {arrival.trace!r} is empty")
    offsets = raw.astype(np.float64)
    if not np.all(np.isfinite(offsets)) or np.any(offsets < 0):
        raise ValueError(f"trace {arrival.trace!r}: offsets must be finite and >= 0")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"trace {arrival.trace!r}: offsets must be non-decreasing")
    return offsets / arrival.time_scale


def generate_arrivals(load: Load, rng: Generator) -> npt.NDArray[np.float64]:
    """Arrival offsets (seconds, non-decreasing) for an open-loop `load`.

    Uses `rng` only for the stochastic processes; the same seed replays the same arrivals.
    """
    arrival = load.arrival
    if arrival is None:
        raise ValueError("closed-loop loads have no arrival offsets (mode: closed)")
    want, horizon = load.requests, load.duration_s

    if isinstance(arrival, Replay):
        offsets = _replay(arrival)
        source = f"trace {arrival.trace!r}"
    elif isinstance(arrival, Bursty):
        offsets = _bursty(arrival, rng)
        window = float(sum(p.duration_s for p in arrival.phases))
        if horizon is not None and horizon > window + 1e-9:
            raise ValueError(
                f"bursty phases cover {window:g}s but duration_s is {horizon:g}s; "
                "extend the phases so the run does not idle"
            )
        source = "bursty phases"
    elif isinstance(arrival, Constant):
        rate = _rate(arrival, load.capacity_rps)
        offsets = _constant(rate, want=want, horizon=horizon)
        source = f"constant rate {rate:g} req/s"
    else:
        rate = _rate(arrival, load.capacity_rps)
        offsets = _poisson(rate, rng, want=want, horizon=horizon)
        source = f"poisson rate {rate:g} req/s"

    if horizon is not None:
        offsets = offsets[offsets < horizon]
    if want is not None:
        if offsets.size < want:
            scope = f"within duration_s={horizon:g}" if horizon is not None else "before it ends"
            raise ValueError(f"{source}: only {offsets.size} of {want} requests arrive {scope}")
        offsets = offsets[:want]
    if offsets.size == 0:
        raise ValueError("no arrivals: check duration_s, the rates, and the trace window")
    return offsets
