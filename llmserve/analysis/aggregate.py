"""Per-repetition and cross-repetition statistics over the steady window (plan Phase 2 item 9).

Steady-state rules are fixed before any data exists (plan §5), so results cannot be cherry-picked:

- **closed loop:** measured = requests dispatched while the client-side in-flight count equals
  the maximum it reaches. The ramp-up (first dispatches, fewer than C in flight) and the drain
  tail (dispatches after workers start going idle) fall outside the window. In-flight is
  recomputed from dispatch/complete timestamps rather than read from the gateway, so the rule
  holds with the gateway on or off.
- **open loop:** measured = requests scheduled to arrive in
  `[warmup_s, duration_s − cooldown_s)`; every request is still sent — the trim only affects
  statistics, so the system stays loaded through warm-up and cooldown. `cooldown_s` needs
  `load.duration_s` to know where the run ends; with a requests-capped run it is ignored.
- **throughput/goodput:** count *completions* whose timestamps fall inside the window, divided by
  the window length — a rate over time, not over the measured population.

Failures are never dropped: every summary carries total counts, a failure breakdown, the success
rate, and an `invalid` flag when more than 1% of requests failed (plan §5 rule 7). An empty
steady set (warm-up longer than the run, fewer requests than workers) falls back to measuring
everything, recorded as `window.mode == "all"` so the fallback stays visible.

Cross-rep blocks are unweighted means of the per-rep statistics (`n` says how many reps
contributed); pooled percentiles over mixed windows would not be meaningful.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llmserve.config.schema import ExperimentConfig
from llmserve.metrics import latency
from llmserve.metrics.records import RequestRecord

_NS = 1e9
INVALID_FAILURE_RATE = 0.01

_LATENCY_METRICS = ("ttft_s", "tpot_s", "itl_s", "e2e_s", "queue_wait_s", "client_lag_s")
_ACROSS_STATS = ("mean", "p50", "p95", "p99")
_THROUGHPUT_KEYS = ("req_s", "prompt_toks_s", "gen_toks_s", "total_toks_s")


@dataclass(frozen=True)
class Window:
    """The steady window of one repetition: which requests are measured, over what time."""

    mode: str  # "closed" | "open" | "all"
    start_ns: int | None
    end_ns: int | None
    measured: list[RequestRecord]

    @property
    def length_s(self) -> float | None:
        if self.start_ns is None or self.end_ns is None:
            return None
        return max(0.0, (self.end_ns - self.start_ns) / _NS)


def window_of(
    records: list[RequestRecord],
    cfg: ExperimentConfig,
    *,
    t0_ns: int,
) -> Window:
    """Pick the steady window for one repetition of `cfg` (module docstring has the rules).

    `t0_ns` is the clock value the repetition started at; open-loop arrival offsets are
    relative to it.
    """
    if cfg.load.mode == "closed":
        window = _closed_window(records)
    else:
        window = _open_window(records, cfg, t0_ns=t0_ns)
    if window.measured or not records:
        return window
    return _all_window(records)


def summarize(records: list[RequestRecord], *, rep: int, window: Window) -> dict[str, Any]:
    """One repetition's summary: counts, failure table, latency blocks, throughput, goodput."""
    n = len(records)
    ok = [r for r in records if r.ok]
    failed = n - len(ok)
    by_status: dict[str, int] = {}
    for r in records:
        if not r.ok:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1

    measured_ok = [r for r in window.measured if r.ok]
    length_s = window.length_s
    start_ns, end_ns = window.start_ns, window.end_ns

    throughput: dict[str, Any] | None = None
    if start_ns is not None and end_ns is not None and length_s is not None and length_s > 0:
        completed = [
            r for r in ok if r.t_complete is not None and start_ns <= r.t_complete <= end_ns
        ]
        prompt_toks = sum(r.prompt_tokens_usage or 0 for r in completed)
        gen_toks = sum(r.output_tokens_usage or 0 for r in completed)
        throughput = {
            "n_completed": len(completed),
            "req_s": len(completed) / length_s,
            "prompt_toks_s": prompt_toks / length_s,
            "gen_toks_s": gen_toks / length_s,
            "total_toks_s": (prompt_toks + gen_toks) / length_s,
        }

    slo_pool = [
        r for r in window.measured if r.slo_ttft_ms is not None and r.slo_tpot_ms is not None
    ]
    goodput_s: float | None = None
    n_slo_pass = 0
    if slo_pool and length_s is not None and length_s > 0:
        for r in slo_pool:
            slo_ttft, slo_tpot = r.slo_ttft_ms, r.slo_tpot_ms
            if not r.ok or slo_ttft is None or slo_tpot is None:
                continue
            t_ttft, t_tpot = latency.ttft(r), latency.tpot(r)
            if t_ttft is None or t_ttft > slo_ttft / 1e3:
                continue
            # A single-token output has no observable TPOT; it can only be judged on TTFT.
            if t_tpot is not None and t_tpot > slo_tpot / 1e3:
                continue
            n_slo_pass += 1
        goodput_s = n_slo_pass / length_s

    return {
        "rep": rep,
        "n_requests": n,
        "n_ok": len(ok),
        "n_failed": failed,
        "success_rate": len(ok) / n if n else None,
        "failures_by_status": dict(sorted(by_status.items())),
        "invalid": bool(n) and failed / n > INVALID_FAILURE_RATE,
        "window": {
            "mode": window.mode,
            "start_s": start_ns / _NS if start_ns is not None else None,
            "end_s": end_ns / _NS if end_ns is not None else None,
            "length_s": length_s,
        },
        "n_measured": len(window.measured),
        "n_measured_ok": len(measured_ok),
        "throughput": throughput,
        "n_slo": len(slo_pool),
        "n_slo_pass": n_slo_pass,
        "goodput_s": goodput_s,
        "latency": {
            "ttft_s": _stats(_values(measured_ok, latency.ttft)),
            "tpot_s": _stats(_values(measured_ok, latency.tpot)),
            "itl_s": _stats([gap for r in measured_ok for gap in latency.inter_token_latencies(r)]),
            "e2e_s": _stats(_values(measured_ok, latency.e2e)),
            "queue_wait_s": _stats(_values(measured_ok, latency.queue_wait)),
            "client_lag_s": _stats(
                [r.client_lag_ns / _NS for r in window.measured if r.client_lag_ns is not None]
            ),
        },
    }


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-repetition aggregation: across-rep mean/std/min/max of each per-rep statistic."""
    return {
        "repetitions": len(summaries),
        "invalid": any(s["invalid"] for s in summaries),
        "invalid_reps": [s["rep"] for s in summaries if s["invalid"]],
        "success_rate": _across([s["success_rate"] for s in summaries]),
        "throughput": {
            key: _across([_throughput(s, key) for s in summaries]) for key in _THROUGHPUT_KEYS
        },
        "goodput_s": _across([s["goodput_s"] for s in summaries]),
        "latency": {
            metric: {
                stat: _across([s["latency"][metric][stat] for s in summaries])
                for stat in _ACROSS_STATS
            }
            for metric in _LATENCY_METRICS
        },
        "per_rep": summaries,
    }


def _closed_window(records: list[RequestRecord]) -> Window:
    """Sweep dispatch/complete events; measured = dispatched at peak in-flight (plan §5 rule 3)."""
    events: list[tuple[int, int, RequestRecord]] = []
    for r in records:
        if r.t_dispatch is not None and r.t_complete is not None:
            events.append((r.t_dispatch, 1, r))
            events.append((r.t_complete, -1, r))
    # At equal timestamps a completion frees its slot before the next dispatch claims it.
    events.sort(key=lambda e: (e[0], 0 if e[1] < 0 else 1))

    in_flight = 0
    at_dispatch: dict[str, int] = {}
    peak = 0
    for _, delta, rec in events:
        in_flight += delta
        if delta > 0:
            at_dispatch[rec.request_id] = in_flight
            peak = max(peak, in_flight)

    if peak == 0:
        return Window("closed", None, None, [])
    measured = [r for r in records if at_dispatch.get(r.request_id, 0) == peak]
    start = min(r.t_dispatch for r in measured if r.t_dispatch is not None)
    ends = [r.t_complete for r in measured if r.ok and r.t_complete is not None]
    if not ends:
        ends = [r.t_complete for r in measured if r.t_complete is not None]
    end = max(ends) if ends else None
    return Window("closed", start, end, _in_order(records, measured))


def _open_window(records: list[RequestRecord], cfg: ExperimentConfig, *, t0_ns: int) -> Window:
    start = t0_ns + int(cfg.measurement.warmup_s * _NS)
    end: int | None
    if cfg.load.duration_s is not None:
        end = t0_ns + int((cfg.load.duration_s - cfg.measurement.cooldown_s) * _NS)
    else:
        # No duration_s → the run ends at its last arrival; +1 ns keeps that arrival inside
        # the exclusive bound (cooldown_s has nothing to trim against and is ignored).
        last = max((r.t_arrival for r in records), default=None)
        end = None if last is None else last + 1
    measured = [r for r in records if r.t_arrival >= start and (end is None or r.t_arrival < end)]
    if not measured:
        return Window("open", start, end, [])
    return Window("open", start, end, _in_order(records, measured))


def _all_window(records: list[RequestRecord]) -> Window:
    starts = [r.t_arrival for r in records]
    ends = [r.t_complete for r in records if r.t_complete is not None]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    return Window("all", start, end, list(records))


def _in_order(records: list[RequestRecord], subset: list[RequestRecord]) -> list[RequestRecord]:
    keep = {r.request_id for r in subset}
    return [r for r in records if r.request_id in keep]


def _values(
    records: list[RequestRecord], metric: Callable[[RequestRecord], float | None]
) -> list[float]:
    return [v for r in records if (v := metric(r)) is not None]


def _stats(values: list[float]) -> dict[str, Any]:
    """count/mean/median/std/P50/P95/P99; every field present, None when count is 0."""
    keys = ("count", "mean", "median", "std", "p50", "p95", "p99")
    if not values:
        return dict(zip(keys, (0, *([None] * 6)), strict=True))
    s = sorted(values)
    return {
        "count": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "std": statistics.stdev(s) if len(s) > 1 else 0.0,
        "p50": _pct(s, 0.50),
        "p95": _pct(s, 0.95),
        "p99": _pct(s, 0.99),
    }


def _pct(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list (numpy's default method)."""
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _throughput(summary: dict[str, Any], key: str) -> float | None:
    t = summary["throughput"]
    return None if t is None else t[key]


def _across(values: list[Any]) -> dict[str, Any]:
    """Across-rep mean/std/min/max of per-rep values; None values (failed reps) drop out."""
    kept = [float(v) for v in values if v is not None]
    if not kept:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(kept),
        "mean": statistics.fmean(kept),
        "std": statistics.stdev(kept) if len(kept) > 1 else None,
        "min": min(kept),
        "max": max(kept),
    }
