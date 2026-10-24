"""Capacity (ρ = 1): the sustained rule, bracketing, and the end-to-end evidence from a run."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
from conftest import RunFactory, experiment_config
from test_analysis_suites import write_manifest

from llmserve.analysis.capacity import RatePoint, capacity_points, find_capacity, ttft_growth
from llmserve.analysis.runs import Run
from llmserve.analysis.suites import load_suite_runs
from llmserve.mock import DelayModel, MockEngine
from llmserve.scheduler.estimators import PrefillFit

PREFILL = PrefillFit(a=0.05, b=2e-4, c=1e-8, r2=1.0, n=100)


def point(
    rate: float, ratio: float = 1.0, growth: float = 1.0, slowdown: float | None = None
) -> RatePoint:
    return RatePoint(
        label=f"r{rate:g}",
        offered_rps=rate,
        arrived_rps=rate,
        achieved_rps=rate * ratio,
        ttft_growth=growth,
        slowdown_growth=growth if slowdown is None else slowdown,
        reps=5,
    )


def test_bracketed_capacity_is_the_last_sustained_rate() -> None:
    cap = find_capacity([point(4, ratio=0.7), point(1), point(2)])
    assert cap.bracketed and cap.monotone
    assert cap.rate_rps == 2
    assert (cap.lower_bound_rps, cap.upper_bound_rps) == (2, 4)
    assert [e.point.offered_rps for e in cap.evidence] == [1, 2, 4]
    assert [e.sustained for e in cap.evidence] == [True, True, False]


def test_unbounded_queueing_is_unsustained_even_when_completions_keep_up() -> None:
    cap = find_capacity([point(1), point(2, growth=math.inf), point(3, growth=2.5)])
    assert cap.rate_rps == 1
    assert [e.sustained for e in cap.evidence] == [True, False, False]


def test_thresholds_are_inclusive_at_the_boundary() -> None:
    cap = find_capacity([point(1, ratio=0.95, growth=2.0), point(2, ratio=0.9499)])
    assert [e.sustained for e in cap.evidence] == [True, False]


def test_completion_ratio_uses_realized_not_nominal_arrivals() -> None:
    # Poisson noise: 10 req/s nominal, 8.5 arrived, all 8.5 completed. Not a capacity loss.
    noisy = RatePoint(
        "r10",
        offered_rps=10,
        arrived_rps=8.5,
        achieved_rps=8.5,
        ttft_growth=1,
        slowdown_growth=1,
        reps=1,
    )
    (e,) = find_capacity([noisy]).evidence
    assert e.completion_ratio == 1.0 and e.sustained


def test_all_sustained_is_not_bracketed_and_gives_no_rate() -> None:
    cap = find_capacity([point(1), point(2)])
    assert not cap.bracketed and cap.rate_rps is None
    assert cap.lower_bound_rps == 2 and cap.upper_bound_rps is None
    assert "extend the sweep upward" in cap.note


def test_none_sustained_is_not_bracketed_and_gives_no_rate() -> None:
    cap = find_capacity([point(1, ratio=0.5), point(2, ratio=0.4)])
    assert not cap.bracketed and cap.rate_rps is None
    assert cap.lower_bound_rps is None and cap.upper_bound_rps == 1
    assert "extend the sweep downward" in cap.note


def test_a_sustained_rate_above_a_failure_is_never_promoted() -> None:
    cap = find_capacity([point(1), point(2, ratio=0.5), point(4)])
    assert cap.rate_rps == 1 and not cap.monotone
    assert "non-monotone" in cap.note


def test_capacity_points_from_real_open_loop_runs(run_factory: RunFactory, tmp_path: Path) -> None:
    root = tmp_path / "results"
    engine = MockEngine(delay=DelayModel(ttft_base_s=0.01, token_base_s=0.001))
    cfg = experiment_config(
        experiment="cap-r20",
        repetitions=1,
        load={"mode": "open", "arrival": {"process": "poisson", "rate": 20}, "duration_s": 1.5},
        measurement={"warmup": {"requests": 0}, "warmup_s": 0.2, "cooldown_s": 0.2},
    )
    run = run_factory(cfg, engine=engine)
    write_manifest(root, "capacity-p128", "load.arrival.rate", [("r20", 20, run, "done")])
    suite = load_suite_runs(root, "capacity-p128")
    assert suite is not None
    (rp,) = capacity_points(suite, PREFILL)
    rep = run.summary["per_rep"][0]
    assert rp.offered_rps == 20
    assert math.isclose(rp.arrived_rps, rep["n_measured"] / rep["window"]["length_s"])
    assert math.isclose(rp.achieved_rps, rep["throughput"]["req_s"])
    assert 0.5 < rp.ttft_growth < 2  # an unloaded mock: TTFT is flat across the window
    cap = find_capacity([rp])
    assert cap.evidence[0].sustained and not cap.bracketed
    # One prompt length: slowdown divides every TTFT by the same constant, so growth is unchanged.
    assert rp.slowdown_growth is not None
    assert math.isclose(rp.slowdown_growth, rp.ttft_growth, rel_tol=1e-12)
    assert "raw TTFT" not in cap.note

    (raw,) = capacity_points(suite, None)
    assert raw.slowdown_growth is None and raw.ttft_growth == rp.ttft_growth
    assert "growth on raw TTFT (no prefill fit" in find_capacity([raw]).note


def test_session1_mixed_sweep_brackets_capacity_on_slowdown() -> None:
    # Session 1, mixed workload: raw TTFT growth 3.08 at 0.05 req/s is prompt-mix noise.
    ratios = {0.05: 1.097, 0.1: 1.019, 0.15: 1.012, 0.2: 1.021, 0.25: 0.770}
    raw = {0.05: 3.08, 0.1: 0.77, 0.15: 0.95, 0.2: 0.84, 0.25: 8.54}
    slowdown = {0.05: 1.1, 0.1: 0.9, 0.15: 1.0, 0.2: 0.95, 0.25: 6.0}
    amended = find_capacity([point(r, ratios[r], raw[r], slowdown[r]) for r in ratios])
    assert amended.rate_rps == 0.2 and amended.monotone
    assert [e.sustained for e in amended.evidence] == [True, True, True, True, False]

    fallback = [replace(point(r, ratios[r], raw[r]), slowdown_growth=None) for r in ratios]
    unfit = find_capacity(fallback)
    assert unfit.rate_rps is None and "capacity < 0.05" in unfit.note
    assert "raw TTFT" in unfit.note


class _StubRun:
    """The two things `ttft_growth` reads from a run: its request frame and per-rep windows."""

    def __init__(self, requests: pd.DataFrame, start_s: float, end_s: float) -> None:
        self._requests = requests
        self.summary = {"per_rep": [{"rep": 0, "window": {"start_s": start_s, "end_s": end_s}}]}

    def requests(self) -> pd.DataFrame:
        return self._requests


def test_slowdown_growth_cancels_a_long_prompt_landing_in_the_last_third() -> None:
    # 30 s window, one request per second. Every request is served unloaded (slowdown 1), but
    # the last third draws mostly 8K-token prompts, so its raw TTFT median is far higher.
    prompts = [128] * 20 + [8192] * 6 + [128] * 4
    rows: list[dict[str, Any]] = [
        {
            "rep": 0,
            "in_window": True,
            "ok": True,
            "t_arrival": int((i + 0.5) * 1e9),
            "prompt_tokens_req": p,
            "ttft_s": PREFILL.predict(p),
        }
        for i, p in enumerate(prompts)
    ]
    run = cast(Run, _StubRun(pd.DataFrame(rows), start_s=0.0, end_s=30.0))
    (raw,) = ttft_growth(run)
    (slow,) = ttft_growth(run, PREFILL)
    assert raw > 2
    assert math.isclose(slow, 1.0)
