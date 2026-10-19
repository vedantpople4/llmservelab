"""Capacity (ρ = 1): the sustained rule, bracketing, and the end-to-end evidence from a run."""

from __future__ import annotations

import math
from pathlib import Path

from conftest import RunFactory, experiment_config
from test_analysis_suites import write_manifest

from llmserve.analysis.capacity import RatePoint, capacity_points, find_capacity
from llmserve.analysis.suites import load_suite_runs
from llmserve.mock import DelayModel, MockEngine


def point(rate: float, ratio: float = 1.0, growth: float = 1.0) -> RatePoint:
    return RatePoint(
        label=f"r{rate:g}",
        offered_rps=rate,
        arrived_rps=rate,
        achieved_rps=rate * ratio,
        ttft_growth=growth,
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
        "r10", offered_rps=10, arrived_rps=8.5, achieved_rps=8.5, ttft_growth=1, reps=1
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
    (rp,) = capacity_points(suite)
    rep = run.summary["per_rep"][0]
    assert rp.offered_rps == 20
    assert math.isclose(rp.arrived_rps, rep["n_measured"] / rep["window"]["length_s"])
    assert math.isclose(rp.achieved_rps, rep["throughput"]["req_s"])
    assert 0.5 < rp.ttft_growth < 2  # an unloaded mock: TTFT is flat across the window
    cap = find_capacity([rp])
    assert cap.evidence[0].sustained and not cap.bracketed
