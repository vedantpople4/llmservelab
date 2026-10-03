from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llmserve.config.schema import Load
from llmserve.workload.arrivals import generate_arrivals


def open_load(**kw: object) -> Load:
    return Load.model_validate({"mode": "open", **kw})


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_constant_offsets_are_exact() -> None:
    load = open_load(arrival={"process": "constant", "rate": 4}, requests=5)
    np.testing.assert_allclose(generate_arrivals(load, rng()), [0.0, 0.25, 0.5, 0.75, 1.0])


def test_constant_horizon_drops_the_boundary() -> None:
    load = open_load(arrival={"process": "constant", "rate": 2}, duration_s=1.0)
    np.testing.assert_allclose(generate_arrivals(load, rng()), [0.0, 0.5])  # t=1.0 is outside


def test_rho_resolves_through_capacity() -> None:
    load = open_load(arrival={"process": "constant", "rho": 0.5}, capacity_rps=100, requests=3)
    np.testing.assert_allclose(generate_arrivals(load, rng()), [0.0, 0.02, 0.04])


def test_more_requests_than_the_horizon_allows_errors() -> None:
    load = open_load(arrival={"process": "constant", "rate": 1}, duration_s=0.5, requests=3)
    with pytest.raises(ValueError, match="only 1 of 3"):
        generate_arrivals(load, rng())


def test_poisson_interarrival_mean_is_the_rate() -> None:
    load = open_load(arrival={"process": "poisson", "rate": 50}, requests=20_000)
    gaps = np.diff(generate_arrivals(load, rng(1)))
    sem = (1 / 50) / np.sqrt(len(gaps))  # 3 sigma of the mean of exponential gaps
    assert abs(gaps.mean() - 1 / 50) < 3 * sem
    assert np.all(gaps > 0)


def test_poisson_stops_at_the_horizon() -> None:
    load = open_load(arrival={"process": "poisson", "rate": 100}, duration_s=2.0)
    offsets = generate_arrivals(load, rng(2))
    assert 0 < offsets[-1] < 2.0
    assert 130 < offsets.size < 270  # ~200 expected at 100 req/s; generous band


def test_poisson_requests_and_duration_together() -> None:
    load = open_load(arrival={"process": "poisson", "rate": 10}, duration_s=0.5, requests=2)
    offsets = generate_arrivals(load, rng(3))
    assert offsets.size == 2
    assert offsets.max() < 0.5


def test_bursty_phase_rates() -> None:
    arrival = {
        "process": "bursty",
        "phases": [
            {"duration_s": 5, "rate": 5},
            {"duration_s": 5, "rate": 50},
            {"duration_s": 5, "rate": 5},
        ],
    }
    offsets = generate_arrivals(open_load(arrival=arrival, duration_s=15.0), rng(4))
    early = int(((offsets >= 0) & (offsets < 5)).sum())  # 25 expected, sigma 5
    mid = int(((offsets >= 5) & (offsets < 10)).sum())  # 250 expected, sigma ~16
    late = int(((offsets >= 10) & (offsets < 15)).sum())
    assert 5 <= early <= 45
    assert 185 <= mid <= 315
    assert 5 <= late <= 45


def test_bursty_phases_must_cover_the_duration() -> None:
    arrival = {"process": "bursty", "phases": [{"duration_s": 5, "rate": 10}]}
    load = open_load(arrival=arrival, duration_s=10.0)
    with pytest.raises(ValueError, match="extend the phases"):
        generate_arrivals(load, rng())


def test_bursty_requests_must_fit_the_phases() -> None:
    arrival = {"process": "bursty", "phases": [{"duration_s": 1, "rate": 1}]}
    with pytest.raises(ValueError, match="only \\d+ of 100"):
        generate_arrivals(open_load(arrival=arrival, requests=100), rng())


def test_replay_scales_and_cuts(tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    trace.write_text("0\n1\n2\n3\n")
    load = open_load(
        arrival={"process": "replay", "trace": str(trace), "time_scale": 2.0},
        duration_s=1.0,
    )
    np.testing.assert_allclose(generate_arrivals(load, rng()), [0.0, 0.5])


def test_replay_caps_request_count(tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    trace.write_text("0\n1\n2\n3\n")
    load = open_load(arrival={"process": "replay", "trace": str(trace)}, requests=2)
    np.testing.assert_allclose(generate_arrivals(load, rng()), [0.0, 1.0])


def test_replay_rejects_unsorted_offsets(tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    trace.write_text("0\n2\n1\n")
    load = open_load(arrival={"process": "replay", "trace": str(trace)}, requests=3)
    with pytest.raises(ValueError, match="non-decreasing"):
        generate_arrivals(load, rng())


def test_replay_missing_trace_is_reported(tmp_path: Path) -> None:
    load = open_load(arrival={"process": "replay", "trace": str(tmp_path / "nope.txt")}, requests=1)
    with pytest.raises(ValueError, match="cannot read trace"):
        generate_arrivals(load, rng())


def test_closed_loop_loads_have_no_arrivals() -> None:
    closed = Load.model_validate({"mode": "closed", "concurrency": 1, "requests": 1})
    with pytest.raises(ValueError, match="closed-loop"):
        generate_arrivals(closed, rng())


def test_empty_window_errors() -> None:
    load = open_load(arrival={"process": "poisson", "rate": 1}, duration_s=1e-9)
    with pytest.raises(ValueError, match="no arrivals"):
        generate_arrivals(load, rng())


def test_same_seed_same_arrivals() -> None:
    load = open_load(arrival={"process": "poisson", "rate": 10}, requests=50)
    a = generate_arrivals(load, np.random.default_rng(7))
    b = generate_arrivals(load, np.random.default_rng(7))
    c = generate_arrivals(load, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
