"""Unit tests for the load driver (plan §Phase 2 item 5).

The streaming call is faked so scheduling behavior is deterministic: concurrency caps, request
caps, duration deadlines, exact scheduled arrival times and client lag.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest

from llmserve.client.capabilities import capabilities
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics.records import RequestRecord, RequestStatus
from llmserve.runner import run as driver
from llmserve.runner.clock import RunClock
from llmserve.workload.spec import RequestSpec

CAPS = capabilities("mock")


def make_cfg(*, gateway: Any = None, **load: Any) -> ExperimentConfig:
    # The driver consumes materialized specs, so `arrival` only has to satisfy the schema;
    # the offsets under test live on the specs themselves.
    if load.get("mode") == "open":
        base: dict[str, Any] = {"arrival": {"process": "constant", "rate": 100}}
    else:
        base = {"concurrency": 1}
    base.update(load)
    cfg: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "driver-test",
        "seed": 7,
        "server": {"kind": "mock", "endpoint": "http://test/v1", "model": "mock"},
        "load": base,
        "workload": {
            "classes": [
                {
                    "name": "c",
                    "prompt": {"distribution": "fixed", "tokens": 8},
                    "output": {"distribution": "fixed", "tokens": 4},
                }
            ]
        },
    }
    if gateway is not None:
        cfg["gateway"] = gateway
    return ExperimentConfig.model_validate(cfg)


def spec(i: int, offset: float = 0.0) -> RequestSpec:
    return RequestSpec(
        request_id=f"r{i:05d}",
        workload_class="c",
        prompt_tokens=8,
        output_tokens=4,
        prompt_ids=tuple(range(100, 108)),
        arrival_offset_s=offset,
    )


def ok_record(spec: RequestSpec, t_arrival: int) -> RequestRecord:
    return RequestRecord(
        request_id=spec.request_id,
        workload_class=spec.workload_class,
        priority=spec.priority,
        prompt_tokens_req=spec.prompt_tokens,
        output_tokens_req=spec.output_tokens,
        status=RequestStatus.OK,
        t_arrival=t_arrival,
        t_first_token=t_arrival + 1,
        t_last_token=t_arrival + 2,
        prompt_tokens_usage=spec.prompt_tokens,
        output_tokens_usage=spec.output_tokens,
    )


def fake_sender(monkeypatch: pytest.MonkeyPatch, *, delay_s: float = 0.0) -> dict[str, int]:
    state = {"in_flight": 0, "peak": 0}

    async def fake(
        _client: httpx.AsyncClient,
        spec: RequestSpec,
        _clock: Any,
        *,
        t_arrival: int,
        **kwargs: Any,
    ) -> RequestRecord:
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        if delay_s:
            await asyncio.sleep(delay_s)
        state["in_flight"] -= 1
        return ok_record(spec, t_arrival)

    monkeypatch.setattr(driver, "stream_completion", fake)
    return state


def drive(cfg: ExperimentConfig, specs: list[RequestSpec]) -> list[RequestRecord]:
    async def run() -> list[RequestRecord]:
        async with httpx.AsyncClient() as client:
            return await driver.drive(cfg, specs, clock=RunClock(), client=client, caps=CAPS)

    return asyncio.run(run())


def test_closed_loop_caps_concurrency_and_request_count(monkeypatch: pytest.MonkeyPatch) -> None:
    state = fake_sender(monkeypatch, delay_s=0.01)
    cfg = make_cfg(mode="closed", concurrency=5, requests=40)

    records = drive(cfg, [spec(i) for i in range(100)])

    assert len(records) == 40
    assert state["peak"] <= 5
    assert all(r.ok for r in records)
    assert all(r.client_lag_ns is None for r in records)  # no schedule, no lag
    assert [r.t_arrival for r in records] == sorted(r.t_arrival for r in records)


def test_closed_loop_stops_at_the_duration_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sender(monkeypatch, delay_s=0.005)
    cfg = make_cfg(mode="closed", concurrency=4, duration_s=0.1)

    records = drive(cfg, [spec(i) for i in range(10_000)])

    # All four workers pass the first deadline check, so each completes at least one request;
    # the deadline then stops the loop well before the 10k specs run out.
    assert 4 <= len(records) < 10_000


def test_open_loop_fires_on_scheduled_offsets_and_records_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sender(monkeypatch)
    cfg = make_cfg(mode="open", requests=5)
    offsets = [0.20, 0.0, 0.10, 0.05, 0.15]  # deliberately shuffled

    started = time.monotonic()
    records = drive(cfg, [spec(i, off) for i, off in enumerate(offsets)])
    elapsed = time.monotonic() - started

    assert len(records) == 5
    assert elapsed >= 0.19  # the arrival loop actually waits out the schedule
    arrivals = [r.t_arrival for r in records]
    assert arrivals == sorted(arrivals)
    assert all(b - a == 50_000_000 for a, b in zip(arrivals, arrivals[1:], strict=False))
    for rec in records:
        assert rec.client_lag_ns is not None
        assert 0 <= rec.client_lag_ns < 100_000_000  # generous bound; P99 < 5 ms is the real bar


def test_open_loop_duration_drops_arrivals_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sender(monkeypatch)
    cfg = make_cfg(mode="open", duration_s=0.1)
    specs = [spec(i, off) for i, off in enumerate([0.0, 0.05, 0.10, 0.15, 0.20])]

    records = drive(cfg, specs)

    assert len(records) == 2
    assert [r.t_arrival - records[0].t_arrival for r in records] == [0, 50_000_000]


def test_open_loop_requests_caps_the_earliest_arrivals(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sender(monkeypatch)
    cfg = make_cfg(mode="open", requests=2)
    specs = [spec(i, off) for i, off in enumerate([0.3, 0.1, 0.2, 0.0])]

    records = drive(cfg, specs)

    assert [r.request_id for r in records] == ["r00003", "r00001"]


def test_empty_workload_yields_no_records(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sender(monkeypatch)
    closed = make_cfg(mode="closed", concurrency=3, requests=10)
    opened = make_cfg(mode="open", requests=10)
    assert drive(closed, []) == []
    assert drive(opened, []) == []


def test_driver_routes_through_the_gateway_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = fake_sender(monkeypatch, delay_s=0.01)
    cfg = make_cfg(
        mode="closed",
        concurrency=5,
        requests=30,
        gateway={"enabled": True, "max_in_flight": 3},
    )

    records = drive(cfg, [spec(i) for i in range(30)])

    assert len(records) == 30
    assert state["peak"] <= 3  # the cap, not the worker count, bounds in-flight
    assert all(r.ok for r in records)
    assert all(r.sched_name == "fifo" for r in records)
    assert all(r.client_lag_ns is None for r in records)
