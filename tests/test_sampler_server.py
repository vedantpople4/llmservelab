"""The `/metrics` scraper (plan Phase 2 item 7): parsing, loud failures, the sampling loop,
and the warm-up's server-idle settle that reads the same endpoint.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from llmserve.client.capabilities import capabilities
from llmserve.metrics.server import (
    SERVER_SCHEMA,
    SamplerError,
    ServerSample,
    ServerSampler,
    lookup,
    parse_prometheus,
    sample_from,
)
from llmserve.runner import storage
from llmserve.runner.experiment import _server_idle


def exposition(
    *, running: float = 1, waiting: float = 0, kv: str = "vllm:gpu_cache_usage_perc"
) -> str:
    lines = [
        "# HELP vllm:num_requests_running Number of running requests.",
        f'vllm:num_requests_running{{model_name="mock"}} {running}',
        f'vllm:num_requests_waiting{{model_name="mock"}} {waiting}',
        f"{kv} 0.5",
        "vllm:num_preemptions_total 3",
        "vllm:prompt_tokens_total 1000",
        "vllm:generation_tokens_total 2000",
    ]
    return "\n".join(lines) + "\n"


def without(series: str) -> str:
    return "\n".join(line for line in exposition().splitlines() if series not in line) + "\n"


def metrics_app(state: dict[str, str]) -> FastAPI:
    """A one-route `/metrics` server whose body the test rewrites in place."""
    app = FastAPI()

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(state["text"])

    return app


def test_parse_prometheus_skips_comments_and_drops_labels() -> None:
    series = parse_prometheus(exposition(running=7))
    assert series["vllm:num_requests_running"] == 7
    assert "vllm:num_requests_waiting" in series
    assert series["vllm:gpu_cache_usage_perc"] == 0.5
    assert all(not name.startswith("#") for name in series)


def test_lookup_falls_back_to_the_older_series_name() -> None:
    series = parse_prometheus(exposition(kv="vllm:kv_cache_usage_perc"))
    assert lookup(series, "kv_usage") == 0.5
    assert lookup(series, "running") == 1


def test_sample_from_names_the_missing_field() -> None:
    series = parse_prometheus(without("vllm:num_requests_waiting"))
    with pytest.raises(SamplerError, match="waiting"):
        sample_from(series, t_ns=1)
    sample = sample_from(parse_prometheus(exposition()), t_ns=42)
    assert sample.t_ns == 42
    assert sample.preemptions_total == 3
    assert sample.prompt_tokens_total == 1000


def test_sampler_collects_samples_at_hz() -> None:
    state = {"text": exposition(running=2)}

    async def run() -> list[ServerSample]:
        transport = httpx.ASGITransport(app=metrics_app(state))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            sampler = ServerSampler(
                time.perf_counter_ns, hz=50, base_url="http://test", client=client
            )
            await sampler.start()  # the validating scrape runs immediately
            state["text"] = exposition(running=0)  # visible from the second scrape on
            await asyncio.sleep(0.06)
            return await sampler.stop()

    samples = asyncio.run(run())
    assert len(samples) >= 2
    assert samples[0].running == 2
    assert samples[0].generation_tokens_total == 2000
    assert any(s.running == 0 for s in samples[1:])
    assert [s.t_ns for s in samples] == sorted(s.t_ns for s in samples)


def test_start_refuses_when_a_series_is_missing() -> None:
    state = {"text": without("vllm:num_requests_waiting")}

    async def run() -> None:
        transport = httpx.ASGITransport(app=metrics_app(state))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            sampler = ServerSampler(
                time.perf_counter_ns, hz=50, base_url="http://test", client=client
            )
            with pytest.raises(SamplerError, match="waiting"):
                await sampler.start()
            assert await sampler.stop() == []

    asyncio.run(run())


def test_mid_run_scrape_failure_lands_in_error() -> None:
    state = {"text": exposition()}

    async def run() -> tuple[list[ServerSample], str | None]:
        transport = httpx.ASGITransport(app=metrics_app(state))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            sampler = ServerSampler(
                time.perf_counter_ns, hz=50, base_url="http://test", client=client
            )
            await sampler.start()
            state["text"] = without("vllm:prompt_tokens_total")  # the backend degraded
            await asyncio.sleep(0.08)
            samples = await sampler.stop()
            return samples, sampler.error

    samples, error = asyncio.run(run())
    assert error is not None and "prompt_tokens_total" in error
    assert len(samples) >= 1  # the samples taken before the failure still count


def test_server_parquet_round_trip(tmp_path: Path) -> None:
    samples = [sample_from(parse_prometheus(exposition()), t_ns=1000)]
    path = tmp_path / "server.parquet"
    storage.write_server(path, samples)
    table = pq.read_table(path)
    assert table.column_names == SERVER_SCHEMA.names
    assert table.to_pylist() == [asdict(s) for s in samples]

    empty = tmp_path / "empty.parquet"
    storage.write_server(empty, [])
    empty_table = pq.read_table(empty)
    assert empty_table.num_rows == 0
    assert empty_table.schema == SERVER_SCHEMA


def test_server_idle_waits_until_both_queues_drain() -> None:
    app = FastAPI()
    calls = 0

    @app.get("/metrics")
    def countdown() -> PlainTextResponse:
        nonlocal calls
        calls += 1
        busy = calls < 3
        return PlainTextResponse(exposition(running=1 if busy else 0, waiting=1 if busy else 0))

    async def run() -> bool | None:
        transport = httpx.ASGITransport(app=app)
        caps = replace(capabilities("mock"), metrics_endpoint=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await _server_idle(client, "http://test/v1", caps, timeout_s=5.0, hz=100)

    assert asyncio.run(run()) is True


def test_server_idle_gives_up_on_the_timeout() -> None:
    state = {"text": exposition(running=1, waiting=1)}

    async def run() -> bool | None:
        transport = httpx.ASGITransport(app=metrics_app(state))
        caps = replace(capabilities("mock"), metrics_endpoint=True)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await _server_idle(client, "http://test/v1", caps, timeout_s=0.05, hz=100)

    assert asyncio.run(run()) is False


def test_server_idle_is_skipped_without_a_metrics_endpoint() -> None:
    async def run() -> bool | None:
        transport = httpx.ASGITransport(app=metrics_app({"text": exposition()}))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # llama.cpp has no /metrics (ADR-009): such a backend is never scraped at all.
            return await _server_idle(
                client, "http://test/v1", capabilities("llamacpp"), timeout_s=1.0, hz=2
            )

    assert asyncio.run(run()) is None
