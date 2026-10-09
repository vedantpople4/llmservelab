"""The mock's continuous-batching simulator (plan Phase 2 item 4).

The engine is driven directly where timing and KV state must be deterministic, and through the
HTTP server where the contract matters: SSE shape, 400 on a request that cannot fit, and the
vLLM-named `/metrics` the harness's server sampler scrapes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import replace

import httpx

from llmserve.client.sse import SseParser
from llmserve.metrics.server import parse_prometheus, sample_from
from llmserve.mock import CbParams, ContinuousBatchEngine, create_app
from llmserve.mock.server import Engine

# Step time is the only thing these tests tune: capacities stay real, delays stay tiny.
PAUSED = CbParams(alpha_s=0.002, beta_s=0.0, gamma_s=0.0, delta_s=0.0)


def app_client(engine: Engine) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://test"
    )


def test_stream_delivers_n_tokens_usage_then_done() -> None:
    engine = ContinuousBatchEngine(params=CbParams.instant())

    async def run() -> list[bool]:
        try:
            async with app_client(engine) as c:
                r = await c.post(
                    "/v1/completions",
                    json={
                        "prompt": [1, 2, 3, 4],
                        "max_tokens": 5,
                        "ignore_eos": True,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )
                assert r.status_code == 200
            parser = SseParser()
            events = parser.feed(r.content)
            parser.close()
            payloads = [json.loads(e.data) for e in events if e.data != "[DONE]"]
            chunks = [p for p in payloads if p["choices"]]
            assert len(chunks) == 5
            assert chunks[-1]["choices"][0]["finish_reason"] == "length"
            assert all(ch["choices"][0]["text"] for ch in chunks)
            usage = [p for p in payloads if not p["choices"] and "usage" in p]
            assert usage[0]["usage"] == {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9,
            }
            assert sum(1 for e in events if e.data == "[DONE]") == 1
            state = engine.metrics()
            assert state["running"] == 0 and state["waiting"] == 0
            assert state["kv_usage"] == 0.0  # every footprint was released
            return chunks
        finally:
            await engine.aclose()  # ASGITransport never runs the lifespan

    asyncio.run(run())


def test_kv_cap_queues_a_request_that_cannot_fit() -> None:
    engine = ContinuousBatchEngine(params=replace(PAUSED, kv_capacity_tokens=40))

    async def run() -> tuple[dict[str, float], dict[str, float], list[bool], list[bool]]:
        first = engine.stream_tokens(32, 4)
        assert await first.__anext__() is False  # token 1 of 4: seq A is admitted and decoding
        admitted = engine.metrics()

        second = engine.stream_tokens(32, 4)
        pending = asyncio.create_task(second.__anext__())  # seq B enqueues itself
        await asyncio.sleep(0)
        blocked = engine.metrics()

        tokens_a = [False, *[tok async for tok in first]]
        tokens_b = [await pending, *[tok async for tok in second]]
        await engine.aclose()
        return admitted, blocked, tokens_a, tokens_b

    admitted, blocked, tokens_a, tokens_b = asyncio.run(run())
    assert admitted["running"] == 1 and admitted["waiting"] == 0
    assert blocked["running"] == 1 and blocked["waiting"] == 1  # 8 free tokens < 32 needed
    assert len(tokens_a) == 4 and tokens_a[-1] is True
    assert len(tokens_b) == 4 and tokens_b[-1] is True


def test_prefill_is_chunked_into_many_steps_before_the_first_token() -> None:
    # 500 prompt tokens with a 100-token budget = 5 one-millisecond prefill steps, so the first
    # token cannot arrive in a single simulated step.
    engine = ContinuousBatchEngine(
        params=CbParams(
            alpha_s=1e-3,
            beta_s=0.0,
            gamma_s=0.0,
            delta_s=0.0,
            max_num_batched_tokens=100,
            prefill_chunk=100,
            kv_capacity_tokens=10_000,
        )
    )

    async def run() -> tuple[float, list[bool]]:
        try:
            stream = engine.stream_tokens(500, 3)
            start = time.perf_counter()
            first = await stream.__anext__()
            elapsed = time.perf_counter() - start
            return elapsed, [first, *[tok async for tok in stream]]
        finally:
            await engine.aclose()

    elapsed, tokens = asyncio.run(run())
    assert 0.004 <= elapsed < 0.5  # >= 5 chunked steps of 1 ms; bounded so a stall fails loudly
    assert len(tokens) == 3 and tokens[-1] is True


def test_kv_exhaustion_preempts_and_both_requests_finish() -> None:
    # A wants 60 + 40 = 100 tokens (the whole capacity); B needs 30 more. Decode-time KV
    # pressure must preempt B (recompute), then re-admit it once A finishes.
    engine = ContinuousBatchEngine(
        params=CbParams(alpha_s=5e-4, beta_s=0.0, gamma_s=0.0, delta_s=0.0, kv_capacity_tokens=100)
    )

    async def run() -> tuple[list[bool], list[bool], dict[str, float]]:
        try:
            a = engine.stream_tokens(60, 40)
            b = engine.stream_tokens(30, 10)
            done_a, done_b = await asyncio.gather(*[_tokens(a), _tokens(b)])
            return done_a, done_b, engine.metrics()
        finally:
            await engine.aclose()

    tokens_a, tokens_b, state = asyncio.run(run())
    assert len(tokens_a) == 40 and tokens_a[-1] is True
    assert len(tokens_b) == 10 and tokens_b[-1] is True  # recompute never loses a delivered token
    assert state["preemptions_total"] >= 1
    assert state["kv_usage"] == 0.0
    assert state["running"] == 0 and state["waiting"] == 0


def test_request_larger_than_the_kv_cap_is_rejected_before_streaming() -> None:
    engine = ContinuousBatchEngine(params=CbParams(kv_capacity_tokens=128))

    async def run() -> httpx.Response:
        try:
            async with app_client(engine) as c:
                return await c.post(
                    "/v1/completions",
                    json={"prompt": [1] * 100, "max_tokens": 50, "stream": True},
                )
        finally:
            await engine.aclose()

    r = asyncio.run(run())
    assert r.status_code == 400
    assert "KV capacity" in r.json()["error"]["message"]


def test_metrics_endpoint_reports_the_simulation_in_vllm_names() -> None:
    engine = ContinuousBatchEngine(params=CbParams.instant())

    async def run() -> tuple[dict[str, float], dict[str, float]]:
        try:
            async with app_client(engine) as c:
                r = await c.post(
                    "/v1/completions",
                    json={
                        "prompt": [1, 2],
                        "max_tokens": 3,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )
                assert r.status_code == 200
                streamed = parse_prometheus((await c.get("/metrics")).text)
                r2 = await c.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 3})
                assert r2.status_code == 200  # non-stream: counted, not simulated
                both = parse_prometheus((await c.get("/metrics")).text)
                return streamed, both
        finally:
            await engine.aclose()

    streamed, both = asyncio.run(run())
    sample = sample_from(streamed, t_ns=0)  # raises unless all six series are present
    assert sample.running == 0 and sample.waiting == 0 and sample.kv_usage == 0.0
    assert sample.preemptions_total == 0.0
    assert sample.prompt_tokens_total == 2
    assert sample.generation_tokens_total == 3
    assert both["vllm:prompt_tokens_total"] == 4
    assert both["vllm:generation_tokens_total"] == 6


async def _tokens(stream: AsyncGenerator[bool, None]) -> list[bool]:
    return [tok async for tok in stream]
