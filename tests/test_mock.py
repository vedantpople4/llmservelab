import asyncio
import json

import httpx

from llmserve.client.sse import SseParser
from llmserve.metrics.server import parse_prometheus, sample_from
from llmserve.mock import DelayModel, MockEngine, create_app


def client(engine: MockEngine | None = None) -> httpx.AsyncClient:
    app = create_app(engine or MockEngine(delay=DelayModel.instant()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_health_version_and_models() -> None:
    async def run() -> None:
        async with client() as c:
            assert (await c.get("/health")).json() == {"status": "ok"}
            assert (await c.get("/version")).json()["version"]
            body = (await c.get("/v1/models")).json()
            assert body["object"] == "list"
            assert body["data"][0]["id"] == "mock"

    asyncio.run(run())


def test_non_stream_returns_usage_for_token_id_prompt() -> None:
    async def run() -> None:
        async with client() as c:
            r = await c.post(
                "/v1/completions",
                json={"prompt": list(range(128)), "max_tokens": 7, "ignore_eos": True},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["usage"] == {"prompt_tokens": 128, "completion_tokens": 7, "total_tokens": 135}
        assert body["choices"][0]["finish_reason"] == "length"

    asyncio.run(run())


def test_stream_delivers_max_tokens_chunks_then_done() -> None:
    async def run() -> None:
        async with client() as c:
            r = await c.post(
                "/v1/completions",
                json={
                    "model": "mock",
                    "prompt": [1, 2, 3, 4],
                    "max_tokens": 5,
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        parser = SseParser()
        events = parser.feed(r.content)
        parser.close()

        texts = [json.loads(e.data) for e in events if e.data != "[DONE]"]
        done = [e for e in events if e.data == "[DONE]"]
        assert len(done) == 1

        chunks = [t for t in texts if t["choices"]]
        assert len(chunks) == 5
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert all(ch["choices"][0]["text"] for ch in chunks)

        usage_chunks = [t for t in texts if not t["choices"] and "usage" in t]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["completion_tokens"] == 5
        assert usage_chunks[0]["usage"]["prompt_tokens"] == 4

    asyncio.run(run())


def test_in_flight_counter_tracks_concurrent_streams() -> None:
    engine = MockEngine(delay=DelayModel(ttft_base_s=0.05, token_base_s=0.0))

    async def run() -> None:
        async with client(engine) as c:

            async def one() -> None:
                r = await c.post(
                    "/v1/completions",
                    json={"prompt": [1], "max_tokens": 1, "stream": True},
                )
                assert r.status_code == 200

            await asyncio.gather(one(), one())

    asyncio.run(run())
    assert engine.in_flight == 0


def test_missing_prompt_is_rejected() -> None:
    async def run() -> None:
        async with client() as c:
            r = await c.post("/v1/completions", json={"max_tokens": 4})
        assert r.status_code == 422

    asyncio.run(run())


def test_text_prompt_is_accepted_with_an_estimated_length() -> None:
    async def run() -> None:
        async with client() as c:
            r = await c.post("/v1/completions", json={"prompt": "hello world", "max_tokens": 2})
        assert r.status_code == 200
        assert r.json()["usage"]["prompt_tokens"] == len("hello world") // 4

    asyncio.run(run())


def test_default_max_tokens_is_16() -> None:
    async def run() -> None:
        async with client() as c:
            r = await c.post(
                "/v1/completions",
                json={"prompt": [1, 2], "stream": True, "stream_options": {"include_usage": True}},
            )
        parser = SseParser()
        events = parser.feed(r.content)
        chunks = [
            json.loads(e.data)
            for e in events
            if e.data != "[DONE]" and json.loads(e.data)["choices"]
        ]
        assert len(chunks) == 16

    asyncio.run(run())


def test_stream_without_include_usage_omits_the_usage_chunk() -> None:
    async def run() -> None:
        async with client() as c:
            r = await c.post(
                "/v1/completions",
                json={"prompt": [1], "max_tokens": 3, "stream": True},
            )
        parser = SseParser()
        events = parser.feed(r.content)
        payloads = [json.loads(e.data) for e in events if e.data != "[DONE]"]
        assert all(p["choices"] for p in payloads)
        assert events[-1].data == "[DONE]"

    asyncio.run(run())


def test_unknown_request_fields_are_ignored() -> None:
    async def run() -> None:
        async with client() as c:
            r = await c.post(
                "/v1/completions",
                json={"prompt": [1], "max_tokens": 2, "temperature": 0.7, "top_p": 0.9, "n": 1},
            )
        assert r.status_code == 200

    asyncio.run(run())


def test_single_token_stream_still_finishes_with_length() -> None:
    # One token is both the first and the last chunk: finish_reason must not be lost (item 4).
    async def run() -> None:
        async with client() as c:
            r = await c.post(
                "/v1/completions",
                json={
                    "prompt": [1],
                    "max_tokens": 1,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
        parser = SseParser()
        events = parser.feed(r.content)
        parser.close()
        payloads = [json.loads(e.data) for e in events if e.data != "[DONE]"]
        chunks = [p for p in payloads if p["choices"]]
        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["finish_reason"] == "length"
        usage = [p for p in payloads if not p["choices"] and "usage" in p]
        assert usage[0]["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    asyncio.run(run())


def test_metrics_endpoint_serves_every_sampler_series() -> None:
    async def run() -> dict[str, float]:
        async with client() as c:
            r = await c.post("/v1/completions", json={"prompt": [7, 8, 9], "max_tokens": 2})
            assert r.status_code == 200  # non-stream: counted, not simulated
            m = await c.get("/metrics")
        assert m.status_code == 200
        return parse_prometheus(m.text)

    sample = sample_from(asyncio.run(run()), t_ns=0)
    assert sample.prompt_tokens_total == 3
    assert sample.generation_tokens_total == 2
    assert sample.running == 0  # delay model: no internal queue or KV, so both are structurally 0
    assert sample.kv_usage == 0.0
