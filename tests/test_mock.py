import asyncio
import json

import httpx

from llmserve.client.sse import SseParser
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
