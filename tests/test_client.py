"""The streaming client: SSE parsing, timestamps, statuses and usage ground truth."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from llmserve.client.capabilities import capabilities
from llmserve.client.openai_stream import stream_completion
from llmserve.metrics import latency
from llmserve.metrics.records import RequestStatus
from llmserve.mock import DelayModel, MockEngine, create_app
from llmserve.runner.clock import RunClock
from llmserve.workload.spec import RequestSpec

CAPS = capabilities("mock")


def spec(**overrides: object) -> RequestSpec:
    base: dict[str, object] = {
        "request_id": "r1",
        "workload_class": "interactive",
        "prompt_tokens": 4,
        "output_tokens": 3,
        "prompt_ids": (1, 2, 3, 4),
    }
    base.update(overrides)
    return RequestSpec(**base)  # type: ignore[arg-type]


def sse(*payloads: dict[str, object]) -> bytes:
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    return body.encode() + b"data: [DONE]\n\n"


def text_chunks(*texts: str, finish: str | None = None) -> list[dict[str, Any]]:
    out = []
    for i, t in enumerate(texts):
        out.append(
            {
                "choices": [
                    {
                        "text": t,
                        "finish_reason": finish if i == len(texts) - 1 else None,
                    }
                ]
            }
        )
    return out


def transport(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


def test_happy_path_records_tokens_and_timestamps() -> None:
    body = sse(
        *text_chunks(" a", " b", " c", finish="length"),
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3}},
    )

    async def run() -> None:
        clock = RunClock()
        t0 = clock()
        async with transport(httpx.MockTransport(lambda r: httpx.Response(200, content=body))) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=t0, caps=CAPS
            )
        assert rec.status is RequestStatus.OK
        assert rec.prompt_tokens_usage == 4
        assert rec.output_tokens_usage == 3
        assert len(rec.chunks) == 3
        assert rec.t_first_token is not None and rec.t_last_token is not None
        assert rec.t_dispatch is not None and rec.t_first_token >= rec.t_dispatch
        assert rec.t_complete is not None and rec.t_complete >= rec.t_last_token
        assert latency.ttft(rec) is not None
        assert latency.tpot(rec) is not None

    asyncio.run(run())


def test_http_error_is_recorded_not_raised() -> None:
    async def run() -> None:
        clock = RunClock()
        h = lambda r: httpx.Response(500, text="boom")  # noqa: E731
        async with transport(httpx.MockTransport(h)) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=clock(), caps=CAPS
            )
        assert rec.status is RequestStatus.HTTP_ERROR
        assert rec.http_status == 500
        assert "boom" in (rec.error or "")

    asyncio.run(run())


def test_connection_error_and_timeout_statuses() -> None:
    async def run() -> None:
        clock = RunClock()

        def refuse(r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=r)

        def hang(r: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=r)

        async with transport(httpx.MockTransport(refuse)) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=clock(), caps=CAPS
            )
        assert rec.status is RequestStatus.CONN_ERROR

        async with transport(httpx.MockTransport(hang)) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=clock(), caps=CAPS
            )
        assert rec.status is RequestStatus.TIMEOUT

    asyncio.run(run())


def test_usage_mismatch_is_length_mismatch() -> None:
    body = sse(
        *text_chunks(" a", " b", " c", finish="length"),
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 7}},
    )

    async def run() -> None:
        clock = RunClock()
        async with transport(httpx.MockTransport(lambda r: httpx.Response(200, content=body))) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=clock(), caps=CAPS
            )
        assert rec.status is RequestStatus.LENGTH_MISMATCH
        assert "requested 3" in (rec.error or "")

    asyncio.run(run())


def test_stream_without_done_is_http_error() -> None:
    body = b'data: {"choices":[{"text":" a"}]}\n\n'

    async def run() -> None:
        clock = RunClock()
        async with transport(httpx.MockTransport(lambda r: httpx.Response(200, content=body))) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=clock(), caps=CAPS
            )
        assert rec.status is RequestStatus.HTTP_ERROR
        assert "[DONE]" in (rec.error or "")

    asyncio.run(run())


def test_malformed_sse_payload_is_http_error() -> None:
    body = b"data: {not json}\n\ndata: [DONE]\n\n"

    async def run() -> None:
        clock = RunClock()
        async with transport(httpx.MockTransport(lambda r: httpx.Response(200, content=body))) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=clock(), caps=CAPS
            )
        assert rec.status is RequestStatus.HTTP_ERROR
        assert "invalid JSON" in (rec.error or "")

    asyncio.run(run())


def test_token_counter_sizes_chunks() -> None:
    body = sse(
        {"choices": [{"text": "hello world"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3}},
    )

    async def run() -> None:
        clock = RunClock()
        async with transport(httpx.MockTransport(lambda r: httpx.Response(200, content=body))) as c:
            rec = await stream_completion(
                c,
                spec(output_tokens=3),
                clock,
                endpoint="http://x/v1",
                model="m",
                t_arrival=clock(),
                caps=CAPS,
                token_counter=lambda text: len(text.split()),
            )
        # The usage block still says 3 tokens, so the record is a success; only the chunk
        # accounting comes from the counter.
        assert rec.status is RequestStatus.OK, rec.error
        assert rec.chunks[0].n_tokens == 2

    asyncio.run(run())


def test_end_to_end_against_mock_server() -> None:
    app = create_app(MockEngine(delay=DelayModel.instant()))

    async def run() -> None:
        clock = RunClock()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            rec = await stream_completion(
                c,
                spec(prompt_tokens=16, output_tokens=8, prompt_ids=tuple(range(16))),
                clock,
                endpoint="http://test/v1",
                model="mock",
                t_arrival=clock(),
                caps=CAPS,
            )
        assert rec.status is RequestStatus.OK, rec.error
        assert len(rec.chunks) == 8
        assert rec.output_tokens_usage == 8
        assert latency.ttft(rec) is not None

    asyncio.run(run())


@pytest.mark.parametrize("n", [1, 2, 64])
def test_end_to_end_chunk_count_matches_output_length(n: int) -> None:
    app = create_app(MockEngine(delay=DelayModel.instant()))

    async def run() -> None:
        clock = RunClock()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            rec = await stream_completion(
                c,
                spec(prompt_tokens=5, output_tokens=n, prompt_ids=(1, 2, 3, 4, 5)),
                clock,
                endpoint="http://test/v1",
                model="mock",
                t_arrival=clock(),
                caps=CAPS,
            )
        assert rec.status is RequestStatus.OK, rec.error
        assert len(rec.chunks) == n

    asyncio.run(run())


def test_payload_is_the_adr005_contract() -> None:
    """Pin what the client actually sends: token IDs, forced length, usage requested."""

    body = sse(
        *text_chunks(" a", " b", " c", finish="length"),
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3}},
    )
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, content=body)

    async def run() -> None:
        clock = RunClock()
        async with transport(httpx.MockTransport(handler)) as c:
            rec = await stream_completion(
                c, spec(), clock, endpoint="http://x/v1", model="m", t_arrival=clock(), caps=CAPS
            )
        assert rec.status is RequestStatus.OK, rec.error

    asyncio.run(run())

    payload = seen["json"]
    assert seen["path"] == "/v1/completions"
    assert payload["prompt"] == [1, 2, 3, 4]
    assert payload["max_tokens"] == 3
    assert payload["min_tokens"] == 3
    assert payload["ignore_eos"] is True
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["model"] == "m"


def test_payload_for_a_backend_without_token_ids() -> None:
    """ADR-009 fallback: text prompt, no forced-length flags, usage still requested."""

    loose = replace(CAPS, token_id_prompts=False, forced_output_length=False)
    body = sse({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3}})
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, content=body)

    async def run() -> None:
        clock = RunClock()
        async with transport(httpx.MockTransport(handler)) as c:
            rec = await stream_completion(
                c,
                spec(prompt_text="hello world"),
                clock,
                endpoint="http://x/v1",
                model="m",
                t_arrival=clock(),
                caps=loose,
            )
        assert rec.status is RequestStatus.OK, rec.error

    asyncio.run(run())

    payload = seen["json"]
    assert payload["prompt"] == "hello world"
    assert "min_tokens" not in payload
    assert "ignore_eos" not in payload
    assert payload["stream_options"] == {"include_usage": True}
