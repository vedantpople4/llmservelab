"""OpenAI-compatible streaming server for the mock backend (ADR-009).

Serves the subset of the API the harness uses: `/health`, `/version`, `/v1/models`,
`GET /metrics` and `POST /v1/completions`, with token-ID prompts, forced output length and
`stream_options.include_usage` — the same contract as pinned vLLM (ADR-005), so the streaming
client's code path is identical against either backend.

Both engines (`--engine delay`, `--engine cb`) are driven through one path: the engine's
`stream_tokens` produces the timing, this module formats it as SSE. `/metrics` speaks vLLM's
series names so `metrics/server.py` can scrape it unchanged; a request whose KV footprint cannot
fit in the simulator is answered 400 before any streaming starts.

Run it with:

    uv run python -m llmserve.mock --port 8001 [--engine cb]
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llmserve.mock.engine import (
    ContinuousBatchEngine,
    MockEngine,
    count_prompt_tokens,
)

MOCK_VERSION = "0.1.0"
# One " x" per generated token: non-empty, whitespace-prefixed, and cheap to count.
TOKEN_TEXT = " x"

Engine = MockEngine | ContinuousBatchEngine


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class CompletionRequest(BaseModel):
    """The fields the harness sends; anything else (temperature, n, ...) is accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    model: str = "mock"
    prompt: list[int] | str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    ignore_eos: bool = False
    stream: bool = False
    stream_options: StreamOptions | None = None


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _text_chunk(
    *, chunk_id: str, created: int, model: str, text: str, finish_reason: str | None
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
    }


def _usage_chunk(
    *, chunk_id: str, created: int, model: str, usage: dict[str, int]
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    }


def _sse(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else _dumps(payload)
    return f"data: {data}\n\n".encode()


def _dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _metrics_text(engine: Engine, totals: dict[str, int]) -> str:
    """vLLM's series names, so `llmserve.metrics.server` scrapes the mock unchanged."""
    state = engine.metrics()
    lines = [
        "# HELP vllm:num_requests_running Number of requests in the engine batch.",
        f"vllm:num_requests_running {state['running']}",
        "# HELP vllm:num_requests_waiting Number of queued requests.",
        f"vllm:num_requests_waiting {state['waiting']}",
        "# HELP vllm:gpu_cache_usage_perc Fraction of KV capacity in use.",
        f"vllm:gpu_cache_usage_perc {state['kv_usage']}",
        "# HELP vllm:num_preemptions_total Preempted sequences (recompute).",
        f"vllm:num_preemptions_total {state['preemptions_total']}",
        "# HELP vllm:prompt_tokens_total Prompt tokens served.",
        f"vllm:prompt_tokens_total {totals['prompt']}",
        "# HELP vllm:generation_tokens_total Generated tokens served.",
        f"vllm:generation_tokens_total {totals['generation']}",
    ]
    return "\n".join(lines) + "\n"


def create_app(engine: Engine | None = None) -> FastAPI:
    engine = engine if engine is not None else MockEngine()
    totals = {"prompt": 0, "generation": 0}  # server-side counters (the engine sees streams only)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await engine.aclose()  # stop the simulator's step loop on shutdown

    app = FastAPI(title="llmserve mock", version=MOCK_VERSION, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"version": MOCK_VERSION}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(_metrics_text(engine, totals))

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llmservelab",
                }
            ],
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(req: CompletionRequest) -> StreamingResponse | JSONResponse:
        prompt_tokens = count_prompt_tokens(req.prompt)
        n = req.max_tokens or 16
        rejected = engine.check_request(prompt_tokens, n)
        if rejected:
            return JSONResponse(
                {
                    "error": {
                        "message": rejected,
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                    }
                },
                status_code=400,
            )
        include_usage = bool(req.stream_options and req.stream_options.include_usage)
        chunk_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        usage = _usage(prompt_tokens, n)

        if not req.stream:
            # Non-stream responses bypass the simulation: usage without waiting (documented).
            totals["prompt"] += prompt_tokens
            totals["generation"] += n
            return JSONResponse(
                {
                    "id": chunk_id,
                    "object": "text_completion",
                    "created": created,
                    "model": engine.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": TOKEN_TEXT * n,
                            "logprobs": None,
                            "finish_reason": "length",
                        }
                    ],
                    "usage": usage,
                }
            )

        async def event_stream() -> AsyncIterator[bytes]:
            totals["prompt"] += prompt_tokens
            stream = engine.stream_tokens(prompt_tokens, n)
            try:
                async for is_last in stream:
                    totals["generation"] += 1
                    yield _sse(
                        _text_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=engine.model,
                            text=TOKEN_TEXT,
                            finish_reason="length" if is_last else None,
                        )
                    )
                if include_usage:
                    yield _sse(
                        _usage_chunk(
                            chunk_id=chunk_id, created=created, model=engine.model, usage=usage
                        )
                    )
                yield _sse("[DONE]")
            finally:
                await stream.aclose()  # releases engine state on a client disconnect too

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
