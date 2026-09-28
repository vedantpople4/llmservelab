"""OpenAI-compatible streaming server for the mock backend (ADR-009).

Serves the subset of the API the harness uses: `/health`, `/version`, `/v1/models` and
`POST /v1/completions`, with token-ID prompts, forced output length and `stream_options.
include_usage` — the same contract as pinned vLLM (ADR-005), so the streaming client's code path
is identical against either backend.

Run it with:

    uv run python -m llmserve.mock --port 8001
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llmserve.mock.engine import MockEngine

MOCK_VERSION = "0.1.0"
# One " x" per generated token: non-empty, whitespace-prefixed, and cheap to count.
TOKEN_TEXT = " x"


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


def create_app(engine: MockEngine | None = None) -> FastAPI:
    engine = engine if engine is not None else MockEngine()
    app = FastAPI(title="llmserve mock", version=MOCK_VERSION)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"version": MOCK_VERSION}

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
        prompt_tokens = MockEngine.prompt_tokens(req.prompt)
        n = req.max_tokens or 16
        include_usage = bool(req.stream_options and req.stream_options.include_usage)
        chunk_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        usage = _usage(prompt_tokens, n)

        if not req.stream:
            text = TOKEN_TEXT * n
            return JSONResponse(
                {
                    "id": chunk_id,
                    "object": "text_completion",
                    "created": created,
                    "model": engine.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "logprobs": None,
                            "finish_reason": "length",
                        }
                    ],
                    "usage": usage,
                }
            )

        async def event_stream() -> AsyncIterator[bytes]:
            engine.acquire()
            try:
                delay = engine.delay
                # Prefill, then decode: the first token pays the whole TTFT.
                await asyncio.sleep(delay.ttft_s(prompt_tokens, engine.in_flight))
                yield _sse(
                    _text_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=engine.model,
                        text=TOKEN_TEXT,
                        finish_reason=None,
                    )
                )
                for i in range(1, n):
                    await asyncio.sleep(delay.token_s(engine.in_flight))
                    last = i == n - 1
                    yield _sse(
                        _text_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=engine.model,
                            text=TOKEN_TEXT,
                            finish_reason="length" if last else None,
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
                engine.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
