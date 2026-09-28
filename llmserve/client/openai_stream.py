"""One request → one `RequestRecord`, with every timestamp the analysis needs.

Design rules for this module (Phase 1 design items 2–3):

- **No retries.** A retry distorts latency; a failure is data and gets its own status.
- **`t_first_token` is set by the first chunk that carries a token**, never by the first byte of
  the response or an empty/role chunk.
- **`usage` is ground truth** for token counts. When the backend guarantees exact output length
  (`ignore_eos` + `min_tokens`), a mismatch is recorded as `length_mismatch`, not as a success.
- **Timestamps come from the caller's clock** (ADR-006); this module never reads a wall clock.

Chunk token counts are derived by tokenizing each chunk's text (documented approximation: BPE can
merge across a chunk boundary). TPOT and all headline metrics use `usage`, not chunk counts, so a
rare miscount only affects the ITL shape.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from llmserve.client.capabilities import Capabilities
from llmserve.client.sse import SseParser
from llmserve.metrics.records import Chunk, RequestRecord, RequestStatus
from llmserve.runner.clock import Clock
from llmserve.workload.spec import RequestSpec

TokenCounter = Callable[[str], int]


@dataclass
class _State:
    """Everything that mutates while the response is being read."""

    chunks: list[Chunk] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error: str | None = None
    done: bool = False
    malformed: bool = False
    t_first_token: int | None = None
    t_last_token: int | None = None
    http_status: int | None = None


def _count_tokens(text: str, counter: TokenCounter | None) -> int:
    if not text:
        return 0
    if counter is None:
        return 1
    return max(1, counter(text))


def _usage_int(usage: dict[str, Any] | None, key: str) -> int | None:
    if usage is None or key not in usage:
        return None
    return int(usage[key])


async def stream_completion(
    client: httpx.AsyncClient,
    spec: RequestSpec,
    clock: Clock,
    *,
    endpoint: str,
    model: str,
    t_arrival: int,
    caps: Capabilities,
    timeout_s: float = 600.0,
    t_dispatch: int | None = None,
    token_counter: TokenCounter | None = None,
) -> RequestRecord:
    """Stream one completion and return its record. Never raises for a request-level failure."""
    url = f"{endpoint.rstrip('/')}/completions"
    prompt: list[int] | str = (
        list(spec.prompt_ids) if caps.token_id_prompts else (spec.prompt_text or "")
    )

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": spec.output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if caps.forced_output_length:
        payload["min_tokens"] = spec.output_tokens
        payload["ignore_eos"] = True

    state = _State()
    status = RequestStatus.OK
    error: str | None = None
    sent_at = t_dispatch if t_dispatch is not None else clock()

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)),
        ) as resp:
            state.http_status = resp.status_code
            if not resp.is_success:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                error = f"HTTP {resp.status_code}: {body[:200]}"
                status = RequestStatus.HTTP_ERROR
            else:
                await _read_stream(resp, clock, token_counter, state)
                if state.malformed:
                    error = state.error
                    status = RequestStatus.HTTP_ERROR
    except httpx.TimeoutException:
        status = RequestStatus.TIMEOUT
        error = error or f"timed out after {timeout_s}s"
    except httpx.TransportError as e:
        status = RequestStatus.CONN_ERROR
        error = f"{type(e).__name__}: {e}"

    if status is RequestStatus.OK:
        status, error = _final_status(error, state, caps, spec)

    return RequestRecord(
        request_id=spec.request_id,
        workload_class=spec.workload_class,
        priority=spec.priority,
        prompt_tokens_req=spec.prompt_tokens,
        output_tokens_req=spec.output_tokens,
        output_tokens_est=spec.output_tokens_est,
        slo_ttft_ms=spec.slo_ttft_ms,
        slo_tpot_ms=spec.slo_tpot_ms,
        status=status,
        t_arrival=t_arrival,
        t_dispatch=sent_at,
        t_first_token=state.t_first_token,
        t_last_token=state.t_last_token,
        t_complete=clock(),
        chunks=tuple(state.chunks),
        prompt_tokens_usage=_usage_int(state.usage, "prompt_tokens"),
        output_tokens_usage=_usage_int(state.usage, "completion_tokens"),
        http_status=state.http_status,
        error=error,
    )


async def _read_stream(
    resp: httpx.Response,
    clock: Clock,
    token_counter: TokenCounter | None,
    state: _State,
) -> None:
    """Drain the SSE body into `state`. httpx transport errors propagate to the caller."""
    parser = SseParser()
    malformed = False
    async for raw in resp.aiter_bytes():
        for event in parser.feed(raw):
            if not _absorb(event.data, clock, token_counter, state):
                malformed = True
                break
        if malformed:
            break
    else:
        # Only reached when the body ended without a malformed event.
        for event in parser.close():
            if event.data != "[DONE]" and state.error is None:
                state.error = "stream ended with a partial SSE event"
    state.malformed = malformed


def _absorb(
    data: str,
    clock: Clock,
    token_counter: TokenCounter | None,
    state: _State,
) -> bool:
    """Fold one SSE event's payload into `state`. Returns False if the payload was malformed."""
    if data == "[DONE]":
        state.done = True
        return True
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        state.error = f"invalid JSON in SSE event: {data[:120]}"
        return False
    if not isinstance(obj, dict):
        state.error = f"unexpected SSE payload type: {type(obj).__name__}"
        return False
    if obj.get("usage"):
        state.usage = obj["usage"]
    choices = obj.get("choices") or []
    if not choices:
        return True
    text = choices[0].get("text") or ""
    n = _count_tokens(text, token_counter)
    if n <= 0:
        return True
    now = clock()
    state.chunks.append(Chunk(t_ns=now, n_tokens=n))
    if state.t_first_token is None:
        state.t_first_token = now
    state.t_last_token = now
    return True


def _final_status(
    error: str | None,
    state: _State,
    caps: Capabilities,
    spec: RequestSpec,
) -> tuple[RequestStatus, str | None]:
    if state.malformed:
        return RequestStatus.HTTP_ERROR, error
    if not state.done:
        return RequestStatus.HTTP_ERROR, error or "stream ended without [DONE]"
    if error is not None:
        return RequestStatus.HTTP_ERROR, error
    if caps.usage_block and state.usage is None:
        return RequestStatus.HTTP_ERROR, "stream completed without a usage block"
    prompt_usage = _usage_int(state.usage, "prompt_tokens")
    output_usage = _usage_int(state.usage, "completion_tokens")
    if caps.token_id_prompts and prompt_usage != spec.prompt_tokens:
        return (
            RequestStatus.LENGTH_MISMATCH,
            f"prompt tokens: requested {spec.prompt_tokens}, usage {prompt_usage}",
        )
    if caps.forced_output_length and output_usage != spec.output_tokens:
        return (
            RequestStatus.LENGTH_MISMATCH,
            f"output tokens: requested {spec.output_tokens}, usage {output_usage}",
        )
    return RequestStatus.OK, error
