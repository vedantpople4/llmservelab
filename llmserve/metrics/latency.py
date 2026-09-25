"""Latency metrics derived from a `RequestRecord` (PRD §8).

All functions return seconds, or None when the metric is undefined for that request. Latency
metrics are defined only for successful requests; failures are counted separately so they are
never silently mixed into percentiles.

TTFT is measured from *arrival*, so it includes time spent in the gateway queue. This is the
latency a user sees. `ttft_server` measures from dispatch and isolates the engine's share.
"""

from __future__ import annotations

from llmserve.metrics.records import RequestRecord

_NS = 1e9


def _span(start: int | None, end: int | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) / _NS


def queue_wait(r: RequestRecord) -> float | None:
    """Time from arrival to dispatch. Defined for failed requests too, if they were dispatched."""
    return _span(r.t_arrival, r.t_dispatch)


def ttft(r: RequestRecord) -> float | None:
    return _span(r.t_arrival, r.t_first_token) if r.ok else None


def ttft_server(r: RequestRecord) -> float | None:
    return _span(r.t_dispatch, r.t_first_token) if r.ok else None


def e2e(r: RequestRecord) -> float | None:
    return _span(r.t_arrival, r.t_last_token) if r.ok else None


def tpot(r: RequestRecord) -> float | None:
    """(last token − first token) / (output_tokens − 1), using the server's token count.

    Chunk count is not token count: engines may stream several tokens per chunk.
    """
    n = r.output_tokens_usage
    if not r.ok or n is None or n < 2:
        return None
    decode = _span(r.t_first_token, r.t_last_token)
    return None if decode is None else decode / (n - 1)


def inter_token_latencies(r: RequestRecord) -> list[float]:
    """Per-token gaps after the first token.

    A chunk carrying k tokens that arrives Δt after the previous chunk counts as k gaps of Δt/k.
    Extra tokens in the *first* chunk have no observable gap and are skipped. With one token per
    chunk, the gaps sum to `t_last_token − t_first_token`, which matches TPOT.
    """
    if not r.ok:
        return []
    gaps: list[float] = []
    for prev, cur in zip(r.chunks, r.chunks[1:], strict=False):
        if cur.n_tokens <= 0:
            continue
        gap = (cur.t_ns - prev.t_ns) / _NS / cur.n_tokens
        gaps.extend([gap] * cur.n_tokens)
    return gaps
