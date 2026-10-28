"""Scrape a backend's Prometheus `/metrics` for server-side state (plan Phase 2 item 7).

Series names differ across versions (`gpu_cache_usage_perc` vs `kv_cache_usage_perc`), so every
sampled field maps to an ordered candidate list. A missing series is a loud `SamplerError` —
a sampler that silently collected nothing would make a run look healthy while its server
metrics are absent. The exception is `OPTIONAL`: the running totals of vLLM's per-request
queue, prefill and decode time histograms and of its tokens-per-step histogram (Phase 4). They
are None when absent (the mock has none). The difference between two samples gives the mean
over the requests finished, or the engine steps run, in between.

The sampler scrapes once at `start()` (so the failure happens before any measured request),
then every `1/hz` seconds until `stop()`. A failed mid-run scrape skips that tick and counts in
`.failures`; only `MAX_CONSECUTIVE_FAILURES` in a row stop the loop, with the reason in `.error`,
and the caller then deactivates the sampler for the remaining repetitions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import pyarrow as pa

SERVER_SCHEMA = pa.schema(
    [
        ("t_ns", pa.int64()),
        ("running", pa.float64()),
        ("waiting", pa.float64()),
        ("kv_usage", pa.float64()),
        ("preemptions_total", pa.float64()),
        ("prompt_tokens_total", pa.float64()),
        ("generation_tokens_total", pa.float64()),
        ("finished_total", pa.float64()),
        ("queue_time_s_total", pa.float64()),
        ("prefill_time_s_total", pa.float64()),
        ("decode_time_s_total", pa.float64()),
        ("steps_total", pa.float64()),
        ("step_tokens_total", pa.float64()),
    ]
)

CANDIDATES: dict[str, tuple[str, ...]] = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_usage": ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"),
    "preemptions_total": ("vllm:num_preemptions_total", "vllm:preemptions_total"),
    "prompt_tokens_total": ("vllm:prompt_tokens_total",),
    "generation_tokens_total": ("vllm:generation_tokens_total",),
}
OPTIONAL: dict[str, tuple[str, ...]] = {
    "finished_total": ("vllm:request_queue_time_seconds_count",),
    "queue_time_s_total": ("vllm:request_queue_time_seconds_sum",),
    "prefill_time_s_total": ("vllm:request_prefill_time_seconds_sum",),
    "decode_time_s_total": ("vllm:request_decode_time_seconds_sum",),
    "steps_total": ("vllm:iteration_tokens_total_count",),
    "step_tokens_total": ("vllm:iteration_tokens_total_sum",),
}


MAX_CONSECUTIVE_FAILURES = 10


class SamplerError(RuntimeError):
    """A scrape produced no usable samples; the sampler cannot run."""


def describe(e: BaseException) -> str:
    """A failure message that is never empty: `httpx.PoolTimeout`'s str() often is."""
    if isinstance(e, SamplerError):
        return str(e)
    text = str(e)
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


@dataclass(frozen=True)
class ServerSample:
    t_ns: int
    running: float
    waiting: float
    kv_usage: float
    preemptions_total: float
    prompt_tokens_total: float
    generation_tokens_total: float
    finished_total: float | None = None
    queue_time_s_total: float | None = None
    prefill_time_s_total: float | None = None
    decode_time_s_total: float | None = None
    steps_total: float | None = None
    step_tokens_total: float | None = None


def parse_prometheus(text: str) -> dict[str, float]:
    """`name{labels} value` lines to a name->value map; comments are skipped, labels dropped.

    When several series share a name (one per model), the last one wins — the harness drives
    exactly one model per run.
    """
    series: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            series[parts[0].split("{", 1)[0]] = float(parts[-1])
        except ValueError:
            continue
    return series


def lookup(series: Mapping[str, float], field: str) -> float | None:
    """The first candidate series present for `field`, or None when the backend has none."""
    for name in {**CANDIDATES, **OPTIONAL}[field]:
        if name in series:
            return series[name]
    return None


def sample_from(series: Mapping[str, float], *, t_ns: int) -> ServerSample:
    """Build one row, or raise `SamplerError` naming the missing field and its candidates."""
    values: dict[str, float] = {}
    for field in CANDIDATES:
        value = lookup(series, field)
        if value is None:
            raise SamplerError(f"missing series for {field!r} (tried {list(CANDIDATES[field])})")
        values[field] = value
    optional: dict[str, Any] = {field: lookup(series, field) for field in OPTIONAL}
    return ServerSample(t_ns=t_ns, **values, **optional)


class ServerSampler:
    """Scrape `GET {base_url}/metrics` at `hz` until `stop()` flushes the samples."""

    def __init__(
        self,
        clock: Callable[[], int],
        *,
        hz: float,
        base_url: str,
        client: httpx.AsyncClient,
    ) -> None:
        self._clock = clock
        self._hz = hz
        self._base = base_url
        self._client = client
        self._samples: list[ServerSample] = []
        self._task: asyncio.Task[None] | None = None
        self.error: str | None = None
        self.failures = 0

    async def scrape(self) -> ServerSample:
        try:
            resp = await self._client.get(f"{self._base}/metrics")
        except httpx.HTTPError as e:
            raise SamplerError(f"GET /metrics failed: {describe(e)}") from e
        if resp.status_code != 200:
            raise SamplerError(f"GET /metrics -> HTTP {resp.status_code}")
        return sample_from(parse_prometheus(resp.text), t_ns=self._clock())

    async def start(self) -> None:
        """Scrape once now so a missing series fails before the first measured request."""
        if self._hz <= 0:
            raise SamplerError("server_hz is 0; the sampler should not have been started")
        self._samples = [await self.scrape()]
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        interval = 1.0 / self._hz
        consecutive = 0
        while True:
            await asyncio.sleep(interval)
            try:
                sample = await self.scrape()
            except Exception as e:
                self.failures += 1
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    self.error = f"{consecutive} consecutive scrapes failed, last: {describe(e)}"
                    return
                continue
            consecutive = 0
            self._samples.append(sample)

    async def stop(self) -> list[ServerSample]:
        """Stop scraping and return the collected rows (in scrape order)."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return list(self._samples)
