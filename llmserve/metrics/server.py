"""Scrape a backend's Prometheus `/metrics` for server-side state (plan Phase 2 item 7).

Series names differ across versions (`gpu_cache_usage_perc` vs `kv_cache_usage_perc`), so every
sampled field maps to an ordered candidate list. A missing series is a loud `SamplerError` —
a sampler that silently collected nothing would make a run look healthy while its server
metrics are absent.

The sampler scrapes once at `start()` (so the failure happens before any measured request),
then every `1/hz` seconds until `stop()`. Mid-run failures land in `.error` and the caller
deactivates the sampler for the remaining repetitions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass

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


class SamplerError(RuntimeError):
    """A scrape produced no usable samples; the sampler cannot run."""


@dataclass(frozen=True)
class ServerSample:
    t_ns: int
    running: float
    waiting: float
    kv_usage: float
    preemptions_total: float
    prompt_tokens_total: float
    generation_tokens_total: float


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
    for name in CANDIDATES[field]:
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
    return ServerSample(t_ns=t_ns, **values)


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

    async def scrape(self) -> ServerSample:
        try:
            resp = await self._client.get(f"{self._base}/metrics")
        except httpx.HTTPError as e:
            raise SamplerError(f"GET /metrics failed: {e}") from e
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
        while True:
            await asyncio.sleep(interval)
            try:
                self._samples.append(await self.scrape())
            except Exception as e:  # SamplerError or transport failure: stop, let `.error` tell
                self.error = str(e)
                return

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
