"""Per-request measurement records and their Parquet schemas.

Timestamps are integer nanoseconds on the benchmark process's monotonic clock
(`time.perf_counter_ns()`), relative to the run's start. Wall-clock time appears only in run
metadata, as the clock anchor (ADR-006).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pyarrow as pa


class RequestStatus(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    CONN_ERROR = "conn_error"
    ABORTED = "aborted"
    # The server generated a different number of tokens than requested. The timings are real,
    # but the request did not match its spec, so it is excluded from latency statistics.
    LENGTH_MISMATCH = "length_mismatch"


@dataclass(frozen=True)
class Chunk:
    """One streamed SSE chunk. A chunk can carry more than one token."""

    t_ns: int
    n_tokens: int


@dataclass(frozen=True, kw_only=True)
class RequestRecord:
    request_id: str
    workload_class: str
    priority: int
    prompt_tokens_req: int
    output_tokens_req: int
    status: RequestStatus
    t_arrival: int

    # What the scheduler believed the output length would be; None means no estimate was used.
    output_tokens_est: int | None = None
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None

    t_dispatch: int | None = None
    t_first_token: int | None = None
    t_last_token: int | None = None
    t_complete: int | None = None
    chunks: tuple[Chunk, ...] = ()

    # Ground truth from the server's `usage` block.
    prompt_tokens_usage: int | None = None
    output_tokens_usage: int | None = None

    http_status: int | None = None
    error: str | None = None

    # Scheduler decision context, captured at dispatch (PRD §26). None when the gateway is off.
    sched_name: str | None = None
    sched_score: float | None = None
    sched_decision_ns: int | None = None
    queue_depth_at_dispatch: int | None = None
    in_flight_at_dispatch: int | None = None
    kv_usage_at_dispatch: float | None = None
    bypass_count: int | None = None

    extra: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is RequestStatus.OK

    def to_row(self) -> dict[str, Any]:
        """Flat row for `REQUEST_SCHEMA`. Chunks go to their own table; see `chunk_rows`."""
        row: dict[str, Any] = {}
        for name in REQUEST_SCHEMA.names:
            if name in ("run_id", "rep"):
                continue
            if name == "status":
                row[name] = self.status.value
            elif name == "n_chunks":
                row[name] = len(self.chunks)
            else:
                row[name] = getattr(self, name)
        return row

    def chunk_rows(self) -> list[dict[str, Any]]:
        return [
            {"request_id": self.request_id, "idx": i, "t_ns": c.t_ns, "n_tokens": c.n_tokens}
            for i, c in enumerate(self.chunks)
        ]


REQUEST_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("rep", pa.int32()),
        ("request_id", pa.string()),
        ("workload_class", pa.string()),
        ("priority", pa.int32()),
        ("prompt_tokens_req", pa.int32()),
        ("output_tokens_req", pa.int32()),
        ("output_tokens_est", pa.int32()),
        ("slo_ttft_ms", pa.float64()),
        ("slo_tpot_ms", pa.float64()),
        ("status", pa.string()),
        ("t_arrival", pa.int64()),
        ("t_dispatch", pa.int64()),
        ("t_first_token", pa.int64()),
        ("t_last_token", pa.int64()),
        ("t_complete", pa.int64()),
        ("n_chunks", pa.int32()),
        ("prompt_tokens_usage", pa.int32()),
        ("output_tokens_usage", pa.int32()),
        ("http_status", pa.int32()),
        ("error", pa.string()),
        ("sched_name", pa.string()),
        ("sched_score", pa.float64()),
        ("sched_decision_ns", pa.int64()),
        ("queue_depth_at_dispatch", pa.int32()),
        ("in_flight_at_dispatch", pa.int32()),
        ("kv_usage_at_dispatch", pa.float64()),
        ("bypass_count", pa.int32()),
    ]
)

CHUNK_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("idx", pa.int32()),
        ("t_ns", pa.int64()),
        ("n_tokens", pa.int32()),
    ]
)
