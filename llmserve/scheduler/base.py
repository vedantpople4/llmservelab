from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    """Service classes (PRD §12). Lower value = more urgent, matching vLLM's priority policy."""

    HIGH = 0
    MEDIUM = 1
    LOW = 2


@dataclass(frozen=True, kw_only=True)
class Request:
    """A waiting request as the scheduler sees it. Times are seconds since run start."""

    request_id: str
    arrival_time: float
    prompt_tokens: int
    # The requested output length. With forced output length this is also the true length, so
    # schedulers must not read it directly; they use `output_tokens_est` (the oracle ablation
    # sets the estimate equal to this value).
    max_output_tokens: int
    output_tokens_est: int | None = None
    priority: int = Priority.MEDIUM
    workload_class: str = "default"
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None


@dataclass(frozen=True)
class SystemState:
    now: float
    queue_depth: int
    in_flight: int
    gpu_utilization: float | None = None
    kv_cache_utilization: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


class Scheduler(ABC):
    name: str

    @abstractmethod
    def select_next(self, waiting: Sequence[Request], state: SystemState) -> Request | None:
        """Pick the next request to admit from `waiting`, or None to hold admission."""
