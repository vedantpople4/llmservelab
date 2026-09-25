from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_time: float
    prompt_tokens: int
    max_output_tokens: int
    priority: int = 0


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
        """Pick the next request to admit, or None to hold admission."""
