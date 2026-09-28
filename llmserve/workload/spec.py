"""Request specifications — the frozen per-request input to a run (ADR-004).

A `RequestSpec` is fully determined before any request is sent: id, class, exact token counts,
the materialized prompt token IDs and the class's priority/SLOs. The workload materializer
(Phase 2) produces these from a config and a seed; the streaming client consumes them.

`prompt_ids` is stored as a tuple so a spec cannot be mutated after a workload file is written.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmserve.scheduler.base import Priority


@dataclass(frozen=True, kw_only=True)
class RequestSpec:
    request_id: str
    workload_class: str
    prompt_tokens: int
    output_tokens: int
    prompt_ids: tuple[int, ...]
    # The same prompt as text, for backends that cannot take token IDs (ADR-009). None when the
    # run only targets token-ID backends, so the text never has to be materialized.
    prompt_text: str | None = None

    # Seconds after run start (open loop); 0 for the closed-loop Phase 1 checks.
    arrival_offset_s: float = 0.0
    priority: int = Priority.MEDIUM
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None
    # The scheduler's view of `output_tokens`; None until an estimator is configured.
    output_tokens_est: int | None = None

    def __post_init__(self) -> None:
        if len(self.prompt_ids) != self.prompt_tokens:
            raise ValueError(
                f"{self.request_id}: prompt_tokens={self.prompt_tokens} but "
                f"len(prompt_ids)={len(self.prompt_ids)}"
            )
        if self.prompt_tokens < 1:
            raise ValueError(f"{self.request_id}: prompt_tokens must be >= 1")
        if self.output_tokens < 1:
            raise ValueError(f"{self.request_id}: output_tokens must be >= 1")


Workload = list[RequestSpec]
