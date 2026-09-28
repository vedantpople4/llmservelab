"""The mock backend's delay model (ADR-009).

Phase 1 ships a parameterised delay model, not a simulator: TTFT grows with prompt length and
per-token latency grows with the number of requests in flight. That is enough for the streaming
client, the arrival process and the smoke check to be developed and tested on a laptop, and it
reproduces the queueing shape the harness has to measure without pretending to be an engine.

The continuous-batching simulator (per-step token budget, chunked prefill, KV cap, preemption)
arrives in Phase 2 and replaces these `predict_*` calls with a real simulation; the server
interface above it does not change.

Mock numbers are never real numbers: runs against this backend are labelled `server.kind: mock`
in metadata and may not appear in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DelayModel:
    """Linear toy model of prefill and decode cost.

    `in_flight` enters the decode term only, which is the qualitative behaviour that makes
    head-of-line blocking and concurrency effects visible to the load driver.
    """

    ttft_base_s: float = 0.02
    ttft_per_prompt_token_s: float = 8e-5
    token_base_s: float = 0.004
    token_in_flight_s: float = 0.001

    @classmethod
    def instant(cls) -> DelayModel:
        """Zero delays: for unit tests and harness-overhead measurements."""
        return cls(
            ttft_base_s=0.0,
            ttft_per_prompt_token_s=0.0,
            token_base_s=0.0,
            token_in_flight_s=0.0,
        )

    def ttft_s(self, prompt_tokens: int, in_flight: int) -> float:
        del in_flight  # prefill is compute-bound; queueing shows up in the client's own timestamps
        return self.ttft_base_s + self.ttft_per_prompt_token_s * prompt_tokens

    def token_s(self, in_flight: int) -> float:
        return self.token_base_s + self.token_in_flight_s * in_flight


class MockEngine:
    """Holds the model identity, the delay model, and the in-flight counter."""

    def __init__(self, model: str = "mock", delay: DelayModel | None = None) -> None:
        self.model = model
        self.delay = delay if delay is not None else DelayModel()
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def acquire(self) -> None:
        # Single event loop, and this runs between awaits, so a plain int is safe here.
        self._in_flight += 1

    def release(self) -> None:
        self._in_flight -= 1

    @staticmethod
    def prompt_tokens(prompt: list[int] | str) -> int:
        """Prompt length: exact for token IDs, an estimate for text (the harness sends IDs)."""
        if isinstance(prompt, str):
            return max(1, len(prompt) // 4)
        return len(prompt)
