"""The mock backend's two engines (ADR-009).

`MockEngine` (Phase 1, `--engine delay`) is a parameterised delay model: TTFT grows with prompt
length and per-token latency grows with the number of requests in flight. That reproduces the
queueing shape the harness measures without pretending to be an engine.

`ContinuousBatchEngine` (Phase 2 item 4, `--engine cb`) is a discrete-time simulator of a
continuous-batching engine: a per-step token budget with chunked prefill, step time
`alpha + beta*prefill_tokens + gamma*decode_seqs + delta*sum(context)`, a KV cap with preemption
(recompute), and vLLM-style counters. It shows queueing, head-of-line blocking and KV pressure
qualitatively — enough to develop schedulers on a laptop.

The server above both sees one interface: `stream_tokens(prompt_tokens, n)` yields one bool per
generated token (True on the last), plus `metrics()`, `check_request()` and `aclose()`.

Mock numbers are never real numbers: runs against this backend are labelled `server.kind: mock`
in metadata and may not appear in the paper.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

DEFAULT_KV_CAPACITY_TOKENS = 16384


def count_prompt_tokens(prompt: list[int] | str) -> int:
    """Prompt length: exact for token IDs, an estimate for text (the harness sends IDs)."""
    if isinstance(prompt, str):
        return max(1, len(prompt) // 4)
    return len(prompt)


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
    """Delay-model engine: the model identity, the delay model, and the in-flight counter."""

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

    async def stream_tokens(self, prompt_tokens: int, n: int) -> AsyncGenerator[bool, None]:
        """Yield one bool per generated token (True on the last).

        The delay model runs inside the stream: prefill pays the TTFT, each decode token pays
        `token_s(in_flight)`, and the in-flight slot is released when the consumer closes the
        generator — a client disconnect included.
        """
        self.acquire()
        try:
            delay = self.delay
            # Prefill, then decode: the first token pays the whole TTFT.
            await asyncio.sleep(delay.ttft_s(prompt_tokens, self._in_flight))
            for i in range(n):
                if i:
                    await asyncio.sleep(delay.token_s(self._in_flight))
                yield i == n - 1
        finally:
            self.release()

    def check_request(self, prompt_tokens: int, n: int) -> str | None:
        """No KV cap in the delay model: every request is admissible."""
        del prompt_tokens, n
        return None

    def metrics(self) -> dict[str, float]:
        """The four engine-side series the sampler reads.

        The delay model has no internal queue and no KV, so those two are structurally zero —
        its "running" is the in-flight counter.
        """
        return {
            "running": float(self._in_flight),
            "waiting": 0.0,
            "kv_usage": 0.0,
            "preemptions_total": 0.0,
        }

    async def aclose(self) -> None:
        """Nothing to release: the delay model has no background loop."""


@dataclass(frozen=True)
class CbParams:
    """Cost model and capacity of the simulated engine (plan Phase 2 item 4)."""

    alpha_s: float = 5e-4
    """Fixed cost of one step."""
    beta_s: float = 2.5e-4
    """Cost per prefill token processed in the step."""
    gamma_s: float = 4e-4
    """Cost per sequence decoded in the step."""
    delta_s: float = 1e-7
    """Cost per resident context token across the batch."""
    max_num_batched_tokens: int = 2048
    """Per-step token budget: decode first, chunked prefill takes what is left."""
    prefill_chunk: int = 512
    """Largest prefill slice one sequence may claim per step (`--chunked-prefill-size`)."""
    kv_capacity_tokens: int = DEFAULT_KV_CAPACITY_TOKENS
    """Total KV blocks; admissions and decodes must fit inside it (`--gpu-memory-utilization`)."""

    @classmethod
    def instant(cls) -> CbParams:
        """Zero step time: for unit tests (the capacities stay real)."""
        return cls(alpha_s=0.0, beta_s=0.0, gamma_s=0.0, delta_s=0.0)


@dataclass(eq=False)
class _Seq:
    """One request inside the simulator; identity comparison (requests are never equal)."""

    prompt: int
    n: int
    generated: int = 0
    prefill_progress: int = 0
    prefill_target: int = 0
    finished: bool = False
    queue: asyncio.Queue[bool] = field(default_factory=asyncio.Queue)

    @property
    def footprint(self) -> int:
        """KV tokens held: the prompt plus every token generated so far."""
        return self.prompt + self.generated


@dataclass
class _Plan:
    """One step, decided before it runs: what computes, what preempts, and how long it takes."""

    dt: float
    decode: list[_Seq]
    prefill: list[tuple[_Seq, int]]
    victims: list[_Seq]


class ContinuousBatchEngine:
    """Discrete-time continuous batching: budget, chunked prefill, KV cap, preemption.

    A background task steps the simulation while `stream_tokens` consumers wait on per-request
    queues. The task starts on the first submit and exits when the engine goes idle; `aclose`
    stops it at server shutdown.
    """

    def __init__(self, model: str = "mock", params: CbParams | None = None) -> None:
        self.model = model
        self.params = params if params is not None else CbParams()
        self.preemptions_total = 0
        self._running: list[_Seq] = []  # admitted, oldest first (FCFS)
        self._waiting: deque[_Seq] = deque()
        self._kv_used = 0
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def kv_capacity(self) -> int:
        return self.params.kv_capacity_tokens

    async def stream_tokens(self, prompt_tokens: int, n: int) -> AsyncGenerator[bool, None]:
        """Queue a request, then yield its tokens as the simulation produces them."""
        bad = self.check_request(prompt_tokens, n)
        if bad:
            raise ValueError(bad)  # the server answers 400 before it ever gets here
        seq = _Seq(prompt=prompt_tokens, n=n)
        self._waiting.append(seq)
        self._ensure_loop()
        try:
            while True:
                is_last = await seq.queue.get()
                yield is_last
                if is_last:
                    return
        finally:
            self._discard(seq)  # normal end or client disconnect: leave the engine clean

    def check_request(self, prompt_tokens: int, n: int) -> str | None:
        """Reject a request whose worst-case KV footprint cannot fit alone (the server 400s)."""
        if prompt_tokens + n > self.kv_capacity:
            return (
                f"prompt + max_tokens = {prompt_tokens + n} exceeds the mock engine's KV "
                f"capacity of {self.kv_capacity} tokens"
            )
        return None

    def metrics(self) -> dict[str, float]:
        capacity = self.kv_capacity
        return {
            "running": float(len(self._running)),
            "waiting": float(len(self._waiting)),
            "kv_usage": self._kv_used / capacity if capacity else 0.0,
            "preemptions_total": float(self.preemptions_total),
        }

    async def aclose(self) -> None:
        """Stop the step loop; in-flight streams are torn down by their consumers."""
        self._closed = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _ensure_loop(self) -> None:
        if not self._closed and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while not self._closed:
                self._admit()
                if not self._running and not self._waiting:
                    return  # idle: the next submit restarts the loop
                plan = self._plan()
                await asyncio.sleep(plan.dt)  # simulated compute (0 still yields to consumers)
                self._commit(plan)
        finally:
            self._task = None

    def _admit(self) -> None:
        """Admit waiting sequences while KV fits them, oldest first (a fat head blocks the tail)."""
        free = self.kv_capacity - self._kv_used
        while self._waiting:
            seq = self._waiting[0]
            need = seq.footprint  # a preempted sequence re-prefills prompt + generated so far
            if need > free:
                break
            self._waiting.popleft()
            seq.prefill_target = need
            seq.prefill_progress = 0
            self._running.append(seq)
            self._kv_used += need
            free -= need

    def _plan(self) -> _Plan:
        p = self.params
        # Decode first: every sequence whose prefill is done emits one token this step.
        decode = [seq for seq in self._running if seq.prefill_progress >= seq.prefill_target]
        # KV: each emission needs a free token; make room by preempting the newest sequences.
        victims: list[_Seq] = []
        free = self.kv_capacity - self._kv_used
        for victim in reversed(self._running):
            if free >= len(decode):
                break
            victims.append(victim)
            free += victim.footprint
            if victim in decode:
                decode.remove(victim)
        # Prefill with the budget the decode batch leaves, chunked, oldest sequence first.
        budget = max(0, p.max_num_batched_tokens - len(decode))
        prefill: list[tuple[_Seq, int]] = []
        for seq in self._running:
            if budget <= 0:
                break
            if seq in victims or seq.prefill_progress >= seq.prefill_target:
                continue
            take = min(p.prefill_chunk, seq.prefill_target - seq.prefill_progress, budget)
            prefill.append((seq, take))
            budget -= take
        participants = set(decode) | {seq for seq, _ in prefill}
        dt = (
            p.alpha_s
            + p.beta_s * sum(t for _, t in prefill)
            + p.gamma_s * len(decode)
            + p.delta_s * sum(seq.footprint for seq in participants)
        )
        return _Plan(dt=dt, decode=decode, prefill=prefill, victims=victims)

    def _commit(self, plan: _Plan) -> None:
        for victim in plan.victims:
            self._running.remove(victim)
            self._kv_used -= victim.footprint
            victim.prefill_progress = 0
            victim.prefill_target = victim.footprint  # recompute on the next admission
            self._waiting.appendleft(victim)
            self.preemptions_total += 1
        for seq, tokens in plan.prefill:
            seq.prefill_progress += tokens
            if seq.prefill_progress >= seq.prefill_target and seq.generated == 0:
                self._emit(seq)  # prefill completion produces the first token
        for seq in plan.decode:
            self._emit(seq)

    def _emit(self, seq: _Seq) -> None:
        seq.generated += 1
        self._kv_used += 1
        done = seq.generated >= seq.n
        if done:
            self._finish(seq)
        seq.queue.put_nowait(done)

    def _finish(self, seq: _Seq) -> None:
        seq.finished = True
        self._running.remove(seq)
        self._kv_used -= seq.footprint

    def _discard(self, seq: _Seq) -> None:
        if seq.finished:
            return
        if seq in self._waiting:
            self._waiting.remove(seq)
        elif seq in self._running:
            self._running.remove(seq)
            self._kv_used -= seq.footprint
