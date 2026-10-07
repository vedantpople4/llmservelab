"""The gateway core: an asyncio admission controller that runs in the benchmark process
(ADR-0002). No HTTP dependency and no clock of its own — the driver submits
`(spec, arrival_ns)` pairs, every timestamp comes from the run clock, and a thin FastAPI
wrapper (a later phase) can expose the same core as a proxy.

The dispatch loop admits waiting requests under `max_in_flight` through the configured
scheduler. It wakes on enqueue, on completion, **and on a 5 ms timer**: a scheduler may hold
admission by returning `None`, and when it does, nothing else will ever wake the loop, so the
timer is what makes a hold end.

Records leave the gateway carrying their dispatch context (PRD §26): scheduler name, decision
time, and the queue depth / in-flight count the decision was made on. With the gateway off,
those fields stay None — the driver sent the request directly.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from llmserve.metrics.records import RequestRecord
from llmserve.runner.clock import Clock
from llmserve.scheduler.base import Request, Scheduler, SystemState
from llmserve.workload.spec import RequestSpec

DISPATCH_TICK_S = 0.005

Sender = Callable[[RequestSpec, int], Awaitable[RequestRecord]]
"""`(spec, arrival_ns) -> record` — the driver's send path, direct or through the gateway."""


@dataclass(frozen=True, kw_only=True)
class _Entry:
    spec: RequestSpec
    arrival_ns: int
    future: asyncio.Future[RequestRecord]


class Gateway:
    def __init__(
        self,
        *,
        max_in_flight: int,
        scheduler: Scheduler,
        clock: Clock,
        send: Sender,
    ) -> None:
        self._max_in_flight = max_in_flight
        self._scheduler = scheduler
        self._clock = clock
        self._send = send
        self._waiting: dict[Request, _Entry] = {}
        self._in_flight = 0
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._admits: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> Gateway:
        self._loop_task = asyncio.create_task(self._dispatch_loop())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        assert self._loop_task is not None, "gateway used without its async context manager"
        self._loop_task.cancel()
        try:
            await self._loop_task
        except asyncio.CancelledError:
            pass
        if self._admits:
            await asyncio.gather(*self._admits, return_exceptions=True)

    async def submit(self, spec: RequestSpec, arrival_ns: int) -> RequestRecord:
        """Enqueue a request and wait for its completion record."""
        future: asyncio.Future[RequestRecord] = asyncio.get_running_loop().create_future()
        waiting_on = Request(
            request_id=spec.request_id,
            arrival_time=arrival_ns / 1e9,
            prompt_tokens=spec.prompt_tokens,
            max_output_tokens=spec.output_tokens,
            output_tokens_est=spec.output_tokens_est,
            priority=spec.priority,
            workload_class=spec.workload_class,
            slo_ttft_ms=spec.slo_ttft_ms,
            slo_tpot_ms=spec.slo_tpot_ms,
        )
        self._waiting[waiting_on] = _Entry(spec=spec, arrival_ns=arrival_ns, future=future)
        self._wake.set()
        return await future

    async def _dispatch_loop(self) -> None:
        while True:
            # Clear before draining: an enqueue or completion during the drain then sets the
            # event again, so the wait below returns immediately instead of losing the wake.
            self._wake.clear()
            self._admit_ready()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=DISPATCH_TICK_S)
            except TimeoutError:
                pass

    def _admit_ready(self) -> None:
        while self._in_flight < self._max_in_flight and self._waiting:
            state = SystemState(
                now=self._clock() / 1e9,
                queue_depth=len(self._waiting),
                in_flight=self._in_flight,
            )
            decided_at = self._clock()
            choice = self._scheduler.select_next(list(self._waiting), state)
            decision_ns = self._clock() - decided_at
            if choice is None:
                return  # held: the 5 ms timer re-polls
            entry = self._waiting.pop(choice)
            self._in_flight += 1
            task = asyncio.create_task(self._admit(entry, state, decision_ns))
            self._admits.add(task)
            task.add_done_callback(self._admits.discard)

    async def _admit(self, entry: _Entry, state: SystemState, decision_ns: int) -> None:
        try:
            record = await self._send(entry.spec, entry.arrival_ns)
            entry.future.set_result(
                dataclasses.replace(
                    record,
                    sched_name=self._scheduler.name,
                    sched_decision_ns=decision_ns,
                    queue_depth_at_dispatch=state.queue_depth,
                    in_flight_at_dispatch=state.in_flight,
                )
            )
        except Exception as e:
            entry.future.set_exception(e)
        finally:
            self._in_flight -= 1
            self._wake.set()
