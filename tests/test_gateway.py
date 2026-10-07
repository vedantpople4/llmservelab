"""Tests for the gateway core (plan §Phase 2 item 6).

The plan's three properties: in-flight never exceeds the cap (hypothesis), held requests are
eventually dispatched (the 5 ms timer), and the scheduler is only offered requests that are
actually waiting — plus the PRD §26 dispatch context on every record.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from hypothesis import given, settings
from hypothesis import strategies as st

from llmserve.gateway.core import Gateway, Sender
from llmserve.metrics.records import RequestRecord, RequestStatus
from llmserve.runner.clock import RunClock
from llmserve.scheduler.base import Request, SystemState
from llmserve.scheduler.fifo import FIFOScheduler
from llmserve.workload.spec import RequestSpec


def spec(i: int) -> RequestSpec:
    return RequestSpec(
        request_id=f"r{i:05d}",
        workload_class="c",
        prompt_tokens=8,
        output_tokens=4,
        prompt_ids=tuple(range(100, 108)),
    )


def ok_record(spec: RequestSpec, t_arrival: int) -> RequestRecord:
    return RequestRecord(
        request_id=spec.request_id,
        workload_class=spec.workload_class,
        priority=spec.priority,
        prompt_tokens_req=spec.prompt_tokens,
        output_tokens_req=spec.output_tokens,
        status=RequestStatus.OK,
        t_arrival=t_arrival,
        t_first_token=t_arrival + 1,
        t_last_token=t_arrival + 2,
    )


def make_sender() -> tuple[Sender, dict[str, int]]:
    state = {"in_flight": 0, "peak": 0, "sent": 0}

    async def send(spec: RequestSpec, arrival_ns: int) -> RequestRecord:
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0)  # yield, so concurrent admits can accumulate
        state["in_flight"] -= 1
        state["sent"] += 1
        return ok_record(spec, arrival_ns)

    return send, state


class RecordingFIFO(FIFOScheduler):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def select_next(self, waiting: Sequence[Request], state: SystemState) -> Request | None:
        self.calls.append([r.request_id for r in waiting])
        return super().select_next(waiting, state)


class HoldingScheduler(FIFOScheduler):
    """Returns None (holds) for the first `holds` calls that see a waiting request."""

    def __init__(self, holds: int) -> None:
        self._left = holds
        self.calls = 0

    def select_next(self, waiting: Sequence[Request], state: SystemState) -> Request | None:
        if waiting:
            self.calls += 1
            if self._left > 0:
                self._left -= 1
                return None
        return super().select_next(waiting, state)


@given(max_in_flight=st.integers(1, 6), n=st.integers(1, 24))
@settings(max_examples=30, deadline=None)
def test_property_in_flight_never_exceeds_the_cap(max_in_flight: int, n: int) -> None:
    async def run() -> None:
        send, state = make_sender()
        gateway = Gateway(
            max_in_flight=max_in_flight,
            scheduler=FIFOScheduler(),
            clock=RunClock(),
            send=send,
        )
        async with gateway:
            records = await asyncio.gather(*(gateway.submit(spec(i), i) for i in range(n)))
        assert len(records) == n
        assert state["sent"] == n
        assert state["peak"] <= max_in_flight

    asyncio.run(run())


def test_held_requests_are_eventually_dispatched_by_the_timer() -> None:
    async def run() -> None:
        send, state = make_sender()
        scheduler = HoldingScheduler(holds=4)
        gateway = Gateway(max_in_flight=1, scheduler=scheduler, clock=RunClock(), send=send)
        async with gateway:
            record = await asyncio.wait_for(gateway.submit(spec(0), 0), timeout=5)
        assert record.ok
        assert state["sent"] == 1
        assert scheduler.calls >= 5  # 4 holds, then an admit — only the 5 ms timer re-polls

    asyncio.run(run())


def test_the_scheduler_is_only_offered_requests_that_are_waiting() -> None:
    async def run() -> None:
        a_started = asyncio.Event()
        release_a = asyncio.Event()

        async def send(s: RequestSpec, arrival_ns: int) -> RequestRecord:
            if s.request_id == "r00000":
                a_started.set()
                await release_a.wait()
            return ok_record(s, arrival_ns)

        scheduler = RecordingFIFO()
        gateway = Gateway(max_in_flight=1, scheduler=scheduler, clock=RunClock(), send=send)
        async with gateway:
            task_a = asyncio.create_task(gateway.submit(spec(0), 0))
            await a_started.wait()  # a is admitted and its send is blocked
            task_b = asyncio.create_task(gateway.submit(spec(1), 1))
            await asyncio.sleep(0.03)  # several ticks pass with a in flight
            release_a.set()
            record_a, record_b = await asyncio.gather(task_a, task_b)

        assert record_a.ok and record_b.ok
        # a is offered once, then never again; b only while a holds the single slot.
        assert scheduler.calls == [["r00000"], ["r00001"]]

    asyncio.run(run())


def test_records_carry_their_dispatch_context() -> None:
    async def run() -> None:
        send, _state = make_sender()
        gateway = Gateway(max_in_flight=2, scheduler=FIFOScheduler(), clock=RunClock(), send=send)
        async with gateway:
            records = await asyncio.gather(gateway.submit(spec(0), 0), gateway.submit(spec(1), 1))

        assert all(r.sched_name == "fifo" for r in records)
        assert all(r.ok for r in records)
        assert all(r.sched_decision_ns is not None and r.sched_decision_ns >= 0 for r in records)
        assert all(
            r.queue_depth_at_dispatch is not None and r.queue_depth_at_dispatch >= 1
            for r in records
        )
        in_flight_at_decision = sorted(
            r.in_flight_at_dispatch for r in records if r.in_flight_at_dispatch is not None
        )
        assert in_flight_at_decision == [0, 1]

    asyncio.run(run())
