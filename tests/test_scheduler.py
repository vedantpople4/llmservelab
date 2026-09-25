import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmserve.scheduler import registry
from llmserve.scheduler.base import Priority, Request, SystemState
from llmserve.scheduler.fifo import FIFOScheduler

STATE = SystemState(now=10.0, queue_depth=2, in_flight=0)


def req(rid: str, arrival: float, prompt: int = 100) -> Request:
    return Request(request_id=rid, arrival_time=arrival, prompt_tokens=prompt, max_output_tokens=10)


def test_fifo_picks_earliest_arrival() -> None:
    late, early = req("b", 2.0, prompt=10), req("a", 1.0, prompt=4000)
    assert FIFOScheduler().select_next([late, early], STATE) is early
    assert FIFOScheduler().select_next([], STATE) is None


@given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=1, max_size=50))
def test_fifo_returns_a_waiting_request_with_minimal_arrival(arrivals: list[float]) -> None:
    waiting = [req(str(i), a) for i, a in enumerate(arrivals)]
    chosen = FIFOScheduler().select_next(waiting, STATE)
    assert chosen in waiting
    assert chosen is not None and chosen.arrival_time == min(arrivals)


def test_priority_direction_matches_vllm() -> None:
    assert Priority.HIGH < Priority.MEDIUM < Priority.LOW


def test_registry() -> None:
    assert isinstance(registry.create("fifo"), FIFOScheduler)
    with pytest.raises(ValueError, match="unknown scheduler"):
        registry.create("nope")
