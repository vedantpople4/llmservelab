import pytest

from llmserve.metrics.latency import RequestTiming
from llmserve.scheduler.base import Request, SystemState
from llmserve.scheduler.fifo import FIFOScheduler


def test_latency_metrics() -> None:
    t = RequestTiming(start=0.0, token_times=[0.5, 0.6, 0.8, 0.9])
    assert t.ttft == pytest.approx(0.5)
    assert t.e2e == pytest.approx(0.9)
    assert t.tpot == pytest.approx(0.4 / 3)
    assert t.inter_token_latencies == pytest.approx([0.1, 0.2, 0.1])


def test_tpot_undefined_for_single_token() -> None:
    assert RequestTiming(start=0.0, token_times=[0.3]).tpot is None


def test_fifo_picks_earliest_arrival() -> None:
    state = SystemState(now=10.0, queue_depth=2, in_flight=0)
    late = Request("b", arrival_time=2.0, prompt_tokens=10, max_output_tokens=10)
    early = Request("a", arrival_time=1.0, prompt_tokens=4000, max_output_tokens=10)
    assert FIFOScheduler().select_next([late, early], state) is early
    assert FIFOScheduler().select_next([], state) is None
