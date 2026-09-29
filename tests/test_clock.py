import time

from llmserve.runner.clock import RunClock


def test_values_are_nanoseconds_since_construction() -> None:
    clock = RunClock()
    first = clock()
    assert first >= 0
    assert clock() >= first


def test_it_advances_while_waiting() -> None:
    clock = RunClock()
    start = clock()
    while clock() - start < 2_000_000:  # 2 ms
        pass
    assert clock() - start >= 2_000_000


def test_wall_anchor_is_close_to_now() -> None:
    anchor = RunClock().wall_anchor_ns
    assert abs(time.time_ns() - anchor) < 5e9  # within 5 s


def test_two_clocks_are_independent() -> None:
    a, b = RunClock(), RunClock()
    assert b() >= 0
    assert a() >= 0
