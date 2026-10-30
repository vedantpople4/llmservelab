"""Scheduler invariants (plan Phase 5 tests): every policy returns a member of `waiting` or None,
chooses the same request whatever the order of `waiting`, and orders by its own key."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from conftest import RunFactory, experiment_config
from hypothesis import given
from hypothesis import strategies as st

from llmserve.scheduler import registry
from llmserve.scheduler.base import Priority, Request, Scheduler, SystemState
from llmserve.scheduler.estimators import CostModel
from llmserve.scheduler.fifo import FIFOScheduler
from llmserve.scheduler.priority import PriorityScheduler
from llmserve.scheduler.shortest import DEFAULT_COST_MODEL, ShortestJobFirst, ShortestPromptFirst

STATE = SystemState(now=10.0, queue_depth=2, in_flight=0)
MODEL = CostModel.load(Path(DEFAULT_COST_MODEL))


def req(
    rid: str,
    arrival: float,
    prompt: int = 100,
    *,
    est: int | None = 10,
    priority: int = Priority.MEDIUM,
) -> Request:
    return Request(
        request_id=rid,
        arrival_time=arrival,
        prompt_tokens=prompt,
        max_output_tokens=10,
        output_tokens_est=est,
        priority=priority,
    )


requests = st.lists(
    st.builds(
        lambda a, p, o, pr: (a, p, o, pr),
        st.floats(min_value=0, max_value=100),
        st.integers(min_value=1, max_value=8192),
        st.one_of(st.none(), st.integers(min_value=1, max_value=1024)),
        st.sampled_from(list(Priority)),
    ),
    min_size=1,
    max_size=40,
).map(lambda rows: [req(f"r{i}", a, p, est=o, priority=pr) for i, (a, p, o, pr) in enumerate(rows)])

POLICIES: list[tuple[str, dict[str, float | str]]] = [
    ("fifo", {}),
    ("spf", {}),
    ("sjf", {}),
    ("priority", {}),
    ("priority", {"aging_s": 5.0}),
]


@pytest.mark.parametrize(("name", "params"), POLICIES)
@given(waiting=requests, order=st.randoms(use_true_random=False))
def test_choice_is_a_waiting_request_independent_of_order(
    name: str, params: dict[str, float | str], waiting: list[Request], order: random.Random
) -> None:
    chosen = registry.create(name, params).select_next(waiting, STATE)
    assert chosen in waiting
    shuffled = list(waiting)
    order.shuffle(shuffled)
    assert registry.create(name, params).select_next(shuffled, STATE) is chosen


@pytest.mark.parametrize(("name", "params"), POLICIES)
def test_an_empty_queue_holds(name: str, params: dict[str, float | str]) -> None:
    assert registry.create(name, params).select_next([], STATE) is None


def _order(scheduler: Scheduler, waiting: list[Request]) -> list[Request]:
    """The admission order when every request in `waiting` is admitted one at a time."""
    left, out = list(waiting), []
    while (chosen := scheduler.select_next(left, STATE)) is not None:
        out.append(chosen)
        left.remove(chosen)
    return out


@given(requests)
def test_fifo_admits_in_arrival_order(waiting: list[Request]) -> None:
    order = _order(FIFOScheduler(), waiting)
    assert [r.arrival_time for r in order] == sorted(r.arrival_time for r in waiting)


@given(requests)
def test_spf_admits_shortest_prompt_first_then_earliest(waiting: list[Request]) -> None:
    order = _order(ShortestPromptFirst(), waiting)
    keys = [(r.prompt_tokens, r.arrival_time) for r in order]
    assert keys == sorted(keys)


@given(requests)
def test_sjf_admits_smallest_predicted_service_time_first(waiting: list[Request]) -> None:
    order = _order(ShortestJobFirst(), waiting)
    costs = [MODEL.service_time(r.prompt_tokens, r.output_tokens_est or 1, 16) for r in order]
    assert costs == sorted(costs)


def test_sjf_weighs_output_length_and_spf_does_not() -> None:
    long_prompt_short_output = req("a", 0.0, prompt=1024, est=4)
    short_prompt_long_output = req("b", 1.0, prompt=128, est=512)
    waiting = [long_prompt_short_output, short_prompt_long_output]
    assert ShortestPromptFirst().select_next(waiting, STATE) is short_prompt_long_output
    assert ShortestJobFirst().select_next(waiting, STATE) is long_prompt_short_output


def test_sjf_without_an_estimate_orders_by_prefill() -> None:
    waiting = [req("a", 0.0, prompt=1024, est=None), req("b", 1.0, prompt=128, est=None)]
    assert ShortestJobFirst().select_next(waiting, STATE) is waiting[1]


@given(requests)
def test_strict_priority_admits_by_class_then_fifo_within_it(waiting: list[Request]) -> None:
    order = _order(PriorityScheduler(), waiting)
    keys = [(r.priority, r.arrival_time) for r in order]
    assert keys == sorted(keys)


def test_aging_lets_a_long_waiting_low_request_overtake() -> None:
    low = req("low", 0.0, priority=Priority.LOW)
    high = req("high", 9.0, priority=Priority.HIGH)
    assert PriorityScheduler().select_next([low, high], STATE) is high
    # At now = 10 the LOW request has waited 10 s, five levels at aging_s = 2, while the HIGH
    # request has gained half a level.
    assert PriorityScheduler(aging_s=2.0).select_next([low, high], STATE) is low
    assert PriorityScheduler(aging_s=50.0).select_next([low, high], STATE) is high


def test_priority_direction_matches_vllm() -> None:
    assert Priority.HIGH < Priority.MEDIUM < Priority.LOW


def test_registry_builds_every_policy_and_checks_params(tmp_path: Path) -> None:
    assert {type(registry.create(n, p)) for n, p in POLICIES} == {
        FIFOScheduler,
        ShortestPromptFirst,
        ShortestJobFirst,
        PriorityScheduler,
    }
    with pytest.raises(ValueError, match="unknown scheduler"):
        registry.create("nope")
    with pytest.raises(ValueError, match="scheduler 'priority'"):
        registry.check("priority", {"aging": 5})
    registry.check("sjf", {"cost_model": str(tmp_path / "absent.json"), "batch": 8})
    with pytest.raises(ValueError, match="aging_s must be >= 0"):
        PriorityScheduler(aging_s=-1)


def test_sjf_loads_the_cost_model_it_is_given(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    flat = {
        "prefill": {"a": 1.0, "b": 0.0, "c": 0.0, "r2": 1.0, "n": 3},
        "decode": {"d0": 1.0, "d1": 0.0, "r2": 1.0, "n": 2},
    }
    path.write_text(json.dumps(flat))
    sjf = ShortestJobFirst(cost_model=str(path))
    # Prefill costs the same for every prompt, so only the output estimate orders them.
    waiting = [req("a", 0.0, prompt=8, est=100), req("b", 1.0, prompt=8000, est=2)]
    assert sjf.select_next(waiting, STATE) is waiting[1]


@pytest.mark.parametrize("name", ["spf", "sjf", "priority"])
def test_each_policy_runs_end_to_end_behind_the_gateway(run_factory: RunFactory, name: str) -> None:
    gateway = {"enabled": True, "max_in_flight": 2, "scheduler": {"name": name}}
    run = run_factory(experiment_config(repetitions=1, gateway=gateway))
    records = run.records(0)
    assert records and all(r.ok for r in records)
    assert {r.sched_name for r in records} == {name}
    assert all(r.output_tokens_est == r.output_tokens_req for r in records)  # the oracle
    assert run.metadata["gateway"]["scheduler"] == name
