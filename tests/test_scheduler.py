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
from llmserve.scheduler.waas import WAAS

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
    ("waas", {}),
    ("waas", {"cost_weight": 0.0, "max_bypass": 1}),
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
        WAAS,
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


@pytest.mark.parametrize("name", ["spf", "sjf", "priority", "waas"])
def test_each_policy_runs_end_to_end_behind_the_gateway(run_factory: RunFactory, name: str) -> None:
    gateway = {"enabled": True, "max_in_flight": 2, "scheduler": {"name": name}}
    run = run_factory(experiment_config(repetitions=1, gateway=gateway))
    records = run.records(0)
    assert records and all(r.ok for r in records)
    assert {r.sched_name for r in records} == {name}
    assert all(r.output_tokens_est == r.output_tokens_req for r in records)  # the oracle
    assert run.metadata["gateway"]["scheduler"] == name
    if name == "waas":
        assert all(r.sched_score is not None and r.bypass_count is not None for r in records)


def _serve(scheduler: Scheduler, arrivals: list[list[Request]]) -> list[tuple[Request, int]]:
    """Each step enqueues `arrivals[step]` and dispatches one request; then the queue drains.
    Returns (request, bypass count reported at dispatch) in dispatch order."""
    waiting: list[Request] = []
    out = []
    for step in range(len(arrivals) + sum(map(len, arrivals))):
        if step < len(arrivals):
            waiting += arrivals[step]
        chosen = scheduler.select_next(waiting, STATE)
        if chosen is None:
            continue
        waiting.remove(chosen)
        _, bypassed = scheduler.on_dispatch(chosen)
        assert bypassed is not None
        out.append((chosen, bypassed))
    assert not waiting
    return out


def _true_bypasses(order: list[Request], r: Request) -> int:
    """Later arrivals dispatched before `r`. Everyone enqueued before `r` was dispatched, so a
    later arrival dispatched earlier overtook it while it waited."""
    before = order[: order.index(r)]
    return sum((o.arrival_time, o.request_id) > (r.arrival_time, r.request_id) for o in before)


@given(
    batches=st.lists(st.lists(st.integers(1, 8192), max_size=4), min_size=1, max_size=30),
    weight=st.sampled_from([0.0, 0.01, 1.0, 1e6]),
    bound=st.integers(1, 6),
)
def test_waas_reports_true_bypasses_and_never_exceeds_the_bound(
    batches: list[list[int]], weight: float, bound: int
) -> None:
    arrivals, n = [], 0
    for step, prompts in enumerate(batches):
        arrivals.append([req(f"r{(n := n + 1):03d}", float(step), p) for p in prompts])
    served = _serve(WAAS(cost_weight=weight, max_bypass=bound), arrivals)
    order = [r for r, _ in served]
    assert sorted(r.request_id for r in order) == sorted(r.request_id for b in arrivals for r in b)
    for r, bypassed in served:
        assert bypassed == _true_bypasses(order, r)
        assert bypassed <= bound


def test_waas_with_zero_cost_weight_is_fifo() -> None:
    waiting = [req("a", 2.0, prompt=10), req("b", 1.0, prompt=8000), req("c", 3.0, prompt=1)]
    assert _order(WAAS(cost_weight=0.0), waiting) == _order(FIFOScheduler(), waiting)


def test_waas_trades_predicted_service_against_age() -> None:
    waas = WAAS(cost_weight=1.0)
    long, short = req("long", 0.0, prompt=8000, est=128), req("short", 5.0, prompt=100, est=128)
    gap = MODEL.service_time(8000, 128, 16) - MODEL.service_time(100, 128, 16)
    assert gap > 5.0  # the short request is worth more than its 5 s younger age
    assert waas.select_next([long, short], STATE) is short
    late = req("short", gap + 1.0, prompt=100, est=128)
    assert WAAS(cost_weight=1.0).select_next([long, late], STATE) is long


def test_a_starving_request_goes_after_max_bypass_overtakes() -> None:
    big = req("big", 0.0, prompt=8000)
    tiny = [[req(f"t{i:03d}", float(i), prompt=1)] for i in range(50)]
    served = _serve(WAAS(cost_weight=1e6, max_bypass=5), [[big, *tiny[0]], *tiny[1:]])
    position = [r.request_id for r, _ in served].index("big")
    assert position == 5 and served[position][1] == 5


def test_waas_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="cost_weight"):
        WAAS(cost_weight=-1)
    with pytest.raises(ValueError, match="max_bypass"):
        WAAS(max_bypass=0)
