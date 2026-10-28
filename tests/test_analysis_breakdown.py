"""Where a request's time goes: the phases add up to E2E and split TTFT by the prefill fit."""

from __future__ import annotations

import math

from conftest import RunFactory, experiment_config

from llmserve.analysis.breakdown import PHASES, Phases, class_breakdown, phases
from llmserve.metrics import latency
from llmserve.metrics.records import RequestRecord, RequestStatus
from llmserve.scheduler.estimators import PrefillFit

S = 1_000_000_000
FLAT_HALF_SECOND = PrefillFit(a=0.5, b=0.0, c=0.0, r2=1.0, n=3)


def record(*, dispatch: float, first: float, last: float, ok: bool = True) -> RequestRecord:
    return RequestRecord(
        request_id="r",
        workload_class="c",
        priority=1,
        prompt_tokens_req=128,
        output_tokens_req=4,
        status=RequestStatus.OK if ok else RequestStatus.TIMEOUT,
        t_arrival=0,
        t_dispatch=int(dispatch * S),
        t_first_token=int(first * S),
        t_last_token=int(last * S),
        t_complete=int(last * S),
    )


def test_phases_split_ttft_into_server_wait_and_unloaded_prefill() -> None:
    p = phases(record(dispatch=1.0, first=3.0, last=10.0), FLAT_HALF_SECOND)
    assert p == Phases(gateway_wait_s=1.0, server_wait_s=1.5, prefill_s=0.5, decode_s=7.0)
    assert p.total_s == 10.0


def test_a_first_token_faster_than_the_fit_is_all_prefill() -> None:
    p = phases(record(dispatch=0.0, first=0.2, last=1.0), FLAT_HALF_SECOND)
    assert p is not None
    assert (p.server_wait_s, p.prefill_s) == (0.0, 0.2)


def test_failed_requests_have_no_phases() -> None:
    assert phases(record(dispatch=0.0, first=1.0, last=2.0, ok=False), FLAT_HALF_SECOND) is None


def test_class_means_add_up_to_mean_e2e(run_factory: RunFactory) -> None:
    shape = {"distribution": "fixed", "tokens": 4}
    run = run_factory(
        experiment_config(
            workload={
                "classes": [
                    {
                        "name": "a",
                        "prompt": {"distribution": "fixed", "tokens": 16},
                        "output": shape,
                    },
                    {
                        "name": "b",
                        "prompt": {"distribution": "fixed", "tokens": 64},
                        "output": shape,
                    },
                ]
            }
        )
    )
    table = class_breakdown(run, FLAT_HALF_SECOND)
    assert table["workload_class"].tolist() == ["a", "b"]
    measured = [r for rep in (0, 1) for r in run.measured(rep)]
    assert int(table["n"].sum()) == len(measured)
    for row in table.itertuples(index=False):
        e2e = [latency.e2e(r) or 0.0 for r in measured if r.workload_class == row.workload_class]
        assert math.isclose(sum(getattr(row, p) for p in PHASES), sum(e2e) / len(e2e))
