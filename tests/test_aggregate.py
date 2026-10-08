"""Steady-window rules and summary statistics (plan Phase 2 item 9).

The window rules are the paper's anti-cherry-picking contract (plan §5), so they get direct
tests: ramp/drain exclusion under closed loop, warm-up/cooldown trimming under open loop,
throughput over completions-in-window, goodput against both SLOs, and the >1% failure
invalidity flag.
"""

from __future__ import annotations

from typing import Any

from llmserve.analysis.aggregate import Window, aggregate, summarize, window_of
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics.records import (
    CHUNK_SCHEMA,
    REQUEST_SCHEMA,
    Chunk,
    RequestRecord,
    RequestStatus,
)

_S = 1_000_000_000  # one second in ns


def rec(
    rid: str,
    *,
    arrival: int,
    dispatch: int | None = None,
    first: int | None = None,
    last: int | None = None,
    complete: int | None = None,
    status: RequestStatus = RequestStatus.OK,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    lag_ns: int | None = None,
    chunks: tuple[Chunk, ...] = (),
    prompt_usage: int = 100,
    output_usage: int = 10,
) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        workload_class="c",
        priority=0,
        prompt_tokens_req=100,
        output_tokens_req=10,
        status=status,
        t_arrival=arrival,
        t_dispatch=dispatch,
        t_first_token=first,
        t_last_token=last,
        t_complete=complete,
        prompt_tokens_usage=prompt_usage if status is RequestStatus.OK else None,
        output_tokens_usage=output_usage if status is RequestStatus.OK else None,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        client_lag_ns=lag_ns,
        chunks=chunks,
    )


def ok_rec(rid: str, *, arrival: int, ttft_ns: int, e2e_ns: int) -> RequestRecord:
    """An ok request with dispatch/token timestamps consistent with the given TTFT and E2E."""
    return rec(
        rid,
        arrival=arrival,
        dispatch=arrival + 1_000,
        first=arrival + ttft_ns,
        last=arrival + e2e_ns,
        complete=arrival + e2e_ns + 1_000,
        chunks=(Chunk(arrival + ttft_ns, 1), Chunk(arrival + e2e_ns, 9)),
    )


def cfg_for(mode: str, **measurement: Any) -> ExperimentConfig:
    load: dict[str, Any] = (
        {"mode": "closed", "concurrency": 4, "requests": 40}
        if mode == "closed"
        else {
            "mode": "open",
            "arrival": {"process": "poisson", "rate": 10},
            "duration_s": 4.0,
        }
    )
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": "agg-test",
            "seed": 7,
            "measurement": measurement,
            "server": {"kind": "mock", "endpoint": "http://test/v1", "model": "mock"},
            "load": load,
            "workload": {
                "classes": [
                    {
                        "name": "c",
                        "prompt": {"distribution": "fixed", "tokens": 8},
                        "output": {"distribution": "fixed", "tokens": 4},
                    }
                ]
            },
        }
    )


def test_closed_window_excludes_ramp_and_counts_completions() -> None:
    # C would be 4; two workers' worth of requests: r0 dispatches at in-flight 1 (ramp) and is
    # excluded, r1..r3 dispatch at the peak of 2. r0 still *completes* inside the window.
    records = [
        rec("r0", arrival=0, dispatch=100, first=500, last=900, complete=1100),
        rec("r1", arrival=10, dispatch=200, first=600, last=1000, complete=1200),
        rec("r2", arrival=20, dispatch=1100, first=1500, last=1900, complete=2100),
        rec("r3", arrival=30, dispatch=1200, first=1600, last=2000, complete=2200),
    ]
    window = window_of(records, cfg_for("closed"), t0_ns=0)
    assert window.mode == "closed"
    assert [r.request_id for r in window.measured] == ["r1", "r2", "r3"]
    assert window.start_ns == 200
    assert window.end_ns == 2200
    assert window.length_s == 2000 / _S

    summary = summarize(records, rep=0, window=window)
    throughput = summary["throughput"]
    assert throughput is not None
    assert throughput["n_completed"] == 4  # r0's completion is inside the window
    assert throughput["req_s"] == 4 / (2000 / _S)
    assert throughput["total_toks_s"] == (4 * 100 + 4 * 10) / (2000 / _S)


def test_closed_window_falls_back_to_all_when_nothing_dispatches() -> None:
    records = [rec("r0", arrival=0, status=RequestStatus.CONN_ERROR)]
    window = window_of(records, cfg_for("closed"), t0_ns=0)
    assert window.mode == "all"
    assert window.measured == records
    summary = summarize(records, rep=0, window=window)
    assert summary["success_rate"] == 0.0
    assert summary["invalid"] is True
    assert summary["failures_by_status"] == {"conn_error": 1}


def test_open_window_trims_warmup_and_cooldown() -> None:
    cfg = cfg_for("open", warmup_s=1.0, cooldown_s=1.0)
    t0 = 5 * _S
    arrivals = [5.0, 5.5, 6.0, 6.5, 7.9, 8.0, 8.5]
    records = [
        rec(f"r{i}", arrival=int(a * _S), dispatch=int(a * _S), complete=int(a * _S) + 10)
        for i, a in enumerate(arrivals)
    ]
    window = window_of(records, cfg, t0_ns=t0)
    assert window.mode == "open"
    assert window.start_ns == 6 * _S  # warm-up trim
    assert window.end_ns == 8 * _S  # cooldown trim; arrival at exactly 8.0 s is excluded
    assert [r.request_id for r in window.measured] == ["r2", "r3", "r4"]


def test_open_cooldown_needs_duration_s_else_it_is_ignored() -> None:
    cfg = cfg_for("open", warmup_s=1.0, cooldown_s=1.0)
    load = cfg.load.model_copy(update={"duration_s": None, "requests": 10})
    cfg = cfg.model_copy(update={"load": load})
    t0 = 0
    records = [
        rec("r0", arrival=int(0.5 * _S)),
        rec("r1", arrival=int(1.5 * _S)),
        rec("r2", arrival=int(3.5 * _S)),
    ]
    window = window_of(records, cfg, t0_ns=t0)
    assert window.end_ns == int(3.5 * _S) + 1  # run end = last arrival (+1 ns); cooldown ignored
    assert [r.request_id for r in window.measured] == ["r1", "r2"]


def test_open_window_falls_back_when_warmup_swallows_the_run() -> None:
    cfg = cfg_for("open", warmup_s=100.0)
    records = [rec("r0", arrival=1, complete=2)]
    window = window_of(records, cfg, t0_ns=0)
    assert window.mode == "all"
    assert window.measured == records


def test_latency_stats_percentiles_and_units() -> None:
    # TTFTs of 1, 2 and 3 ms over a trivial all-window.
    records = [
        ok_rec("r0", arrival=0, ttft_ns=1_000_000, e2e_ns=4_000_000),
        ok_rec("r1", arrival=0, ttft_ns=2_000_000, e2e_ns=5_000_000),
        ok_rec("r2", arrival=0, ttft_ns=3_000_000, e2e_ns=6_000_000),
    ]
    window = Window("all", 0, 10 * _S, records)
    summary = summarize(records, rep=0, window=window)
    ttft = summary["latency"]["ttft_s"]
    assert ttft["count"] == 3
    assert ttft["mean"] == 0.002
    assert ttft["median"] == 0.002
    assert ttft["p50"] == 0.002
    assert abs(ttft["p95"] - 0.0029) < 1e-12  # linear interpolation between 2 ms and 3 ms
    assert ttft["std"] > 0
    itl = summary["latency"]["itl_s"]
    assert itl["count"] == 3 * 9  # 3 records x 9 gaps after the first token
    assert summary["latency"]["client_lag_s"]["count"] == 0  # no lag under closed loop


def test_goodput_counts_only_requests_meeting_both_slos() -> None:
    slow = rec(
        "slow",
        arrival=0,
        dispatch=0,
        first=int(0.7 * _S),  # TTFT 700 ms > 500 ms SLO
        last=int(0.79 * _S),
        complete=int(0.8 * _S),
        slo_ttft_ms=500,
        slo_tpot_ms=40,
    )
    fast = rec(
        "fast",
        arrival=0,
        dispatch=0,
        first=int(0.1 * _S),  # TTFT 100 ms, TPOT 10 ms — both inside the SLOs
        last=int(0.19 * _S),
        complete=int(0.2 * _S),
        slo_ttft_ms=500,
        slo_tpot_ms=40,
    )
    window = Window("all", 0, _S, [slow, fast])
    summary = summarize([slow, fast], rep=0, window=window)
    assert summary["n_slo"] == 2
    assert summary["n_slo_pass"] == 1
    assert summary["goodput_s"] == 1.0  # 1 passing req over a 1 s window


def test_goodput_is_none_without_slos() -> None:
    records = [ok_rec("r0", arrival=0, ttft_ns=1_000, e2e_ns=2_000)]
    summary = summarize(records, rep=0, window=Window("all", 0, _S, records))
    assert summary["goodput_s"] is None
    assert summary["n_slo"] == 0


def test_invalid_flag_is_strictly_more_than_one_percent() -> None:
    ok_recs = [ok_rec(f"r{i}", arrival=0, ttft_ns=1_000, e2e_ns=2_000) for i in range(99)]
    one_bad = ok_recs + [rec("bad", arrival=0, status=RequestStatus.TIMEOUT)]
    assert summarize(one_bad, rep=0, window=Window("all", 0, _S, one_bad))["invalid"] is False
    two_bad = ok_recs[:98] + [
        rec("b1", arrival=0, status=RequestStatus.TIMEOUT),
        rec("b2", arrival=0, status=RequestStatus.HTTP_ERROR),
    ]
    summary = summarize(two_bad, rep=0, window=Window("all", 0, _S, two_bad))
    assert summary["invalid"] is True
    assert summary["failures_by_status"] == {"http_error": 1, "timeout": 1}


def test_empty_run_summarizes_without_crashing() -> None:
    summary = summarize([], rep=0, window=Window("closed", None, None, []))
    assert summary["n_requests"] == 0
    assert summary["success_rate"] is None
    assert summary["invalid"] is False
    assert summary["throughput"] is None
    assert summary["latency"]["ttft_s"] == {
        "count": 0,
        "mean": None,
        "median": None,
        "std": None,
        "p50": None,
        "p95": None,
        "p99": None,
    }


def test_aggregate_across_reps() -> None:
    r0 = [ok_rec("a", arrival=0, ttft_ns=1_000_000, e2e_ns=2_000_000)]
    r1 = [ok_rec("a", arrival=0, ttft_ns=3_000_000, e2e_ns=4_000_000)]
    s0 = summarize(r0, rep=0, window=Window("all", 0, _S, r0))
    s1 = summarize(r1, rep=1, window=Window("all", 0, _S, r1))
    cross = aggregate([s0, s1])
    assert cross["repetitions"] == 2
    assert cross["invalid"] is False
    assert cross["success_rate"]["mean"] == 1.0
    ttft = cross["latency"]["ttft_s"]["mean"]  # across-rep block of the per-rep means
    assert ttft["n"] == 2
    assert abs(ttft["mean"] - 0.002) < 1e-12  # mean of the two per-rep means
    assert len(cross["per_rep"]) == 2


def test_schemas_are_untouched() -> None:
    # The writers depend on these exact schemas; a drift here breaks every stored run.
    assert "run_id" in REQUEST_SCHEMA.names and "rep" in REQUEST_SCHEMA.names
    assert CHUNK_SCHEMA.names == ["request_id", "idx", "t_ns", "n_tokens"]
