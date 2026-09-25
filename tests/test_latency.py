import pytest

from llmserve.metrics import latency
from llmserve.metrics.records import Chunk, RequestRecord, RequestStatus

MS = 1_000_000


def record(
    chunks: list[tuple[int, int]],
    *,
    status: RequestStatus = RequestStatus.OK,
    dispatch_ms: int | None = 100,
    output_tokens: int | None = None,
) -> RequestRecord:
    cs = tuple(Chunk(t_ns=t * MS, n_tokens=n) for t, n in chunks)
    return RequestRecord(
        request_id="r0",
        workload_class="default",
        priority=1,
        prompt_tokens_req=128,
        output_tokens_req=sum(n for _, n in chunks),
        status=status,
        t_arrival=0,
        t_dispatch=None if dispatch_ms is None else dispatch_ms * MS,
        t_first_token=cs[0].t_ns if cs else None,
        t_last_token=cs[-1].t_ns if cs else None,
        chunks=cs,
        output_tokens_usage=sum(n for _, n in chunks) if output_tokens is None else output_tokens,
    )


def test_one_token_per_chunk() -> None:
    r = record([(500, 1), (600, 1), (800, 1), (900, 1)])
    assert latency.queue_wait(r) == pytest.approx(0.1)
    assert latency.ttft(r) == pytest.approx(0.5)
    assert latency.ttft_server(r) == pytest.approx(0.4)
    assert latency.e2e(r) == pytest.approx(0.9)
    assert latency.tpot(r) == pytest.approx(0.4 / 3)
    assert latency.inter_token_latencies(r) == pytest.approx([0.1, 0.2, 0.1])


def test_tpot_counts_tokens_not_chunks() -> None:
    # 2 chunks, 5 tokens: 4 decode gaps over 400 ms.
    r = record([(500, 1), (900, 4)])
    assert latency.tpot(r) == pytest.approx(0.1)
    assert latency.inter_token_latencies(r) == pytest.approx([0.1] * 4)


def test_itl_sums_to_decode_time_when_tpot_matches() -> None:
    r = record([(500, 1), (700, 2), (1000, 3)])
    itl = latency.inter_token_latencies(r)
    assert len(itl) == 5
    assert sum(itl) == pytest.approx(0.5)
    assert sum(itl) / len(itl) == pytest.approx(latency.tpot(r))


def test_tpot_undefined_for_single_token() -> None:
    assert latency.tpot(record([(300, 1)])) is None


@pytest.mark.parametrize("status", [s for s in RequestStatus if s is not RequestStatus.OK])
def test_failed_requests_have_no_latency(status: RequestStatus) -> None:
    r = record([(500, 1), (600, 1)], status=status)
    assert latency.ttft(r) is None
    assert latency.e2e(r) is None
    assert latency.tpot(r) is None
    assert latency.inter_token_latencies(r) == []
    # A failed request that was dispatched still has a queue wait.
    assert latency.queue_wait(r) == pytest.approx(0.1)


def test_request_that_never_produced_a_token() -> None:
    r = record([], status=RequestStatus.TIMEOUT, dispatch_ms=None, output_tokens=0)
    assert latency.queue_wait(r) is None
    assert latency.ttft(r) is None
