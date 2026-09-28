import pyarrow as pa

from llmserve.metrics.records import (
    CHUNK_SCHEMA,
    REQUEST_SCHEMA,
    Chunk,
    RequestRecord,
    RequestStatus,
)


def test_rows_conform_to_parquet_schemas() -> None:
    ok = RequestRecord(
        request_id="r1",
        workload_class="interactive",
        priority=0,
        prompt_tokens_req=128,
        output_tokens_req=2,
        status=RequestStatus.OK,
        t_arrival=0,
        t_dispatch=5,
        t_first_token=10,
        t_last_token=20,
        chunks=(Chunk(10, 1), Chunk(20, 1)),
        output_tokens_usage=2,
        sched_name="fifo",
    )
    failed = RequestRecord(
        request_id="r2",
        workload_class="interactive",
        priority=0,
        prompt_tokens_req=128,
        output_tokens_req=2,
        status=RequestStatus.HTTP_ERROR,
        t_arrival=3,
        http_status=503,
        error="Service Unavailable",
    )
    rows = [dict(r.to_row(), run_id="run", rep=0) for r in (ok, failed)]
    table = pa.Table.from_pylist(rows, schema=REQUEST_SCHEMA)
    assert table.column("status").to_pylist() == ["ok", "http_error"]
    assert table.column("t_first_token").to_pylist() == [10, None]
    assert table.column("n_chunks").to_pylist() == [2, 0]

    chunks = pa.Table.from_pylist(ok.chunk_rows(), schema=CHUNK_SCHEMA)
    assert chunks.column("idx").to_pylist() == [0, 1]


def test_to_row_covers_every_schema_column() -> None:
    r = RequestRecord(
        request_id="r",
        workload_class="c",
        priority=1,
        prompt_tokens_req=1,
        output_tokens_req=1,
        status=RequestStatus.OK,
        t_arrival=0,
    )
    assert set(r.to_row()) | {"run_id", "rep"} == set(REQUEST_SCHEMA.names)
