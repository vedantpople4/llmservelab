"""Run-directory layout and the writers for every file a benchmark produces (plan Phase 2 item 8).

```
results/<experiment>/<run_id>/          run_id = <utc-ts>-<config_hash8>-<git_sha7>
  config.resolved.yaml  metadata.json  workload.parquet  summary.json
  rep-00/ workload.parquet requests.parquet chunks.parquet summary.json
         gpu.parquet server.parquet events.jsonl
  rep-01/ ...
```

`workload.parquet` at the run root is rep 0's file — the same bytes `llmserve generate-workload
--rep 0` produces for this config, so the exit check's "identical workload file" has one obvious
path; each repetition keeps its own copy under `rep-NN/` because seeds differ per rep (ADR-0004).
`gpu.parquet`, `server.parquet` and `events.jsonl` are written per rep by the samplers (Phase 2
item 7), for the reps where the sampler actually ran.

Every write is deterministic given its inputs (sorted JSON keys, fixed parquet schema), so
reruns with the same seed are byte-comparable.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from llmserve.config.schema import ExperimentConfig
from llmserve.metrics.gpu import GPU_SCHEMA, GpuSample
from llmserve.metrics.records import CHUNK_SCHEMA, REQUEST_SCHEMA, RequestRecord
from llmserve.metrics.server import SERVER_SCHEMA, ServerSample


def make_run_id(config_hash: str, git_sha: str | None, now_utc: datetime) -> str:
    """`20261004T120304Z-1a2b3c4d-9f8e7d6`, per the plan's `<utc-ts>-<hash8>-<sha7>`."""
    sha = git_sha or "unknown"
    return f"{now_utc.strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:8]}-{sha[:7]}"


def rep_dir(run_dir: Path, rep: int) -> Path:
    return run_dir / f"rep-{rep:02d}"


def write_config(cfg: ExperimentConfig, path: Path) -> None:
    """The resolved config (refs expanded, defaults visible) as YAML."""
    path.write_text(yaml.safe_dump(cfg.canonical(), sort_keys=False, width=100))


def write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_requests(path: Path, records: list[RequestRecord], *, run_id: str, rep: int) -> None:
    """`requests.parquet` on the schema from `metrics.records`; `run_id`/`rep` come from here."""
    rows = [{**r.to_row(), "run_id": run_id, "rep": rep} for r in records]
    pq.write_table(pa.Table.from_pylist(rows, schema=REQUEST_SCHEMA), path)


def write_chunks(path: Path, records: list[RequestRecord]) -> None:
    rows = [row for r in records for row in r.chunk_rows()]
    pq.write_table(pa.Table.from_pylist(rows, schema=CHUNK_SCHEMA), path)


def write_gpu(path: Path, samples: list[GpuSample]) -> None:
    """`gpu.parquet` on the schema from `metrics.gpu`; empty is a valid (schema-only) file."""
    pq.write_table(pa.Table.from_pylist([asdict(s) for s in samples], schema=GPU_SCHEMA), path)


def write_server(path: Path, samples: list[ServerSample]) -> None:
    pq.write_table(pa.Table.from_pylist([asdict(s) for s in samples], schema=SERVER_SCHEMA), path)


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    """`events.jsonl`, one sorted-key JSON object per line; written even when there are none."""
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events))
