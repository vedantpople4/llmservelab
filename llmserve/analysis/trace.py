"""One request's timeline (plan Phase 4 item 5): `llmserve trace <run> --request <id>`.

Answers "why was this request slow?" from what the run stored: the request's own timestamps,
its chunks, vLLM's `/metrics` state sampled nearest before each step, and any `events.jsonl`
entry that names the request or happened while it was in flight. Times are seconds after the
request's arrival.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from llmserve.analysis.breakdown import phases
from llmserve.analysis.runs import Run
from llmserve.metrics import latency
from llmserve.metrics.records import RequestRecord
from llmserve.scheduler.estimators import PrefillFit

_NS = 1e9


def trace(run: Run, request_id: str, *, rep: int = 0, prefill: PrefillFit | None = None) -> str:
    """The timeline as text; raises KeyError when the rep has no such request."""
    records = {r.request_id: r for r in run.records(rep)}
    if request_id not in records:
        raise KeyError(f"no request {request_id!r} in rep {rep} of {run.run_id}")
    r = records[request_id]
    rep_path = run.path / f"rep-{rep:02d}"
    server = _server(rep_path / "server.parquet")
    measured = any(m.request_id == request_id for m in run.measured(rep))

    lines = [
        f"run {run.run_id} · rep {rep} · request {r.request_id} · class {r.workload_class} · "
        f"{r.status.value}" + ("" if measured else " · outside the steady window"),
        f"prompt {r.prompt_tokens_req} tokens (server counted {_n(r.prompt_tokens_usage)}) · "
        f"output {r.output_tokens_req} requested ({_n(r.output_tokens_usage)} generated)",
    ]
    if r.error:
        lines.append(f"error: {r.error} (HTTP {_n(r.http_status)})")
    lines += ["", f"{'t (s)':>10}  {'event':<12}  detail"]

    def step(t_ns: int | None, event: str, detail: str = "") -> None:
        if t_ns is None:
            return
        state = _state_at(server, t_ns)
        detail = " · ".join(part for part in (detail, state) if part)
        lines.append(f"{(t_ns - r.t_arrival) / _NS:>+10.3f}  {event:<12}  {detail}".rstrip())

    step(r.t_arrival, "arrival")
    step(r.t_dispatch, "dispatch", _ms("gateway wait", latency.queue_wait(r)))
    step(r.t_first_token, "first token", _first_token(r, prefill))
    step(r.t_last_token, "last token", _decode(r))
    step(r.t_complete, "complete")

    events = _events(rep_path / "events.jsonl", r)
    if events:
        lines += ["", "events:"]
        for e in events:
            t = (int(e["t_ns"]) - r.t_arrival) / _NS
            rest = {k: v for k, v in e.items() if k != "t_ns"}
            lines.append(f"{t:>+10.3f}  {json.dumps(rest, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _first_token(r: RequestRecord, prefill: PrefillFit | None) -> str:
    detail = _ms("TTFT", latency.ttft(r))
    p = phases(r, prefill) if prefill is not None else None
    if p is not None:
        detail += (
            f" = gateway wait {p.gateway_wait_s * 1e3:.1f} + server wait "
            f"{p.server_wait_s * 1e3:.1f} + unloaded prefill {p.prefill_s * 1e3:.1f} ms"
        )
    return detail


def _decode(r: RequestRecord) -> str:
    gaps = latency.inter_token_latencies(r)
    decode = None
    if r.ok and r.t_first_token is not None and r.t_last_token is not None:
        decode = (r.t_last_token - r.t_first_token) / _NS
    parts = [_ms("decode", decode), f"{len(r.chunks)} chunks"]
    parts.append(_ms("TPOT", latency.tpot(r)))
    if gaps:
        parts.append(
            f"ITL P50 {np.median(gaps) * 1e3:.1f} ms, max {max(gaps) * 1e3:.1f} ms "
            f"at token {gaps.index(max(gaps)) + 2}"
        )
    return " · ".join(p for p in parts if p)


def _server(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas().sort_values("t_ns", ignore_index=True)


def _state_at(server: pd.DataFrame, t_ns: int) -> str:
    """vLLM's state at the last `/metrics` sample at or before `t_ns`."""
    if server.empty:
        return ""
    i = int(np.searchsorted(server["t_ns"].to_numpy(), t_ns, side="right")) - 1
    if i < 0:
        return ""
    s = server.iloc[i]
    age = (t_ns - int(s["t_ns"])) / _NS
    return (
        f"vLLM running {s['running']:g}, waiting {s['waiting']:g}, "
        f"KV {100 * s['kv_usage']:.1f}% ({age:.1f} s before)"
    )


def _events(path: Path, r: RequestRecord) -> list[dict[str, Any]]:
    """Events naming the request, plus unattributed ones while it was in flight."""
    if not path.exists():
        return []
    end = r.t_complete or r.t_last_token or r.t_arrival
    out = []
    for line in path.read_text().splitlines():
        e = json.loads(line)
        named = e.get("request_id") == r.request_id
        during = "request_id" not in e and r.t_arrival <= int(e["t_ns"]) <= end
        if named or during:
            out.append(e)
    return sorted(out, key=lambda e: int(e["t_ns"]))


def _ms(name: str, seconds: float | None) -> str:
    return "" if seconds is None else f"{name} {seconds * 1e3:.1f} ms"


def _n(value: int | None) -> str:
    return "?" if value is None else str(value)
