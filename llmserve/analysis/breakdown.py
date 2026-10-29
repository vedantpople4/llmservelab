"""Where a request's time goes (plan Phase 4 item 4): four consecutive phases that sum to E2E.

- **gateway wait**: arrival to dispatch, time in the harness's own queue (zero without a gateway).
- **server wait**: dispatch to first token, minus the unloaded prefill. It is everything the
  engine adds on top of computing the prompt alone: time in vLLM's waiting queue, and prefill
  slowed by sharing steps with running decodes. vLLM's `/metrics` histograms split the two per
  rep; the per-request records cannot.
- **prefill**: the unloaded TTFT the H3 fit predicts for the prompt length, capped at the
  observed dispatch-to-first-token time so no phase is negative.
- **decode**: first token to last token.

Means, not medians, are aggregated: means of the phases add up to the mean E2E, so a stacked
bar is the whole request.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
import pandas as pd

from llmserve.analysis.runs import Run
from llmserve.metrics import latency
from llmserve.metrics.records import RequestRecord
from llmserve.scheduler.estimators import PrefillFit

PHASES = ("gateway_wait_s", "server_wait_s", "prefill_s", "decode_s")
COHORT_GAP_S = 0.25


@dataclass(frozen=True)
class Phases:
    gateway_wait_s: float
    server_wait_s: float
    prefill_s: float
    decode_s: float

    @property
    def total_s(self) -> float:
        return self.gateway_wait_s + self.server_wait_s + self.prefill_s + self.decode_s


def phases(r: RequestRecord, prefill: PrefillFit) -> Phases | None:
    """The four phases of an ok request; None for a failed one."""
    waited = latency.queue_wait(r)
    to_first = latency.ttft_server(r)
    e2e = latency.e2e(r)
    if waited is None or to_first is None or e2e is None:
        return None
    computing = min(to_first, max(0.0, prefill.predict(r.prompt_tokens_req)))
    return Phases(
        gateway_wait_s=waited,
        server_wait_s=to_first - computing,
        prefill_s=computing,
        decode_s=e2e - waited - to_first,
    )


def class_breakdown(run: Run, prefill: PrefillFit) -> pd.DataFrame:
    """Mean seconds per phase for each workload class over the measured requests of every rep.

    Columns: `workload_class`, `n`, then `PHASES`; one row per class, sorted by name."""
    rows = []
    for rep_path in run.rep_dirs():
        for r in run.measured(int(rep_path.name.removeprefix("rep-"))):
            p = phases(r, prefill)
            if p is not None:
                rows.append((r.workload_class, *astuple(p)))
    frame = pd.DataFrame(rows, columns=["workload_class", *PHASES])
    grouped = frame.groupby("workload_class", sort=True)
    out = grouped[list(PHASES)].mean()
    out.insert(0, "n", grouped.size())
    return out.reset_index()


def dispatch_cohorts(run: Run, *, gap_s: float = COHORT_GAP_S) -> pd.DataFrame:
    """Measured ok requests grouped into cohorts dispatched together: a new cohort starts after a
    gap of more than `gap_s` between consecutive dispatches. Closed-loop requests of one fixed
    length finish together and are replaced together, so vLLM prefills each cohort as one batch
    and every member's TTFT is the cohort's prefill, not its own.

    Columns: `rep`, `size`, `prompt_tokens` (the cohort's sum), `ttft_s` (its members' median)."""
    rows = []
    for rep_path in run.rep_dirs():
        rep = int(rep_path.name.removeprefix("rep-"))
        records = sorted(
            (r for r in run.measured(rep) if latency.ttft_server(r) is not None),
            key=lambda r: r.t_dispatch or 0,
        )
        if not records:
            continue
        dispatch_s = np.array([r.t_dispatch or 0 for r in records], dtype=float) / 1e9
        starts = np.r_[0, np.flatnonzero(np.diff(dispatch_s) > gap_s) + 1, len(records)]
        for lo, hi in zip(starts[:-1], starts[1:], strict=True):
            members = records[lo:hi]
            rows.append(
                (
                    rep,
                    len(members),
                    sum(r.prompt_tokens_req for r in members),
                    float(np.median([latency.ttft_server(r) or 0.0 for r in members])),
                )
            )
    return pd.DataFrame(rows, columns=["rep", "size", "prompt_tokens", "ttft_s"])
