"""Scheduler comparisons (plan Phase 5 items 4 to 7): the in-flight cap trade-off and the
policy matrix, read from suite runs.

**In-flight cap** (`inflight-cap`). Per swept cap, across-rep mean throughput and the pooled
gateway wait and TTFT of the ok, in-window requests. `choose_cap` picks the smallest cap whose
generated tok/s is at least `CAP_THROUGHPUT_SHARE` of the largest swept cap's, which stands for
uncapped.

**Policy matrix** (`matrix-*`, `xcheck-vllm-priority`). Each point is one run, identified by its
config, not its label: the policy is `gateway.scheduler.name`, or `vllm-priority` for a run with
no gateway against a priority-scheduling vLLM; ρ is `load.arrival.rho`; the seed is `seed`.
Requests are the ok, in-window ones, pooled across the seeds of a (policy, ρ) cell. Policies at
one seed see identical requests, so size deciles are cut once per ρ over the pooled requests and
every policy is binned by the same edges.

- **Slowdown** is E2E ÷ the cost model's unloaded service time for the request's prompt and
  requested output at decode batch 1.
- **Max queue wait** is the longest gateway wait (dispatch − arrival) of any ok request.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from llmserve.analysis.runs import Run
from llmserve.analysis.suites import SuiteRuns
from llmserve.scheduler.estimators import CostModel

CAP_THROUGHPUT_SHARE = 0.95
QUANTILES = (50, 95, 99)
DECILES = 10
VLLM_PRIORITY = "vllm-priority"
POLICY_ORDER = ("fifo", "spf", "sjf", "priority", VLLM_PRIORITY)


@dataclass(frozen=True)
class Cell:
    """One matrix point: the run and the (policy, ρ, seed) its config names."""

    policy: str
    rho: float
    seed: int
    run: Run


def policy_of(run: Run) -> str:
    cfg = run.config
    if cfg.gateway.enabled:
        return cfg.gateway.scheduler.name
    return VLLM_PRIORITY if cfg.server.scheduling_policy == "priority" else "none"


def cells(suite: SuiteRuns) -> list[Cell]:
    out = []
    for point in suite.points:
        cfg = point.run.config
        rho = cfg.load.arrival.rho if cfg.load.mode == "open" else None
        if rho is None:
            continue
        out.append(Cell(policy_of(point.run), float(rho), cfg.seed, point.run))
    return sorted(out, key=lambda c: (c.rho, _policy_rank(c.policy), c.seed))


def _policy_rank(policy: str) -> tuple[int, str]:
    return (
        (POLICY_ORDER.index(policy), "") if policy in POLICY_ORDER else (len(POLICY_ORDER), policy)
    )


def measured(run: Run) -> pd.DataFrame:
    """The ok, in-window requests of every rep of `run`."""
    frame = run.requests()
    if frame.empty:
        return frame
    return frame[frame["ok"] & frame["in_window"]]


def matrix_requests(suite: SuiteRuns) -> pd.DataFrame:
    """Every measured request of every cell, with `policy`, `rho` and `seed` columns."""
    frames = []
    for cell in cells(suite):
        frame = measured(cell.run)
        if frame.empty:
            continue
        frames.append(frame.assign(policy=cell.policy, rho=cell.rho, seed=cell.seed))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def class_ttft(requests: pd.DataFrame) -> pd.DataFrame:
    """Per (ρ, policy, class): n and TTFT P50/P95/P99 in seconds."""
    rows = []
    for (rho, policy, cls), group in requests.groupby(["rho", "policy", "workload_class"]):
        ttft = group["ttft_s"].dropna().to_numpy()
        row = {"rho": rho, "policy": policy, "workload_class": cls, "n": len(ttft)}
        for q in QUANTILES:
            row[f"p{q}"] = float(np.percentile(ttft, q)) if len(ttft) else float("nan")
        rows.append(row)
    return _sorted(pd.DataFrame(rows))


def policy_summary(suite: SuiteRuns) -> pd.DataFrame:
    """Per (ρ, policy): runs, mean generated tok/s and goodput across runs, and the max gateway
    wait over every measured request."""
    groups: dict[tuple[float, str], list[Cell]] = {}
    for cell in cells(suite):
        groups.setdefault((cell.rho, cell.policy), []).append(cell)
    rows = []
    for (rho, policy), group in groups.items():
        tput = [v for c in group for v in c.run.rep_metric("throughput.gen_toks_s")]
        good = [v for c in group for v in c.run.rep_metric("goodput_s")]
        waits = [measured(c.run)["queue_wait_s"].max() for c in group]
        rows.append(
            {
                "rho": rho,
                "policy": policy,
                "runs": len(group),
                "gen_toks_s": float(np.mean(tput)) if tput else float("nan"),
                "goodput_s": float(np.mean(good)) if good else float("nan"),
                "max_wait_s": float(np.nanmax(waits)) if waits else float("nan"),
            }
        )
    return _sorted(pd.DataFrame(rows))


def slowdown_deciles(requests: pd.DataFrame, model: CostModel) -> pd.DataFrame:
    """Per (ρ, policy, size decile): the decile's service-time range, n, and slowdown P50/P95."""
    if requests.empty:
        return pd.DataFrame()
    service = np.array(
        [
            model.service_time(p, o, 1)
            for p, o in zip(
                requests["prompt_tokens_req"], requests["output_tokens_req"], strict=True
            )
        ]
    )
    frame = requests.assign(service_s=service, slowdown=requests["e2e_s"] / service)
    rows = []
    for rho, at_rho in frame.groupby("rho"):
        edges = np.unique(np.quantile(at_rho["service_s"], np.linspace(0, 1, DECILES + 1)))
        decile = np.clip(np.searchsorted(edges, at_rho["service_s"], side="right") - 1, 0, None)
        decile = np.minimum(decile, len(edges) - 2)
        for (policy, d), group in at_rho.assign(decile=decile).groupby(["policy", "decile"]):
            s = group["slowdown"].dropna().to_numpy()
            rows.append(
                {
                    "rho": rho,
                    "policy": policy,
                    "decile": int(d) + 1,
                    "service_lo_s": float(edges[d]),
                    "service_hi_s": float(edges[d + 1]),
                    "n": len(s),
                    "p50": float(np.percentile(s, 50)) if len(s) else float("nan"),
                    "p95": float(np.percentile(s, 95)) if len(s) else float("nan"),
                }
            )
    return _sorted(pd.DataFrame(rows), "decile")


def _sorted(frame: pd.DataFrame, *extra: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    rank = frame["policy"].map(_policy_rank)
    keys = ["rho", "_rank", *extra, *(["workload_class"] if "workload_class" in frame else [])]
    return frame.assign(_rank=rank).sort_values(keys).drop(columns="_rank").reset_index(drop=True)


# --- in-flight cap ---


def inflight_cap_table(suite: SuiteRuns) -> pd.DataFrame:
    """Per swept cap: reps, mean generated tok/s, gateway wait P50 and max, E2E P95, and TTFT
    P95 per class (`ttft_p95_<class>` columns)."""
    rows = []
    for point in suite.points:
        if point.value is None:
            continue
        frame = measured(point.run)
        tput = point.run.rep_metric("throughput.gen_toks_s")
        row: dict[str, float | int] = {
            "cap": int(point.value),
            "reps": len(tput),
            "gen_toks_s": float(np.mean(tput)) if tput else float("nan"),
            "wait_p50_s": float(frame["queue_wait_s"].median()),
            "wait_max_s": float(frame["queue_wait_s"].max()),
            "e2e_p95_s": float(frame["e2e_s"].quantile(0.95)),
        }
        for cls, group in sorted(frame.groupby("workload_class"), key=lambda kv: str(kv[0])):
            row[f"ttft_p95_{cls}"] = float(group["ttft_s"].quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cap").reset_index(drop=True) if rows else pd.DataFrame()


def choose_cap(table: pd.DataFrame) -> int | None:
    """The smallest cap with ≥ 95% of the largest cap's generated tok/s; None with < 2 caps."""
    if len(table) < 2:
        return None
    target = CAP_THROUGHPUT_SHARE * float(table.iloc[-1]["gen_toks_s"])
    keep = table[table["gen_toks_s"] >= target]
    return int(keep.iloc[0]["cap"]) if not keep.empty else None
