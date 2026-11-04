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
- **Prefill interference** buckets interactive requests by the long-context prompt tokens still
  prefilling when they were dispatched. vLLM schedules a running prefill's next chunk before any
  waiting prompt, so those tokens sit ahead of the new request whatever the gateway's order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from llmserve.analysis.profile import profile
from llmserve.analysis.runs import Run
from llmserve.analysis.stats import bootstrap_percentile_diff
from llmserve.analysis.suites import SuiteRuns
from llmserve.scheduler.estimators import CostModel

CAP_THROUGHPUT_SHARE = 0.95
QUANTILES = (50, 95, 99)
DECILES = 10
VLLM_PRIORITY = "vllm-priority"
POLICY_ORDER = ("fifo", "spf", "sjf", "priority", "waas", VLLM_PRIORITY)
THRESHOLD_TAG = "@lpt"
MIN_COMPARE = 20  # as H2 and H4 require per condition
INTERFERENCE_EDGES = (-1, 0, 2048, 4096, 8192, float("inf"))
INTERFERENCE_LABELS = ("none", "≤ 2k", "2k–4k", "4k–8k", "> 8k")


@dataclass(frozen=True)
class Cell:
    """One matrix point: the run and the (policy, ρ, seed) its config names."""

    policy: str
    rho: float
    seed: int
    run: Run


def policy_of(run: Run) -> str:
    """The gateway scheduler, or `vllm-priority`/`none` without a gateway, with its params as
    `[k=v,...]` when any are set, suffixed with `@lpt<T>` when vLLM ran with a long-prefill token
    threshold T."""
    cfg = run.config
    if cfg.gateway.enabled:
        spec = cfg.gateway.scheduler
        name = spec.name
        if spec.params:
            name += (
                "["
                + ",".join(
                    f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(spec.params.items())
                )
                + "]"
            )
    else:
        name = VLLM_PRIORITY if cfg.server.scheduling_policy == "priority" else "none"
    threshold = cfg.server.long_prefill_token_threshold
    return f"{name}{THRESHOLD_TAG}{threshold}" if threshold else name


def cells(suite: SuiteRuns) -> list[Cell]:
    out = []
    for point in suite.points:
        cfg = point.run.config
        rho = getattr(cfg.load.arrival, "rho", None)
        if rho is None:
            continue
        out.append(Cell(policy_of(point.run), float(rho), cfg.seed, point.run))
    return sorted(out, key=lambda c: (c.rho, _policy_rank(c.policy), c.seed))


def split_policy(policy: str) -> tuple[str, str, int]:
    """A `policy_of` label's scheduler name, params tag (`""` when none) and threshold (0 when
    unset)."""
    tagged, _, threshold = policy.partition(THRESHOLD_TAG)
    base, _, params = tagged.partition("[")
    return base, params.removesuffix("]"), int(threshold or 0)


def _policy_rank(policy: str) -> tuple[int, str, int, str]:
    base, params, threshold = split_policy(policy)
    known = base in POLICY_ORDER
    rank = POLICY_ORDER.index(base) if known else len(POLICY_ORDER)
    return rank, "" if known else base, threshold, params


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
    """Per (ρ, policy): runs, mean generated tok/s and goodput across runs, the max gateway
    wait over every measured request, and the runs' mean GPU clock."""
    groups: dict[tuple[float, str], list[Cell]] = {}
    for cell in cells(suite):
        groups.setdefault((cell.rho, cell.policy), []).append(cell)
    rows = []
    for (rho, policy), group in groups.items():
        tput = [v for c in group for v in c.run.rep_metric("throughput.gen_toks_s")]
        good = [v for c in group for v in c.run.rep_metric("goodput_s")]
        waits = [measured(c.run)["queue_wait_s"].max() for c in group]
        clocks = [m for c in group if not np.isnan(m := _gpu_mhz(c.run))]
        rows.append(
            {
                "rho": rho,
                "policy": policy,
                "runs": len(group),
                "gen_toks_s": float(np.mean(tput)) if tput else float("nan"),
                "goodput_s": float(np.mean(good)) if good else float("nan"),
                "max_wait_s": float(np.nanmax(waits)) if waits else float("nan"),
                "gpu_mhz": float(np.mean(clocks)) if clocks else float("nan"),
            }
        )
    return _sorted(pd.DataFrame(rows))


def _gpu_mhz(run: Run) -> float:
    power = profile(run).power
    return power.gpu_freq_mhz if power else float("nan")


def paired(suite: SuiteRuns, baseline: str, cls: str, q: float = 95.0) -> pd.DataFrame:
    """Per run: ρ, seed, policy, its mean GPU clock, `cls` TTFT `pct_q` over its measured
    requests, and that minus the `baseline` run's at the same ρ and seed (NaN without one).
    A paired suite runs a seed's policies back to back, so each difference compares runs on a
    similar machine state; the clock column shows how similar."""
    rows = []
    for cell in cells(suite):
        frame = measured(cell.run)
        ttft = frame.loc[frame["workload_class"] == cls, "ttft_s"].dropna()
        rows.append(
            {
                "rho": cell.rho,
                "seed": cell.seed,
                "policy": cell.policy,
                "gpu_mhz": _gpu_mhz(cell.run),
                "ttft": float(np.percentile(ttft, q)) if len(ttft) else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    base = frame[frame["policy"] == baseline].set_index(["rho", "seed"])["ttft"]
    keys = pd.MultiIndex.from_frame(frame[["rho", "seed"]])
    frame["diff"] = frame["ttft"].to_numpy() - base.reindex(keys).to_numpy()
    ordered = frame.assign(_rank=frame["policy"].map(_policy_rank))
    return (
        ordered.sort_values(["rho", "seed", "_rank"]).drop(columns="_rank").reset_index(drop=True)
    )


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


def prefill_interference(
    suite: SuiteRuns,
    victim: str = "interactive",
    blocker: str = "long_context",
    by: tuple[str, ...] = ("rho",),
) -> pd.DataFrame:
    """`interference` over every request of every cell of `suite`, warm-up included."""
    frames = [cell.run.requests().assign(rho=cell.rho, policy=cell.policy) for cell in cells(suite)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return interference(pd.concat(frames, ignore_index=True), victim, blocker, by)


def interference(
    requests: pd.DataFrame, victim: str, blocker: str, by: tuple[str, ...] = ("rho",)
) -> pd.DataFrame:
    """Per (`by` columns, bucket): how long measured `victim` requests wait inside the engine
    given the `blocker` prefill already in flight when they were dispatched. `by` defaults to
    ρ alone and may add `policy`.

    A blocker is in flight from its dispatch to its first token. Every ok blocker of the same
    rep counts, warm-up included, since it occupies the engine either way. The bucket is the
    summed prompt tokens of blockers in flight at the victim's dispatch, and server TTFT is
    dispatch to first token."""
    rows = []
    for keys, rep in requests[requests["ok"]].groupby([*by, "run_id", "rep"]):
        victims = rep[(rep["workload_class"] == victim) & rep["in_window"]]
        blockers = rep[rep["workload_class"] == blocker]
        if victims.empty:
            continue
        at = victims["t_dispatch"].to_numpy()[:, None]
        inside = (blockers["t_dispatch"].to_numpy() <= at) & (
            blockers["t_first_token"].to_numpy() > at
        )
        ahead = inside @ blockers["prompt_tokens_req"].to_numpy()
        server = (victims["t_first_token"] - victims["t_dispatch"]).to_numpy() / 1e9
        rows += [(*keys[: len(by)], a, s) for a, s in zip(ahead, server, strict=True)]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=[*by, "tokens_ahead", "server_ttft_s"])
    frame["bucket"] = pd.cut(frame["tokens_ahead"], INTERFERENCE_EDGES, labels=INTERFERENCE_LABELS)
    out = (
        frame.groupby([*by, "bucket"], observed=True)["server_ttft_s"]
        .agg(n="size", p50=lambda s: s.quantile(0.5), p95=lambda s: s.quantile(0.95))
        .reset_index()
    )
    out["share"] = out["n"] / out.groupby(list(by))["n"].transform("sum")
    return _sorted(out, "bucket") if "policy" in by else out


def against_baseline(
    requests: pd.DataFrame, baseline: str, q: float = 95.0, metric: str = "ttft_s"
) -> pd.DataFrame:
    """Per (ρ, policy other than `baseline`, class): `pct_q(policy) − pct_q(baseline)` of
    `metric` over measured requests, with its percentile bootstrap CI (`stats`, B = 10,000,
    seed 0). Cells with fewer than `MIN_COMPARE` requests on either side are left out."""
    rows = []
    for (rho, cls), at in requests.groupby(["rho", "workload_class"]):
        base = at.loc[at["policy"] == baseline, metric].dropna().to_numpy()
        if len(base) < MIN_COMPARE:
            continue
        for policy, group in at[at["policy"] != baseline].groupby("policy"):
            other = group[metric].dropna().to_numpy()
            if len(other) < MIN_COMPARE:
                continue
            ci = bootstrap_percentile_diff(other, base, q)
            rows.append(
                {
                    "rho": rho,
                    "policy": policy,
                    "workload_class": cls,
                    "n": ci.n_a,
                    "n_base": ci.n_b,
                    "value": ci.a,
                    "base": ci.b,
                    "diff": ci.diff,
                    "lo": ci.lo,
                    "hi": ci.hi,
                }
            )
    return _sorted(pd.DataFrame(rows))


def output_prior(suite: SuiteRuns) -> dict[str, int]:
    """Per class, the median `usage` output tokens of every ok request of every cell: the
    history the `class_prior` estimator reads (ADR-0014)."""
    frames = [cell.run.requests() for cell in cells(suite)]
    frame = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else None
    if frame is None or frame.empty:
        return {}
    ok = frame[frame["ok"]]
    medians = ok.groupby("workload_class")["output_tokens_usage"].median()
    return {str(cls): int(round(v)) for cls, v in sorted(medians.items())}


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
