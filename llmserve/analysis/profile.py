"""What the engine and the Mac did during a run (plan Phase 4 items 1 to 3), from the samples the
run stored: vLLM's `/metrics` totals (`server.parquet`) and `powermetrics` (`power.parquet`).

- **vLLM's split.** vLLM times each finished request's queue, prefill and decode itself. The
  totals' change over a rep divided by the requests it finished is that rep's mean, which splits
  the breakdown's server wait into queueing and prefill slowed by shared steps. The per-request
  records cannot.
- **Energy.** Each power sample covers the `elapsed_ns` before it, so a rep's joules are
  `combined_power × elapsed` summed. Per output token, that is the rep's energy (idle and
  warm-up included) over the tokens its ok requests generated.
- **KV at dispatch.** The KV usage of the last `/metrics` sample at or before each request's
  dispatch, the harness-side stand-in for `kv_usage_at_dispatch` when there is no gateway.

Totals and energy cover whole reps, warm-up and cool-down included; reps are pooled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from llmserve.analysis.runs import Run

NOMINAL = "Nominal"


@dataclass(frozen=True)
class EngineSplit:
    """vLLM's own per-request means, and its batch size, over the reps of a run."""

    finished: int
    queue_s: float
    prefill_s: float
    decode_s: float
    tokens_per_step: float


@dataclass(frozen=True)
class MacPower:
    gpu_active: float  # mean of 1 - idle ratio
    gpu_freq_mhz: float
    power_w: float  # time-weighted combined CPU + GPU + ANE
    joules_per_token: float | None  # None when the run generated no tokens
    throttled_share: float  # share of samples with thermal pressure above Nominal


@dataclass(frozen=True)
class Profile:
    run: Run
    running_mean: float
    kv_peak: float
    engine: EngineSplit | None  # None without vLLM's timing totals
    power: MacPower | None  # None without power samples


def profile(run: Run) -> Profile:
    server = run.server()
    return Profile(
        run=run,
        running_mean=float(server["running"].mean()) if len(server) else float("nan"),
        kv_peak=float(server["kv_usage"].max()) if len(server) else float("nan"),
        engine=_engine(server),
        power=_power(run),
    )


def _engine(server: pd.DataFrame) -> EngineSplit | None:
    totals = ["finished_total", "queue_time_s_total", "prefill_time_s_total"]
    totals += ["decode_time_s_total", "steps_total", "step_tokens_total"]
    if server.empty or server[totals].isna().any().any():
        return None
    ordered = server.sort_values(["rep", "t_ns"])
    grouped = ordered.groupby("rep")[totals]
    delta = (grouped.last() - grouped.first()).sum()
    finished, steps = delta["finished_total"], delta["steps_total"]
    if finished <= 0 or steps <= 0:
        return None
    return EngineSplit(
        finished=int(finished),
        queue_s=float(delta["queue_time_s_total"] / finished),
        prefill_s=float(delta["prefill_time_s_total"] / finished),
        decode_s=float(delta["decode_time_s_total"] / finished),
        tokens_per_step=float(delta["step_tokens_total"] / steps),
    )


def _power(run: Run) -> MacPower | None:
    samples = run.power()
    if samples.empty:
        return None
    seconds = samples["elapsed_ns"].to_numpy(dtype=float) / 1e9
    joules = float((samples["combined_power_mw"].to_numpy(dtype=float) / 1e3 * seconds).sum())
    requests = run.requests()
    tokens = float(requests.loc[requests["ok"], "output_tokens_usage"].fillna(0).sum())
    return MacPower(
        gpu_active=float(samples["gpu_active_ratio"].mean()),
        gpu_freq_mhz=float(samples["gpu_freq_mhz"].mean()),
        power_w=joules / float(seconds.sum()),
        joules_per_token=joules / tokens if tokens > 0 else None,
        throttled_share=float((samples["thermal_pressure"] != NOMINAL).mean()),
    )


def kv_at_dispatch(run: Run) -> pd.DataFrame:
    """One row per measured ok request: `rep`, `workload_class`, `kv_usage` (0 to 1) at its
    dispatch, `ttft_s`. Empty without server samples."""
    server = run.server()
    columns = ["rep", "workload_class", "kv_usage", "ttft_s"]
    if server.empty:
        return pd.DataFrame(columns=columns)
    requests = run.requests()
    requests = requests[requests["in_window"] & requests["ok"]]
    frames = []
    for rep, reqs in requests.groupby("rep"):
        samples = server[server["rep"] == rep].sort_values("t_ns")
        i = np.searchsorted(samples["t_ns"].to_numpy(), reqs["t_dispatch"].to_numpy(), "right") - 1
        kv = np.where(i >= 0, samples["kv_usage"].to_numpy()[np.maximum(i, 0)], np.nan)
        frames.append(reqs[["rep", "workload_class", "ttft_s"]].assign(kv_usage=kv))
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)[columns]
    return out.dropna(subset=["kv_usage"]).reset_index(drop=True)
