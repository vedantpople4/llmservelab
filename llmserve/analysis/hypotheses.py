"""Verdicts for the Phase 3 hypotheses (plan Appendix B), each by a rule fixed before any data.

A verdict is `supported`, `refuted`, `inconclusive` (the data exist but cannot decide), or
`not-run` (an input suite or run is missing or has no usable points). Missing inputs never raise.

**H1, saturation** (concurrency sweep, 128/128). Per swept C, across-rep means of
`throughput.gen_toks_s` (T) and `latency.ttft_s.p99` (L).
- plateau = the smallest swept C whose next swept point (a doubling on the power-of-2 sweep)
  adds < 10% generated tok/s: `T(C_next) < 1.10 · T(C)`.
- knee = the smallest swept C with `L(C) > 2 · L(C_min)`, where C_min is the smallest swept C
  (C = 1 on the study sweep).
- supported iff both occur inside the sweep and knee ≥ plateau (tail latency blows up only once
  extra concurrency stops buying throughput, background.md §4); refuted iff both occur and
  knee < plateau; inconclusive if either is absent or fewer than 3 points have both metrics.

**H2, head-of-line blocking** (`h2-mixed` vs `h2-interactive`, open loop at equal ρ). TTFT of
the ok, in-window requests of class `interactive`, pooled across reps. For q ∈ {95, 99}, the
percentile bootstrap CI (`stats.bootstrap_percentile_diff`, B = 10,000, seed 0) of
`pct_q(mixed) − pct_q(homogeneous)`.
- supported iff the CI lower bound is > 0 for P95 or for P99;
- refuted iff for both P95 and P99 the CI upper bound is < 10% of the homogeneous percentile
  (any increase is smaller than 10%, with 95% confidence);
- inconclusive otherwise, or when either condition has fewer than 20 interactive requests.
P99 from fewer than 1,000 requests per condition is marked indicative (plan §5 rule 2).

**H3, prefill cost** (prompt-length sweep). Per-request `(prompt_tokens_req, ttft_s)` over the
ok, in-window requests of points run closed loop at C = 1 (other points are excluded: queueing
would confound prefill). Fit `TTFT ≈ a + b·p + c·p²` (`estimators.fit_prefill`).
- supported iff R² ≥ 0.95; refuted iff R² < 0.95; inconclusive with < 3 distinct lengths.

**H4, shortest prompt first** (`matrix-mixed`, plan Phase 5). Per ρ ≥ 0.8, TTFT of the ok,
in-window `interactive` requests of the SPF and FIFO points, each pooled across seeds. The
percentile bootstrap CI (B = 10,000, seed 0) of `P95(SPF) − P95(FIFO)`.
- supported iff the CI upper bound is < 0 at every ρ ≥ 0.8 with both policies;
- refuted iff at every such ρ the CI lower bound is > −10% of FIFO's P95 (any reduction is
  smaller than 10%, with 95% confidence);
- inconclusive otherwise, or when a condition has fewer than 20 interactive requests;
- not-run when no ρ ≥ 0.8 has both an SPF and a FIFO point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from llmserve.analysis.policies import matrix_requests
from llmserve.analysis.runs import Run
from llmserve.analysis.stats import DiffCI, bootstrap_percentile_diff
from llmserve.analysis.suites import SuiteRuns, sweep_table
from llmserve.scheduler.estimators import PrefillFit, fit_prefill

Status = Literal["supported", "refuted", "inconclusive", "not-run"]

H1_THROUGHPUT = "throughput.gen_toks_s"
H1_TAIL = "latency.ttft_s.p99"
PLATEAU_GAIN = 0.10
KNEE_FACTOR = 2.0
H1_MIN_POINTS = 3

H2_CLASS = "interactive"
H2_QUANTILES = (95.0, 99.0)
H2_EQUIVALENCE_MARGIN = 0.10
H2_MIN_SAMPLES = 20
P99_MIN_SAMPLES = 1000

H3_MIN_R2 = 0.95

H4_CLASS = "interactive"
H4_MIN_RHO = 0.8
H4_QUANTILE = 95.0
H4_EQUIVALENCE_MARGIN = 0.10

STATEMENTS = {
    "H1": "Throughput saturates with concurrency, and tail TTFT degrades sharply only at or "
    "after that saturation point.",
    "H2": "In a mixed workload, short interactive requests see worse tail TTFT than the same "
    "requests in a homogeneous workload at equal load (head-of-line blocking).",
    "H3": "At concurrency 1, TTFT is a quadratic in prompt length: TTFT ≈ a + b·p + c·p², "
    "with R² ≥ 0.95.",
    "H4": "In the mixed workload at ρ ≥ 0.8, shortest prompt first gives interactive requests "
    "lower P95 TTFT than FIFO.",
}


@dataclass(frozen=True)
class Verdict:
    id: str
    statement: str
    status: Status
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _verdict(hid: str, status: Status, summary: str, **evidence: Any) -> Verdict:
    return Verdict(hid, STATEMENTS[hid], status, summary, evidence)


# --- H1 ---


def h1(suite: SuiteRuns | None) -> Verdict:
    if suite is None or not suite.points:
        return _verdict("H1", "not-run", "The concurrency sweep has no usable runs.")
    verdict = judge_h1(sweep_table(suite, (H1_THROUGHPUT, H1_TAIL)))
    verdict.evidence["suite"] = suite.suite
    return verdict


def judge_h1(table: pd.DataFrame) -> Verdict:
    """Apply the H1 rule to a `sweep_table` holding `H1_THROUGHPUT` and `H1_TAIL`."""
    wide = table.pivot_table(index="value", columns="metric", values="mean", aggfunc="first")
    if H1_THROUGHPUT not in wide or H1_TAIL not in wide:
        return _verdict("H1", "inconclusive", "The sweep lacks throughput or P99 TTFT.")
    wide = wide[[H1_THROUGHPUT, H1_TAIL]].dropna().sort_index()
    cs = [float(c) for c in wide.index]
    tput = [float(v) for v in wide[H1_THROUGHPUT]]
    tail = [float(v) for v in wide[H1_TAIL]]
    evidence: dict[str, Any] = {
        "concurrency": cs,
        "gen_toks_s": tput,
        "ttft_p99_s": tail,
        "plateau_c": None,
        "knee_c": None,
    }
    if len(cs) < H1_MIN_POINTS:
        return _verdict(
            "H1", "inconclusive", f"Only {len(cs)} sweep points have both metrics.", **evidence
        )
    plateau = next(
        (cs[i] for i in range(len(cs) - 1) if tput[i + 1] < (1 + PLATEAU_GAIN) * tput[i]), None
    )
    knee = next((c for c, v in zip(cs, tail, strict=True) if v > KNEE_FACTOR * tail[0]), None)
    evidence.update(plateau_c=plateau, knee_c=knee, baseline_c=cs[0])
    if plateau is None or knee is None:
        missing = " and ".join(
            name for name, v in (("throughput plateau", plateau), ("P99 knee", knee)) if v is None
        )
        return _verdict("H1", "inconclusive", f"No {missing} inside C ≤ {cs[-1]:g}.", **evidence)
    if knee >= plateau:
        return _verdict(
            "H1",
            "supported",
            f"Throughput plateaus at C = {plateau:g} and P99 TTFT knees at C = {knee:g}.",
            **evidence,
        )
    return _verdict(
        "H1",
        "refuted",
        f"P99 TTFT knees at C = {knee:g}, before the throughput plateau at C = {plateau:g}.",
        **evidence,
    )


# --- H2 ---


def interactive_ttft(run: Run, workload_class: str = H2_CLASS) -> npt.NDArray[np.float64]:
    """TTFT (s) of the ok, in-window requests of `workload_class`, pooled across reps."""
    frame = run.requests()
    if frame.empty:
        return np.empty(0)
    keep = frame["ok"] & frame["in_window"] & (frame["workload_class"] == workload_class)
    values: npt.NDArray[np.float64] = frame.loc[keep, "ttft_s"].dropna().to_numpy(np.float64)
    return values


def h2(mixed: Run | None, homogeneous: Run | None) -> Verdict:
    if mixed is None or homogeneous is None:
        missing = [
            n for n, r in (("h2-mixed", mixed), ("h2-interactive", homogeneous)) if r is None
        ]
        return _verdict("H2", "not-run", f"Missing run: {', '.join(missing)}.")
    verdict = judge_h2(interactive_ttft(mixed), interactive_ttft(homogeneous))
    verdict.evidence.update(mixed_run=mixed.run_id, homogeneous_run=homogeneous.run_id)
    return verdict


def judge_h2(mixed_ttft: npt.ArrayLike, homogeneous_ttft: npt.ArrayLike) -> Verdict:
    a = np.asarray(mixed_ttft, dtype=np.float64)
    b = np.asarray(homogeneous_ttft, dtype=np.float64)
    evidence: dict[str, Any] = {"n_mixed": len(a), "n_homogeneous": len(b)}
    if min(len(a), len(b)) < H2_MIN_SAMPLES:
        return _verdict(
            "H2",
            "inconclusive",
            f"Too few interactive requests ({len(a)} mixed, {len(b)} homogeneous; "
            f"need {H2_MIN_SAMPLES} each).",
            **evidence,
        )
    cis = {q: bootstrap_percentile_diff(a, b, q) for q in H2_QUANTILES}
    for q, ci in cis.items():
        evidence[f"p{q:g}"] = {
            "mixed_s": ci.a,
            "homogeneous_s": ci.b,
            "diff_s": ci.diff,
            "ci_lo_s": ci.lo,
            "ci_hi_s": ci.hi,
        }
    evidence["p99_indicative"] = min(len(a), len(b)) < P99_MIN_SAMPLES
    evidence["bootstrap"] = {"resamples": cis[95.0].resamples, "seed": cis[95.0].seed}
    worse = [q for q, ci in cis.items() if ci.lo > 0]
    if worse:
        ci = cis[worse[0]]
        return _verdict(
            "H2",
            "supported",
            f"Interactive P{worse[0]:g} TTFT is {_pct(ci.diff, ci.b)} higher in the mixed "
            f"workload (95% CI {ci.lo * 1e3:+.1f} to {ci.hi * 1e3:+.1f} ms).",
            **evidence,
        )
    if all(ci.hi < H2_EQUIVALENCE_MARGIN * ci.b for ci in cis.values()):
        return _verdict(
            "H2",
            "refuted",
            "Mixing raises interactive P95 and P99 TTFT by less than 10% (95% CI upper bounds).",
            **evidence,
        )
    return _verdict(
        "H2",
        "inconclusive",
        "No significant P95/P99 increase, but the CIs do not exclude a ≥ 10% one.",
        **evidence,
    )


def _pct(diff: float, base: float) -> str:
    return f"{diff / base:+.0%}" if base > 0 else f"{diff * 1e3:+.1f} ms"


# --- H3 ---


def prefill_points(suite: SuiteRuns) -> pd.DataFrame:
    """`(prompt_tokens, ttft_s)` of every ok, in-window request at closed-loop C = 1."""
    frames = []
    for point in suite.points:
        load = point.run.config.load
        if load.mode != "closed" or load.concurrency != 1:
            continue
        frame = point.run.requests()
        if frame.empty:
            continue
        keep = frame[frame["ok"] & frame["in_window"]]
        frames.append(
            pd.DataFrame(
                {
                    "prompt_tokens": keep["prompt_tokens_req"].astype(float),
                    "ttft_s": keep["ttft_s"].astype(float),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["prompt_tokens", "ttft_s"])
    return pd.concat(frames, ignore_index=True).dropna()


def h3(suite: SuiteRuns | None) -> Verdict:
    if suite is None or not suite.points:
        return _verdict("H3", "not-run", "The prompt-length sweep has no usable runs.")
    points = prefill_points(suite)
    verdict = judge_h3(points["prompt_tokens"], points["ttft_s"])
    verdict.evidence["suite"] = suite.suite
    return verdict


def judge_h3(prompt_tokens: npt.ArrayLike, ttft_s: npt.ArrayLike) -> Verdict:
    p = np.asarray(prompt_tokens, dtype=np.float64)
    distinct = len(np.unique(p[np.isfinite(p)]))
    if distinct < 3:
        return _verdict(
            "H3",
            "inconclusive",
            f"Only {distinct} distinct prompt lengths at C = 1 (need 3).",
            n=len(p),
        )
    fit = fit_prefill(p, ttft_s)
    evidence = {"a": fit.a, "b": fit.b, "c": fit.c, "r2": fit.r2, "n": fit.n}
    if fit.r2 >= H3_MIN_R2:
        return _verdict("H3", "supported", f"Quadratic prefill fit R² = {fit.r2:.3f}.", **evidence)
    return _verdict(
        "H3", "refuted", f"Quadratic prefill fit R² = {fit.r2:.3f} < {H3_MIN_R2}.", **evidence
    )


def prefill_fit(verdict: Verdict) -> PrefillFit | None:
    """The fit an H3 verdict was judged on, or None when there was none."""
    e = verdict.evidence
    if "r2" not in e:
        return None
    return PrefillFit(a=e["a"], b=e["b"], c=e["c"], r2=e["r2"], n=e["n"])


# --- H4 ---


def h4(suite: SuiteRuns | None) -> Verdict:
    if suite is None or not suite.points:
        return _verdict("H4", "not-run", "matrix-mixed has no usable runs.")
    requests = matrix_requests(suite)
    samples: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    if not requests.empty:
        keep = requests[(requests["workload_class"] == H4_CLASS) & (requests["rho"] >= H4_MIN_RHO)]
        for rho, at_rho in keep.groupby("rho"):
            by_policy = {
                policy: group["ttft_s"].dropna().to_numpy(np.float64)
                for policy, group in at_rho.groupby("policy")
            }
            if "spf" in by_policy and "fifo" in by_policy:
                samples[float(rho)] = (by_policy["spf"], by_policy["fifo"])
    verdict = judge_h4(samples)
    verdict.evidence["suite"] = suite.suite
    return verdict


def judge_h4(
    samples: dict[float, tuple[npt.ArrayLike, npt.ArrayLike]],
) -> Verdict:
    """`samples` maps ρ to (SPF interactive TTFT, FIFO interactive TTFT)."""
    if not samples:
        return _verdict("H4", "not-run", f"No ρ ≥ {H4_MIN_RHO:g} has both SPF and FIFO runs.")
    evidence: dict[str, Any] = {"per_rho": {}}
    cis: dict[float, DiffCI] = {}
    for rho, (spf, fifo) in sorted(samples.items()):
        a = np.asarray(spf, dtype=np.float64)
        b = np.asarray(fifo, dtype=np.float64)
        entry: dict[str, Any] = {"n_spf": len(a), "n_fifo": len(b)}
        evidence["per_rho"][rho] = entry
        if min(len(a), len(b)) < H2_MIN_SAMPLES:
            continue
        ci = bootstrap_percentile_diff(a, b, H4_QUANTILE)
        cis[rho] = ci
        entry.update(spf_s=ci.a, fifo_s=ci.b, diff_s=ci.diff, ci_lo_s=ci.lo, ci_hi_s=ci.hi)
    short = [rho for rho in samples if rho not in cis]
    if short:
        return _verdict(
            "H4",
            "inconclusive",
            f"Too few interactive requests at ρ {', '.join(f'{r:g}' for r in sorted(short))} "
            f"(need {H2_MIN_SAMPLES} per policy).",
            **evidence,
        )
    worst = max(cis, key=lambda rho: cis[rho].hi)
    if all(ci.hi < 0 for ci in cis.values()):
        ci = cis[worst]
        return _verdict(
            "H4",
            "supported",
            f"SPF lowers interactive P95 TTFT at every ρ ≥ {H4_MIN_RHO:g}; the smallest gain is "
            f"{_pct(ci.diff, ci.b)} at ρ {worst:g} (95% CI {ci.lo:+.2f} to {ci.hi:+.2f} s).",
            **evidence,
        )
    if all(ci.lo > -H4_EQUIVALENCE_MARGIN * ci.b for ci in cis.values()):
        return _verdict(
            "H4",
            "refuted",
            f"SPF lowers interactive P95 TTFT by less than 10% at every ρ ≥ {H4_MIN_RHO:g} "
            "(95% CI lower bounds).",
            **evidence,
        )
    return _verdict(
        "H4",
        "inconclusive",
        f"SPF's P95 change at ρ {worst:g} is {_pct(cis[worst].diff, cis[worst].b)} "
        f"(95% CI {cis[worst].lo:+.2f} to {cis[worst].hi:+.2f} s), not a clear gain or a "
        "clear null.",
        **evidence,
    )
