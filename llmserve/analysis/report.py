"""The Baseline Performance Report (plan Phase 3 item 7): `baseline.md`, its figures and the
fitted cost model, generated from suite manifests alone.

Rules:

- **Inputs** are the suite manifests under `<results_root>/_suites/` (`suites.load_suite_runs`).
  A suite with no manifest renders as "not yet run" with the command that runs it.
- **Reportability** (ADR-0009, ADR-0013): by default every run whose `Run.reportable` is False
  is excluded and listed with its reasons. With `dev=True` they are included, and a banner marks
  the report and every figure title as development results. Dev mode also falls back to the Mac
  dev suites (`dev-concurrency`, `dev-prompt-length`) when the study suite is absent.
- **Determinism:** the same inputs give byte-identical output. Nothing here reads the clock;
  every row is sorted; stale figures and cost models from an earlier build are removed.
- **Cost model:** prefill from the H3 fit; decode from per-rep TPOT P50 vs C over the
  concurrency sweep (the output-length sweep runs at a single C, so it cannot fit a slope).
  `cost_model.json` is written only when both fits exist.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from llmserve.analysis import plots
from llmserve.analysis.capacity import Capacity, capacity_points, find_capacity
from llmserve.analysis.hypotheses import (
    H2_CLASS,
    Verdict,
    h1,
    h2,
    h3,
    interactive_ttft,
    prefill_fit,
    prefill_points,
)
from llmserve.analysis.runs import Run, latest_run
from llmserve.analysis.stats import Estimate, mean_ci
from llmserve.analysis.suites import SuiteRuns, load_suite_runs, suite_file_for, sweep_table
from llmserve.scheduler.estimators import CostModel, DecodeFit, PrefillFit, fit_decode

DEV_BANNER = "DEVELOPMENT RESULTS — not reportable (ADR-0009/0013)"

CONCURRENCY = ("concurrency-p128", "dev-concurrency")
CONCURRENCY_MIXED = ("concurrency-mixed",)
PROMPT_LENGTH = ("prompt-length", "dev-prompt-length")
OUTPUT_LENGTH = ("output-length",)
CAPACITY = (
    ("p128", "capacity-p128"),
    ("mixed", "capacity-mixed"),
    ("interactive", "capacity-interactive"),
)
BURSTY = ("bursty",)
H2_MIXED = ("h2-mixed",)
H2_HOMOGENEOUS = ("h2-interactive",)
DEV_SUITES = frozenset({"dev-concurrency", "dev-prompt-length"})
BASELINE = tuple(f"e{n:02d}" for n in range(1, 8))
EXIT_SUCCESS_RATE = 0.99

SWEEP_METRICS = (
    "throughput.gen_toks_s",
    "latency.ttft_s.p50",
    "latency.ttft_s.p99",
    "latency.tpot_s.p50",
    "latency.tpot_s.p99",
)


@dataclass(frozen=True)
class Excluded:
    suite: str
    label: str
    run_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Exhibit:
    """A figure or table in the report and the runs it was drawn from."""

    ref: str
    title: str
    suite_files: tuple[str, ...]
    runs: tuple[tuple[str, Run], ...]  # (point label, run)


class _Report:
    """One build: loaded suites, exclusions, and the exhibits written so far."""

    def __init__(self, root: Path, out_dir: Path, dev: bool) -> None:
        self.root = root
        self.out_dir = out_dir
        self.dev = dev
        self.excluded: list[Excluded] = []
        self.exhibits: list[Exhibit] = []
        self._suites: dict[str, SuiteRuns | None] = {}

    def suite(self, names: Sequence[str]) -> SuiteRuns | None:
        """The first of `names` with a manifest (dev suites only in dev mode), filtered."""
        for name in names:
            if name in DEV_SUITES and not self.dev:
                continue
            loaded = self._load(name)
            if loaded is not None:
                return loaded
        return None

    def run(self, names: Sequence[str]) -> Run | None:
        suite = self.suite(names)
        return suite.points[0].run if suite is not None and suite.points else None

    def missing(self, names: Sequence[str]) -> str:
        """Why `names` has nothing to show: not yet run, or no reportable points."""
        name = names[0]
        for candidate in names:
            if candidate in self._suites and self._suites[candidate] is not None:
                return f"`{candidate}` has no reportable runs (see Excluded runs)."
        return f"Not yet run. Run it with `llmserve suite {suite_file_for(name)}`."

    def title(self, text: str) -> str:
        return f"{DEV_BANNER}\n{text}" if self.dev else text

    def save(self, fig: Figure, name: str, suites: Iterable[SuiteRuns], title: str) -> str:
        figures = self.out_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures / f"{name}.png", dpi=110, metadata={"Software": None})
        self.cite(f"figures/{name}.png", title, suites)
        return f"![{title}](figures/{name}.png)"

    def cite(self, ref: str, title: str, suites: Iterable[SuiteRuns]) -> None:
        suites = list(suites)
        self.exhibits.append(
            Exhibit(
                ref=ref,
                title=title,
                suite_files=tuple(s.suite_file for s in suites),
                runs=tuple((p.label, p.run) for s in suites for p in s.points),
            )
        )

    def _load(self, name: str) -> SuiteRuns | None:
        if name not in self._suites:
            loaded = load_suite_runs(self.root, name)
            if loaded is not None and not self.dev:
                for point in loaded.points:
                    reasons = point.run.unreportable_reasons
                    if reasons:
                        self.excluded.append(
                            Excluded(name, point.label, point.run.run_id, tuple(reasons))
                        )
                loaded = loaded.keep(lambda run: run.reportable)
            self._suites[name] = loaded
        return self._suites[name]


def build_report(results_root: Path, out_dir: Path, *, dev: bool = False) -> Path:
    """Write `baseline.md`, `figures/*.png` and (when fittable) `cost_model.json` to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted((out_dir / "figures").glob("*.png")):
        stale.unlink()
    (out_dir / "cost_model.json").unlink(missing_ok=True)

    r = _Report(results_root, out_dir, dev)
    concurrency = r.suite(CONCURRENCY)
    prompt = r.suite(PROMPT_LENGTH)
    verdicts = [
        h1(concurrency),
        h2(r.run(H2_MIXED), r.run(H2_HOMOGENEOUS)),
        h3(prompt),
    ]
    body = [
        *_baseline_section(r),
        *_capacity_section(r, prefill_fit(verdicts[2])),
        *_concurrency_section(r, verdicts[0]),
        *_prompt_section(r, prompt, verdicts[2]),
        *_output_section(r),
        *_bursty_section(r),
        *_h2_section(r, verdicts[1]),
        *_cost_model_section(r, concurrency, verdicts[2]),
    ]
    lines = ["# Baseline performance report", ""]
    if dev:
        lines += [f"> **{DEV_BANNER}**", ""]
    lines += [
        f"Generated by `llmserve report` from `{results_root.name}/_suites/`.",
        "",
        *_verdict_section(verdicts),
        *body,
        *_excluded_section(r),
        *_provenance_section(r),
    ]
    path = out_dir / "baseline.md"
    path.write_text("\n".join(lines).rstrip("\n") + "\n")
    return path


# --- sections ---


def _verdict_section(verdicts: Sequence[Verdict]) -> list[str]:
    lines = ["## Summary of hypotheses", "", "| ID | Hypothesis | Verdict | Evidence |"]
    lines.append("|---|---|---|---|")
    for v in verdicts:
        lines.append(f"| {v.id} | {v.statement} | **{v.status}** | {v.summary} |")
    lines.append("")
    lines.append(
        "Decision rules are fixed in `llmserve/analysis/hypotheses.py` and "
        "`llmserve/analysis/capacity.py` before any data exists (plan §5)."
    )
    return [*lines, ""]


def _baseline_section(r: _Report) -> list[str]:
    """E01–E07, run as plain benchmarks (`configs/baseline/`); the Phase 3 exit check needs
    each at 5 reps with ≥ 99% success."""
    lines = [
        "## Baseline experiments (E01–E07)",
        "",
        "| Exp | Load | Workload | Reps | Success ≥ 99% | TTFT P50 (ms) | TTFT P99 (ms) "
        "| TPOT P50 (ms) | gen tok/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    shown: list[tuple[str, Run]] = []
    for exp in BASELINE:
        run = latest_run(r.root, exp)
        if run is not None and not r.dev and run.unreportable_reasons:
            r.excluded.append(
                Excluded("baseline", exp, run.run_id, tuple(run.unreportable_reasons))
            )
            run = None
        if run is None:
            lines.append(
                f"| {exp} | — | — | — | — | Not yet run: "
                f"`llmserve benchmark configs/baseline/{exp}.yaml` | | | |"
            )
            continue
        shown.append((exp, run))
        cfg = run.config
        load = f"closed C={cfg.load.concurrency}" if cfg.load.mode == "closed" else "open"
        workload = ", ".join(c.name for c in cfg.workload.classes)
        rates = run.rep_metric("success_rate")
        ok = "yes" if rates and min(rates) >= EXIT_SUCCESS_RATE else "no"
        cells = [
            _est(mean_ci(run.rep_metric(metric)), scale, fmt)
            for metric, scale, fmt in (
                ("latency.ttft_s.p50", 1e3, ".1f"),
                ("latency.ttft_s.p99", 1e3, ".1f"),
                ("latency.tpot_s.p50", 1e3, ".2f"),
                ("throughput.gen_toks_s", 1.0, ".1f"),
            )
        ]
        lines.append(
            f"| {exp} | {load} | {workload} | {len(rates)} | {ok} | " + " | ".join(cells) + " |"
        )
    lines += ["", "Cells: across-rep mean [t-based 95% CI].", ""]
    if shown:
        r.exhibits.append(
            Exhibit(
                ref="table: baseline experiments",
                title="E01–E07",
                suite_files=tuple(f"configs/baseline/{exp}.yaml" for exp, _ in shown),
                runs=tuple(shown),
            )
        )
    return lines


def _capacity_section(r: _Report, prefill: PrefillFit | None) -> list[str]:
    lines = [
        "## Capacity (ρ = 1)",
        "",
        "The queueing test was amended in October 2026 to compare slowdown (TTFT ÷ the H3 fit's "
        "unloaded TTFT for the prompt length) instead of raw TTFT, because at low rates a "
        "heterogeneous prompt mix moves the raw median of each third with whichever long "
        "prompts land in it.",
        "",
        "| Workload | ρ = 1 (req/s) | Bracketed | Note |",
    ]
    lines.append("|---|---|---|---|")
    details: list[str] = []
    for workload, name in CAPACITY:
        suite = r.suite((name,))
        if suite is None or not suite.points:
            lines.append(f"| {workload} | — | — | {r.missing((name,))} |")
            continue
        cap = find_capacity(capacity_points(suite, prefill))
        r.cite(f"table: capacity ({workload})", f"Capacity evidence, {workload}", [suite])
        rate = f"{cap.rate_rps:g}" if cap.rate_rps is not None else "—"
        lines.append(f"| {workload} | {rate} | {'yes' if cap.bracketed else 'no'} | {cap.note} |")
        details += _capacity_evidence(workload, cap)
    return [*lines, "", *details]


def _capacity_evidence(workload: str, cap: Capacity) -> list[str]:
    lines = [
        f"**{workload}:** per offered rate (sustained iff completion ratio ≥ 0.95 and slowdown "
        "growth ≤ 2, raw TTFT growth without a prefill fit).",
        "",
        "| Offered (req/s) | Arrived (req/s) | Achieved (req/s) | Completion ratio "
        "| TTFT growth | Slowdown growth | Sustained |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in cap.evidence:
        p = e.point
        lines.append(
            f"| {p.offered_rps:g} | {p.arrived_rps:.3g} | {p.achieved_rps:.3g} "
            f"| {e.completion_ratio:.3f} "
            f"| {_num(p.ttft_growth, '.2f')} | {_num(p.slowdown_growth, '.2f')} "
            f"| {'yes' if e.sustained else 'no'} |"
        )
    return [*lines, ""]


def _concurrency_section(r: _Report, verdict: Verdict) -> list[str]:
    lines = ["## Concurrency sweep (H1)", ""]
    for names, workload in ((CONCURRENCY, "128/128"), (CONCURRENCY_MIXED, "mixed")):
        s = r.suite(names)
        lines += [f"### {workload}", ""]
        if s is None or not s.points:
            lines += [r.missing(names), ""]
            continue
        table = sweep_table(s, SWEEP_METRICS)
        lines += _sweep_markdown(table, "C")
        title = f"Latency vs concurrency ({workload}, {s.suite})"
        lines.append(
            r.save(
                plots.latency_vs_concurrency(table, title=r.title(title)),
                f"latency_vs_concurrency_{s.suite}",
                [s],
                title,
            )
        )
        title = f"Throughput vs concurrency ({workload}, {s.suite})"
        lines += [
            "",
            r.save(
                plots.throughput_vs_concurrency(table, title=r.title(title)),
                f"throughput_vs_concurrency_{s.suite}",
                [s],
                title,
            ),
            "",
        ]
    e = verdict.evidence
    lines += [
        f"H1: **{verdict.status}**. {verdict.summary} "
        f"(plateau C = {_num(e.get('plateau_c'))}, knee C = {_num(e.get('knee_c'))})",
        "",
    ]
    return lines


def _prompt_section(r: _Report, suite: SuiteRuns | None, verdict: Verdict) -> list[str]:
    lines = ["## Prompt length vs TTFT (H3)", ""]
    if suite is None or not suite.points:
        return [*lines, r.missing(PROMPT_LENGTH), ""]
    fit = prefill_fit(verdict)
    title = f"TTFT vs prompt length at C = 1 ({suite.suite})"
    lines.append(
        r.save(
            plots.prompt_length_ttft(prefill_points(suite), fit, title=r.title(title)),
            f"prompt_length_ttft_{suite.suite}",
            [suite],
            title,
        )
    )
    lines.append("")
    if fit is not None:
        lines += [
            f"Fit: TTFT ≈ {_poly(fit.a, fit.b, fit.c)} s "
            f"(R² = {fit.r2:.4f}, n = {fit.n} requests).",
            "",
        ]
    lines += [f"H3: **{verdict.status}**. {verdict.summary}", ""]
    return lines


def _output_section(r: _Report) -> list[str]:
    lines = ["## Output length vs TPOT (C = 8)", ""]
    suite = r.suite(OUTPUT_LENGTH)
    if suite is None or not suite.points:
        return [*lines, r.missing(OUTPUT_LENGTH), ""]
    r.cite("table: output length vs TPOT", "Output length vs TPOT", [suite])
    return [*lines, *_sweep_markdown(sweep_table(suite, SWEEP_METRICS), "output tokens")]


def _bursty_section(r: _Report) -> list[str]:
    lines = ["## Bursty arrivals", ""]
    suite = r.suite(BURSTY)
    if suite is None or not suite.points:
        return [*lines, r.missing(BURSTY), ""]
    run = suite.points[0].run
    requests, server = run.requests(), run.server()
    if not requests.empty:
        requests = requests[requests["rep"] == 0]
    if not server.empty:
        server = server[server["rep"] == 0]
    title = f"Bursty arrivals over time, rep 0 ({run.run_id})"
    lines.append(
        r.save(
            plots.bursty_timeseries(requests, server, title=r.title(title)),
            "bursty_timeseries",
            [suite],
            title,
        )
    )
    if server.empty:
        lines += ["", "No server `/metrics` samples: queue depth and KV usage are empty."]
    return [*lines, ""]


def _h2_section(r: _Report, verdict: Verdict) -> list[str]:
    lines = ["## Head-of-line blocking (H2)", ""]
    mixed_suite, homo_suite = r.suite(H2_MIXED), r.suite(H2_HOMOGENEOUS)
    mixed, homo = r.run(H2_MIXED), r.run(H2_HOMOGENEOUS)
    for names, run in ((H2_MIXED, mixed), (H2_HOMOGENEOUS, homo)):
        if run is None:
            lines += [f"`{names[0]}`: {r.missing(names)}", ""]
    if mixed is None or homo is None or mixed_suite is None or homo_suite is None:
        return [*lines, f"H2: **{verdict.status}**. {verdict.summary}", ""]
    title = f"TTFT CDF of {H2_CLASS} requests: mixed vs homogeneous workload"
    samples = {"mixed": interactive_ttft(mixed), "homogeneous": interactive_ttft(homo)}
    lines.append(
        r.save(
            plots.ttft_cdf(samples, title=r.title(title)),
            "h2_ttft_cdf",
            [mixed_suite, homo_suite],
            title,
        )
    )
    lines.append("")
    e = verdict.evidence
    if "p95" in e:
        lines += [
            "| Percentile | Mixed (ms) | Homogeneous (ms) | Δ (ms) | 95% CI of Δ (ms) |",
            "|---|---|---|---|---|",
        ]
        for q in ("p95", "p99"):
            d = e[q]
            lines.append(
                f"| {q.upper()} | {d['mixed_s'] * 1e3:.1f} | {d['homogeneous_s'] * 1e3:.1f} "
                f"| {d['diff_s'] * 1e3:+.1f} | {d['ci_lo_s'] * 1e3:+.1f} to "
                f"{d['ci_hi_s'] * 1e3:+.1f} |"
            )
        lines.append("")
        lines.append(
            f"Requests: {e['n_mixed']} mixed, {e['n_homogeneous']} homogeneous; bootstrap "
            f"B = {e['bootstrap']['resamples']}, seed {e['bootstrap']['seed']}."
            + (" P99 is indicative (< 1,000 requests)." if e["p99_indicative"] else "")
        )
        lines.append("")
    by_class = mixed.requests()
    keep = by_class[by_class["ok"] & by_class["in_window"]]
    per_class = {
        f"mixed: {name}": group["ttft_s"].dropna().to_numpy()
        for name, group in sorted(keep.groupby("workload_class"), key=lambda kv: str(kv[0]))
    }
    per_class[f"homogeneous: {H2_CLASS}"] = samples["homogeneous"]
    title = "TTFT CDF per workload class (h2-mixed and h2-interactive)"
    lines.append(
        r.save(
            plots.ttft_cdf(per_class, title=r.title(title)),
            "ttft_cdf_workloads",
            [mixed_suite, homo_suite],
            title,
        )
    )
    return [*lines, "", f"H2: **{verdict.status}**. {verdict.summary}", ""]


def _cost_model_section(r: _Report, concurrency: SuiteRuns | None, h3v: Verdict) -> list[str]:
    lines = ["## Cost model", ""]
    prefill = prefill_fit(h3v)
    decode = _decode_fit(concurrency)
    if prefill is None or decode is None:
        missing = [n for n, f in (("prefill", prefill), ("decode", decode)) if f is None]
        return [*lines, f"Not fitted: no {' or '.join(missing)} data.", ""]
    CostModel(prefill=prefill, decode=decode).save(r.out_dir / "cost_model.json")
    return [
        *lines,
        f"Prefill: TTFT ≈ {_poly(prefill.a, prefill.b, prefill.c)} s (R² = {prefill.r2:.4f}).",
        f"Decode: TPOT ≈ {decode.d0:.4g} {decode.d1:+.4g}·B s (R² = {decode.r2:.4f}, "
        f"n = {decode.n} rep points).",
        "",
        "Written to `cost_model.json`.",
        "",
    ]


def _decode_fit(suite: SuiteRuns | None) -> DecodeFit | None:
    if suite is None:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in suite.points:
        if point.value is None:
            continue
        values = point.run.rep_metric("latency.tpot_s.p50")
        xs += [point.value] * len(values)
        ys += values
    if len(set(xs)) < 2:
        return None
    return fit_decode(xs, ys)


def _excluded_section(r: _Report) -> list[str]:
    lines = ["## Excluded runs", ""]
    if r.dev:
        return [*lines, "None: development mode includes every run (see banner).", ""]
    if not r.excluded:
        return [*lines, "None.", ""]
    lines += ["| Suite | Point | Run | Reasons |", "|---|---|---|---|"]
    for e in sorted(r.excluded, key=lambda e: (e.suite, e.label, e.run_id)):
        lines.append(f"| {e.suite} | {e.label} | `{e.run_id}` | {'; '.join(e.reasons)} |")
    return [*lines, ""]


def _provenance_section(r: _Report) -> list[str]:
    lines = ["## Provenance", ""]
    if not r.exhibits:
        return [*lines, "No figures or tables.", ""]
    for fig in r.exhibits:
        lines += [
            f"**{fig.ref}** ({fig.title}). Suite files: "
            + ", ".join(f"`{f}`" for f in fig.suite_files),
            "",
            "| Point | Run | Config hash | Git SHA | Server (version) | Hardware | Thermal drift |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, run in fig.runs:
            md = run.metadata
            version = (md.get("backend_version") or {}).get("value")
            gpu = (md.get("gpu") or {}).get("name")
            lines.append(
                f"| {label} | `{run.run_id}` | `{str(md.get('config_hash'))[:12]}` "
                f"| `{md.get('git_sha')}`{' (dirty)' if md.get('git_dirty') else ''} "
                f"| {run.config.server.kind} ({version}) | {gpu or md.get('platform')} "
                f"| {_drift(run.thermal_drift)} |"
            )
        lines.append("")
    return lines


def _sweep_markdown(table: pd.DataFrame, axis: str) -> list[str]:
    cols = (
        ("throughput.gen_toks_s", "gen tok/s", 1.0, ".1f"),
        ("latency.ttft_s.p50", "TTFT P50 (ms)", 1e3, ".1f"),
        ("latency.ttft_s.p99", "TTFT P99 (ms)", 1e3, ".1f"),
        ("latency.tpot_s.p50", "TPOT P50 (ms)", 1e3, ".2f"),
        ("latency.tpot_s.p99", "TPOT P99 (ms)", 1e3, ".2f"),
    )
    lines = [
        f"| {axis} | " + " | ".join(c[1] for c in cols) + " | reps |",
        "|---|" + "---|" * (len(cols) + 1),
    ]
    for value, rows in table.groupby("value", sort=True):
        by_metric = {row.metric: row for row in rows.itertuples()}
        cells = []
        for metric, _, scale, fmt in cols:
            row = by_metric.get(metric)
            est = Estimate(row.mean, row.lo, row.hi, row.n) if row is not None else None
            cells.append(_est(est, scale, fmt))
        n = max((int(row.n) for row in by_metric.values()), default=0)
        lines.append(f"| {value:g} | " + " | ".join(cells) + f" | {n} |")
    lines.append("")
    lines.append("Cells: across-rep mean [t-based 95% CI].")
    return [*lines, ""]


def _est(est: Estimate | None, scale: float, fmt: str) -> str:
    if est is None or not math.isfinite(est.mean):
        return "—"
    text = format(est.mean * scale, fmt)
    if math.isfinite(est.lo) and math.isfinite(est.hi):
        text += f" [{format(est.lo * scale, fmt)}, {format(est.hi * scale, fmt)}]"
    return text


def _drift(drift: float | None) -> str:
    return "—" if drift is None else f"{drift:+.1%}"


def _poly(a: float, b: float, c: float) -> str:
    return f"{a:.4g} {b:+.4g}·p {c:+.4g}·p²"


def _num(value: Any, fmt: str = "g") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—" if value is None else "∞"
    return format(value, fmt)


# --- one run, for `llmserve analyze` ---


def describe_run(run: Run) -> str:
    """Multi-line human summary of one run: counts, validity, reportability, latencies with
    across-rep 95% CIs, and throughput."""
    md, summary = run.metadata, run.summary
    per_rep = summary["per_rep"]
    total = sum(s["n_requests"] for s in per_rep)
    ok = sum(s["n_ok"] for s in per_rep)
    measured = sum(s["n_measured"] for s in per_rep)
    failures: dict[str, int] = {}
    for s in per_rep:
        for status, count in s["failures_by_status"].items():
            failures[status] = failures.get(status, 0) + count
    version = (md.get("backend_version") or {}).get("value")
    reasons = run.unreportable_reasons
    failed = ", ".join(f"{k} {v}" for k, v in sorted(failures.items())) or "none"
    lines = [
        f"run {run.run_id}  (experiment {run.experiment})",
        f"  server: {run.config.server.kind} {version or ''}".rstrip()
        + f", model {run.config.server.model}",
        f"  config {str(md.get('config_hash'))[:12]}, git {md.get('git_sha')}"
        + (" (dirty)" if md.get("git_dirty") else ""),
        f"  reps {len(per_rep)}; requests {total}, ok {ok}, failed {total - ok} ({failed}); "
        f"measured {measured}",
        "  valid: "
        + (
            "no (reps " + ", ".join(map(str, summary.get("invalid_reps", []))) + ")"
            if summary["invalid"]
            else "yes"
        ),
        "  reportable: " + ("yes" if not reasons else "no (" + "; ".join(reasons) + ")"),
        f"  thermal drift (probe after vs before, TPOT P50): {_drift(run.thermal_drift)}",
        "  latency (across-rep mean [95% CI], ms):",
    ]
    for metric, name in (("ttft_s", "TTFT"), ("tpot_s", "TPOT"), ("e2e_s", "E2E")):
        cells = [
            f"{stat.upper()} "
            + _est(mean_ci(run.rep_metric(f"latency.{metric}.{stat}")), 1e3, ".1f")
            for stat in ("p50", "p95", "p99")
        ]
        lines.append(f"    {name:<5}" + "  ".join(cells))
    tput = [
        f"{label} {_est(mean_ci(run.rep_metric(f'throughput.{key}')), 1.0, '.2f')}"
        for key, label in (
            ("req_s", "req/s"),
            ("gen_toks_s", "gen tok/s"),
            ("total_toks_s", "total tok/s"),
        )
    ]
    lines.append("  throughput: " + "; ".join(tput))
    goodput = run.rep_metric("goodput_s")
    if goodput:
        lines.append(f"  goodput: {_est(mean_ci(goodput), 1.0, '.2f')} req/s meeting both SLOs")
    return "\n".join(lines)
