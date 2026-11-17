# LLMServeLab

A research platform for studying LLM-serving performance and request scheduling.

**Research question:** Can a workload-aware scheduling policy improve tail latency and serving
efficiency compared with FIFO under heterogeneous LLM workloads, on one Apple-silicon machine
serving a 7B model through vLLM (ADR-0013)?

**Why:** Production inference creates tradeoffs between latency, throughput, memory, and scheduling
that model-quality work doesn't cover.

## Findings so far

Measured on the M5 MacBook Air with vllm-metal 0.31.0 and Qwen2.5-7B-Instruct-AWQ. Every number
comes from stored runs through `llmserve report` or `llmserve.analysis`. Write-ups:
[where the time goes](docs/findings/where-the-time-goes.md) (Phase 4) and
[WAAS on a threshold server](docs/findings/waas-on-a-threshold-server.md) (Phases 6 and 7).

**Short answer to the research question: not on this platform.** The interactive tail in a mixed
workload comes mostly from long prompts prefilling inside vLLM, not from the order requests are
admitted. Setting vLLM's `--long-prefill-token-threshold 256` removes most of it. On that
server, sending requests straight to vLLM with no gateway beats every gateway ordering tried
below capacity, including WAAS. Past capacity (ρ 1.2), WAAS shortens the interactive tail only
by making generation-heavy requests wait far longer.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: throughput plateaus and P99 TTFT knees with concurrency | Refuted, with a caveat | The P99 knee at C = 2 comes before the plateau at C = 4, but closed-loop cohorts batch their prefills, so the knee is not saturation |
| H2: short requests see a worse tail in a mixed workload | Supported | Interactive P95 TTFT 15.3 s mixed against 2.7 s homogeneous at equal load (diff CI +10.4 to +13.6 s) |
| H3: prefill time fits prompt length | Supported | R² 0.999 |
| H4: shortest prompt first beats FIFO on the interactive tail at ρ ≥ 0.8 | Inconclusive | Gateway cap 16: ρ 0.8 33.3 s to 14.8 s (CI −35.6 to −4.3 s); ρ 0.95 27.1 s to 22.8 s (CI −13.1 to +1.3 s) |
| H5: WAAS beats FIFO on interactive P95 TTFT at no throughput cost | Inconclusive | Held-out seeds, threshold server: +0.37 s at ρ 0.85, −0.49 s at ρ 0.95 (CI −4.28 to +2.13 s), throughput within 1% |
| H6: WAAS gain against KV capacity and long-context share | Running | 36-point `kv-pressure` suite, results expected 17 November 2026 |

Key measurements behind the short answer, interactive P95 TTFT:

- **Server threshold.** At matched GPU clocks, threshold 256 cut FIFO's interactive P95 by
  about 10 s at ρ 0.8 and 0.95. Long-context P95 rose by 8 to 12 s, and throughput held.
- **No gateway against a capped gateway** (threshold server, ρ 0.95): 5.8 s with no gateway
  against 51.5 s for gateway FIFO and 28.4 s for gateway SPF at in-flight cap 16, at equal
  throughput. vLLM's own priority scheduling adds nothing over FCFS.
- **The cap is the cost.** At ρ 0.95 a gateway cap of 32 or more never fills, so ordering has no
  queue to act on. Cap 16 is the only cap that queues, and it alone produces the long tail.
- **Overload** (ρ 1.2, one clock-matched seed): WAAS 23.5 s against 39.4 s for FIFO and 38.7 s
  with no gateway, at equal throughput. Its generation-heavy P95 rose to 155 s, against 38 s
  and 18 s.
- **GPU clock confound.** The fanless Air's sustained GPU clock varies about 830 to 1170 MHz between
  runs, which alone moves interactive P95 several-fold at high load. Comparisons above pair runs
  back to back and report clocks.

## Status

- **Phase 0 (foundations):** data model, config schema, ADRs and CI are in place.
- **Phase 1 (serving baseline):** the streaming client, token-ID prompt builder, env check and
  smoke check are built and pass against the in-package mock backend — 100 sequential requests,
  0 failures, 0 usage mismatches. Against real vLLM (vllm-metal on the Mac) only 20- and
  30-request checks have run so far; the 100-request check is still to do.
- **Phase 2 (benchmark harness):** done. `llmserve benchmark <config>` runs warm-up, N
  repetitions, parquet output, per-rep and cross-rep summaries and full metadata against the
  mock or vllm-metal on the Mac. Known gaps carried into Phase 3 are listed in the
  implementation plan.
- **Phase 3 (baseline characterization):** done. `llmserve suite` runs the resumable sweeps in
  `configs/suites/`, and `llmserve report` builds the Baseline Performance Report (H1–H3
  verdicts, capacity, cost model) from them. Measured capacity is 0.4 req/s for 128/128 and
  interactive, 0.2 req/s for the mixed workload. Run order: [Phase 3 runbook](docs/runbooks/phase3.md).
- **Phase 4 (profiling):** done. powermetrics and vLLM `/metrics` sampling, per-request phase
  breakdowns and `llmserve trace`. Decode is 79% to 98% of end-to-end time, and in the mixed
  workload long-context prefills stall running requests for up to 5.6 s per step.
- **Phase 5 (scheduling baselines):** done. A gateway with an in-flight cap and FIFO, SPF, SJF
  and Priority schedulers. Greedy orders starve generation-heavy requests at ρ 0.95.
- **Phase 6 (WAAS):** done. Age-weighted cost order with a bypass bound
  ([ADR-0014](docs/adr/0014-waas-design-from-phase-5.md)), output estimators, the server
  threshold sweep, tuning (cost_weight 2, max_bypass 32) and the held-out H5 evaluation.
- **Phase 7 (evaluation):** in progress. Done: the gateway against no-gateway cross-check, the
  in-flight cap sweep, the overload suite and the regression baseline. Running: H6
  (`kv-pressure`). Left: resizing the `paper_01` matrix and one public trace replay.
- **Phase 8 (report and release):** not started.

Measurement platform: an M5 MacBook Air running vLLM 0.31.0 through vllm-metal, serving
`Qwen2.5-7B-Instruct-AWQ` (ADR-0013, [runbook](docs/runbooks/mac.md)). The earlier free-T4 route
(ADR-0011, Kaggle and Colab) is superseded and kept only as unused code.

See [docs/PLAN.md](docs/PLAN.md) and the detailed [implementation plan](docs/IMPLEMENTATION_PLAN.md).

## Setup

```bash
uv sync --extra dev --extra mock
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy llmserve tests
uv run llmserve validate $(find configs -name '*.yaml' -not -path 'configs/workloads/*')
```

### Phase 1 exit check (no GPU needed)

```bash
uv run python -m llmserve.mock --port 8001      # shell 1: mock OpenAI-compatible server
uv run python scripts/smoke.py configs/dev/smoke_mock.yaml   # shell 2: 100 sequential requests
```

Exit code 0 means every request ended `ok` (no timeout, HTTP error, connection error or usage
mismatch). Against vLLM on the Mac, start the server per [the Mac runbook](docs/runbooks/mac.md)
and run the same script on `configs/baseline/e01.yaml`.

## Layout

| Path | Contents |
|---|---|
| `llmserve/config` | Experiment config schema (pydantic), loader, config hash |
| `llmserve/workload` | Request specs, length distributions, prompt builder (exact token counts) |
| `llmserve/client` | SSE parser, streaming client (one request → `RequestRecord`), backend capabilities, API-key auth |
| `llmserve/mock` | Mock delay model + OpenAI-compatible SSE server for laptop development (ADR-009) |
| `llmserve/scheduler` | `Scheduler` interface plus FIFO, SJF, priority, WAAS |
| `llmserve/metrics` | TTFT/TPOT/ITL, records, GPU (NVML), server (vLLM `/metrics`) |
| `llmserve/runner` | Clock, env check, experiment runner: config → warm-up → run → store |
| `llmserve/analysis` | Aggregation, statistics, plots |
| `configs/` | Committed experiment configs; every figure maps to one |
| `docker/` | vLLM compose file (pinned image, §6 flags) — GPU host only |
| `docs/runbooks/` | [Mac](docs/runbooks/mac.md) (the measurement platform, ADR-0013); [Phase 3](docs/runbooks/phase3.md) run order; Kaggle and Colab (superseded T4 routes, ADR-0011) |
| `scripts/` | `smoke.py` (Phase 1 exit check), `envelope.py` (back-of-envelope model) |
| `results/` | Raw run output (git-ignored) |
| `docs/adr/` | Architecture decision records |
