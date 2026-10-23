# LLMServeLab

A research platform for studying LLM-serving performance and request scheduling.

**Research question:** Can a workload-aware scheduling policy improve tail latency and serving
efficiency compared with FIFO under heterogeneous LLM workloads, on one Apple-silicon machine
serving a 7B model through vLLM (ADR-0013)?

**Why:** Production inference creates tradeoffs between latency, throughput, memory, and scheduling
that model-quality work doesn't cover.

**Findings:** None yet. Results go here only after experiments produce them.

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
- **Phase 3 (baseline characterization):** code complete and sized for the Mac; runs pending. `llmserve suite`
  runs the resumable sweeps in `configs/suites/`, and `llmserve report` builds the Baseline
  Performance Report (H1–H3 verdicts, capacity, cost model) from them. The run order and
  session plan are in the [Phase 3 runbook](docs/runbooks/phase3.md).

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
