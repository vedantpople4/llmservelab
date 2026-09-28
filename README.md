# LLMServeLab

A research platform for studying LLM-serving performance and request scheduling.

**Research question:** Can a workload-aware scheduling policy improve tail latency and GPU efficiency
compared with FIFO under heterogeneous LLM workloads?

**Why:** Production inference creates tradeoffs between latency, throughput, memory, and scheduling
that model-quality work doesn't cover.

**Findings:** None yet. Results go here only after experiments produce them.

## Status

- **Phase 0 (foundations):** data model, config schema, ADRs and CI are in place.
- **Phase 1 (serving baseline):** the streaming client, token-ID prompt builder, env check and
  smoke check are built and pass against the in-package mock backend — 100 sequential requests,
  0 failures, 0 usage mismatches. The vLLM run on a rented GPU is still pending:
  `docker/compose.yml` is authored but has never been executed.

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
mismatch). On a GPU host the same check runs against vLLM via `docker/compose.yml`, using
`configs/baseline/e01.yaml`.

## Layout

| Path | Contents |
|---|---|
| `llmserve/config` | Experiment config schema (pydantic), loader, config hash |
| `llmserve/workload` | Request specs, length distributions, prompt builder (exact token counts) |
| `llmserve/client` | SSE parser, streaming client (one request → `RequestRecord`), backend capabilities |
| `llmserve/mock` | Mock delay model + OpenAI-compatible SSE server for laptop development (ADR-009) |
| `llmserve/scheduler` | `Scheduler` interface plus FIFO, SJF, priority, WAAS |
| `llmserve/metrics` | TTFT/TPOT/ITL, records, GPU (NVML), server (vLLM `/metrics`) |
| `llmserve/runner` | Clock, env check, experiment runner: config → warm-up → run → store |
| `llmserve/analysis` | Aggregation, statistics, plots |
| `configs/` | Committed experiment configs; every figure maps to one |
| `docker/` | vLLM compose file (pinned image, §6 flags) — GPU host only |
| `scripts/` | `smoke.py` (Phase 1 exit check), `envelope.py` (back-of-envelope model) |
| `results/` | Raw run output (git-ignored) |
| `docs/adr/` | Architecture decision records |
