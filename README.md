# LLMServeLab

A research platform for studying LLM-serving performance and request scheduling.

**Research question:** Can a workload-aware scheduling policy improve tail latency and GPU efficiency
compared with FIFO under heterogeneous LLM workloads?

**Why:** Production inference creates tradeoffs between latency, throughput, memory, and scheduling
that model-quality work doesn't cover.

**Findings:** None yet. Results go here only after experiments produce them.

## Status

Phase 1 (serving baseline). See [docs/PLAN.md](docs/PLAN.md) and the detailed
[implementation plan](docs/IMPLEMENTATION_PLAN.md).

## Setup

```bash
uv sync --extra dev
uv run pytest
```

## Layout

| Path | Contents |
|---|---|
| `llmserve/workload` | Prompt/output length control, arrival processes, trace replay |
| `llmserve/scheduler` | `Scheduler` interface plus FIFO, SJF, priority, WAAS |
| `llmserve/metrics` | TTFT/TPOT/ITL, GPU (NVML), server (vLLM `/metrics`) |
| `llmserve/runner` | Experiment runner: config → warm-up → run → store |
| `llmserve/analysis` | Aggregation, statistics, plots |
| `configs/` | Committed experiment configs; every figure maps to one |
| `results/` | Raw run output (git-ignored) |
