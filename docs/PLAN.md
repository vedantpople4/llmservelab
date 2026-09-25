# Plan

Summary plan. The detailed architecture and phase-by-phase work items are in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), which takes precedence where the two differ
(for example, the mock server now lives in `llmserve/mock/` rather than `scripts/`).

Rule from the PRD: measure first, invent second. The scheduler comes last.

## Two decisions to make before writing much code

### 1. Where does "the scheduler" sit?

vLLM already schedules internally: continuous batching, chunked prefill, preemption. A scheduler in
front of vLLM only controls **admission order**, and it only has an effect when requests actually wait
in *our* queue. So the gateway must cap in-flight requests (e.g. `max_in_flight = 16`). Otherwise
everything passes straight into vLLM's queue and every policy measures the same.

| Option | What it controls | Cost |
|---|---|---|
| A. Gateway admission control (FastAPI proxy with a queue + in-flight cap) | Which request enters vLLM next | Low; independent of vLLM version |
| B. vLLM built-in `--scheduling-policy priority` | vLLM's own waiting queue, via per-request priority | Low; limited to a priority number |
| C. Patch vLLM's scheduler class | Batch composition, prefill/decode interleaving | High; tied to one pinned vLLM version |

**Recommendation:** Start with A as the main research surface, and use B as a cross-check. Only
consider C if Phase 5 data shows admission order can't fix the problem. Write this down in the paper's
System Design section, because a reviewer will ask about it.

### 2. Where do experiments run?

A Mac has no CUDA, so vLLM benchmarks need a rented GPU (L4 or A10 is cheapest for a 7B model).
To keep GPU hours low:

- Build a **mock OpenAI-compatible streaming server** (`scripts/mock_server.py`) whose TTFT grows
  with prompt length and whose per-token delay grows with concurrency. Develop and test the whole
  harness on the laptop against it.
- Rent the GPU only for real runs, and keep one hardware SKU for the whole study. Record it in the
  metadata.

## Phases and milestones

Each milestone ends with a check that can be verified. Rough time estimates assume part-time work.

| # | Milestone | Done when | Est. |
|---|---|---|---|
| 0 | Fundamentals | You can explain prefill vs decode, KV cache, PagedAttention, continuous batching. Read Orca, vLLM, Sarathi-Serve, DistServe. Notes in `docs/background.md` | 1 wk |
| 1 | Serving baseline | vLLM + one 7B model in Docker on a GPU; streaming client records per-token timestamps; 100 sequential requests, 0 unexplained failures | 1 wk |
| 2 | Benchmark harness | `llmserve benchmark configs/baseline/e01.yaml` runs warm-up → workload → Parquet + `metadata.json` (git commit, GPU, CUDA, driver, vLLM version, seed). Workload generator is deterministic under a seed (unit-tested). Mock server lets this run on the laptop | 2 wk |
| 3 | Baseline characterization | E01–E07 × 5 reps; plots of latency/throughput vs concurrency, prompt length vs TTFT; bursty and mixed runs. Verify H1 and H2. **Deliverable: Baseline Performance Report** | 2 wk |
| 4 | GPU profiling | NVML sampler (util, mem, power) + scraping vLLM `/metrics` (KV-cache usage, running/waiting); one Nsight Systems trace explaining where time goes | 1 wk |
| 5 | Scheduling baselines | Gateway with in-flight cap; FIFO, ShortestPromptFirst, Priority behind the `Scheduler` interface; compared on identical seeded workloads. Find the concrete weakness (likely head-of-line blocking of short requests behind long prefills) | 2 wk |
| 6 | WAAS | Scheduler designed from the Phase 5 evidence. Include starvation protection (age term) and run the oracle / estimated / no output-length ablation | 2–3 wk |
| 7 | Evaluation | Automated matrix (3 schedulers × 4 workloads × 5 concurrencies × 3 reps); report means, P50/P95/P99, CIs, and throughput cost next to latency gains | 1–2 wk |
| 8 | Report + release | `make reproduce`, 8–12 page paper, README findings filled with measured numbers only | 2 wk |

MVP = end of Phase 4 (PRD §42). Version 1.0 = end of Phase 8.

## Build order inside Phase 1–2 (first code to write)

1. `llmserve/workload/distributions.py`: seeded prompt/output length sampling. Build prompts to an
   exact token count with the model's tokenizer, not by counting words.
2. `llmserve/runner/client.py`: async httpx streaming client that records `start` and every token
   timestamp into `RequestTiming`. Send `ignore_eos: true` with `max_tokens` so output length is
   controlled.
3. `scripts/mock_server.py`: fake backend for laptop development.
4. `llmserve/workload/generator.py`: arrival processes (constant, Poisson, bursty) plus a
   concurrency limiter.
5. `llmserve/runner/experiment.py` + `llmserve/cli.py benchmark`: wire config → run → Parquet +
   metadata.
6. `llmserve/analysis/aggregate.py`: per-run percentiles and cross-repetition stats.

## Pitfalls to design around early

- **Client-side bottleneck:** at high concurrency, the Python load generator can become the
  bottleneck. Use asyncio (not threads), and sanity-check it against the mock server at high
  concurrency.
- **Tokenization mismatch:** token counts must come from the server's tokenizer; use the `usage`
  field in responses as ground truth.
- **Prefix caching:** identical synthetic prompts get cached and make TTFT look better than it is.
  Randomize prompt content per request, or disable prefix caching explicitly and record that choice.
- **Clock:** use `time.perf_counter()` for durations and wall-clock time only for metadata.
- **Pin versions:** vLLM image tag, model revision hash, CUDA/driver versions. Store all of them in
  every run's metadata.
