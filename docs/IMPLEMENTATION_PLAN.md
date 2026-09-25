# LLMServeLab: Implementation Plan

This is the detailed engineering plan: architecture, design decisions, and phase-by-phase work
items. It expands [PLAN.md](PLAN.md) and the PRD. Where the two differ, this document wins, and the
reason is stated.

Guiding rule from the PRD: **measure first, invent second.** Every phase ends with an exit check
that can be verified. Nothing in Phase 6 (WAAS) is fixed until Phase 5 data justifies it.

---

## Part I: Architecture

### 1. System overview

```
                        ┌───────────────────────── benchmark process (one clock) ─────────────────────────┐
 experiment.yaml ──►    │  Config loader ─► Workload materializer ─► workload.parquet (frozen, seeded)    │
                        │                                   │                                              │
                        │                                   ▼                                              │
                        │                 Load driver (closed-loop workers | open-loop arrival clock)      │
                        │                                   │ RequestSpec + t_arrival                      │
                        │                                   ▼                                              │
                        │            Gateway core: queue + Scheduler.select_next + in-flight cap           │
                        │                                   │ t_dispatch                                   │
                        │                                   ▼                                              │
                        │         Streaming client (httpx, SSE) ── records every chunk timestamp           │
                        │                                   │                                              │
                        │   Samplers: NVML/DCGM thread (10 Hz) · vLLM /metrics scraper (2 Hz) · event log  │
                        └───────────────────────────────────┼──────────────────────────────────────────────┘
                                                            ▼
                                       vLLM OpenAI server (pinned image) ── GPU
                                                            │
             results/<exp>/<run_id>/rep-NN/{requests,chunks,gpu,server}.parquet, events.jsonl, metadata.json
                                                            ▼
                               Analysis: per-run summaries ─► cross-rep stats ─► plots ─► report
```

### 2. Architecture decisions (write each as an ADR in `docs/adr/`)

**ADR-001: The scheduler controls admission order at a gateway with a cap on requests in flight.**
vLLM already does continuous batching, chunked prefill and preemption. A scheduler in front of it
only has an effect when requests wait in *our* queue, so the gateway caps in-flight requests
(`max_in_flight`). vLLM's own `--scheduling-policy priority` is a cross-check (Phase 5). We don't
patch vLLM's scheduler unless the Phase 5 data shows that admission order can't fix the problem.

**ADR-002: The gateway core is a library; experiments run it in-process.**
`llmserve/gateway/core.py` is an asyncio admission controller with no HTTP dependency. Experiments
run it in the benchmark process, so all timestamps (arrival, dispatch, first token) come from one
monotonic clock and no proxy hop adds latency. A thin FastAPI wrapper (`gateway/app.py`) exposes
the same core as an OpenAI-compatible proxy. It serves demos, external clients, and a one-time
check that proxy mode and in-process mode give the same latency within noise.
*Why not only a proxy:* the proxy adds a network hop and a second clock domain. Aligning client and
proxy timestamps is error-prone, and a reviewer will ask about it.

**ADR-003: Closed-loop and open-loop load are different experiment modes.**
- *Closed-loop* (`concurrency: C`): C workers, and each sends its next request when the previous
  one finishes. Used for E01–E07 and saturation curves (H1).
- *Open-loop* (`arrival: poisson|constant|bursty|replay`): requests arrive on a clock that doesn't
  depend on completions. Used for scheduler evaluation (H4–H6), because queueing, and therefore
  scheduling, only matters when arrivals can exceed service capacity.

The scaffold's `e01.yaml` mixes the two (`type: constant` together with `concurrency`). The new
schema separates them.

Open-loop load is expressed as **utilization ρ = arrival rate / measured capacity**, where capacity
is the saturation throughput from Phase 3 on the same hardware. Results then mean the same thing on
different GPUs, and "ρ = 0.9" is easier to read than "15 req/s".

**ADR-004: Workloads are materialized before a run and frozen.**
The generator writes `workload.parquet` (one row per request: arrival offset, class, prompt token
IDs hash, prompt length, output length, priority, SLO, output-length estimate) before any request is
sent. Every scheduler in a comparison replays *the same file*, which makes the comparison paired.
Trace replay is the same code path with an external file.

**ADR-005: Prompts are sent as token-ID arrays to `/v1/completions`.**
vLLM accepts `prompt: [int, ...]`. That gives exact prompt lengths with no chat template and no
detokenize/retokenize drift. Token IDs are drawn from real text (a bundled public-domain corpus,
tokenized once with the pinned tokenizer), starting at a random offset per request. Every prompt
also starts with a unique nonce block, so prefix caching can't produce cache hits. Prefix caching is
*also* disabled on the server (`--no-enable-prefix-caching`), and the setting is asserted at run
start. Output length is forced with `max_tokens = min_tokens = N` and `ignore_eos: true`. The
`usage` block (`stream_options.include_usage`) is ground truth, and any mismatch is logged per
request.

**ADR-006: Time.** Durations use `time.perf_counter_ns()`. At run start we record the pair
`(perf_counter_ns, time.time_ns)` once, and use it to align GPU and server samples to the client
clock. Samplers run on the same host and use the same monotonic clock.

**ADR-007: The unit of statistical replication is a run, not a request.**
Requests within a run are correlated through shared queue state. Percentiles are computed per run.
Confidence intervals come from variation across repetitions. See §5.

**ADR-008: Configs are validated by pydantic models.** One schema serves `llmserve validate`, the
runner, and the analysis code, and it is tested. Add `pydantic>=2` as a dependency.

**ADR-009: Local backends on a Mac for development.** `server.kind` accepts `mock`, `mlx`,
`llamacpp` and `vllm`. Each backend declares its capabilities (token-ID prompts, forced output
length, continuous batching, `/metrics`, prefix-cache control), and the harness adapts. Phases 1–2
can be completed mostly on a Mac. Only `kind: vllm` results may appear in reports.

The full ADRs are in [docs/adr/](adr/README.md).

### 3. Target module layout

This extends PRD §24. Changes from PLAN.md: the mock server moves into the package so tests can
import it, and new `config/`, `client/`, `gateway/` and `mock/` packages are added.

```
llmserve/
  config/     schema.py (pydantic models) · loader.py (YAML → model, `ref` resolution, hash)
  workload/   spec.py (RequestSpec, Workload) · distributions.py · arrivals.py · prompts.py
              classes.py (workload classes → specs) · generator.py (materialize) · replay.py (traces)
  client/     sse.py (streaming parser) · openai_stream.py (one request → RequestRecord)
  gateway/    core.py (AdmissionController) · app.py (FastAPI proxy)
  scheduler/  base.py · fifo.py · shortest.py · priority.py · workload_aware.py
              estimators.py (output-length + cost models) · registry.py (name → class)
  metrics/    latency.py · records.py (RequestRecord schema) · gpu.py (NVML/DCGM) · server.py
  runner/     env_check.py · metadata.py · run.py (one repetition) · experiment.py (reps)
              suite.py (matrix expansion, resumable) · storage.py (Parquet/JSON layout)
  analysis/   aggregate.py · stats.py · plots.py · report.py · trace.py (per-request timeline)
  mock/       engine.py (discrete-time continuous-batching simulator) · server.py (OpenAI SSE)
  cli.py
configs/      baseline/ · workloads/ · schedulers/ · suites/
experiments/  paper_01.yaml (the suite that produces the paper's figures)
docker/       compose.yml (vllm + dcgm-exporter) · Dockerfile (harness image)
scripts/      calibrate_mock.py · nsys_profile.sh · fetch_traces.py
paper/        figures.yaml (figure id → suite/config → plot function) · main.tex
reproduce.py  Makefile
```

### 4. Data model

**Request record** (`requests.parquet`, one row per request). This replaces the scaffold's
two-field `RequestTiming`:

| Column | Type | Notes |
|---|---|---|
| `run_id`, `rep`, `request_id` | str, int, str | `request_id` is stable across schedulers (from the workload file) |
| `workload_class`, `priority`, `slo_ttft_ms`, `slo_tpot_ms` | | from the spec |
| `prompt_tokens_req`, `output_tokens_req`, `output_tokens_est` | int | est is the scheduler's view; req is the oracle value |
| `t_arrival`, `t_enqueue`, `t_dispatch`, `t_first_token`, `t_last_token`, `t_complete` | int64 ns | on the client monotonic clock, relative to run start |
| `prompt_tokens_usage`, `output_tokens_usage` | int | from the server; must match `*_req` |
| `n_chunks` | int | stream chunks can carry more than one token |
| `status` | enum | `ok`, `timeout`, `http_error`, `conn_error`, `aborted`, `length_mismatch` |
| `http_status`, `error` | int, str | |
| `sched_name`, `sched_score`, `queue_depth_at_dispatch`, `in_flight_at_dispatch`, `kv_usage_at_dispatch`, `sched_decision_ns`, `bypass_count` | | the scheduler's decision context (PRD §26) |

Derived metrics live in `metrics/latency.py` as pure functions of a record:

- `queue_wait = t_dispatch − t_arrival`
- `ttft = t_first_token − t_arrival`. This is the user-visible TTFT and the headline metric.
- `ttft_server = t_first_token − t_dispatch`
- `e2e = t_last_token − t_arrival`
- `tpot = (t_last_token − t_first_token) / (output_tokens_usage − 1)`. This fixes the scaffold,
  which counted chunks instead of tokens.
- ITL is taken from `chunks.parquet`. A chunk that carries k tokens counts as k ITL samples of Δt/k,
  and this choice is documented.
- `slowdown = e2e / e2e_isolated(prompt, output)`, where isolated latency comes from the Phase 3
  single-stream fit. This is the standard fairness measure for size-based scheduling.

Every derived metric returns `None` for a failed request instead of raising an error.

**Chunk table** (`chunks.parquet`): `request_id, idx, t_ns, n_tokens`.
**GPU samples** (`gpu.parquet`): `t_ns, gpu_idx, util_pct, mem_used, mem_total, power_w, temp_c,
sm_clock, mem_clock`, plus `sm_active`, `sm_occupancy` and `dram_active` when DCGM is available.
**Server samples** (`server.parquet`): `t_ns, running, waiting, kv_usage, preemptions_total,
prompt_tokens_total, generation_tokens_total`.
**Events** (`events.jsonl`): enqueue, dispatch (with the full scheduler score vector for the top-k
candidates), complete, error, sampler gaps.

**Run directory:**
```
results/<experiment>/<run_id>/          run_id = <utc-ts>-<config_hash8>-<git_sha7>
  config.resolved.yaml  metadata.json  workload.parquet
  rep-00/ requests.parquet chunks.parquet gpu.parquet server.parquet events.jsonl summary.json
  rep-01/ ...
  summary.json   (cross-rep statistics)
```

**metadata.json** contains the PRD §16 fields plus:
- `git_dirty`. Paper runs refuse to start on a dirty tree unless `--allow-dirty` is passed.
- `config_hash`, `uv_lock_hash`, host name, CPU model
- GPU name and UUID, driver, CUDA
- vLLM version (from `/version`) and the full vLLM launch args
- model revision SHA, tokenizer revision
- `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, prefix
  caching state
- seed and child-seed keys, gateway settings, scheduler name and parameters
- `clock_anchor`, and the list of which samplers were active

### 5. Statistics and measurement rules

These rules are fixed before any data exists, so results can't be cherry-picked.

1. **Repetitions:** at least 5 per configuration, and 10 for the headline scheduler comparison.
   With n = 5, a two-sided Wilcoxon signed-rank test can't go below p = 0.0625, so p-values alone
   are not enough. Report effect sizes with bootstrap 95% CIs over repetitions (paired by workload
   seed), and use a paired t-test as a secondary check.
2. **Tail stability:** P99 needs about 1,000 requests per run to be meaningful (about 10 samples
   beyond the percentile). Runs used for P99 claims have at least 1,000 measured requests. Below
   that, the report shows P95 and marks P99 as indicative.
3. **Steady state:**
   - *Closed-loop:* count only requests dispatched while in-flight = C. The ramp-up and drain
     tails are excluded, and the window is recorded in `summary.json`.
   - *Open-loop:* discard requests that arrive in the first `warmup_s` and the last `cooldown_s`,
     but keep sending them, so the system stays loaded.
4. **Throughput** = tokens (or requests) completed inside the steady window ÷ window length.
   Report input, output and total tokens/s, and req/s (PRD §8.5).
5. **Goodput** = req/s that meet *both* the TTFT and TPOT SLOs of their class (from DistServe).
   This is the single number that captures the latency/throughput trade-off (PRD §41: measure it,
   don't hide it).
6. **Order effects:** runs in a comparison are interleaved in randomized order (for example
   rep 0: FIFO → WAAS → SPF; rep 1: SPF → FIFO → WAAS), and the order is recorded. This keeps
   thermal drift and noisy neighbours from biasing one scheduler.
7. **Failures** are never dropped silently. The success rate is part of every summary. A run with
   more than 1% unexplained failures is marked invalid and appears in a failures table.
8. **Claim format** (PRD §20): absolute difference, percentage difference, and CI, for example
   "P95 TTFT 2.80 s → 2.10 s (−25%, 95% CI −19% to −31%, 10 paired reps)".

### 6. Hardware and model sizing

- **Model:** `Qwen/Qwen2.5-7B-Instruct` in bf16, pinned by revision SHA. Weights take about 15 GB.
- **KV cost per token:** 2 (K, V) × 28 layers × 4 KV heads × 128 head dim × 2 bytes ≈ **56 KiB**.
- **GPU (one SKU for the whole study):** a 24 GB card (L4 or A10-class). With
  `gpu_memory_utilization = 0.90`, about 21.6 GB is usable. Minus weights and activations, that
  leaves roughly 5–6 GB of KV, or about 100k tokens: about 12 concurrent 8K-token requests, or about
  400 concurrent 256-token requests. KV pressure (H6) is then reachable at realistic mixes. A 48 GB
  or 80 GB card would hide it unless load is pushed very high.
- **Back-of-envelope model** (goes in `docs/background.md`, and is compared with measurements in
  Phase 3):
  - Prefill is roughly compute-bound: FLOPs ≈ 2 × 7.6e9 × prompt tokens. For 1K tokens that is
    about 15.6 TFLOP, so tens to a few hundred ms depending on the card and MFU.
  - Decode at small batch is memory-bound: time per step ≈ bytes read (weights + KV) ÷ memory
    bandwidth. On a ~300 GB/s card that is about 50 ms per token at batch 1. On a ~600 GB/s card it
    is about 25 ms.

  Fill these in with the chosen card's datasheet numbers.
- **Server settings** (recorded, and held constant unless they are the variable under test):
  - `--max-model-len 10240` (8K prompt + 1K output + margin)
  - `--gpu-memory-utilization 0.90`
  - `--no-enable-prefix-caching`
  - `--max-num-seqs 256`
  - `--max-num-batched-tokens`: vLLM's default, recorded. This is the chunked-prefill budget and a
    key variable for H4, because it sets how much a long prefill can delay decodes.
  - `--disable-log-requests`
  - the image is pinned by digest

**GPU budget (rough; recompute after Phase 1 measures real per-request cost):**

| Phase | Work | Est. GPU-hours |
|---|---|---|
| 1 | bring-up, 100-request check | 2–3 |
| 2 | harness validation on real GPU | 2–3 |
| 3 | E01–E07 × 5 reps, saturation sweeps, bursty | 10–15 |
| 4 | profiling runs, one Nsight trace | 3–5 |
| 5 | 3 policies × 3 workloads × 3 loads × 5 reps + cap sweep + vLLM-priority cross-check | 12–18 |
| 6 | WAAS iteration + ablations | 10–15 |
| 7 | 180-run matrix + headline 10-rep comparison | 30–40 |
| **Total** | including about 30% debugging overhead | **≈ 90–130** |

E01 (single stream, 128 output tokens) is the slowest per request. If per-request time turns out
large, cut E01 to 100 requests. Its variance is low, so fewer requests are enough.

Everything except the final numbers can be developed and tested on the mock server (§ Phase 2),
which keeps paid GPU time focused on measurement.

---

## Part II: Phases

Each phase lists: **goal**, **work items**, **key design details**, **tests**, **exit check**,
and **risks**. Estimates assume part-time work and match PLAN.md.

### Phase 0: Fundamentals and foundations (1 week)

*Status:* work items 2–5 are done (data model, config schema, ADRs, CI). Work item 1 is yours:
the reading and `docs/background.md` §1–3. The quantitative model (§4) and
`scripts/envelope.py` are already in place.

**Goal:** understand the system well enough to predict results, and fix the scaffold's data model
before anything builds on it.

Work items:
1. Read Orca (iteration-level scheduling), vLLM/PagedAttention, Sarathi-Serve (chunked prefill,
   stall-free batching), DistServe (goodput, prefill/decode disaggregation), and FastServe or
   another preemptive LLM scheduler as related work. Write `docs/background.md`:
   - prefill vs decode, and the KV cache
   - continuous batching, and chunked prefill
   - where head-of-line blocking can still happen in vLLM
   - the back-of-envelope model from §6, with predicted TTFT/TPOT ranges for the chosen GPU
2. Write ADR-001…009 (§2) in `docs/adr/`.
3. Replace the scaffold data model:
   - `metrics/records.py`: the `RequestRecord` dataclass plus a Parquet schema (a pyarrow schema
     constant).
   - `metrics/latency.py`: pure metric functions that take token counts from usage, return `None`
     on failure, and use ns timestamps. Delete the "wall-clock" wording.
   - `scheduler/base.py`: extend `Request` with `output_tokens_est`, `workload_class`,
     `slo_ttft_ms`, `slo_tpot_ms` and `enqueue_time`. Define priority direction: *lower value =
     more urgent*, matching vLLM, with `HIGH=0, MEDIUM=1, LOW=2`.
4. `config/schema.py`: pydantic models for the schema in Appendix A, and `llmserve validate`.
   Migrate `configs/baseline/e01.yaml`.
5. Set up CI (GitHub Actions): `ruff check`, `ruff format --check`, `mypy --strict`, `pytest`.

Tests: metric functions (multi-token chunks, single token, failure), config validation (valid,
invalid mode combinations, unknown distribution, defaults applied), and a config-hash stability
test.

Exit check: CI green; `llmserve validate configs/baseline/e01.yaml` passes; background note can
predict (within 2×) TTFT for a 1K prompt and TPOT at batch 1 on the chosen GPU.

### Phase 1: Serving baseline (1 week)

**Goal:** one model serving reliably, and a streaming client whose timestamps can be trusted.

Work items:
1. `docker/compose.yml`:
   - a `vllm` service using the pinned `vllm/vllm-openai` image (by digest) with the §6 flags
   - a model cache volume
   - `HF_TOKEN` from the environment
   - a healthcheck on `/health`
   - an optional `dcgm-exporter` service
2. `client/sse.py`: an incremental SSE parser (handles `data:` lines, `[DONE]`, keep-alives, and a
   JSON event split across TCP reads).
3. `client/openai_stream.py`:
   - one async function `stream_completion(client, spec, clock) -> RequestRecord`
   - sends the token-ID prompt, `max_tokens = min_tokens`, `ignore_eos`, and
     `stream_options.include_usage`
   - records `t_first_token` at the first chunk that contains text (not the role chunk), plus every
     chunk time and each chunk's token count
   - has a request timeout, and **no retries** (retries distort latency; failures are data)
4. `runner/env_check.py`: checks that the server is healthy, the `/version` matches the expected
   one, the model matches, prefix caching is off (read from the launch args in config and
   cross-checked with an identical-prompt TTFT probe), and the GPU is idle before the run.
5. `workload/prompts.py`:
   - loads the bundled corpus, tokenizes it once with the pinned tokenizer, and caches the token
     IDs in `~/.cache/llmserve/`
   - `build_prompt(n_tokens, rng) -> list[int]`: a nonce block, then a corpus slice starting at a
     random offset
6. `scripts/smoke.py`: 100 sequential requests, prints a table, and exits non-zero on any failure
   or any usage mismatch.

Tests: SSE parser against recorded byte streams (split at every byte offset); prompt builder gives
exact length, deterministic output under a seed, and different prompts for different requests.

Exit check (PRD §28): 100 sequential requests, 0 unexplained failures, and 0 usage mismatches.
Measured TTFT/TPOT fall inside the Phase 0 predicted range, or the gap is explained in
`docs/background.md`.

Risks:
- *vLLM rejects `min_tokens` or token-ID prompts in the pinned version.* Check on day 1. The
  fallback is a text prompt with a verified round-trip.
- *First-token detection is off by one chunk.* Validate against the server's TTFT histogram.

### Phase 2: Benchmark harness and mock server (2 weeks)

**Goal:** `llmserve benchmark <config>` runs the whole pipeline (warm-up → workload → storage →
summary) against either the mock or vLLM, deterministically.

Work items:
1. `workload/distributions.py`: `fixed`, `uniform(min,max)`, `choice(values, weights)` (the PRD's
   "mixed"), `lognormal(median, sigma, min, max)`, and `empirical(path)`. Each is a small class with
   `sample(rng, n) -> np.ndarray[int]`.
2. `workload/arrivals.py`:
   - `constant(rate)`: deterministic gaps
   - `poisson(rate)`: exponential gaps
   - `bursty(phases=[(duration, rate)...])`: a piecewise-constant Poisson process. The PRD's
     5 → 50 → 5 req/s pattern is the default preset.
   - `replay`: timestamps from a file, with an optional `time_scale` to hit a target ρ
3. `workload/classes.py` + `generator.py`:
   - classes (interactive, long-context, generation-heavy, batch), each with a weight, a
     prompt/output distribution, a priority and SLOs
   - `materialize(config, rep) -> Workload`, written to `workload.parquet`
   - seeding uses `np.random.SeedSequence(seed).spawn` with **independent child streams** for
     arrivals, class assignment, lengths and prompt content. Changing one dimension then doesn't
     reshuffle the others. Repetition r uses `SeedSequence(seed, spawn_key=(r,))`.
4. `mock/engine.py`: a discrete-time simulator of a continuous-batching engine:
   - a per-step token budget (like `max_num_batched_tokens`), with prefills split into chunks
   - step time = `α + β·prefill_tokens_in_step + γ·decode_seqs_in_step + δ·Σ context_len`
   - a KV-capacity limit, with preemption (recompute) when it is exceeded
   - exposes `/metrics` with vLLM-style names (running, waiting, KV usage, preemptions)

   The mock then shows queueing, head-of-line blocking and KV pressure *qualitatively*, which is
   enough to build and debug schedulers on a laptop. `mock/server.py` serves it as an
   OpenAI-compatible SSE endpoint (FastAPI + uvicorn, as optional `mock` extras).
5. Load driver in `runner/run.py`:
   - closed-loop: C worker tasks
   - open-loop: a single arrival task that sleeps until each arrival offset and then enqueues
   - measures **client lag** (actual send time − scheduled send time) per request
6. `gateway/core.py`, v1 (FIFO only for now):
   - an `asyncio` queue and a semaphore sized `max_in_flight`
   - a dispatch loop that wakes on enqueue, on completion, and on a 5 ms timer. The timer is
     needed because a scheduler may *hold* admission by returning `None`.
   - releases the in-flight slot when a request completes
   - when `gateway.enabled: false`, the driver calls the client directly
7. Samplers:
   - `metrics/gpu.py`: an NVML **thread**, not an asyncio task, because NVML calls block. It
     samples at 10 Hz and writes to a ring buffer that is flushed at the end.
   - `metrics/server.py`: scrapes `/metrics` at 2 Hz and parses the Prometheus text format with a
     version-keyed name map (`gpu_cache_usage_perc` vs `kv_cache_usage_perc`, and so on). It fails
     loudly if an expected series is missing.
   - Both are optional per config, and absence is recorded in metadata.
8. `runner/metadata.py`, `storage.py`, `experiment.py`: the resolved config, metadata, the
   workload file, warm-up (a separate small workload whose results are discarded, followed by
   waiting until `running == waiting == 0` and GPU utilization is below 5%), then N repetitions
   and per-rep summaries.
9. `analysis/aggregate.py` v1: per-rep summary (count, success rate, mean/median/std/P50/P95/P99
   for TTFT, TPOT, ITL, E2E and queue wait, plus throughput and goodput over the steady window)
   and cross-rep aggregation.
10. CLI: `validate`, `generate-workload`, `benchmark`, and `run --config` (an alias of `benchmark`,
    since the PRD lists both).

Tests:
- *Determinism:* the same seed gives a byte-identical `workload.parquet`, and a different rep
  gives a different one. Changing only `output` leaves arrivals and prompt lengths unchanged.
- *Arrivals:* the Poisson rate is within ±3σ over 10k samples; bursty phase rates are correct.
- *Distributions:* support bounds hold, and `choice` weights are right.
- *Gateway:* in-flight never exceeds the cap (a property test with `hypothesis`); held requests are
  eventually dispatched; the scheduler is only offered requests that are actually waiting.
- *Integration:* start the mock with a pytest fixture, run a 200-request closed-loop and a 30 s
  open-loop config end to end, and check the output files and schema.
- *Harness overhead:* against a mock with zero delay, at C = 128 and 200 req/s, client lag P99 must
  be under 5 ms. If it isn't, switch to `uvloop`, and to `aiohttp` if needed, before Phase 3.

Exit check (PRD §29): one command runs a complete benchmark on the mock in CI, and on vLLM on the
GPU. Rerunning with the same seed gives an identical workload file. Every PRD §16 metadata field
is present.

Risks:
- *The Python client becomes the bottleneck at C ≥ 64.* The lag metric makes this visible, and the
  fix is a better event loop or several client processes, each with its own share of the workload.
- *The mock gets mistaken for reality.* Mock results never appear in the paper, and they are
  labeled `server.kind: mock` in metadata.

### Phase 3: Baseline characterization (2 weeks, GPU)

**Goal:** a Baseline Performance Report that tests H1 and H2 and provides the numbers that later
phases depend on.

Work items:
1. Configs for E01–E07 (`configs/baseline/`). E07's "mixed" is a named class mix in
   `configs/workloads/mixed.yaml`, for example 60% interactive (128–512 in, 128 out), 25%
   long-context (2K–8K in, 128 out) and 15% generation-heavy (256 in, 512–1K out).
2. Sweeps (`configs/suites/`):
   - *Concurrency sweep:* C ∈ {1, 2, 4, …, 128} for 128/128 and for the mixed workload. This gives
     latency and throughput vs concurrency and locates the saturation point (H1).
   - *Prompt-length sweep* at C = 1: {128 … 8192} → TTFT. Fit `TTFT ≈ a + b·p + c·p²` (H3).
   - *Output-length sweep* at C = 8 → TPOT.
   - *Capacity:* the open-loop rate at which queue wait grows without bound, per workload.
     **This defines ρ = 1 for every later phase.**
   - *Bursty* (5 → 50 → 5 req/s) with FIFO. Plot queue depth, TTFT and KV usage over time.
3. `analysis/plots.py` v1:
   - latency vs concurrency, and throughput vs concurrency (with CI bands)
   - prompt length vs TTFT (with the fitted curve)
   - TTFT CDFs per workload
   - time series for the bursty run
4. Heterogeneity test for H2: compare P95/P99 TTFT of the *short* requests in the mixed workload
   with the same short requests in a homogeneous workload at the same total ρ. This result directly
   motivates Phase 5.
5. Fit the cost model (`scheduler/estimators.py`): prefill time as a function of prompt length and
   decode time per token as a function of batch size. It is used for slowdown and for SJF/WAAS cost
   estimates.
6. Calibrate the mock (`scripts/calibrate_mock.py`): fit α, β, γ and δ to the real data so that
   laptop experiments point in the same direction as real ones.
7. Write `docs/reports/baseline.md` (the deliverable), with numbers and figures generated by
   `llmserve report`.

Exit check: E01–E07 × 5 reps complete with ≥ 99% success. H1 and H2 are each marked
supported/refuted with evidence. Capacity (ρ = 1) is measured for each workload. Every figure has a
committed config.

Risks:
- *vLLM's chunked prefill already removes most head-of-line blocking (H2 is weak).* That is a
  result worth reporting. The research then focuses on queueing under overload (ρ ≥ 0.85) and on
  KV pressure, where admission order still matters.

### Phase 4: GPU profiling (1 week, GPU)

**Goal:** answer "where does time actually go?" (PRD §31).

Work items:
1. Enable DCGM profiling fields (`SM_ACTIVE`, `SM_OCCUPANCY`, `DRAM_ACTIVE`, `PIPE_TENSOR_ACTIVE`)
   through `dcgm-exporter` where the cloud allows it. **NVML "utilization" only means "a kernel was
   running"**. It reads close to 100% during memory-bound decode, which misleads. The report states
   this explicitly.
2. Plots:
   - GPU utilization vs request rate
   - KV usage vs latency (scatter of `kv_usage_at_dispatch` against TTFT)
   - memory over time
   - power, and joules per output token (a stretch metric that falls out of the power samples)
3. One Nsight Systems trace (`scripts/nsys_profile.sh`) of about 20 s of the mixed workload at
   ρ = 0.8, plus one vLLM torch-profiler capture (`VLLM_TORCH_PROFILER_DIR`, `/start_profile`,
   `/stop_profile`). Break a step down into attention, MLP, sampling and CPU scheduling overhead.
   Show the same breakdown for a prefill-heavy step and a decode-heavy step.
4. Time decomposition per request: queue wait, server-side wait (from vLLM's queue metric),
   prefill, and decode. Show it as stacked bars per workload class.
5. `llmserve trace <run> --request <id>` prints the PRD §40 timeline.

Exit check (**MVP, PRD §42**): all MVP items are working end to end, and there is a written answer,
backed by a trace, to "where does the time go" for each workload class.

### Phase 5: Scheduling baselines (2 weeks)

**Goal:** implement the simple policies, compare them fairly, and find the concrete weakness that
WAAS must fix.

Work items:
1. `scheduler/shortest.py`: `ShortestPromptFirst` (key: `prompt_tokens`) and `ShortestJobFirst`
   (key: predicted cost from the Phase 3 cost model with `prompt + est_output`). Ties break by
   arrival time. Use a heap for static keys.
2. `scheduler/priority.py`: strict class priority, FIFO within a class, and optional aging.
3. `scheduler/registry.py`: config name → class plus params. Add a scheduler-decision-latency
   microbenchmark (P99 < 100 µs at a queue depth of 1,000).
4. **Choose the in-flight cap:** sweep `max_in_flight` ∈ {4, 8, 16, 32, 64, uncapped} with FIFO at
   ρ = 0.9. Pick the smallest cap that keeps ≥ 95% of uncapped throughput. Record the trade-off
   curve, since it is itself a figure.
5. Comparison matrix (all paired on the same workload files, in interleaved order):
   {FIFO, SPF, SJF, Priority} × {interactive-only, mixed, long-context-heavy} × ρ ∈ {0.5, 0.8, 0.95}
   × 5 reps.
6. Cross-check (ADR-001): the same workloads with no gateway and vLLM `--scheduling-policy
   priority`, with per-request `priority` set from the class. How close does vLLM's native priority
   come to gateway Priority?
7. Analysis for each policy:
   - P50/P95/P99 TTFT per class
   - slowdown by request-size decile, to show where SJF starves long requests
   - max queue wait
   - goodput and throughput

   Write `docs/reports/scheduling_baselines.md` with *where each policy wins and where it fails*.

Tests: scheduler invariants as property tests (returns a member of `waiting` or `None`;
deterministic given the same inputs; FIFO order; SPF order; within-class FIFO for Priority).

Exit check: a written, data-backed problem statement for WAAS. It should read something like "at
ρ ≥ 0.8 in mixed workloads, FIFO inflates short-request P95 TTFT by X×; SPF fixes it but gives long
requests Y× slowdown and unbounded max wait; neither responds to KV pressure (Z preemptions at
ρ = 0.95)". Tested hypotheses: H4.

### Phase 6: WAAS, the Workload-Aware Adaptive Scheduler (2–3 weeks)

**Goal:** a scheduler designed from Phase 5 evidence, with guarantees and ablations.

The PRD requires the final design to come from experiments. Below is the **starting design space**,
to be narrowed using Phase 5 findings. It is not a commitment.

1. **Ordering score** (a candidate): for each waiting request r,
   `score(r) = α·urgency(r) + β·age(r)/age_ref − γ·cost(r)/cost_ref`, and select the maximum.
   - `urgency` comes from TTFT slack: `slack = (t_arrival + slo_ttft) − now − predicted_ttft(r)`.
     Urgency grows sharply as slack goes to 0, and a request whose slack is already negative is
     deprioritized ("already lost") unless the age term forces it.
   - `cost` is the Phase 3 prefill + decode model applied to `(prompt_tokens, output_tokens_est)`.
2. **Bounded bypass (starvation guarantee):** each request counts how many times a later arrival
   was dispatched ahead of it. At `bypass_count ≥ K` it is dispatched next regardless of score.
   This gives a provable bound on how long any request can be overtaken, and max wait is measured
   to confirm it.
3. **KV-aware admission gate** (the PRD's `δ·memory_pressure`): admit r only if
   `kv_usage + kv_bytes(prompt + est_output)/kv_capacity ≤ θ`. Otherwise, try smaller candidates
   within the bypass bound, or hold (return `None`). The aim is to prevent vLLM preemptions, which
   waste the prefill work already done.
4. **Adaptive in-flight cap** (optional): an AIMD controller on `max_in_flight` driven by the
   observed `ttft_server` and preemption rate.
5. **Output-length estimators** (`estimators.py`), all behind one interface. The ablation of PRD
   §41 compares:
   - *oracle*: the true `output_tokens_req`
   - *class prior*: the per-class median from history
   - *online*: an EWMA per class, updated through a new `Scheduler.on_complete(request, record)`
     hook
   - *none*: a constant
6. Interface additions (backward-compatible): `on_enqueue`, `on_complete`, and a `SystemState`
   filled from the live `/metrics` scrape (KV usage, running, waiting). The scrape runs every 2 Hz,
   so the gateway also keeps its own running estimate of KV usage between scrapes (admitted tokens
   − completed tokens).
7. Tuning protocol: choose α, β, γ, K and θ on a **tuning** workload seed set with the calibrated
   mock, then confirm on real GPU tuning runs. Freeze them. Evaluate on **held-out** seeds in
   Phase 7. Tuning and evaluation seeds never overlap; this is what prevents cherry-picking.
8. Ablations (each one removes a single component): no urgency, no age/bypass, no cost, no KV gate,
   and each output estimator. Run each on the mixed workload at ρ ∈ {0.8, 0.95}.

Tests: the bypass bound holds under adversarial inputs (a property test that feeds a stream of tiny
requests and confirms that a large request is dispatched within K); the gate never admits past θ
under the running KV estimate; the score is deterministic.

Exit check: WAAS beats the best Phase 5 baseline on the targeted metric (for example, short-class
P95 TTFT) on tuning seeds, with a measured throughput and goodput cost and a verified starvation
bound. Ablations show which components matter.

### Phase 7: Research evaluation (1–2 weeks, GPU)

**Goal:** the automated, reproducible evaluation matrix (PRD §34), which tests H5 and H6.

Work items:
1. `runner/suite.py`:
   - expands a suite YAML (a cartesian product with overrides) into runs
   - **resumable**: a run is skipped if a complete result with the same config hash exists
   - randomized interleaving (§5 rule 6)
   - a failure in one run doesn't stop the suite, and failed runs are logged to a failures table
   - `--dry-run` prints the plan with estimated GPU time
2. `experiments/paper_01.yaml`: 3 schedulers (FIFO, best baseline from Phase 5, WAAS) × 4
   workloads (interactive, long-context, generation-heavy, mixed-bursty) × 5 load levels
   (ρ ∈ {0.5, 0.7, 0.85, 0.95, 1.05}) × 3 reps = **180 runs**, all on held-out seeds. The PRD
   phrases load levels as "concurrency levels"; open-loop ρ is the correct equivalent for
   scheduler evaluation (ADR-003).
3. Headline comparison: mixed workload at ρ = 0.85 and 0.95, FIFO vs WAAS, 10 paired reps, at least
   2,000 requests per run.
4. KV-pressure study (H6): vary KV capacity through `--gpu-memory-utilization` ∈ {0.6, 0.75, 0.9},
   or through the mix of long-context requests, and show how WAAS's advantage changes.
5. One public trace replay (for example the Azure LLM inference traces or BurstGPT, license
   permitting) scaled to ρ = 0.85, to address the "synthetic workloads" risk.
6. Performance regression test (PRD §39): a 5-minute standard workload with stored baseline
   summaries. `llmserve regress` flags any metric that moves more than 10%. Run it before and after
   every change that touches the harness.
7. `analysis/stats.py`: paired bootstrap CIs, paired t-test, and the claim-format helper.
   `analysis/report.py` produces the results tables.

Exit check: the whole suite runs from one command, and each hypothesis H1–H6 has a verdict with
CIs. Every figure is produced from `paper/figures.yaml` without manual steps.

### Phase 8: Report and release (2 weeks)

Work items:
1. `reproduce.py experiments/paper_01.yaml` and `make reproduce`: environment check → suite →
   analysis → figures. Also `make reproduce-mock`, which runs a small version of the whole pipeline
   on CPU in about 10 minutes. It lets reviewers without a GPU verify the pipeline, with numbers
   clearly marked as non-representative.
2. `docker compose up` brings up vLLM, dcgm-exporter and the harness image, as PRD §5 describes.
3. Paper (8–12 pages; PRD §35 structure). System Design explains ADR-001/002/003. Methodology
   explains §5 and §6. Limitations covers one GPU SKU, one model, synthetic plus one trace, the
   output-length estimator, and the fact that only admission order is controlled, not batch
   composition.
4. Publish summarized results (`results/summaries/`, committed) and raw data as a release asset or
   dataset (Zenodo/HF), with a DOI if practical.
5. README findings: only numbers from `summary.json`, in the claim format.
6. Documentation:
   - `docs/architecture.md` (components and interfaces)
   - `docs/benchmarking.md` (methodology and repeatability assumptions)
   - `docs/extending.md` ("add a scheduler in 20 lines", covering PRD §48 item 11)

Exit check (PRD §48): a fresh clone on a new GPU VM reproduces the headline figure within the
reported CI, following only the README.

---

## Part III: Cross-cutting concerns

### Testing pyramid

| Layer | What | Where it runs |
|---|---|---|
| Unit | metrics, distributions, arrivals, config, schedulers, SSE parser, stats | CI, < 30 s |
| Property | scheduler invariants, gateway cap, workload determinism, bypass bound | CI (`hypothesis`) |
| Integration | generator → gateway → mock → records → summary | CI, < 2 min |
| Smoke | 100 sequential requests on real vLLM | GPU VM, before every study |
| Regression | standard 5-min workload vs stored baseline, ±10% flag | GPU VM, before and after harness changes |

### Observability

Every request has a `request_id` that is stable across schedulers. `events.jsonl` records every
scheduler decision with the scores of the top candidates. `llmserve trace` rebuilds the §40
timeline for any request, so "why was request 128 slow?" can always be answered. A Grafana
dashboard is post-MVP (PRD §27). If it gets built, it is fed by the same `/metrics` plus a gateway
`/metrics` endpoint.

### Dependencies to add

| Package | Why | Group |
|---|---|---|
| `pydantic>=2` | config schema | core |
| `transformers` (tokenizer only) | exact token-ID prompts | core |
| `fastapi`, `uvicorn` | mock server, gateway proxy | `serve` extra |
| `hypothesis` | property tests | dev |
| `uvloop` | event loop, if the lag check requires it | core (Linux) |
| `nvidia-ml-py` | already present | `gpu` extra |

### Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Chunked prefill in vLLM already solves head-of-line blocking | Medium | Changes the paper's angle | Focus on overload and KV pressure; the negative result is still reported (Phase 3) |
| No queue forms, so all policies look the same | High if ignored | Invalidates Phase 5–7 | Gateway cap (ADR-001), open-loop ρ ≥ 0.8 (ADR-003) |
| Load generator becomes the bottleneck | Medium | Wrong latencies | Client-lag metric plus a Phase 2 threshold |
| Output length can't be known in advance | Certain | Weaker WAAS | Oracle/estimated/none ablation (Phase 6) |
| vLLM changes metric names or flags | High across versions | Broken scrapers | Pin the image digest; version-keyed name map; fail loudly |
| Cloud noise (noisy neighbours, thermal throttling) | Medium | Variance | Interleaved order; record clocks and temperature; CIs |
| GPU budget overrun | Medium | Schedule | Mock-first development; `--dry-run` time estimates; resumable suites |
| Tuning leaks into evaluation | Medium | Credibility | Separate tuning and held-out seed sets (Phase 6) |

### Timeline (part-time)

| Weeks | Phase | GPU needed |
|---|---|---|
| 1 | 0: Fundamentals and foundations | no |
| 2 | 1: Serving baseline | yes (bring-up) |
| 3–4 | 2: Harness + mock | mostly no |
| 5–6 | 3: Baseline characterization | yes |
| 7 | 4: GPU profiling → **MVP** | yes |
| 8–9 | 5: Scheduling baselines | yes |
| 10–12 | 6: WAAS | mock + GPU |
| 13–14 | 7: Evaluation | yes |
| 15–16 | 8: Report + release → **v1.0** | minimal |

---

## Appendix A: Config schema (v1)

```yaml
schema_version: 1
experiment: e07_mixed_c32
description: Mixed workload, closed loop, concurrency 32 (PRD §11 E07)
seed: 42
repetitions: 5

server:
  kind: vllm                      # vllm | mlx | llamacpp | mock  (ADR-009)
  endpoint: http://localhost:8000/v1
  model: Qwen/Qwen2.5-7B-Instruct
  model_revision: <commit-sha>    # asserted by env_check
  expected_vllm_version: "<pinned>"
  prefix_caching: false           # asserted by env_check
  scheduling_policy: fcfs         # fcfs | priority (vLLM-native cross-check)

gateway:
  enabled: false                  # false = direct to vLLM (Phase 3 baselines)
  max_in_flight: 16
  scheduler:
    name: fifo                    # fifo | spf | sjf | priority | waas
    params: {}
  output_estimator: oracle        # oracle | class_prior | online | none

load:
  mode: closed                    # closed | open
  concurrency: 32                 # closed only
  # arrival:                      # open only
  #   process: poisson            # constant | poisson | bursty | replay
  #   rho: 0.85                   # or `rate:` in req/s; rho needs capacity_rps
  #   phases: [{duration_s: 10, rate: 5}, {duration_s: 10, rate: 50}, {duration_s: 10, rate: 5}]
  #   trace: data/traces/azure_conv.csv
  requests: 2000
  # capacity_rps: 11.2            # measured in Phase 3; required when arrival uses rho

workload:
  ref: ../workloads/mixed.yaml    # relative to this file; or inline `classes:`

measurement:
  warmup: {requests: 20, settle_timeout_s: 60}
  steady_window: auto             # closed: in-flight == C; open: trim warmup_s/cooldown_s
  warmup_s: 30
  cooldown_s: 30
  request_timeout_s: 600
  samplers: {gpu_hz: 10, server_hz: 2, dcgm: auto}
```

The implemented schema is `llmserve/config/schema.py`, which is authoritative.

`configs/workloads/mixed.yaml`:
```yaml
classes:
  - name: interactive
    weight: 0.60
    priority: high
    slo: {ttft_ms: 500, tpot_ms: 100}
    prompt: {distribution: uniform, min: 128, max: 512}
    output: {distribution: fixed, tokens: 128}
  - name: long_context
    weight: 0.25
    priority: medium
    slo: {ttft_ms: 3000, tpot_ms: 100}
    prompt: {distribution: choice, values: [2048, 4096, 8192], weights: [0.5, 0.3, 0.2]}
    output: {distribution: fixed, tokens: 128}
  - name: generation_heavy
    weight: 0.15
    priority: low
    slo: {ttft_ms: 2000, tpot_ms: 150}
    prompt: {distribution: fixed, tokens: 256}
    output: {distribution: choice, values: [512, 1024]}
```
SLO values are placeholders. Set them from Phase 3 isolated-latency data, for example at 5× the
isolated P50, before any scheduler work, and don't change them afterwards.

## Appendix B: Hypotheses → evidence map

| ID | Evidence | Phase | Figure |
|---|---|---|---|
| H1 | concurrency sweep: throughput plateau + P99 knee | 3 | latency/throughput vs C |
| H2 | short-request P95/P99 in mixed vs homogeneous at equal ρ | 3 | TTFT CDF overlay |
| H3 | R² of the prefill-time fit on prompt length | 3 | prompt length vs TTFT |
| H4 | SPF vs FIFO short-class TTFT | 5 | per-class TTFT bars |
| H5 | WAAS vs FIFO P95 TTFT with throughput ratio, paired CIs | 7 | headline figure |
| H6 | WAAS gain vs KV capacity and long-context share | 7 | gain vs pressure heatmap |

## Appendix C: First two weeks, as issues

1. `records.py` + metric functions rewrite (usage-based TPOT, `None` on failure, ns clock)
2. Extend `Request` (estimate, class, SLOs, priority direction)
3. Pydantic config schema + `llmserve validate` + migrate `e01.yaml`
4. CI workflow (ruff, mypy, pytest)
5. `docs/background.md` + ADR-001…008
6. `docker/compose.yml` with pinned vLLM and a healthcheck
7. SSE parser + streaming client + byte-split tests
8. Token-ID prompt builder + corpus cache
9. `scripts/smoke.py`: 100 sequential requests (Phase 1 exit check)
10. Seeded distributions + arrivals + determinism tests
