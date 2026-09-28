# Implementation Plan: Phase 1 — Serving baseline (end-to-end, runnable on this Mac)

Phase 1's goal (`docs/IMPLEMENTATION_PLAN.md` §Phase 1): **one model serving reliably, and a
streaming client whose timestamps can be trusted.** Exit check (PRD §28): 100 sequential requests,
0 unexplained failures, 0 usage mismatches.

## Overview

Build the full Phase 1 chain — mock backend → SSE parser → streaming client → prompt builder →
env check → smoke run — so `uv run python scripts/smoke.py configs/dev/smoke_mock.yaml` closes the
exit check locally against the in-package mock. The vLLM/Docker half is authored but can only be
closed on a rented GPU (ADR-009: only `kind: vllm` results may appear in reports).

## Architecture decisions

- **Mock pulled forward from Phase 2.** Phase 1 needs a server to run 100 requests against and
  there is no local GPU. A minimal delay-model streaming server (`llmserve/mock/`) is built now;
  the full continuous-batching simulator (`mock/engine.py` with KV cap, preemption) stays in Phase 2.
- **Token-ID prompts end to end** (ADR-005): the mock accepts `prompt: [int, ...]` exactly like
  vLLM, so the client code path is identical for both backends.
- **Capability table keyed by `server.kind`** (`llmserve/client/capabilities.py`) implements
  ADR-009's "each backend declares its capabilities" without a schema change: token-ID prompts,
  forced output length, `/version`, `/metrics`, prefix-cache control.
- **Tokenizer behind a protocol.** `workload/prompts.py` takes anything with `encode/decode`; the
  real one loads the pinned Qwen `tokenizer.json` (cached under `~/.cache/llmserve/`), tests inject
  a stub so CI never needs the network.
- **No retries, ever** (Phase 1 design item 3): failures are data and get their own status.
- **All timestamps from one clock**: `time.perf_counter_ns()` captured by a `Clock` callable and
  relative to run start (ADR-006).

## Task list

### Checkpoint A — parser and backend

- [x] **T1: `llmserve/mock/` minimal streaming backend**
  `mock/engine.py` (delay model: TTFT = a + b·prompt_tokens + queue, per-token delay grows with
  in-flight), `mock/server.py` (`create_app()` factory: `/health`, `/version`, `/v1/models`,
  `POST /v1/completions` SSE with token-ID prompts, `ignore_eos`/`max_tokens`, final
  `usage` chunk, `[DONE]`), `python -m llmserve.mock --port 8001` entry point.
  *Accept:* in-process test streams N tokens with `usage["completion_tokens"] == N`; `prompt_tokens`
  echoes the token-ID prompt length; health/version respond.
- [x] **T2: `llmserve/client/sse.py` incremental parser**
  `SseParser.feed(bytes) -> list[SseEvent]`, `.close()`. Handles `data:`/multi-line data, `event:`,
  comments (`: keep-alive`), `\r\n`, and a JSON payload split across TCP reads.
  *Accept:* recorded stream replayed split at **every byte offset** yields identical events;
  `[DONE]` is surfaced as a terminal event; trailing partial line is buffered, not lost.
- [x] **T3: `llmserve/workload/spec.py` — `RequestSpec`**
  Frozen dataclass: request_id, workload_class, priority, SLOs, prompt/output token counts,
  materialized `prompt_ids`. `Workload = list[RequestSpec]`.
  *Accept:* constructible from a `WorkloadClass` + rng; no mutable shared state.

### Checkpoint B — client, prompts, checks

- [x] **T4: `llmserve/client/openai_stream.py`**
  `async stream_completion(client, spec, clock, *, t_arrival, cfg) -> RequestRecord`. Records
  `t_first_token` at the first chunk that carries a token (not the role/empty chunk), every chunk
  timestamp + token count, and `usage`. Statuses: `ok`, `timeout`, `http_error`, `conn_error`,
  `length_mismatch`. No retries; timeout from `measurement.request_timeout_s`.
  *Accept:* against the mock: ok record with `prompt_tokens_usage`/`output_tokens_usage` equal to
  spec; usage mismatch ⇒ `LENGTH_MISMATCH`; non-200 ⇒ `HTTP_ERROR`; refused port ⇒ `CONN_ERROR`;
  slow mock ⇒ `TIMEOUT`.
- [x] **T5: `llmserve/client/capabilities.py`**
  Capability flags per `server.kind` (ADR-009) used by T4/T6 to adapt.
  *Accept:* table covers mock/mlx/llamacpp/vllm; typed, exhaustive.
- [x] **T6: `llmserve/workload/prompts.py` + bundled corpus**
  `Tokenizer` protocol; `load_qwen_tokenizer()` (downloads/caches `tokenizer.json`);
  `build_prompt(n_tokens, rng) -> list[int]` = unique nonce block + corpus slice at a random
  offset, exact length. Bundled public-domain corpus at `llmserve/data/corpus.txt`.
  *Accept:* exact length for n ∈ {1, 17, 128, 4096}; same seed ⇒ same prompt; different requests ⇒
  different prompts (prefix-cache can't hit); tests pass with a stub tokenizer and no network.
- [x] **T7: `llmserve/runner/env_check.py`**
  Checks `/health` ok, `/version` == `expected_version` (when set), model matches
  `/v1/models`, prefix caching off (config + identical-prompt TTFT probe), GPU idle (only
  `kind: vllm` with NVML; skipped and recorded otherwise).
  *Accept:* passes against mock; each failure raises with a specific message.
- [x] **T8: `scripts/smoke.py` + `configs/dev/smoke_mock.yaml`**
  100 sequential requests from a config; prints count / success rate / TTFT+TPOT p50 p95 /
  throughput; exits non-zero on any non-`ok` status or usage mismatch.
  *Accept:* `uv run python scripts/smoke.py configs/dev/smoke_mock.yaml` → exit 0;
  a deliberately broken server (e.g. `--endpoint` pointed at a bad port) → exit 1.

### Checkpoint C — repo hygiene

- [x] **T9: dependencies + CI**
  `pyproject.toml`: extra `mock = ["fastapi", "uvicorn"]`, dep `tokenizers>=0.19`;
  CI syncs `--extra dev --extra mock`; regenerate `uv.lock`.
  *Accept:* `uv sync --locked --extra dev --extra mock` works; `uv run llmserve validate
  $(find configs -name '*.yaml' -not -path 'configs/workloads/*')` passes.
- [x] **T10: `docker/compose.yml` (authored, GPU-unverified)**
  Pinned `vllm/vllm-openai` image, model cache volume, `HF_TOKEN`, `/health` healthcheck, and the
  §6 flags (`--max-model-len 10240 --gpu-memory-utilization 0.90 --no-enable-prefix-caching
  --max-num-seqs 256 --disable-log-requests`).
  *Accept:* file lints (`docker compose config`); marked in README as needing a GPU to run.
- [x] **T11: schema gap — add `n_chunks` to `REQUEST_SCHEMA`**
  Plan §4 lists it; the column is missing. Populate from `len(chunks)`.
  *Accept:* records round-trip through Parquet; existing tests still pass.

### Final gate
- [x] `uv run ruff check . && uv run ruff format --check . && uv run mypy llmserve tests && uv run pytest -q`
- [x] smoke exit check green against the mock
- [x] README + IMPLEMENTATION_PLAN status updated ("Phase 1 code complete; GPU run pending")

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No network for `tokenizer.json` / corpus | High | cache on first fetch; tests use a stub tokenizer; fail with an actionable message |
| Per-chunk token counts ≠ true token boundaries | Med | TPOT uses `usage`, not chunks (already in `metrics/latency.py`); chunk counts documented as tokenizer-derived approximation; record `n_chunks` |
| vLLM rejects `min_tokens` / token-ID prompts | Med | check on GPU day 1; fallback recorded in the plan (text prompt + verified round-trip) |
| `docker/compose.yml` untested until GPU | Low | `docker compose config` locally; real check deferred and explicit |

## Open questions (non-blocking)

1. Rent the GPU now (L4/A10) or finish Phase 2 first on the mock? — Phase 1's *real* exit check
   needs 2–3 GPU-hours.
2. `t_enqueue` column is in the plan's data model but absent from `REQUEST_SCHEMA`; it belongs to
   gateway work (Phase 2/5). Deferring unless you want it now.
