# ADR-0012: vllm-metal on a Mac is the scheduler development platform

Status: accepted (October 2026); its "never reported" rule is superseded by ADR-0013

## Context

GPU time on the free T4 (ADR-0011) is quota-bound, and Phases 5–6 (scheduling baselines, WAAS)
need many short iterations against a real continuous-batching engine. The mock (ADR-009)
batches only qualitatively. The Mac dev surfaces from ADR-009/0010 (mlx, llama.cpp, Ollama,
LM Studio) take text prompts, cannot force output length, and expose no `/metrics`, so the
gateway, samplers and length checks run in a degraded mode on all of them.

vllm-metal is a vLLM hardware plugin for Apple Silicon: upstream vLLM supplies the API server,
scheduler and paged block manager; MLX supplies the model layers. A bring-up on an M5 MacBook
(24 GB, October 2026, vLLM 0.31.0 + vllm-metal 0.31.0) observed:

- It serves the **same checkpoint as the T4**, `Qwen/Qwen2.5-7B-Instruct-AWQ`, through
  mlx-lm's AWQ repack (load log: `quantization=auto_awq`).
- Token-ID prompts, `ignore_eos` + `max_tokens` (exact output length, `usage` agrees),
  `stream_options.include_usage`, `/health`, `/version` and `/v1/models` all work.
- `/metrics` exposes `vllm:num_requests_running`, `num_requests_waiting`,
  `kv_cache_usage_perc` and `num_preemptions_total`, which the server sampler already maps.
- `--no-enable-prefix-caching` is honored (identical-prompt TTFT probe: 180 ms → 179 ms).
- Continuous batching is real: at closed-loop concurrency 8 the server ran 8 requests at once,
  generation throughput rose 2.6× over concurrency 1, and TPOT rose 47 → 139 ms.
- `min_tokens` is **rejected** with a 400 (`does not support sampling controls backed by vLLM
  logits processors`).
- With the default `--gpu-memory-utilization` (0.92) the KV pool (11 GB) exceeded free memory
  and the server warned that blocks would page. At 0.5 the pool is 52,784 tokens (2.8 GB) with
  no paging.

## Decision

Add `server.kind: vllm-metal`, a development backend with its own capability row, and use it
for building and debugging schedulers and the gateway on the Mac. The capability table gains
`min_tokens_param`, so the client sends `min_tokens` only to backends that accept it; vLLM on
CUDA keeps receiving it, unchanged.

Mac runs use `--gpu-memory-utilization 0.5` so the KV pool never pages into swap.

## Consequences

- Phases 5–6 can iterate on a real vLLM scheduler without spending GPU quota, with the same
  model, tokenizer, request payloads and `/metrics` as the T4 runs.
- `vllm-metal` results are development results. They are never compared with T4 numbers and
  never appear in reports: only `kind: vllm` is reportable (ADR-009). The harness runs a newer
  vLLM here (0.31.0) than on the T4 (0.10.2), so engine behavior may differ too.
- There are no GPU samples on the Mac (no NVML); GPU-utilization questions stay on the T4.
- The Mac's KV pool is far smaller than the T4's (≈ 53k vs ≈ 130–140k tokens), so KV pressure
  appears at lower load. That helps when developing KV-aware scheduling and must be kept in
  mind when choosing load levels.
- Unified memory is shared with macOS and every open app. Close heavy apps and keep the Mac on
  power during runs; background load shows up as latency.
