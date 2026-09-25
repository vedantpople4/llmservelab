# ADR-0009: Local backends on a Mac for development

Status: accepted (September 2026)

## Context

Rented GPU time is the scarcest resource. Much of Phases 1–2 (streaming client, token counting,
metadata, storage) can be checked against any real model, and a Mac with Apple Silicon can run a
small Qwen2.5 model through MLX or llama.cpp.

## Decision

`server.kind` accepts `mock`, `mlx`, `llamacpp` and `vllm`. Each backend declares its
capabilities (token-ID prompts, forced output length, continuous batching, Prometheus metrics,
prefix-cache control), and the harness adapts: for example, it records actual output lengths when
they can't be forced, and disables the server sampler when there is no `/metrics`. Mac dev configs
use the Qwen2.5 family so the tokenizer matches the GPU model.

## Consequences

- Phases 1–2 can be completed mostly without a GPU.
- Only `kind: vllm` results may appear in reports; the analysis layer enforces this.
- Mac results are affected by quantization, unified memory and background load, and must never be
  compared with GPU numbers.
