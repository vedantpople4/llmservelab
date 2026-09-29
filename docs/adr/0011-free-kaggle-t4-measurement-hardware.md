# ADR-0011: Free Kaggle T4 is the measurement hardware

Status: accepted (September 2026)

## Context

Every reported number must come from one fixed hardware SKU (PLAN.md §2). The original plan
was a rented 24 GB card with `Qwen2.5-7B-Instruct` in bf16. Kaggle offers free GPU notebooks
with a rolling 30 GPU-hour weekly quota, no card required, which removes money from the
critical path — but it constrains the SKU in three ways: its only usable accelerator is a
Tesla T4 ×2 (the P100 is sm_60 and unsupported by vLLM; the `machine_shape:
"NvidiaTeslaT4"` metadata key is what selects T4s, not the CLI `--accelerator` flag), Turing
has no bf16 silicon, and each T4 has 16 GB, not 24.

## Decision

The whole study runs on **one Tesla T4 (16 GB) serving
`Qwen/Qwen2.5-7B-Instruct-AWQ` (INT4 weights, fp16 activations)**. Runs take the first GPU of
a Kaggle T4 ×2 session (`CUDA_VISIBLE_DEVICES=0`) so they stay single-GPU; the second T4
idles. §6 is re-derived for this SKU: ≈ 7.5–8 GB of KV after weights (≈ 130–140k tokens),
which keeps the KV-pressure hypothesis (H6) reachable at the planned load mixes.

## Consequences

- GPU time is free but quota-bound: 30 h/week means the ≈ 90–130 GPU-hour plan spans about
  4–5 quota weeks; sessions cap at 9–12 hours and long runs resume in a fresh notebook.
- Kaggle has no Docker, so `docker/compose.yml` remains the GPU-host reference while the
  notebook path (`docs/runbooks/kaggle.md`) pins vLLM by pip version. Both must use the §6
  flags so the two routes are interchangeable.
- AWQ kernels on sm_75 must be verified at Phase 1 bring-up; if they fail, the fallback is
  Qwen2.5-7B GPTQ-INT4 on the same card — same model, same precision class, new pin.
- Reported numbers are fp16-on-T4, AWQ weights. They never get compared against bf16 output
  or another card, and limitations says so.
