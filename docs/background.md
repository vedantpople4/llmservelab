# Background: LLM inference fundamentals

Phase 0 deliverable. §4 is filled in. §1–3 are **yours to write** after the reading. The prompts
in each section are the questions a reviewer, or a PhD interviewer, will ask. Exit check: you can
answer them without notes, and you can predict TTFT and TPOT on the chosen GPU to within 2×.

## 1. Reading list

| Paper | Read for | Done |
|---|---|---|
| Orca (Yu et al., OSDI '22) | Iteration-level scheduling: why batching per *step* beats batching per *request* | ☐ |
| vLLM / PagedAttention (Kwon et al., SOSP '23) | KV-cache fragmentation, block tables, preemption by recompute or swap | ☐ |
| Sarathi-Serve (Agrawal et al., OSDI '24) | Chunked prefill, stall-free batching, the prefill/decode interference trade-off | ☐ |
| DistServe (Zhong et al., OSDI '24) | Goodput under TTFT/TPOT SLOs; why prefill and decode want different resources | ☐ |
| FastServe (Wu et al., 2023) or similar | Preemptive, size-aware scheduling for LLMs: the closest related work to WAAS | ☐ |
| vLLM docs: "Optimization and Tuning" | `max_num_seqs`, `max_num_batched_tokens`, chunked prefill defaults in the pinned version | ☐ |

For each paper, write 5–10 lines: the problem, the key mechanism, the headline result, and
**what it means for LLMServeLab**. The last point is the one that matters.

## 2. Concepts (write in your own words)

- **Prefill vs decode.** Why is prefill compute-bound and decode memory-bound? What does that mean
  for how TTFT scales with prompt length, and how TPOT scales with batch size?
- **KV cache.** What is stored, how big is it per token for our model (§4), and what happens when
  it runs out? (Preemption: which request is chosen, and what work is lost?)
- **PagedAttention.** What problem does block-based allocation solve that contiguous allocation
  doesn't?
- **Continuous batching.** Why can a new request join a running batch at the next step? What does
  that do to the TPOT of requests already running?
- **Chunked prefill.** How does `max_num_batched_tokens` bound the interference a long prompt
  causes to running decodes? Where can head-of-line blocking still happen *with* chunked prefill?
  (Hint: in the waiting queue, before a request gets any budget at all.)
- **Where a gateway scheduler can and cannot help.** Tie this to ADR-0001: which of the effects
  above can admission order influence?

## 3. Predictions (fill in before Phase 1, check after)

| Quantity | Predicted | Measured (Phase 1/3) | Explanation of the gap |
|---|---|---|---|
| TTFT, 1K prompt, batch 1 | | | |
| TTFT, 8K prompt, batch 1 | | | |
| TPOT, batch 1 | | | |
| TPOT, batch 32 | | | |
| Saturation concurrency for 128/128 | | | |
| Max concurrent 8K-prompt requests before preemption | | | |

## 4. Back-of-envelope model for Qwen2.5-7B

Run `uv run python scripts/envelope.py --preset <gpu>` for the numbers. The script uses the
formulas below.

**Model:** 7.62B parameters, 28 layers, hidden size 3584, 4 KV heads (GQA) × 128 head dim.
Weights in bf16 are about 15.2 GB; the 4-bit MLX build is about 4.3 GB.

**KV cache per token:** 2 (K and V) × 28 layers × 4 KV heads × 128 × 2 bytes = **57,344 B ≈ 56
KiB**. A 24 GB card at `gpu_memory_utilization = 0.9` leaves about 5 GB for KV, roughly 85k
tokens. That is about ten 8K-token requests at once, so KV pressure (H6) is reachable on a 24 GB
card.

**Prefill (TTFT at batch 1)** is compute-bound:

    FLOPs ≈ 2 · params · p  +  2 · layers · p² · hidden     (linear layers + causal attention)
    time  ≈ FLOPs / (peak FLOPs · MFU)

The attention term is about 10% of the total at 8K tokens, so TTFT is close to linear in prompt
length, curving slightly upward at long prompts. This is the basis of H3.

**Decode (TPOT)**: each step reads all weights plus every running sequence's KV cache, and does
2 · params FLOPs per sequence:

    time/step ≈ max( (weights + batch · context · 56 KiB) / (bandwidth · efficiency),
                     2 · params · batch / (peak FLOPs · MFU) )

At small batch the weight read dominates, so TPOT barely changes from batch 1 to 8 while total
throughput grows about 8×. That is why batching works. As batch and context grow, KV reads and then
compute take over, and TPOT rises. This is the saturation knee in H1.

**Example (L4, approximate datasheet values: 121 TFLOPS bf16, 300 GB/s; 50% MFU, 80% bandwidth
efficiency):**

| | Estimate |
|---|---|
| TTFT, 1K prompt | ≈ 260 ms |
| TTFT, 8K prompt | ≈ 2.3 s |
| TPOT, batch 1 | ≈ 64 ms (≈ 16 tok/s) |
| TPOT, batch 32 | ≈ 71 ms (≈ 450 tok/s total) |

Real numbers will differ: MFU for short prompts is lower than 50%, CUDA graphs and kernel launch
overheads add fixed costs, and the chunked-prefill budget changes how prefill and decode share a
step. Explaining those gaps is part of the Phase 3 report.

**Mac development (ADR-0009):** run the script with your chip's memory bandwidth and
`--weights-gb 4.3` for the 4-bit model. Use it only to check that the harness sees the expected
*shape* (TTFT growing with prompt length, TPOT flat at small batch). Never compare Mac numbers with
GPU numbers.
