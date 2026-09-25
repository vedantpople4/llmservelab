# Learning path for Phase 0

A one-week path through the material behind `docs/background.md`. Each stage says which part of
the homework it feeds. Work through them in order: each assumes the previous one.

Legend: ★ = required for the homework, ○ = optional depth.

## Stage 1: How a transformer generates text (day 1)

Goal: be able to explain why generation happens one token at a time, and what gets recomputed or
cached at each step.

- ★ Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
  Attention, Q/K/V, and layers, with pictures.
- ○ 3Blue1Brown's "Neural networks" video series, chapters on transformers and attention, if you
  want visual intuition first.
- ○ Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT). Read `model.py` and
  `sample.py`. The generation loop is only a few lines, and the KV cache is what those lines are
  missing.

**Self-check:** why does generating token *n* need the keys and values of all previous tokens? What
would it cost to recompute them at every step?

## Stage 2: Inference arithmetic (day 2) → background.md §2 and §3

Goal: be able to predict TTFT and TPOT from hardware specs. This is the core of the Phase 0 exit
check.

- ★ kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/).
  FLOPs vs memory bandwidth, KV cache size, and why decode is memory-bound. Read it with a
  calculator open.
- ★ [How To Scale Your Model, ch. 7: All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)
  (Google DeepMind). The best modern treatment of prefill vs decode, KV-cache sizing and
  disaggregation. The earlier chapter
  [All the Transformer Math You Need to Know](https://jax-ml.github.io/scaling-book/transformers/)
  covers the FLOP counts.
- ★ Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html).
  Compute-bound vs memory-bound vs overhead-bound. The overhead part explains why short prompts
  get low MFU.
- ○ Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
  (MLSys '23). The research paper behind much of the above.
- ○ The model's own [config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json).
  Check the numbers in `scripts/envelope.py` against it.

**Homework:** redo the KV-cache-per-token and batch-1 TPOT calculations in background.md §4 by
hand, then run `scripts/envelope.py` and compare. Fill in the "Predicted" column of §3.

## Stage 3: Serving systems, the core papers (days 3–4) → background.md §1 and §2

Goal: know what vLLM does internally, so you know what a gateway scheduler can and cannot change
(ADR-0001).

Read in this order. For each paper, use the note template at the end of this file.

1. ★ Anyscale, [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference).
   A blog-length introduction; read it before Orca.
2. ★ Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
   (OSDI '22). Iteration-level scheduling. Focus on §3 (the problem) and §4 (selective batching).
3. ★ Kwon et al., [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
   (SOSP '23). This is vLLM. Focus on KV fragmentation, block tables, and preemption (swap vs
   recompute).
4. ★ Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
   (OSDI '24). Chunked prefill and stall-free batching. **The most important paper for H2 and H4**:
   it explains how much head-of-line blocking vLLM already removes.
5. ★ Zhong et al., [DistServe](https://arxiv.org/abs/2401.09670) (OSDI '24). Goodput under
   TTFT/TPOT SLOs, and why prefill and decode interfere. This is where our goodput metric comes
   from.
6. ★ Wu et al., [FastServe: Fast Distributed Inference Serving for LLMs](https://arxiv.org/abs/2305.05920).
   Preemptive, size-aware scheduling. The closest related work to WAAS.

Optional depth on scheduling (useful before Phase 5–6; skim abstracts now):

- ○ Fu et al., [Efficient LLM Scheduling by Learning to Rank](https://arxiv.org/abs/2408.15792)
  (NeurIPS '24). You can't predict exact output length, but you can predict its *rank*. Directly
  relevant to the output-estimator ablation. Code: [vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr).
- ○ Zheng et al., [Response Length Perception and Sequence Scheduling](https://arxiv.org/abs/2305.13144)
  (NeurIPS '23). Output-length prediction with the LLM itself.
- ○ Sun et al., [Llumnix: Dynamic Scheduling for LLM Serving](https://arxiv.org/abs/2406.03243)
  (OSDI '24). Priorities and rescheduling across instances; relevant to the multi-GPU stretch goal.
- ○ Patel et al., [Splitwise](https://arxiv.org/abs/2311.18677). Prefill/decode split across
  machines.
- ○ Zheng et al., [SGLang](https://arxiv.org/abs/2312.07104). RadixAttention and prefix caching:
  why we disable prefix caching (ADR-0005).

Read the code too:

- ★ [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): a vLLM-style engine in about 1,200
  lines of Python. Read its scheduler and block manager after the PagedAttention paper; they are
  much easier to follow than vLLM's own.
- ○ [vLLM docs](https://docs.vllm.ai/): "Optimization and Tuning" (`max_num_seqs`,
  `max_num_batched_tokens`, chunked prefill) and the metrics page. Check the pages for the vLLM
  version you will pin.
- ○ [vLLM blog](https://blog.vllm.ai/): posts on the V1 engine architecture.

**Homework:** background.md §2. The key question is where head-of-line blocking can still happen
with chunked prefill. Answer it using Sarathi-Serve and the nano-vllm scheduler.

## Stage 4: Measurement methodology (day 5) → ADR-0003, ADR-0007

Goal: understand why the plan insists on open-loop load, paired runs and confidence intervals.
This is what separates a research result from a benchmark screenshot.

- ★ Schroeder, Wierman and Harchol-Balter, [Open Versus Closed: A Cautionary Tale](https://www.usenix.org/conference/nsdi-06/open-versus-closed-cautionary-tale)
  (NSDI '06). Why closed-loop load hides scheduling effects. This is the paper behind ADR-0003.
  Read the eight principles at the end.
- ★ Gil Tene, "How NOT to Measure Latency" (QCon SF 2015):
  [slides](https://videog.infoq.com/downloads/pdfdownloads/presentations/QConSF2015-GilTene-HowNOTtomeasureLatency.pdf).
  Coordinated omission: why the harness measures latency from the *scheduled* arrival time and
  tracks client lag.
- ★ Hoefler and Belli, [Scientific Benchmarking of Parallel Computing Systems](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
  (SC '15). Twelve rules for reporting performance: percentiles, variability, confidence
  intervals, no cherry-picking. Keep it open while writing the paper.
- ○ Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!"
  (ASPLOS '09). How small environment changes create fake speedups; the reason runs are
  interleaved.
- ○ Mor Harchol-Balter, *Performance Modeling and Design of Computer Systems* (book). Queueing
  theory and SRPT/SJF scheduling. Chapters on M/G/1 and size-based scheduling explain why SJF
  helps the median but hurts large jobs, which is the trade-off WAAS must manage.

**Self-check:** at utilization ρ = 0.5, would you expect FIFO and SJF to differ much? At ρ = 0.95?
Why? (Queueing theory predicts the answer before any experiment runs.)

## Stage 5: Hands-on on your Mac (day 6) → background.md §3

Goal: see prefill vs decode with your own eyes before writing any harness code.

```bash
uv pip install mlx-lm      # in a separate venv is fine
mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "$(python -c 'print("hello " * 2000)')" --max-tokens 128 --verbose true
```

`--verbose` prints prompt tokens/s (prefill) and generation tokens/s (decode). Repeat with prompt
lengths of about 128, 512, 2K and 8K tokens, and with the 7B 4-bit model if memory allows.

**Homework:** plot prompt length against prefill time. Is it roughly linear? Is decode speed
roughly constant as prompt length grows, and does it drop at 8K? Compare with
`scripts/envelope.py --tflops <mac> --bandwidth-gbs <mac> --weights-gb 4.3`, and write down the
gap. These are Mac numbers: they check your understanding, not the paper. Check `mlx_lm.generate
--help` for your version in case the flag names differ.

- Tools reference: [mlx-lm](https://github.com/ml-explore/mlx-lm) and
  [llama.cpp](https://github.com/ggml-org/llama.cpp) (see its server documentation for
  `--parallel` and `--metrics`).

## Stage 6: Write it up (day 7)

Finish background.md §1–3. The exit check is to explain, without notes:

1. prefill vs decode, and which hardware limit each hits
2. KV-cache size per token for Qwen2.5-7B, and what happens when it runs out
3. continuous batching and chunked prefill, and what each fixes
4. where head-of-line blocking still happens in vLLM, and so where a gateway scheduler can help
5. why scheduler experiments need open-loop load

## Paper note template

```markdown
### <Paper> (<venue, year>)
- Problem:
- Key mechanism:
- Headline result (with the baseline it beats):
- Assumptions / limitations:
- What it means for LLMServeLab: (which hypothesis, ADR, or phase it affects)
```
