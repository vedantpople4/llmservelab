# ADR-0013: The M5 MacBook with vllm-metal is the measurement platform

Status: accepted (October 2026). Supersedes ADR-0011 and the "never reported" rule of ADR-0012.

## Context

ADR-0011 put every reported number on a free Kaggle or Colab T4. In practice the T4 route never
produced a run: sessions are quota-bound, disconnect, and need setup each time, while the whole
pipeline (suites, analysis, report) already runs end to end on the Mac against vllm-metal
(ADR-0012) with the same checkpoint, tokenizer, payloads and `/metrics`. The study needs one
fixed platform. The owner chose the Mac.

Closed-loop probes on the M5 MacBook Air (24 GB, macOS 27, vLLM 0.31.0 + vllm-metal 0.31.0,
`Qwen/Qwen2.5-7B-Instruct-AWQ` at revision `b25037543e9394b818fdfca67ab2a00ecc7dd641`,
`--gpu-memory-utilization 0.5`, one rep each, October 2026):

| Probe | TTFT p50 | TPOT p50 | Throughput |
|---|---|---|---|
| 128/128, C = 1 | 0.23 s | 47 ms | 20 gen tok/s |
| 128/128, C = 16 | 3.5 s | 200 ms | 71 gen tok/s, 0.55 req/s |
| 128/128, C = 32 | 7.4 s | 292 ms | 93 gen tok/s, 0.73 req/s |
| 128/128, C = 64 | 9.6 s | 537 ms | 102 gen tok/s, 0.80 req/s |
| 2048/16, C = 1 | 2.5 s | 47 ms | 640 prompt tok/s |
| 4096/16, C = 1 | 5.4 s | 48 ms | 670 prompt tok/s |
| 8192/16, C = 1 | 12.8 s | 58 ms | 600 prompt tok/s |
| mixed workload, C = 16 | 7.2 s | 267 ms | 0.22 req/s |

Against the T4 envelope, prefill is about 3× slower and batch-1 decode about 2× slower, and the
KV pool holds 52,784 tokens rather than about 130–140k.

## Decision

- **Platform.** Every reported number comes from this one machine: an M5 MacBook Air, 24 GB,
  serving `Qwen/Qwen2.5-7B-Instruct-AWQ` at the pinned revision through vLLM 0.31.0 +
  vllm-metal 0.31.0, with the server flags in `docs/runbooks/mac.md`. `server.kind:
  vllm-metal` is the one reportable kind (`MEASUREMENT_KIND` in `llmserve/analysis/runs.py`).
  The `vllm` kind and the T4 notebooks stay in the code as an unused route.
- **Sizing.** Configs are sized for the Mac's measured capacity (0.8 req/s for 128/128, about
  0.22 req/s for the mixed workload): Phase 3 is about 25 machine-hours. The suite estimator's
  Mac model is refit from the table above.
- **SLOs.** Until Phase 3 freezes SLOs from isolated latency, the placeholders are the PRD's T4
  values scaled by the measured speed ratios: TTFT ×3, TPOT ×2.
- **Run hygiene, enforced in code.** A fanless laptop adds three failure modes. Each one makes a
  run unreportable instead of relying on a checklist:
  - Thermal drift: a fixed probe batch (16 requests at C=1, 128-token prompts, 32 output
    tokens) runs before the first repetition and after the last; its TPOT p50 after differs
    from before by more than 10% in either direction (a slower "before" means the run started on
    an already-slowed machine). Comparing repetitions instead would confound heat with workload,
    since each repetition samples its own workload.
  - Sleep: any time asleep during the run (a closed lid sleeps the Mac even mid-run).
  - Battery power at any point of the run; the source is polled every 15 s.
  Runs hold `caffeinate` against idle sleep, and the "before" probe repeats, a minute apart, until
  its TPOT p50 is at most 58 ms (a cool Mac probes at 45-55 ms) or 20 minutes pass, so heat from
  one experiment does not carry into the next.

  Amended October 2026 after Phase 3 sessions 2-4: the probe grew from 4 requests (one slow
  request moved the median past 10%), drift became two-sided, battery is polled rather than read
  at the ends, and the cool-down gate was added (E06 started at 67-91 ms right after E05).

## Consequences

- **The research question narrows.** It becomes WAAS vs FIFO for one Apple-silicon machine
  serving a 7B model. "GPU efficiency" means throughput, KV-cache use and, once Phase 4 adds
  `powermetrics` sampling, tokens per joule. The paper states that transfer to datacenter GPUs
  is untested.
- **H1–H5 are unchanged** in form. The Mac's slow prefill and small KV pool should make
  head-of-line blocking stronger, which helps H4/H5 show an effect and limits how far it
  generalizes.
- **H6 is redesigned.** Above `--gpu-memory-utilization 0.5` the KV pool pages into swap, so KV
  capacity varies below it (0.3 / 0.4 / 0.5) and through the long-context share.
- **Phase 4 is redesigned.** No NVML, DCGM or Nsight. Profiling uses `powermetrics` (GPU active
  residency, frequency, power, thermal pressure; needs a sudoers rule), vLLM's `/metrics`, and
  an Instruments Metal trace. The answer to "where does the time go" is coarser.
- **An open anomaly.** TTFT at C ≥ 8 far exceeds prefill cost (3.5 s at C = 16 for a 0.23 s
  prefill). Phase 4 must explain it before Phase 5 builds on Mac TTFT.
- **Long runs occupy the machine.** Runs need the Mac idle, on power and with the lid open.
  Phase 7's matrix must be resized for the Mac before it runs.
- **Versions are part of the result.** vllm-metal is young. Results hold for the pinned vLLM and
  vllm-metal versions and are rerun, not reused, after an upgrade.
