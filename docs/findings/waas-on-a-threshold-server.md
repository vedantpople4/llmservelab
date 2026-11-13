# WAAS on a threshold server: Phase 6 and H5 findings

This records the server-threshold experiment, the WAAS tuning, and the held-out H5 evaluation
that ADR-0014 planned. The platform is the one ADR-0013 describes: the M5 MacBook Air with
vllm-metal 0.31.0 and Qwen2.5-7B-Instruct-AWQ. Every point uses E07's mixed workload behind the
gateway with an in-flight cap of 16.

The numbers come from `llmserve report` (sections "Long-prefill threshold", "WAAS tuning" and
the H5 verdict) over the suites `threshold-paired`, `waas-paired` and `eval-mixed`, and from
`analysis.policies` over `xcheck-paired` and `xcheck-control`. TTFT is the interactive class's
P95 unless stated otherwise.

## Short answer

Setting vLLM's `--long-prefill-token-threshold 256` removes most of the interactive tail. On
that server, WAAS (cost_weight 2, max_bypass 32) does not beat FIFO on held-out seeds. H5 is
inconclusive. At ρ 0.95 the P95 change is −0.49 s, with a 95% CI of −4.28 to +2.13 s.

On the same server, sending requests straight to vLLM with no gateway beats every gateway
ordering tried, on every class, at equal throughput. The gain comes from removing the gateway's
in-flight cap of 16, not from vLLM's priority scheduling, which adds nothing over FCFS.

## The GPU clock confound

The fanless Air's sustained GPU clock varies from about 890 to 1100 MHz between runs. At ρ 0.95
that alone moves interactive P95 3 to 5×. The thermal probe compares TPOT before and after
each run and does not see it. The unpaired suites (`prefill-threshold`, `waas-tuning`)
compared runs hours apart and are confounded by it.

The paired suites run each seed's conditions back to back, alternating which goes first.
`llmserve report` prints the mean GPU clock of every run next to its TTFT. Below, "matched"
means the two runs of a pair are within about 15 MHz.

## Long-prefill threshold (`threshold-paired`)

FIFO, threshold unset (0) against 256, at ρ 0.8 and 0.95, seeds 42 to 44.

| ρ | Seed | Unset | 256 | Δ interactive | Δ long-context | Clocks, unset / 256 |
|---|---|---|---|---|---|---|
| 0.8 | 42 | 13.5 s | 3.8 s | −9.7 s | +8.4 s | 1041 / 1053 MHz |
| 0.8 | 43 | 14.5 s | 5.2 s | −9.3 s | +4.4 s | 1013 / 1053 MHz |
| 0.8 | 44 | 30.9 s | 6.1 s | −24.8 s | −4.8 s | 911 / 1029 MHz |
| 0.95 | 42 | 21.7 s | 11.6 s | −10.2 s | +11.7 s | 1057 / 1060 MHz |
| 0.95 | 43 | 27.0 s | 18.1 s | −8.9 s | −4.7 s | 989 / 1064 MHz |
| 0.95 | 44 | 65.2 s | 49.3 s | −16.0 s | −8.3 s | 978 / 1054 MHz |

- Seed 42 is matched at both ρ. On it, the threshold cuts interactive P95 by about 10 s, and
  long-context P95 rises by 8 to 12 s. That is the expected trade: a long prompt's prefill now
  shares each step with the waiting prompts.
- Mean generated tok/s did not fall (24.2 to 25.9 at ρ 0.8, 28.9 to 29.2 at ρ 0.95).
- The threshold run had the higher clock in all six pairs, whichever ran first. So the clock may
  partly respond to the threshold, for example through lower sustained power from shorter
  prefill steps. These runs cannot separate that from chance.

## WAAS tuning (`waas-paired`)

FIFO and WAAS at cost_weight 0.5, 1 and 2 (max_bypass 32), ρ 0.95, threshold 256, class-prior
output estimates, tuning seeds 42 to 44. An earlier unpaired grid (`waas-tuning`) showed no
visible effect of max_bypass 8 against 32.

| Seed | FIFO | w0.5 Δ | w1 Δ | w2 Δ | Clocks |
|---|---|---|---|---|---|
| 42 | 43.5 s | −1.5 s | −8.8 s | −33.3 s | 876 to 1039 MHz, not matched |
| 43 | 16.0 s | −1.6 s | −0.3 s | −4.4 s | 1066 to 1078 MHz |
| 44 | 37.7 s | −6.1 s | −2.4 s | −4.9 s | 1067 to 1070 MHz |

cost_weight 2 was the only setting below FIFO on every seed, and it was frozen for the
evaluation. It is the largest weight tried, so the optimum may be higher. Throughput was flat
(28.5 to 29.1 tok/s). The worst gateway wait rose from 59 s under FIFO to 155 s under w2.

## H5 (`eval-mixed`)

FIFO against WAAS at cost_weight 2 and max_bypass 32, threshold 256, ρ 0.85 and 0.95, held-out
seeds 1000 to 1004. The rule is in `llmserve/analysis/hypotheses.py` and was fixed before any
WAAS run.

| ρ | FIFO P95 | WAAS P95 | Δ (95% CI) | Throughput, WAAS / FIFO |
|---|---|---|---|---|
| 0.85 | 4.56 s | 4.93 s | +0.37 s (−0.55 to +1.28) | 0.992 |
| 0.95 | 8.22 s | 7.73 s | −0.49 s (−4.28 to +2.13) | 0.995 |

Each condition pools 392 to 437 interactive requests. Throughput stays above 95% of FIFO's. No
CI excludes zero, and the ρ 0.95 CI does not exclude a 10% gain, so the verdict is
inconclusive.

Per seed, at ρ 0.85 the two policies are within 0.25 s on four seeds. At ρ 0.95 WAAS cuts P95
by 3.9 s and 4.4 s on seeds 1000 and 1002, where FIFO's tail reached 12 s, and is flat on 1001
and 1003, where it stayed near 5 s. Seed 1004 ran at 900 to 1030 MHz against 1040 to 1080 for
the rest, and its WAAS runs were the slower-clocked of each pair. It is the only seed where WAAS
lost, by 1.2 s and 2.2 s, with long-context P95 14 to 23 s worse in the same runs. The rule has
no clock adjustment, so it counts. The worst gateway wait rose from 7.0 s to 10.5 s at ρ 0.85
and from 39 s to 49 s at ρ 0.95.

## Gateway against no gateway (`xcheck-paired`, `xcheck-control`)

Threshold 256, ρ 0.85 and 0.95, tuning seeds 42 to 44. `xcheck-paired` runs gateway FIFO and
SPF (in-flight cap 16, FCFS server) and vLLM's native priority with no gateway
(`--scheduling-policy priority`), each seed's blocks back to back. `xcheck-control` adds no
gateway on a FCFS server, run a day later. P95 TTFT, pooled over seeds:

| ρ | Setup | Interactive | Generation-heavy | Long-context | Generated tok/s |
|---|---|---|---|---|---|
| 0.85 | Gateway FIFO | 15.6 s | 4.6 s | 38.7 s | 26.5 |
| 0.85 | Gateway SPF | 6.7 s | 4.0 s | 39.7 s | 27.0 |
| 0.85 | No gateway, priority | 5.1 s | 3.9 s | 36.6 s | 27.0 |
| 0.85 | No gateway, FCFS | 4.7 s | 4.4 s | 34.1 s | 25.7 |
| 0.95 | Gateway FIFO | 51.5 s | 49.8 s | 67.6 s | 28.4 |
| 0.95 | Gateway SPF | 28.4 s | 10.5 s | 69.9 s | 29.0 |
| 0.95 | No gateway, priority | 5.8 s | 4.9 s | 40.9 s | 29.7 |
| 0.95 | No gateway, FCFS | 5.8 s | 4.6 s | 35.5 s | 28.6 |

- **The cap is the cost.** Without a gateway, vLLM batches every arrival and nothing waits in
  front of it. With the cap, requests queued up to 35 s (ρ 0.85) and 96 s (ρ 0.95) at the
  gateway under FIFO. At clock-matched pairs, no gateway on FCFS against gateway FIFO is 5.6 s
  against 17.2 s (ρ 0.85, seed 44, 1003 / 1023 MHz) and 4.2 s against 16.5 s (ρ 0.95, seed 43,
  1065 / 1059 MHz).
- **Priority adds nothing.** Per run, no gateway on FCFS and on priority are within 1 s on
  interactive P95 except where clocks differ, for example 4.2 s against 4.8 s at ρ 0.95 seed 43
  (1065 / 1070 MHz).
- **Throughput is flat** across all four setups (25.7 to 29.7 generated tok/s), and no setup
  improved one class at another's expense.
- `xcheck-control` ran on a different day from `xcheck-paired`, so its comparisons are across
  sessions. Its clocks (996 to 1103 MHz) sit at or above the other runs', so seed-level
  comparisons above use pairs within about 20 MHz.

The cap of 16 was chosen in Phase 5 (`inflight-cap`) on a server without the threshold, as the
smallest cap keeping 95% of uncapped throughput. On a threshold server it still costs no
throughput to remove, and it makes the tail it was meant to shape. Admission ordering only
helped because the cap created a queue to order.

## What it means

The tail WAAS was designed against came mostly from long prefills inside vLLM (ADR-0014,
context item 3). The server threshold addresses that directly. On the held-out seeds it leaves
FIFO's ρ 0.95 P95 at 5 to 12 s, against 16 to 44 s on the tuning seeds at the same threshold.
That leaves too little queueing for an admission order to recover. WAAS still helps on seeds
whose FIFO tail is long, and it costs longer worst-case waits.

## Open questions

- **The cap on a threshold server.** A cap sweep with the threshold set (for example 16, 32,
  64 and none) would show whether any cap keeps a queue worth ordering without the tail cost.
  If none does, the gateway's role on this platform is overload protection, not ordering.
- **WAAS without the threshold.** WAAS against FIFO on an unset server would show whether
  gateway ordering alone recovers the tail. It would not count toward H5.
- **The clock-threshold link.** A run that logs GPU clock per engine step under both
  thresholds would show whether the higher clock follows the threshold.
- **Higher cost_weight.** w2 sits at the edge of the tuned grid.
