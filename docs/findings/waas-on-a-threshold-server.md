# WAAS on a threshold server: Phase 6 and H5 findings

This records the server-threshold experiment, the WAAS tuning, and the held-out H5 evaluation
that ADR-0014 planned. The platform is the one ADR-0013 describes: the M5 MacBook Air with
vllm-metal 0.31.0 and Qwen2.5-7B-Instruct-AWQ. Every point uses E07's mixed workload behind the
gateway with an in-flight cap of 16.

The numbers come from `llmserve report` (sections "Long-prefill threshold", "WAAS tuning" and
the H5 verdict) over the suites `threshold-paired`, `waas-paired` and `eval-mixed`, and from
`analysis.policies` over `xcheck-paired`, `xcheck-control`, `cap-paired`, `overload-mixed` and `kv-pressure`. TTFT is the interactive class's
P95 unless stated otherwise.

## Short answer

No gateway ordering beat sending requests straight to a threshold server below capacity, and
WAAS does not help there. Past capacity WAAS shortens the interactive tail only by starving
generation-heavy requests.

Setting vLLM's `--long-prefill-token-threshold 256` removes most of the interactive tail. On
that server, WAAS (cost_weight 2, max_bypass 32) does not beat FIFO on held-out seeds. H5 is
inconclusive. At ρ 0.95 the P95 change is −0.49 s, with a 95% CI of −4.28 to +2.13 s.

On the same server, sending requests straight to vLLM with no gateway beats every gateway
ordering tried, on every class, at equal throughput. The gain comes from removing the gateway's
in-flight cap of 16, not from vLLM's priority scheduling, which adds nothing over FCFS. At
ρ 0.95 a cap of 32 or more never fills, so the gateway has no queue to order.

Past capacity (`overload-mixed`, ρ 1.2), the one clock-matched seed shows WAAS cutting
interactive P95 from 39 s to 24 s against gateway FIFO and no gateway, while generation-heavy
P95 rises from 18 to 38 s to 155 s. At ρ 1.05 nothing queued at the gateway.

Shrinking the KV pool (H6, `kv-pressure`) makes vLLM preempt, and the interactive tail grows by
two orders of magnitude, but WAAS gains nothing at any pool size or long-context share. Its
interactive P95 is below FIFO's in 5 of 12 (pool, share, seed) cells and above in 7.

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

## In-flight cap on a threshold server (`cap-paired`)

Gateway FIFO, threshold 256, ρ 0.95, caps 16, 32, 64 and 256 (vLLM's `--max-num-seqs`, so
uncapped). Seeds 42 and 43 each run their four caps back to back on one server, in alternating
order. Per run:

| Seed | Cap | GPU MHz | Interactive | Generation-heavy | Long-context | Generated tok/s | Max gateway wait |
|---|---|---|---|---|---|---|---|
| 42 | 16 | 1087 | 10.8 s | 3.1 s | 37.9 s | 28.1 | 23.3 s |
| 42 | 32 | 1031 | 4.5 s | 2.0 s | 35.2 s | 28.1 | 0.0 s |
| 42 | 64 | 889 | 8.4 s | 6.5 s | 47.8 s | 25.5 | 0.0 s |
| 42 | 256 | 932 | 5.1 s | 3.5 s | 43.3 s | 28.5 | 0.0 s |
| 43 | 16 | 963 | 37.1 s | 43.9 s | 53.2 s | 28.6 | 44.8 s |
| 43 | 32 | 970 | 5.4 s | 6.8 s | 44.4 s | 30.5 | 0.0 s |
| 43 | 64 | 958 | 5.3 s | 6.8 s | 44.8 s | 28.4 | 0.0 s |
| 43 | 256 | 964 | 5.4 s | 7.7 s | 45.1 s | 30.5 | 0.0 s |

- **Only cap 16 queues.** At 32 and above no request waited at the gateway, so those runs are
  the same as no gateway, and their interactive P95 (4.5 to 5.4 s) matches `xcheck-control`.
- **Cap 16 is the tail.** Seed 43's clocks are matched (958 to 970 MHz), and cap 16 is 37.1 s
  against 5.3 to 5.4 s. On seed 42 cap 16 had the fastest clock and was still the worst run.
  Seed 42's cap 64 run had the slowest clock (889 MHz) and the next-worst tail.
- **Throughput does not depend on the cap** above 16 (25.5 to 30.5 tok/s, tracking the clock).

So at ρ 0.95 on this server, vLLM never holds more than 32 of these requests at once. A cap
that leaves room for that has nothing to queue and nothing for an admission order to act on,
and a cap that doesn't adds a gateway wait that dominates the tail. Gateway ordering can only
matter in sustained overload (ρ > 1), where some queue is unavoidable.

## Overload (`overload-mixed`)

Gateway FIFO and WAAS (cost_weight 2, max_bypass 32), both at in-flight cap 32, against no
gateway on a FCFS server. Threshold 256, ρ 1.05 and 1.2 against the pre-threshold capacity
estimate (0.15 req/s), held-out seeds 1000 and 1001. Each seed and ρ runs its three setups back
to back in alternating order. Per run, P95 TTFT:

| ρ | Seed | Setup | GPU MHz | Interactive | Generation-heavy | Long-context | Generated tok/s | Max gateway wait |
|---|---|---|---|---|---|---|---|---|
| 1.05 | 1000 | FIFO | 1134 | 5.0 s | 4.4 s | 43.1 s | 28.5 | 0.0 s |
| 1.05 | 1000 | WAAS | 1065 | 6.4 s | 5.3 s | 45.0 s | 28.7 | 0.0 s |
| 1.05 | 1000 | No gateway | 872 | 11.2 s | 9.0 s | 70.4 s | 24.2 | 0.0 s |
| 1.05 | 1001 | FIFO | 1033 | 5.1 s | 3.8 s | 35.9 s | 35.3 | 0.0 s |
| 1.05 | 1001 | WAAS | 1041 | 5.3 s | 3.2 s | 35.8 s | 35.3 | 0.0 s |
| 1.05 | 1001 | No gateway | 1068 | 4.6 s | 3.2 s | 32.2 s | 36.8 | 0.0 s |
| 1.2 | 1000 | FIFO | 1167 | 5.8 s | 5.0 s | 42.5 s | 31.2 | 2.1 s |
| 1.2 | 1000 | WAAS | 983 | 38.6 s | 204.8 s | 78.9 s | 24.5 | 204.5 s |
| 1.2 | 1000 | No gateway | 831 | 197.4 s | 193.2 s | 307.7 s | 15.3 | 0.0 s |
| 1.2 | 1001 | FIFO | 1019 | 39.4 s | 37.8 s | 56.6 s | 33.3 | 39.6 s |
| 1.2 | 1001 | WAAS | 1012 | 23.5 s | 154.7 s | 61.8 s | 33.3 | 166.2 s |
| 1.2 | 1001 | No gateway | 1008 | 38.7 s | 17.6 s | 81.7 s | 33.3 | 0.0 s |

All requests completed in every run.

- **ρ 1.05 is not overload.** The cap of 32 never filled, so the gateway setups match no
  gateway. On seed 1001 (1033 to 1068 MHz) all three are within 0.7 s on interactive P95. The
  capacity estimate predates the threshold, and the server now keeps up at this rate.
- **At ρ 1.2 WAAS trades classes.** Seed 1001 is matched (1008 to 1019 MHz) and throughput is
  identical. WAAS cuts interactive P95 by 16 s against both other setups and raises
  generation-heavy P95 to 155 s, against 38 s under FIFO and 18 s with no gateway. The worst
  gateway wait rises from 40 s to 166 s. max_bypass 32 bounds how often a request is passed
  over, not how long it waits, and past capacity 32 bypasses add up to minutes.
- **No gateway against gateway FIFO** at ρ 1.2 seed 1001 ties on interactive (38.7 against
  39.4 s), is better on generation-heavy (17.6 against 37.8 s) and worse on long-context (81.7
  against 56.6 s). Holding the queue in vLLM rather than the gateway moves the tail between
  classes without shrinking it.
- **Seed 1000 at ρ 1.2 is confounded.** Its three runs span 831 to 1167 MHz, and the tail
  tracks the clock: the slowest run (no gateway) fell to 15.3 tok/s and a 197 s interactive
  P95, while the fastest (FIFO) barely queued. At ρ 1.2 a 30% clock drop pushes effective load
  far past capacity, so this seed says nothing about the setups.

With one clean seed per ρ, these are observations, not a test. They do not show a free win
for WAAS past capacity: its interactive gain is paid for by a generation-heavy tail four to
nine times longer.

## KV pressure, H6 (`kv-pressure`)

Gateway FIFO and WAAS (cost_weight 2, max_bypass 32) at in-flight cap 32, against no gateway,
at ρ 0.95 on a FCFS threshold-256 server. Three KV pool sizes (`--gpu-memory-utilization` 0.5,
0.45 and 0.4) times two long-context shares (25%, the E07 mix, and 40%, taken from interactive),
held-out seeds 1000 and 1001. The server restarts once per pool size, and each (share, seed)
runs its three setups back to back in alternating order.

The plan named 0.3, 0.4 and 0.5. vllm-metal sizes the pool as a fraction of the 19.07 GB Metal
limit minus the model and overhead (6.51 GB), so 0.3 leaves a negative KV budget and the engine
refuses to start. 0.4 is close to the floor: its 19.5k tokens hold two 10,240-token requests.

Each cell reads FIFO / WAAS / no gateway. P95 TTFT in seconds:

| Pool (KV tokens) | Long share | Seed | GPU MHz | Interactive | Generation-heavy | Preemptions | Generated tok/s |
|---|---|---|---|---|---|---|---|
| 0.5 (52.8k) | 25% | 1000 | 1027 to 1100 | 3.8 / 5.9 / 5.8 | 3.6 / 5.4 / 5.6 | 0 / 0 / 0 | 28.5 / 28.6 / 28.6 |
| 0.5 (52.8k) | 25% | 1001 | 1020 to 1031 | 4.8 / 5.3 / 4.7 | 2.8 / 3.3 / 2.5 | 0 / 0 / 0 | 32.6 / 32.6 / 32.6 |
| 0.5 (52.8k) | 40% | 1000 | 1015 to 1018 | 12.7 / 14.8 / 14.4 | 5.0 / 5.1 / 4.8 | 15 / 15 / 14 | 25.2 / 25.7 / 25.7 |
| 0.5 (52.8k) | 40% | 1001 | 1002 to 1007 | 35.1 / 29.6 / 21.5 | 33.2 / 37.3 / 12.7 | 10 / 11 / 20 | 28.6 / 28.6 / 27.2 |
| 0.45 (36.2k) | 25% | 1000 | 999 to 1014 | 12.8 / 10.3 / 8.4 | 7.3 / 6.3 / 7.1 | 11 / 9 / 12 | 27.5 / 27.5 / 28.1 |
| 0.45 (36.2k) | 25% | 1001 | 1016 to 1022 | 4.7 / 5.3 / 5.0 | 2.6 / 3.2 / 2.6 | 0 / 0 / 0 | 32.6 / 31.6 / 32.6 |
| 0.45 (36.2k) | 40% | 1000 | 978 to 986 | 147.0 / 139.5 / 158.7 | 78.5 / 401.1 / 103.7 | 155 / 194 / 170 | 15.8 / 13.7 / 15.6 |
| 0.45 (36.2k) | 40% | 1001 | 1011 to 1019 | 37.5 / 36.8 / 34.3 | 53.9 / 34.4 / 23.6 | 41 / 28 / 25 | 27.9 / 28.5 / 29.0 |
| 0.4 (19.5k) | 25% | 1000 | 1017 to 1023 | 23.5 / 23.9 / 23.4 | 12.3 / 14.3 / 12.9 | 41 / 38 / 33 | 25.2 / 24.3 / 25.9 |
| 0.4 (19.5k) | 25% | 1001 | 1007 to 1033 | 28.6 / 34.0 / 48.6 | 29.7 / 39.6 / 52.0 | 32 / 57 / 62 | 31.4 / 31.2 / 30.2 |
| 0.4 (19.5k) | 40% | 1000 | 976 to 983 | 283.8 / 270.0 / 269.8 | 208.3 / 191.2 / 192.6 | 184 / 186 / 174 | 16.9 / 18.1 / 17.9 |
| 0.4 (19.5k) | 40% | 1001 | 770 to 866 | 927.2 / 1067.6 / 820.4 | 916.4 / 1336.2 / 825.2 | 318 / 323 / 335 | 15.8 / 11.3 / 12.8 |

All requests completed in every run. ρ is against the 0.15 req/s capacity measured at the E07
mix, so the 40% share is a heavier load at the same ρ.

- **KV pressure dominates.** Averaged over setups and seeds, interactive P95 is 5.1 s at pool 0.5
  and share 25%, 21 s at 0.5 and 40%, 92 s at 0.45 and 40%, and 606 s at 0.4 and 40%.
  Preemptions go from 0 to 250 per run, and throughput falls from 31 to 15 generated tok/s.
  At 0.4 and 40% the server cannot keep up, which is overload by memory rather than by rate.
- **WAAS gains nothing at any level.** On interactive P95 it is below FIFO in 5 of 12 cells and
  above in 7, with no trend in pool size or share. It wins by more than 2 s in four cells: 5.5 s
  (0.5, 40%, seed 1001), 2.5 s (0.45, 25%, seed 1000), 7.5 s (0.45, 40%, seed 1000) and 13.8 s
  (0.4, 40%, seed 1000, against a 284 s tail). In the 0.45, 40%, seed 1000 cell its
  generation-heavy P95 is 401 s against FIFO's 79 s.
- **The gateway queues only at the 40% share.** At 25% the cap of 32 never filled at any pool
  size, so the gateway setups match no gateway and differences are run-to-run noise.
- **A cap can limit preemption, sometimes.** At 0.4 and 25% on seed 1001, FIFO with cap 32
  preempted 32 times against 62 with no gateway, and its interactive P95 was 28.6 s against
  48.6 s. Seed 1000 in the same cell shows no difference (41 against 33 preemptions, 23.5
  against 23.4 s).
- **Clocks.** Every cell is within 30 MHz except two. (0.5, 25%, seed 1000) has FIFO at 1100 MHz
  against 1027 and 1028, and (0.4, 40%, seed 1001) spans 770 to 866 MHz, so its tails, all above
  800 s, track the clock.

There was no pre-registered rule for H6. Read against its statement (the WAAS gain over FIFO
grows with KV pressure and long-context share), it is not supported: there is no consistent gain
at any level to grow.

## What it means

The tail WAAS was designed against came mostly from long prefills inside vLLM (ADR-0014,
context item 3). The server threshold addresses that directly. On the held-out seeds it leaves
FIFO's ρ 0.95 P95 at 5 to 12 s, against 16 to 44 s on the tuning seeds at the same threshold.
That leaves too little queueing for an admission order to recover. WAAS still helps on seeds
whose FIFO tail is long, and it costs longer worst-case waits.

## Open questions

- **A KV-aware gate.** Under KV pressure the lever is preemption, not order. A gateway that holds
  a long-context request while the KV pool is nearly full might prevent the preemption cascades
  at 0.4 and 0.45. One seed hints that a plain cap sometimes does this.
- **Overload with an aging bound.** A WAAS variant that bounds wait time rather than bypass
  count might keep the interactive gain at ρ 1.2 without starving generation-heavy requests.
  More matched seeds would be needed to tell.
- **WAAS without the threshold.** WAAS against FIFO on an unset server would show whether
  gateway ordering alone recovers the tail. It would not count toward H5.
- **The clock-threshold link.** A run that logs GPU clock per engine step under both
  thresholds would show whether the higher clock follows the threshold.
- **Higher cost_weight.** w2 sits at the edge of the tuned grid.
