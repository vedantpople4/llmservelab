# Finish Phase 7: plan

Phase 7's exit check (IMPLEMENTATION_PLAN.md) asks for three things that don't exist yet. The
`paper_01` matrix has to run from one command, H6 needs a verdict with CIs in `llmserve report`,
and every figure has to come from `paper/figures.yaml`. The plan also asks for one public trace
replay. This plan does all four as five PRs, each with its own evidence, then runs the Mac work.

What the earlier Phase 7 results change:

- **Setups.** The comparison is gateway FIFO, gateway WAAS and no gateway, all on the threshold-256
  server. SPF, SJF and Priority dropped out in Phase 5, and no gateway now beats every gateway
  order below capacity (`docs/findings/waas-on-a-threshold-server.md`). Both gateways use cap
  32, since cap 16 makes the tail by itself.
- **Load levels.** ρ 0.5, 0.85 and 1.2. Nothing below about 1.05 queues at a cap-32 gateway, so
  0.7 and 0.95 add little over 0.85. 1.2 is the only level where ordering has anything to do.
- **Seeds.** New held-out seeds from 2000. 1000 and 1001 have been used by three suites.

## Budget

Each point is a 20-minute window plus warm-up and the 10-minute cool-down, about 37 minutes.

| Work | Points | Mac hours |
|---|---|---|
| Capacity, generation-heavy workload | about 5 | 3 |
| Capacity, trace mix | about 5 | 3 |
| `paper_01`, 3 setups × 4 workloads × 3 ρ × **2 seeds** | 72 | 44 |
| Trace replay, 3 setups × 2 trace windows, ρ 0.85 | 6 | 4 |
| **Total** | about 88 | about 54 |

The 2-seed matrix is the recommendation. 1 seed halves `paper_01` to 22 hours but gives no
within-cell check on the clock confound. The original plan's 180 runs would take about 110 hours.

## PR 1. H6 verdict in the report

- [ ] Add `h6(suite)` to `llmserve/analysis/hypotheses.py` and call it from `build_report`.
- [ ] Rule: per (pool, share), the bootstrap CI of WAAS P95 minus FIFO P95 on interactive TTFT.
  Supported if the gain grows from pool 0.5 to 0.4 at both shares with CIs excluding zero at
  0.4. The docstring says the rule was written after `kv-pressure` ran.
- [ ] Test with a synthetic suite where the gain grows, and one where it doesn't.
- [ ] Evidence: `llmserve report` prints an H6 verdict on the stored `kv-pressure` runs.

## PR 2. Trace workload

The replay arrival process exists but reads offsets only, so lengths would still come from the
synthetic classes. A trace has to supply lengths too.

- [ ] `scripts/azure_trace.py` downloads `AzureLLMInferenceTrace_conv.csv` (Azure Public
  Dataset, 2023, CC BY 4.0) into `configs/traces/` with an attribution header. It writes two
  windows of about 250 consecutive requests each, one quiet and one bursty by per-minute
  count, as `offset_s,prompt_tokens,output_tokens`.
- [ ] Add a `workload.trace` source in `llmserve/config/schema.py` that takes per-request lengths
  and arrival offsets from that file. Class labels come from a fixed rule (prompt ≥ 2048 tokens
  is long_context, output ≥ 256 is generation_heavy, otherwise interactive), so per-class
  metrics and the WAAS class prior still work. Prompts longer than `max_model_len` minus the
  output are dropped and counted.
- [ ] `load.arrival.time_scale` stretches the trace to the target ρ. The conv trace runs at 5.5
  req/s, so ρ 0.85 is a slow-down of about 30×.
- [ ] Tests: the same file and seed give identical requests, and labels follow the rule.
- [ ] Evidence: a mock-server run of each window completes, with its class counts printed.

Trace facts, from the downloaded file: 19,366 requests over 58 minutes. Prompt tokens have
median 1,020, P99 4,142 and max 14,050, with one request over 10,240 in total. Output tokens
have median 129 and P99 601. Arrivals per minute range from 37 to 507.

## PR 3. Capacities

- [ ] A capacity suite for the generation-heavy workload and one for the trace mix, on the
  threshold-256 server, using the existing capacity rule.
- [ ] Run both on the Mac, about 6 hours.
- [ ] Evidence: `llmserve report` brackets both capacities.

Interactive (0.4 req/s), long-context-heavy (0.1) and mixed (0.15) keep their Phase 3 and
Phase 5 capacities, so ρ stays comparable with every earlier result. The report says those were
measured without the threshold.

## PR 4. `paper_01`

- [ ] `experiments/paper_01.yaml` from a generator in `scripts/matrix_suites.py`. Workloads are
  interactive, long-context-heavy, generation-heavy and mixed-bursty (the bursty arrival process
  over the mixed classes). Setups and ρ are as above.
- [ ] Order: one server for the whole matrix, and each (workload, ρ, seed) runs its three setups
  back to back in alternating order, as every paired suite has.
- [ ] Add the trace-replay points as a second suite in the same PR.
- [ ] Runner script with the usual guards (AC power, GPU apps, server up, flags in the log).
- [ ] Evidence: `llmserve suite experiments/paper_01.yaml --plan` lists 72 points. Then the Mac
  run, about 48 hours with the trace points.

## PR 5. Figures from `paper/figures.yaml`

- [ ] `paper/figures.yaml` maps each figure id to a suite, a metric and a plot type.
  `llmserve report` builds every figure from it.
- [ ] Figures: the H5 headline, gain against pressure for H6 (the heatmap from Appendix B of
  the implementation plan), per-class TTFT for each `paper_01` workload, and the trace replay.
- [ ] Evidence: deleting `docs/reports/figures/` and rerunning `llmserve report` recreates
  every file.

## Order

PR 1 and PR 2 are independent and come first. PR 3 needs PR 2 for the trace mix. PR 4 needs
PR 3's capacities. PR 5 needs PR 4's results for its last figures, but its code can land
earlier. The Mac is idle during PR 1 and PR 2, so PR 3's generation-heavy capacity run can
start as soon as its suite exists.

## Risks

- **The trace's ρ is approximate.** Its mix differs from the synthetic classes, so PR 3
  measures its capacity rather than borrowing mixed's.
- **The GPU clock.** Every comparison stays paired within a (workload, ρ, seed) block, and the
  report prints clocks next to TTFT.
- **Interruptions.** Power or another GPU app stops the runner between points, and it resumes
  on the remaining points. 54 hours will see several stops.
