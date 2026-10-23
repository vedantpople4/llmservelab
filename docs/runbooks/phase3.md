# Phase 3 runbook: baseline characterization on the Mac

How to produce the Baseline Performance Report: which suites and benchmarks to run, in what
order, across overnight sessions, and how to build the report. The platform is the M5 MacBook
with vllm-metal (ADR-0013); setup and run hygiene are in [mac.md](mac.md). Only `kind:
vllm-metal` runs from a clean tree, with ≤ 1% failures, no thermal drift above 10%, no sleep and
no battery power are reportable.

## Before the first session

- **Model pinned.** Configs pin `server.model_revision` to the HF commit
  `b25037543e9394b818fdfca67ab2a00ecc7dd641`, and the serve command passes the same value to
  `--revision`. Change both together or neither.
- **Plan a suite before running it.** `uv run llmserve suite <file> --plan` lists its points
  and a rough duration (Mac model refit from probes, within about 2×).
- **Commit first.** Results go to `results/` (git-ignored); the tree they ran from must be clean.

## How a suite runs

A suite skips every point that already has a valid run with an identical config, so after an
interruption, rerun the same command and it continues from the first unfinished point. A point
that was running when it stopped starts over; reps inside a point are not resumable. `--only
LABEL ...` runs a subset to fit a session.

## Order

Capacity comes first because H2 needs it.

| # | What | Command | Estimate |
|---|---|---|---|
| 1 | Capacity, 128/128 | `llmserve suite configs/suites/capacity_p128.yaml` | ~55 min |
| 2 | Capacity, mixed | `llmserve suite configs/suites/capacity_mixed.yaml` | ~1.5 h |
| 3 | Capacity, interactive only | `llmserve suite configs/suites/capacity_interactive.yaml` | ~55 min |
| 4 | Prompt length (H3) | `llmserve suite configs/suites/prompt_length.yaml` | ~1 h |
| 5 | Output length | `llmserve suite configs/suites/output_length.yaml` | ~1.3 h |
| 6 | Bursty | `llmserve suite configs/suites/bursty.yaml` | ~20 min |
| 7 | H2, mixed | `llmserve suite configs/suites/h2_mixed.yaml --set load.capacity_rps=<ρ=1 of #2>` | ~1.5 h |
| 8 | H2, interactive | `llmserve suite configs/suites/h2_interactive.yaml --set load.capacity_rps=<ρ=1 of #3>` | ~1.5 h |
| 9 | Concurrency, 128/128 (H1) | `llmserve suite configs/suites/concurrency_p128.yaml` | ~3.9 h |
| 10 | Concurrency, mixed | `llmserve suite configs/suites/concurrency_mixed.yaml` | ~3.4 h |
| 11 | E01–E07 | `llmserve benchmark configs/baseline/e0N.yaml` for N = 1..7 | ~9.2 h total |

Read ρ = 1 for steps 7 and 8 from the report's capacity table: run `llmserve report results -o
docs/reports` after step 3. If the table says the sweep did not bracket capacity, widen the
rates in the capacity suite and rerun it before going on.

Run a session as one command under `caffeinate -i`, so the Mac cannot idle-sleep between
suites. Session 1, for example:

```bash
caffeinate -i sh -c 'for s in capacity_p128 capacity_mixed capacity_interactive \
    prompt_length output_length bursty; do
  uv run llmserve suite configs/suites/$s.yaml
done'
```

Suggested overnight sessions of about 8 hours:

1. Steps 1–6 (~5.9 h).
2. Steps 7–9 (~6.9 h).
3. Step 10 and E01–E03 (~5.8 h).
4. E04–E07 (~6.7 h).

## Build the report

```bash
uv run llmserve report results -o docs/reports
```

This writes `docs/reports/baseline.md`, `docs/reports/figures/*.png` and
`docs/reports/cost_model.json`. Runs that are not reportable are excluded and listed with the
reason. Sections for anything not yet run show the command that produces them. The Phase 3
exit check is met when E01–E07 show 5 reps with "Success ≥ 99%: yes", H1 and H2 have a verdict
other than not-run, and capacity is bracketed for every workload.
