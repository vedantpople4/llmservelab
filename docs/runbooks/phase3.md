# Phase 3 runbook: baseline characterization on the T4

How to produce the Baseline Performance Report: which suites and benchmarks to run, in what
order, across Colab sessions, and how to build the report. The route is the Colab notebook
(`notebooks/colab.ipynb`, `docs/runbooks/colab.md`) with `QUICK = False`. Only `kind: vllm`
runs from a clean tree with ≤ 1% failures are reportable.

## Before the first session

- **Model pinned.** Configs pin `server.model_revision` to the HF commit
  `b25037543e9394b818fdfca67ab2a00ecc7dd641`, and the notebook passes the same value to
  `vllm serve --revision`. Change both together or neither.
- **Plan a suite before running it.** `uv run llmserve suite <file> --plan` lists its points
  and a rough duration (T4 envelope, within about 2×).
- **Check a suite on the Mac first.** The dev suites (`configs/suites/dev/`) and
  `llmserve report --dev` exercise the whole path against vllm-metal (`docs/runbooks/mac.md`).

## How a suite runs on Colab

Set `SUITE` (and optionally `SUITE_ARGS`) in the notebook's step 5b. Results go to
`MyDrive/llmservelab/results/`, shared by every session. A suite skips every point that already
has a valid run with an identical config, so after a disconnect, rerun the same cell in a new
session and it continues from the first unfinished point. A point that was running when the
session dropped starts over; reps inside a point are not resumable. `--only LABEL ...` runs a
subset to fit a session.

## Order

Capacity comes first because H2 needs it. Each row is one notebook run of step 5b, or step 4
(`BENCHMARK_CONFIG`) for E01–E07.

| # | What | Command | Estimate |
|---|---|---|---|
| 1 | Capacity, 128/128 | `llmserve suite configs/suites/capacity_p128.yaml` | ~22 min |
| 2 | Capacity, mixed | `llmserve suite configs/suites/capacity_mixed.yaml` | ~28 min |
| 3 | Capacity, interactive only | `llmserve suite configs/suites/capacity_interactive.yaml` | ~27 min |
| 4 | Prompt length (H3) | `llmserve suite configs/suites/prompt_length.yaml` | ~49 min |
| 5 | Output length | `llmserve suite configs/suites/output_length.yaml` | ~37 min |
| 6 | Bursty | `llmserve suite configs/suites/bursty.yaml` | ~29 min |
| 7 | H2, mixed | `llmserve suite configs/suites/h2_mixed.yaml --set load.capacity_rps=<ρ=1 of #2>` | ~31 min |
| 8 | H2, interactive | `llmserve suite configs/suites/h2_interactive.yaml --set load.capacity_rps=<ρ=1 of #3>` | ~31 min |
| 9 | Concurrency, 128/128 (H1) | `llmserve suite configs/suites/concurrency_p128.yaml` | ~2.2 h |
| 10 | Concurrency, mixed | `llmserve suite configs/suites/concurrency_mixed.yaml` | ~3.5 h |
| 11 | E01–E07 | `llmserve benchmark configs/baseline/e0N.yaml` for N = 1..7 | ~11 h total |

Read ρ = 1 for steps 7 and 8 from the report's capacity table: run
`llmserve report <drive results> -o <drive results>/report` after step 3 (step 5b does this
automatically). If the table says the sweep did not bracket capacity, widen the rates in the
capacity suite and rerun it before going on.

Suggested sessions, each well under Colab's ~12 h cap with ~10 min of setup:

1. Steps 1–6 (~3.2 h).
2. Steps 7–9 (~3.2 h).
3. Step 10 (~3.5 h), split with `--only` if a session runs short.
4. Onward: E01–E07 (one or two benchmarks per session).

## Open decision: the E01–E07 budget

E01–E07 as committed run 1000–2000 requests × 5 reps each. The estimates are E01 49 min, E02
38, E03 16, E04 73, E05 193, E06 170, E07 131: about 11 h. With the suites (~10 h) Phase 3
needs about 21 GPU-hours, above the plan's ~15 h. E05 and E06 alone are ~3 h each and are not
resumable mid-run. Options:

- Run them as committed and accept ~3 quota weeks for Phase 3.
- Lower `load.requests` in E04–E07 (e.g. 300) before running anything; the steady window stays
  large at C = 8–32 and each benchmark fits a session comfortably.
- Drop E01: its config matches `concurrency-p128` point `c001` apart from the experiment name.

Decide before step 11; changing a config after it has run means rerunning it.

## Build the report

Copy `MyDrive/llmservelab/results/` to the repo's `results/` (or point the command at it), then:

```bash
uv run llmserve report results -o docs/reports
```

This writes `docs/reports/baseline.md`, `docs/reports/figures/*.png` and
`docs/reports/cost_model.json`. Runs that are not reportable are excluded and listed with the
reason. Sections for anything not yet run show the command that produces them. The Phase 3
exit check is met when E01–E07 show 5 reps with "Success ≥ 99%: yes", H1 and H2 have a verdict
other than not-run, and capacity is bracketed for every workload.
