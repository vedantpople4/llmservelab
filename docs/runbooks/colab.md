# Colab runbook: free T4 sessions on Google Colab

This is the Colab route to the same measurement setup as [the Kaggle runbook](kaggle.md):
**one Tesla T4, `Qwen/Qwen2.5-7B-Instruct-AWQ`, pip-installed vLLM 0.10.2, no Docker.** The
card, model, vLLM pin and §6 flags are identical to the Kaggle route (ADR-0011), so the
notebook cells differ only in paths, GPU selection and how results survive the session.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vedantpople4/llmservelab/blob/main/notebooks/colab.ipynb)

The badge opens [`notebooks/colab.ipynb`](../../notebooks/colab.ipynb). The notebook is the
source of truth for the commands; this runbook explains them.

**Status:** authored, not yet verified end to end on Colab. The vLLM bring-up observations in
the Kaggle runbook (V1 engine on Turing, AWQ on sm_75, the bf16 → fp16 downgrade) were made on
Kaggle. Confirm them in `/content/vllm.log` on the first Colab run, then update this line.

## One platform per study

A Colab T4 and a Kaggle T4 are the same SKU, but the host CPU, NVIDIA driver and CUDA version
differ. Client lag depends on the host, and kernel behaviour can depend on the driver.
**Never mix Colab and Kaggle runs in one comparison, figure or cross-rep aggregate.** Pick one
platform for a whole phase. `metadata.json` records `platform`, `gpu.driver` and `gpu.cuda`,
so a mixed set can be spotted after the fact, but the rule is not to create one.

## One-time setup

1. **Runtime → Change runtime type → Hardware accelerator: T4 GPU.** Colab gives one T4, so
   `CUDA_VISIBLE_DEVICES` is not needed.
2. Optional: add an `HF_TOKEN` under Colab's **Secrets** panel (key icon) and grant the
   notebook access. Weight downloads get faster and Hugging Face stops warning about
   unauthenticated requests.

Then **Runtime → Run all.** The first code cell asks for Google Drive access; grant it and
the rest runs unattended.

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `QUICK` | `True` | `True`: 20 smoke requests and `configs/dev/e01_quick.yaml` (1 rep × 30 requests). Checks the pipeline in about 15 min. `False`: 100 smoke requests and `configs/baseline/e01.yaml` (5 reps × 200), the real exit checks. Only `QUICK = False` runs are recorded results. |
| `REPO_REF` | `main` | Branch to clone. Point it at a PR branch to test that branch before merge. |

## What each step does

Every step runs through `run(step, cmd)` or `with timed(step):`. Each prints a start and end
line with its wall time, appends a row to `timings.jsonl`, writes its output to
`logs/<step>.log`, and then copies everything to Drive. A failing step still records its
timing and saves before raising.

| Step | Work | Quick | Full |
|---|---|---|---|
| `1_setup` | clone, `nvidia-smi`, install uv, `uv sync --locked` with the `gpu` extra | 3 to 6 min | same |
| `2_server_start` | start vLLM with the §6 flags, wait for `/health` (first start downloads ~6 GB) | 4 to 10 min | same |
| `2b_vllm_log_check` | grep `vllm.log` for the engine (V1/V0) and dtype lines | instant | same |
| `3_env_check` | health, model, prefix caching off, GPU idle under 5% | ~30 s | same |
| `3b_smoke` | sequential requests at 128 in / 128 out; `--skip-env-check` since `3_env_check` just ran | ~1.5 min | 5 to 8 min |
| `4_benchmark` | `llmserve benchmark`: warm-up, reps, parquet, summaries, metadata | ~3 min | 50 to 80 min |

These times are estimates for a T4 at 25 to 40 ms per output token. `timings.jsonl` holds
the measured ones; update this table after the first verified run. The smoke step's summary
line gives wall time per 100 requests, and a full benchmark takes about 10× that.

`nvidia-smi` must say `Tesla T4`. Any other GPU (an L4, or an A100 on a paid tier) is a
different SKU, and its numbers cannot sit next to T4 numbers.

The vLLM flags must match `docker/compose.yml` and the Kaggle runbook exactly. Change all
three together or none.

## Phase 3 suites

Step 5b runs a suite from `configs/suites/` and writes to `MyDrive/llmservelab/results/`, a
folder shared by all sessions, so an interrupted suite resumes in the next session. The order
and session plan are in [the Phase 3 runbook](phase3.md). The server cell passes
`--revision MODEL_REVISION`, which must equal `server.model_revision` in the configs.

## Where output goes

```
MyDrive/llmservelab/sessions/<UTC timestamp>/
  logs/timings.jsonl      one row per step: step, started_utc, elapsed_s, status, quick, cmd
  logs/<step>.log         full stdout and stderr of that step
  vllm.log                the server log
  results/<exp>/<run_id>/ the benchmark run directory (parquet, summaries, metadata.json)
```

Steps write to local disk (`/content/logs`, `/content/vllm.log`, `results/`) and copy to
Drive only between steps, so Drive I/O never overlaps a measurement. A disconnect loses at
most the step that was running.

## Warnings

The setup cell sets `PYTHONWARNINGS=ignore::DeprecationWarning,ignore::FutureWarning`,
`HF_HUB_DISABLE_PROGRESS_BARS=1`, `TOKENIZERS_PARALLELISM=false` and `UV_NO_PROGRESS=1`.
These hide library noise only.

Never hide these. They describe the measurement:

- vLLM's engine line (`V1 engine` or `Falling back to V0`) and the bf16 → fp16 downgrade.
  Keep `VLLM_LOGGING_LEVEL` at its default; the log check needs the INFO lines.
- Harness output: `env_check` failures, `INVALID: >1% failures`, `sampler ... stopped`.
- `Error in sitecustomize`. It means the `.venv` imports something from the platform's
  Python. Report it instead of hiding it.

## Session discipline

- **Colab does not publish its free GPU quota**, and it varies with demand. Disconnect the
  runtime (**Runtime → Disconnect and delete runtime**) whenever you are not measuring.
- **Keep the tab open during long runs.** A running cell usually keeps the session alive,
  but a closed tab or sleeping laptop can drop it, and sessions cap at about 12 hours.
- **A new session starts from setup.** `1_setup` and `2_server_start` cost about 10 minutes
  of GPU time each session, so fit several runs into one session where you can.

## Traps

| Trap | Consequence | Fix |
|---|---|---|
| runtime left on CPU or a non-T4 GPU | vLLM fails, or numbers from the wrong SKU | Runtime → Change runtime type → T4 GPU; check `1_setup.log` |
| `REPO_REF` names a branch without `configs/dev/e01_quick.yaml` | `4_benchmark` fails: config not found | use `main` after this notebook's PR merges, or the PR branch before |
| editing files in the clone | `llmserve benchmark` refuses a dirty tree | commit and push instead; `--allow-dirty` runs are not recorded results |
| one-off `uv pip install vllm` | `Failed to spawn: vllm` | install only through `uv sync --locked --extra gpu` |
| other GPU work while the server runs | `env_check` fails the idle check; results polluted | one job at a time on the GPU |
| `QUICK = True` results in a report | 1 rep × 30 requests is not a measurement | report only `QUICK = False` runs |
| mixing Colab and Kaggle runs | host and driver become hidden variables | one platform per phase (see above) |
| idle runtime left connected | free quota drains | disconnect when not measuring |
