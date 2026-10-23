# Mac runbook: the measurement platform (ADR-0013)

Upstream vLLM on Apple Silicon through the vllm-metal plugin, serving
`Qwen/Qwen2.5-7B-Instruct-AWQ` at a pinned revision. Every reported number comes from this
setup on one machine (ADR-0013). The Phase 3 run order is in [phase3.md](phase3.md).

**Verified** on an M5 MacBook Air, 24 GB, macOS 27, vLLM 0.31.0 + vllm-metal 0.31.0
(October 2026).

## Install (once)

Into its own virtualenv, separate from the project's `.venv`:

```bash
uv venv --python 3.12 ~/.venv-vllm-metal
VIRTUAL_ENV=~/.venv-vllm-metal uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl" \
  "https://github.com/vllm-project/vllm-metal/releases/download/v0.31.0/vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl"
```

The two versions move together: each vllm-metal release names its vLLM release in
`.github/vllm-release-tag.commit`. An upgrade changes the platform. Update both URLs and
`server.expected_version` in `configs/baseline/e0*.yaml` and `configs/dev/e01_metal*.yaml`,
then rerun everything reported; results do not carry across versions.

Homebrew also works (`brew install vllm-project/vllm-metal/vllm-metal`), but it refuses to
install with outdated Xcode Command Line Tools.

## Serve

```bash
~/.venv-vllm-metal/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
  --host 127.0.0.1 --port 8000 --max-model-len 10240 --dtype float16 \
  --no-enable-prefix-caching --max-num-seqs 256 --gpu-memory-utilization 0.5
```

The revision matches `server.model_revision` in the configs. Change both together or neither.
The first start downloads ~5.5 GB of weights (about 3 minutes). Later starts take ~15 s.
Check the log for `KV cache size: 52,784 tokens` and for the absence of `KV commit probe: the
pool ... is larger than this machine's free memory`. That warning means the KV pool will
page into swap. Close apps or lower `--gpu-memory-utilization`.

## Before a measured run

A run is unreportable if the Mac drifts thermally, sleeps, or runs on battery (ADR-0013), so:

- Plug in the charger and keep the lid open. A closed lid sleeps the Mac even mid-run, and
  `caffeinate` (which runs hold automatically) only prevents idle sleep.
- Quit heavy apps. The server shares unified memory with everything else, and background load
  shows up as latency.
- Commit first. Runs from a dirty tree are unreportable.
- Check the server: `uv run python -m llmserve.runner.env_check configs/baseline/e01.yaml`.

## Run

```bash
uv run llmserve benchmark configs/baseline/e01.yaml   # one experiment
uv run llmserve suite configs/suites/capacity_p128.yaml --plan   # points and a duration estimate
uv run llmserve suite configs/suites/capacity_p128.yaml          # resumable sweep
uv run llmserve analyze results/e01/<run>                         # one run's summary
uv run llmserve report results -o docs/reports                    # the Baseline report
```

Quick checks that are never reported: `configs/dev/e01_metal.yaml` (C = 1, 30 requests),
`configs/dev/e01_metal_c8.yaml` (C = 8) and the dev suites in `configs/suites/dev/`.

## What the Mac measures

Closed-loop probes (one rep each, October 2026; full table in ADR-0013):

| Load | TTFT p50 | TPOT p50 | Throughput |
|---|---|---|---|
| 128/128, C = 1 | 0.23 s | 47 ms | 20 gen tok/s |
| 128/128, C = 64 | 9.6 s | 537 ms | 102 gen tok/s |
| 8192-token prompt, C = 1 | 12.8 s | 58 ms | 600 prompt tok/s |
| mixed workload, C = 16 | 7.2 s | 267 ms | 0.22 req/s |

TTFT at C ≥ 8 is much higher than prefill cost alone explains. Phase 4 investigates it.

Platform notes:

- `min_tokens` is rejected by the server, so the client omits it (`min_tokens_param`); output
  length is forced with `ignore_eos` + `max_tokens`.
- There is no NVML. `/metrics` gives running and waiting counts, KV-cache use and preemptions.
- The KV pool is ≈ 53k tokens at `--gpu-memory-utilization 0.5`. Higher values page into swap.
- The model's `generation_config.json` overrides default sampling (temperature 0.7, repetition
  penalty 1.05). Output length is forced, so this changes the text, not the token count.

## Stop

```bash
pkill -f "vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ"
```

The server holds about 8 to 9 GB of unified memory (weights plus KV pool) while it runs.
