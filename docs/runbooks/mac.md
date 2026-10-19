# Mac runbook: vllm-metal for scheduler development (ADR-0012)

Upstream vLLM on Apple Silicon through the vllm-metal plugin, serving the same
`Qwen/Qwen2.5-7B-Instruct-AWQ` checkpoint as the T4. Use it to build and debug schedulers
and the gateway. Results are development results: never compared with T4 numbers, never in
reports.

**Verified** on an M5 MacBook, 24 GB, macOS 27, vLLM 0.31.0 + vllm-metal 0.31.0 (October 2026).

## Install (once)

Into its own virtualenv, separate from the project's `.venv`:

```bash
uv venv --python 3.12 ~/.venv-vllm-metal
VIRTUAL_ENV=~/.venv-vllm-metal uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl" \
  "https://github.com/vllm-project/vllm-metal/releases/download/v0.31.0/vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl"
```

The two versions move together: each vllm-metal release names its vLLM release in
`.github/vllm-release-tag.commit`. When upgrading, update both URLs and
`server.expected_version` in `configs/dev/e01_metal*.yaml`.

Homebrew also works (`brew install vllm-project/vllm-metal/vllm-metal`), but it refuses to
install with outdated Xcode Command Line Tools.

## Serve

```bash
~/.venv-vllm-metal/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 127.0.0.1 --port 8000 --max-model-len 10240 --dtype float16 \
  --no-enable-prefix-caching --max-num-seqs 256 --gpu-memory-utilization 0.5
```

The first start downloads ~5.5 GB of weights (about 3 minutes). Later starts take ~15 s.
Check the log for `KV cache size: 52,784 tokens` and for the absence of `KV commit probe: the
pool ... is larger than this machine's free memory`. That warning means the KV pool will
page into swap. Close apps or lower `--gpu-memory-utilization`.

## Run

```bash
uv run python -m llmserve.runner.env_check configs/dev/e01_metal.yaml
uv run python scripts/smoke.py configs/dev/e01_metal.yaml --requests 20 --skip-env-check
uv run llmserve benchmark configs/dev/e01_metal.yaml       # concurrency 1, 30 requests
uv run llmserve benchmark configs/dev/e01_metal_c8.yaml    # concurrency 8, 64 requests
```

First observed numbers (one rep each, for orientation only):

| Config | TTFT p50 | TPOT p50 | Generation | Server `running` max |
|---|---|---|---|---|
| `e01_metal` (c = 1) | 231 ms | 47 ms | 20.6 tok/s | 1 |
| `e01_metal_c8` (c = 8) | 1.88 s | 139 ms | 54.3 tok/s | 8 |

TTFT at c = 8 is much higher than prefill cost alone explains. Investigate it before relying
on Mac TTFT for scheduler decisions.

## Differences from the T4 route

| | T4 (vLLM 0.10.2, CUDA) | Mac (vLLM 0.31.0, vllm-metal) |
|---|---|---|
| Reportable | yes (`kind: vllm`) | no (`kind: vllm-metal`) |
| `min_tokens` | sent | rejected by the server; the client omits it (`min_tokens_param`) |
| GPU sampler | NVML at 10 Hz | none |
| KV pool | ≈ 130–140k tokens | ≈ 53k tokens at `--gpu-memory-utilization 0.5` |
| Interference | dedicated card | shared unified memory; close heavy apps, stay on power |

The server also logs that the model's `generation_config.json` overrides default sampling
(temperature 0.7, repetition penalty 1.05). Output length is forced with `ignore_eos`, so this
changes the generated text, not the number of tokens.

## Stop

```bash
pkill -f "vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ"
```

The server holds about 8 to 9 GB of unified memory (weights plus KV pool) while it runs.
