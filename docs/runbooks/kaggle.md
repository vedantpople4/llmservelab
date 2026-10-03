# Kaggle runbook: free T4 sessions for real measurements (ADR-0011)

The whole study runs here: **one Tesla T4, `Qwen/Qwen2.5-7B-Instruct-AWQ`, pip-installed
vLLM, no Docker, no money.** Kaggle gives 30 GPU-hours per rolling week to a phone-verified
account. This notebook flow replaces `docker/compose.yml` for the Kaggle route; both must use
the §6 flags so the routes stay interchangeable.

## One-time setup

1. Verify a phone number (Kaggle's anti-abuse gate for accelerators) under **Settings →
   Phone Verification**.
2. Create a notebook at **kaggle.com/code → + New Notebook**.
3. **Settings → Accelerator → GPU T4 x2.** The account may offer *GPU P100* as the default —
   **never use it**: P100 is sm_60, and vLLM's kernels need sm_70+. If the session lands on
   P100 (`nvidia-smi` says "Tesla P100"), stop it and re-select T4 ×2.
4. **Settings → Internet → On** for setup (git clone, pip, model download). Keep it on for the
   first check cell too — the prompt tokenizer is fetched from HF once and cached; later
   measurement sessions can run with internet off if you want zero network during runs.

### Driving Kaggle from the CLI (optional)

The CLI does not persist a UI-selected GPU; the metadata file does. In `kernel-metadata.json`:

```json
{
  "id": "<kaggle-user>/llmservelab-phase1",
  "title": "llmservelab phase 1",
  "code_file": "phase1.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "isInternetEnabled": true
}
```

`machine_shape: "NvidiaTeslaT4"` is what selects T4 ×2 — the `--accelerator` flag alone is
not honored on push.

## Notebook flow (four cells)

### Cell 1 — environment (internet on)

```python
!git clone https://github.com/vedantpople4/llmservelab.git /kaggle/working/llmservelab
%cd /kaggle/working/llmservelab
!pip install -q uv
# vllm==0.10.2 comes from the locked `gpu` extra (pin matches docker/compose.yml's VLLM_IMAGE).
# Do NOT `uv pip install vllm` instead: `uv run` exact-syncs the env and deletes undeclared
# packages, which surfaces as "Failed to spawn: vllm" in Cell 2.
!uv sync --locked --extra dev --extra mock --extra gpu
```

Verify the SKU before anything else (only GPU 0 will be used):

```python
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Expect `0, Tesla T4, 16269 MiB` and `1, Tesla T4, 16269 MiB`. A P100 here means the session
has the wrong accelerator — stop and re-create it.

### Cell 2 — server (§6 flags)

```python
import subprocess, time, urllib.request

env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"}
server = subprocess.Popen(
    [
        "uv",
        "run",
        "vllm",
        "serve",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "10240",
        "--gpu-memory-utilization",
        "0.90",
        "--dtype",
        "float16",
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        "256",
        "--disable-log-requests",
    ],
    stdout=open("/kaggle/working/vllm.log", "w"),
    stderr=subprocess.STDOUT,
    env=env,
)

for _ in range(180):  # first start downloads ~6 GB of weights
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        print("vLLM healthy")
        break
    except Exception:
        time.sleep(5)
else:
    print(open("/kaggle/working/vllm.log").read()[-4000:])
    raise RuntimeError("vLLM did not become healthy")
```

vLLM is pinned to the same version as `docker/compose.yml`'s `VLLM_IMAGE`. Two log lines
look alarming but are **expected** on this SKU:

- `Compute Capability < 8.0 is not supported by the V1 Engine. Falling back to V0` — the
  pinned vLLM serves Turing on its V0 engine; the T4 numbers in this study are V0-engine
  numbers, and that is part of the SKU's identity.
- a bf16 → fp16 downgrade warning — Qwen2.5's config is bf16 and Turing has no bf16 silicon;
  Cell 2 pins `--dtype float16` so the downgrade is explicit rather than incidental.

**Phase 1's first task is to verify AWQ kernels actually run on sm_75.** vLLM documents
compute capability ≥ 7.5 (T4 named) and AWQ as Turing-supported, so this is a bring-up
check, not a known-broken path — if it still fails, switch to
`Qwen/Qwen2.5-7B-Instruct-GPTQ-INT4` (same card, same precision class; ADR-0011).

### Cell 3 — env check + smoke (the Phase 1 exit check)

```python
!uv run python -m llmserve.runner.env_check configs/baseline/e01.yaml
!uv run python scripts/smoke.py configs/baseline/e01.yaml
```

`env_check` asserts health, served model, `prefix_caching: false` plus the identical-prompt
TTFT probe, and **GPU idle < 5%** — so don't run anything else in the notebook while the
server is up (vLLM loaded but not serving shows ~0% util, which passes). `smoke` exits 0 only
with 100/100 `ok` and no usage mismatches: that is the Phase 1 exit check.

### Cell 4 — artifacts

Notebook **Commit ("Save & Run All")** preserves the outputs as the run record. Copy anything
beyond stdout (e.g. `results/`) into `/kaggle/datasets/…` via `kaggle datasets push`, or
attach it to the commit, before the session ends — `/kaggle/working` does not persist across
sessions.

## Quota discipline

- **30 GPU-hours per rolling week**, shared across T4 and P100 sessions. Quota ticks while a
  GPU session is alive: **stop the session when you are not measuring**.
- Sessions cap at ~9–12 hours; a run that outlives its session resumes from Cell 1 in a new
  notebook (model cache is gone — re-download or keep it as a Kaggle dataset).
- Phase budget (plan §6): Phases 1–2 ≈ 5 h, Phase 3 ≈ 15 h, Phases 4–7 ≈ 55–80 h — about
  3–5 quota weeks for everything.

## Traps (all verified the hard way)

| Trap | Consequence | Fix |
|---|---|---|
| P100 default session | vLLM kernels fail (sm_60) | Settings → T4 ×2, or `machine_shape` in metadata |
| `kaggle kernels push --accelerator` | silently lands on P100 | `machine_shape: "NvidiaTeslaT4"` |
| Docker habit | Kaggle has no Docker | this runbook's pip path; compose is for GPU hosts |
| AWQ kernels on Turing | server fails at model load | documented as supported (CC ≥ 7.5, AWQ ✅ Turing); verify at Cell 2, fallback GPTQ-INT4 (ADR-0011) |
| `--dtype auto` | some vLLM versions hard-error on a bf16 config with T4 | Cell 2 pins `--dtype float16` |
| V0-engine fallback warning | looks like a broken install | expected on the pinned vLLM; record it, T4 numbers are V0-engine numbers |
| `uv pip install vllm` + `uv run` | `Failed to spawn: vllm` — `uv run` exact-syncs and deletes undeclared packages | install vllm via `uv sync --extra gpu` (it is a locked dependency) |
| Model name mismatch | env check: `model mismatch` | configs pin `Qwen/Qwen2.5-7B-Instruct-AWQ`, and so does `vllm serve` |
| Idle GPU session | quota drains for free | stop the session when idle |
