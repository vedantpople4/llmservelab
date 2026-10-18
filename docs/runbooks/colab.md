# Colab runbook: free T4 sessions on Google Colab

This is the Colab route to the same measurement setup as [the Kaggle runbook](kaggle.md):
**one Tesla T4, `Qwen/Qwen2.5-7B-Instruct-AWQ`, pip-installed vLLM 0.10.2, no Docker.** The
card, model, vLLM pin and §6 flags are identical to the Kaggle route (ADR-0011), so the
notebook cells differ only in paths, GPU selection and how results survive the session.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vedantpople4/llmservelab/blob/main/notebooks/colab.ipynb)

The badge opens [`notebooks/colab.ipynb`](../../notebooks/colab.ipynb), which holds the cells below. Keep the two in sync.

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

1. Open a new notebook at colab.research.google.com.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU.** Colab gives one T4, so
   `CUDA_VISIBLE_DEVICES` is not needed.
3. If the repo is private, create a GitHub personal access token with read access to it.
   A plain `git clone` of a private repo hangs on `Username for 'https://github.com':`.

## Notebook flow (five cells)

### Cell 1: environment

```python
!git clone https://github.com/vedantpople4/llmservelab.git /content/llmservelab
# private repo: !git clone https://<PAT>@github.com/vedantpople4/llmservelab.git /content/llmservelab
%cd /content/llmservelab
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q uv
!uv sync --locked --extra dev --extra mock --extra gpu
```

`nvidia-smi` must report `Tesla T4`. Anything else (an L4, an A100 on a paid tier) is a
different SKU, and its numbers cannot sit next to T4 numbers. Change the runtime type and
reconnect.

Use `%cd`, not `!cd`. Each `!` line runs in its own shell, so `!cd` changes nothing.

`uv sync --locked` installs into `.venv/`, isolated from Colab's preinstalled torch and
transformers. vLLM comes only from the locked `gpu` extra; a one-off `uv pip install vllm`
leaves it undeclared and `uv run vllm` fails with `Failed to spawn: vllm`.

### Cell 2: server (§6 flags)

```python
import os, subprocess, time, urllib.request

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
    stdout=open("/content/vllm.log", "w"),
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
)

for _ in range(180):  # first start downloads ~6 GB of weights
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        print("vLLM healthy")
        break
    except Exception:
        time.sleep(5)
else:
    print(open("/content/vllm.log").read()[-4000:])
    raise RuntimeError("vLLM did not become healthy")
```

The flags must match `docker/compose.yml` and the Kaggle runbook exactly; change all three
together or none.

### Cell 3: env check and smoke (Phase 1 exit check)

```python
!uv run python -m llmserve.runner.env_check configs/baseline/e01.yaml
!uv run python scripts/smoke.py configs/baseline/e01.yaml
```

`env_check` asserts health, the served model, prefix caching off (config plus an
identical-prompt TTFT probe) and GPU utilization under 5%. Run nothing else on the GPU while
the server is up. `smoke` exits 0 only with 100/100 `ok` and no usage mismatches.

### Cell 4: benchmark (Phase 2 exit check, GPU half)

```python
!uv run llmserve benchmark configs/baseline/e01.yaml
```

The benchmark refuses a dirty working tree. A fresh clone is clean (`.venv/` and `results/`
are gitignored). If you edit anything in the clone, commit and push it first; `--allow-dirty`
exists for debugging, and a dirty run is not a recorded result.

### Cell 5: save results

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/llmservelab
!cp -r results /content/drive/MyDrive/llmservelab/
!cp /content/vllm.log /content/drive/MyDrive/llmservelab/vllm-$(date +%Y%m%dT%H%M%S).log
```

Run this after every benchmark, not once at the end. `/content` is wiped on disconnect, and a
disconnect can come without warning.

## Session discipline

- **Colab does not publish its free GPU quota**, and it varies with demand. Disconnect the
  runtime (**Runtime → Disconnect and delete runtime**) whenever you are not measuring.
- **Idle sessions disconnect** after roughly 90 minutes without notebook interaction, and
  sessions cap at about 12 hours. Split long suites into runs that each finish, and save,
  well inside one session.
- **A new session starts from Cell 1.** The weights download again (~6 GB). Pointing
  `HF_HOME` at Drive avoids that, but reads from Drive are slower than local disk, so it
  costs startup time. Never let it overlap a measured run.

## Traps

| Trap | Consequence | Fix |
|---|---|---|
| Runtime left on CPU or a non-T4 GPU | vLLM fails, or numbers from the wrong SKU | Runtime → Change runtime type → T4 GPU; check `nvidia-smi` |
| `!cd` instead of `%cd` | later cells run in `/content`, configs not found | `%cd /content/llmservelab` |
| private repo | `git clone` hangs on a username prompt | clone with a PAT, or make the repo public |
| one-off `uv pip install vllm` | `Failed to spawn: vllm` | install only through `uv sync --locked --extra gpu` |
| other GPU work while the server runs | `env_check` fails the idle check; results polluted | one job at a time on the GPU |
| results only in `/content` | lost on disconnect | Cell 5 after every run |
| mixing Colab and Kaggle runs | host and driver become hidden variables | one platform per phase (see above) |
| idle runtime left connected | free quota drains | disconnect when not measuring |
