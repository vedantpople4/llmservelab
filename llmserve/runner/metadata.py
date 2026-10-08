"""Run metadata: everything needed to reproduce or audit a benchmark (PRD §16, plan §4).

Every key is always present; a value that this host or backend cannot provide is `null`, never
omitted — "present but null" is what makes the exit check's field-by-field comparison possible.

Captured here: identity (run/config/git), the machine (host, platform, Python), the GPU (NVML,
best-effort), the resolved server/gateway/measurement settings, the seed's child keys, the clock
anchor (ADR-0006), which samplers were active (item 7 adds some; their absence is recorded), and
the warm-up's outcome. vLLM's launch args and the docker-side sizing flags are deployment facts
this process does not own, so they are null until the harness launches the server itself.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

from llmserve.config.schema import ExperimentConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]


def git_state(cwd: Path = _REPO_ROOT) -> tuple[str | None, bool]:
    """`(commit sha, dirty)` of the checkout containing this code.

    Outside a git checkout (an installed wheel) the sha is None and the tree counts as clean:
    there is nothing to compare against.
    """

    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    return sha, bool(porcelain)


def uv_lock_hash(cwd: Path = _REPO_ROOT) -> str | None:
    lock = cwd / "uv.lock"
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def gpu_info() -> dict[str, Any]:
    """GPU name/UUID/driver/CUDA from NVML; all-null when NVML or the GPU is absent."""
    info: dict[str, Any] = {"name": None, "uuid": None, "driver": None, "cuda": None}
    try:
        import pynvml
    except ImportError:
        return info
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["name"] = pynvml.nvmlDeviceGetName(handle).decode(errors="replace")
        info["uuid"] = pynvml.nvmlDeviceGetUUID(handle).decode(errors="replace")
        info["driver"] = pynvml.nvmlSystemGetDriverVersion().decode(errors="replace")
        cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        info["cuda"] = f"{cuda // 1000}.{(cuda % 1000) // 10}"
    except Exception:  # NVML errors mean "cannot tell", not "no GPU"
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return info


def build_metadata(
    *,
    cfg: ExperimentConfig,
    run_id: str,
    config_hash: str,
    endpoint: str,
    git_sha: str | None,
    git_dirty: bool,
    clock_anchor_ns: int,
    started_utc: str,
    finished_utc: str,
    warmup: dict[str, Any],
    backend_version: dict[str, Any],
) -> dict[str, Any]:
    """Assemble `metadata.json`. `endpoint` is the one actually driven (CLI `--endpoint`)."""
    return {
        "run_id": run_id,
        "experiment": cfg.experiment,
        "description": cfg.description,
        "schema_version": cfg.schema_version,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        # --- provenance ---
        "config_hash": config_hash,
        "uv_lock_hash": uv_lock_hash(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        # --- the seed and every stream it spawns (ADR-0004) ---
        "seed": cfg.seed,
        "repetitions": cfg.repetitions,
        "child_seed_keys": [[cfg.seed, rep] for rep in range(cfg.repetitions)],
        # --- the server under test ---
        "server": {
            "kind": cfg.server.kind,
            "endpoint": endpoint,
            "model": cfg.server.model,
            "model_revision": cfg.server.model_revision,
            "expected_version": cfg.server.expected_version,
            "prefix_caching": cfg.server.prefix_caching,
            "scheduling_policy": cfg.server.scheduling_policy,
        },
        "backend_version": backend_version,
        "tokenizer_revision": None,
        # Deployment-side vLLM facts this process does not own (docker/Kaggle flags); recorded
        # so the field exists and is filled when the harness launches the server itself.
        "vllm_launch_args": None,
        "max_model_len": None,
        "gpu_memory_utilization": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "gpu": gpu_info(),
        # --- how it was driven ---
        "load": {
            "mode": cfg.load.mode,
            "concurrency": cfg.load.concurrency,
            "requests": cfg.load.requests,
            "duration_s": cfg.load.duration_s,
            "arrival": cfg.load.arrival.model_dump() if cfg.load.arrival else None,
        },
        "gateway": {
            "enabled": cfg.gateway.enabled,
            "max_in_flight": cfg.gateway.max_in_flight,
            "scheduler": cfg.gateway.scheduler.name,
            "scheduler_params": cfg.gateway.scheduler.params,
            "output_estimator": cfg.gateway.output_estimator,
        },
        "measurement": {
            "warmup_requests": cfg.measurement.warmup.requests,
            "warmup_settle_timeout_s": cfg.measurement.warmup.settle_timeout_s,
            "warmup_s": cfg.measurement.warmup_s,
            "cooldown_s": cfg.measurement.cooldown_s,
            "request_timeout_s": cfg.measurement.request_timeout_s,
        },
        "warmup": warmup,
        "clock_anchor_ns": clock_anchor_ns,
        "monotonic_clock": "time.perf_counter_ns, relative to run start (ADR-0006)",
        "samplers": {
            "active": [],
            "gpu_hz": cfg.measurement.samplers.gpu_hz,
            "server_hz": cfg.measurement.samplers.server_hz,
            "dcgm": cfg.measurement.samplers.dcgm,
            "note": "gpu/server samplers land with Phase 2 item 7; none were active here",
        },
    }
