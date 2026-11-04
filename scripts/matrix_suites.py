"""Write the Phase 5 comparison-matrix suites (plan Phase 5 item 5, resized for the Mac).

    uv run python scripts/matrix_suites.py --max-in-flight 16

The plan's {FIFO, SPF, SJF, Priority} × 3 workloads × ρ {0.5, 0.8, 0.95} × 5 reps is ~90 h on the
Mac. The resized matrix keeps the full ρ axis on the mixed workload, where H4 lives, and runs the
other two workloads at ρ 0.8 only:

- `matrix-mixed`: 4 policies × ρ {0.5, 0.8, 0.95} × 3 seeds = 36 runs.
- `matrix-interactive`, `matrix-long-context-heavy`: 4 policies × ρ 0.8 × 2 seeds = 8 runs each.
- `xcheck-vllm-priority` (plan Phase 5 item 6): the mixed points at ρ {0.8, 0.95} with no
  gateway, against a vLLM started with `--scheduling-policy priority` and each request carrying
  its class priority. Each point pairs with `matrix-mixed`'s gateway Priority point at the same
  ρ and seed. 2 × 3 = 6 runs.
- `prefill-threshold` (ADR-0014): FIFO on the mixed workload against a vLLM started with
  `--long-prefill-token-threshold` 0 (unset), 256 and 1024, at ρ {0.8, 0.95}, 2 seeds = 12 runs.
  Points are grouped by threshold, so the server restarts twice; run each group with `--only`.
- `threshold-paired`: `prefill-threshold`'s unset and 256 thresholds at ρ {0.8, 0.95} on 3
  seeds = 12 runs. `prefill-threshold` ran each threshold as one block, and the unset block's
  GPU clock averaged ~8% below the others'. Here each seed runs both thresholds back to back,
  in reversed order on alternate seeds. Restart the server before each (seed, threshold) block
  and run its two points with `--only`.
- `waas-tuning` (ADR-0014): WAAS's `cost_weight` × `max_bypass` grid and FIFO on the mixed
  workload at ρ 0.95, 2 tuning seeds = 18 runs, against a vLLM started with
  `--long-prefill-token-threshold 256`, the threshold `prefill-threshold` chose. Schedulers see
  the class-prior output estimate (`configs/output_priors/mixed.json`), not the oracle.
  Superseded by `waas-paired`: the GPU clock varied ~20% between its runs, which moved the
  tail more than the settings did.
- `waas-paired`: FIFO and WAAS at cost_weight {0.5, 1, 2} (max_bypass 32) under the same
  conditions, 3 tuning seeds = 12 runs. Each seed runs its policies back to back, in reversed
  order on alternate seeds, so slow drift in the machine's state falls evenly on every policy.

Every point is one repetition. Points are ordered seed, then ρ, then policy, so the policies of
one cell run back to back on the same machine state, and a session that stops early leaves
whole seeds. Policies at one seed see identical requests: a rep's workload is drawn from the
config seed alone. The in-flight cap comes from the `inflight-cap` suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

POLICIES = ("fifo", "spf", "sjf", "priority")
SEED = 42
SUITES = Path(__file__).resolve().parent.parent / "configs" / "suites"

# The classes of configs/workloads/mixed.yaml, verbatim apart from weights.
CLASSES: dict[str, dict[str, Any]] = {
    "interactive": {
        "priority": "high",
        "slo": {"ttft_ms": 1500, "tpot_ms": 200},
        "prompt": {"distribution": "uniform", "min": 128, "max": 512},
        "output": {"distribution": "fixed", "tokens": 128},
    },
    "long_context": {
        "priority": "medium",
        "slo": {"ttft_ms": 9000, "tpot_ms": 200},
        "prompt": {
            "distribution": "choice",
            "values": [2048, 4096, 8192],
            "weights": [0.5, 0.3, 0.2],
        },
        "output": {"distribution": "fixed", "tokens": 128},
    },
    "generation_heavy": {
        "priority": "low",
        "slo": {"ttft_ms": 6000, "tpot_ms": 300},
        "prompt": {"distribution": "fixed", "tokens": 256},
        "output": {"distribution": "choice", "values": [512, 1024]},
    },
}


def classes(weights: dict[str, float]) -> list[dict[str, Any]]:
    return [{"name": name, "weight": w, **CLASSES[name]} for name, w in weights.items()]


# (suite, description of the workload, class weights, capacity in req/s, ρ values, seeds)
MATRICES: list[tuple[str, str, dict[str, float], float, list[float], int]] = [
    (
        "matrix-mixed",
        "E07's mixed workload (capacity 0.15 req/s, capacity-mixed-long)",
        {"interactive": 0.60, "long_context": 0.25, "generation_heavy": 0.15},
        0.15,
        [0.5, 0.8, 0.95],
        3,
    ),
    (
        "matrix-interactive",
        "the interactive class alone (capacity 0.4 req/s, capacity-interactive)",
        {"interactive": 1.0},
        0.4,
        [0.8],
        2,
    ),
    (
        "matrix-long-context-heavy",
        "the mixed classes with long_context at half the arrivals "
        "(capacity 0.1 req/s, capacity-long-context-heavy)",
        {"interactive": 0.35, "long_context": 0.50, "generation_heavy": 0.15},
        0.1,
        [0.8],
        2,
    ),
]


def suite(
    name: str,
    workload: str,
    weights: dict[str, float],
    capacity: float,
    rhos: list[float],
    seeds: int,
    max_in_flight: int,
) -> dict[str, Any]:
    load: dict[str, Any] = {
        "mode": "open",
        "arrival": {"process": "poisson"},
        "duration_s": 1200,
        "capacity_rps": capacity,
    }
    points = [
        {
            "label": f"{policy}-rho{round(rho * 100):03d}-s{k}",
            "set": {
                "seed": SEED + k,
                "gateway.scheduler.name": policy,
                "load.arrival.rho": rho,
            },
        }
        for k in range(seeds)
        for rho in rhos
        for policy in POLICIES
    ]
    return {
        "suite": name,
        "description": (
            f"Phase 5 comparison matrix on {workload}: {', '.join(POLICIES)} behind the gateway "
            f"(max_in_flight {max_in_flight}) at rho {', '.join(f'{r:g}' for r in rhos)}, "
            f"{seeds} seeds, Poisson for 1200 s (warm-up 120 s, cool-down 60 s trimmed). "
            "Generated by scripts/matrix_suites.py; edit that, not this file."
        ),
        "base": "../baseline/e07.yaml",
        "set": {
            "repetitions": 1,
            "gateway": {"enabled": True, "max_in_flight": max_in_flight},
            "load": load,
            "workload": {"classes": classes(weights)},
            "measurement.warmup_s": 120,
            "measurement.cooldown_s": 60,
            "measurement.request_timeout_s": 1800,
        },
        "points": points,
    }


def xcheck() -> dict[str, Any]:
    mixed = suite(*MATRICES[0][:4], [0.8, 0.95], 3, max_in_flight=1)
    mixed["set"]["gateway"] = {"enabled": False}
    mixed["set"]["server.scheduling_policy"] = "priority"
    points = [p for p in mixed["points"] if p["set"]["gateway.scheduler.name"] == "priority"]
    for p in points:
        del p["set"]["gateway.scheduler.name"]
        p["label"] = p["label"].replace("priority-", "vllm-priority-")
    return {
        **mixed,
        "suite": "xcheck-vllm-priority",
        "description": (
            "Phase 5 cross-check: E07's mixed workload with no gateway against vLLM's native "
            "priority scheduling (serve with --scheduling-policy priority), at rho 0.8 and 0.95, "
            "3 seeds; pairs with matrix-mixed's priority points. Generated by "
            "scripts/matrix_suites.py; edit that, not this file."
        ),
        "points": points,
    }


THRESHOLDS = (0, 256, 1024)
PAIRED_SEEDS = 3


def prefill_threshold(max_in_flight: int) -> dict[str, Any]:
    mixed = suite(*MATRICES[0][:4], [0.8, 0.95], 2, max_in_flight)
    mixed["set"]["gateway.scheduler.name"] = "fifo"
    fifo = [p for p in mixed["points"] if p["set"].pop("gateway.scheduler.name") == "fifo"]
    points = [
        {
            "label": f"t{t:04d}-{p['label'].removeprefix('fifo-')}",
            "set": {**p["set"], "server.long_prefill_token_threshold": t},
        }
        for t in THRESHOLDS
        for p in fifo
    ]
    return {
        **mixed,
        "suite": "prefill-threshold",
        "description": (
            "Phase 6 server experiment (ADR-0014): E07's mixed workload under FIFO behind the "
            f"gateway (max_in_flight {max_in_flight}) against a vLLM started with "
            f"--long-prefill-token-threshold {', '.join(map(str, THRESHOLDS))} (0 = unset), at "
            "rho 0.8 and 0.95, 2 seeds. Restart the server for each threshold and run its points "
            "with --only. Generated by scripts/matrix_suites.py; edit that, not this file."
        ),
        "points": points,
    }


PAIRED_THRESHOLDS = (0, 256)


def threshold_paired(max_in_flight: int) -> dict[str, Any]:
    by_threshold = prefill_threshold(max_in_flight)
    base = by_threshold["points"][0]["set"]
    points = []
    for k in range(PAIRED_SEEDS):
        for t in PAIRED_THRESHOLDS if k % 2 == 0 else reversed(PAIRED_THRESHOLDS):
            for rho in (0.8, 0.95):
                points.append(
                    {
                        "label": f"s{k}-t{t:04d}-rho{round(rho * 100):03d}",
                        "set": {
                            **base,
                            "seed": SEED + k,
                            "load.arrival.rho": rho,
                            "server.long_prefill_token_threshold": t,
                        },
                    }
                )
    return {
        **by_threshold,
        "suite": "threshold-paired",
        "description": (
            "Phase 6 server experiment, paired (ADR-0014): E07's mixed workload under FIFO behind "
            f"the gateway (max_in_flight {max_in_flight}) against a vLLM started with "
            f"--long-prefill-token-threshold {' and '.join(map(str, PAIRED_THRESHOLDS))} "
            f"(0 = unset), at rho 0.8 and 0.95, {PAIRED_SEEDS} seeds, each seed's thresholds back "
            "to back in alternating order. Restart the server before each (seed, threshold) "
            "block and run its points with --only. Generated by scripts/matrix_suites.py; edit "
            "that, not this file."
        ),
        "points": points,
    }


TUNING_THRESHOLD = 256
TUNING_RHO = 0.95
COST_WEIGHTS = (0.25, 0.5, 1.0, 2.0)
MAX_BYPASSES = (8, 32)


def waas_tuning(max_in_flight: int) -> dict[str, Any]:
    mixed = suite(*MATRICES[0][:4], [TUNING_RHO], 2, max_in_flight)
    mixed["set"]["server.long_prefill_token_threshold"] = TUNING_THRESHOLD
    mixed["set"]["gateway"] |= {
        "output_estimator": "class_prior",
        "output_prior": "configs/output_priors/mixed.json",
    }
    fifo = [p for p in mixed["points"] if p["set"]["gateway.scheduler.name"] == "fifo"]
    points = []
    for p in fifo:
        seed = p["label"].rsplit("-", 1)[1]
        points.append({"label": f"fifo-{seed}", "set": p["set"]})
        for w in COST_WEIGHTS:
            for b in MAX_BYPASSES:
                points.append(
                    {
                        "label": f"waas-w{round(w * 100):03d}-b{b:02d}-{seed}",
                        "set": {
                            **p["set"],
                            "gateway.scheduler.name": "waas",
                            "gateway.scheduler.params": {"cost_weight": w, "max_bypass": b},
                        },
                    }
                )
    return {
        **mixed,
        "suite": "waas-tuning",
        "description": (
            "Phase 6 WAAS tuning (ADR-0014): E07's mixed workload behind the gateway "
            f"(max_in_flight {max_in_flight}) at rho {TUNING_RHO:g}, FIFO and WAAS at cost_weight "
            f"{', '.join(f'{w:g}' for w in COST_WEIGHTS)} x max_bypass "
            f"{', '.join(map(str, MAX_BYPASSES))}, 2 tuning seeds, class-prior output estimates, "
            f"against a vLLM started with --long-prefill-token-threshold {TUNING_THRESHOLD}. "
            "Generated by scripts/matrix_suites.py; edit that, not this file."
        ),
        "points": points,
    }


PAIRED_COST_WEIGHTS = (0.5, 1.0, 2.0)
PAIRED_MAX_BYPASS = 32


def waas_paired(max_in_flight: int) -> dict[str, Any]:
    tuning = waas_tuning(max_in_flight)
    base = {k: v for k, v in tuning["points"][0]["set"].items() if k != "seed"}
    settings: list[tuple[str, dict[str, Any]]] = [("fifo", {})]
    for w in PAIRED_COST_WEIGHTS:
        params = {"cost_weight": w, "max_bypass": PAIRED_MAX_BYPASS}
        label = f"waas-w{round(w * 100):03d}-b{PAIRED_MAX_BYPASS:02d}"
        settings.append(
            (label, {"gateway.scheduler.name": "waas", "gateway.scheduler.params": params})
        )
    points = []
    for k in range(PAIRED_SEEDS):
        for label, extra in settings if k % 2 == 0 else reversed(settings):
            points.append({"label": f"s{k}-{label}", "set": {**base, "seed": SEED + k, **extra}})
    return {
        **tuning,
        "suite": "waas-paired",
        "description": (
            "Phase 6 WAAS tuning, paired (ADR-0014): E07's mixed workload behind the gateway "
            f"(max_in_flight {max_in_flight}) at rho {TUNING_RHO:g}, FIFO and WAAS at cost_weight "
            f"{', '.join(f'{w:g}' for w in PAIRED_COST_WEIGHTS)} (max_bypass {PAIRED_MAX_BYPASS}), "
            f"{PAIRED_SEEDS} tuning seeds, each seed's policies back to back in alternating "
            "order, class-prior output estimates, against a vLLM started with "
            f"--long-prefill-token-threshold {TUNING_THRESHOLD}. Generated by "
            "scripts/matrix_suites.py; edit that, not this file."
        ),
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-in-flight", type=int, required=True)
    args = parser.parse_args()
    for name, workload, weights, capacity, rhos, seeds in MATRICES:
        path = SUITES / f"{name.replace('-', '_')}.yaml"
        body = suite(name, workload, weights, capacity, rhos, seeds, args.max_in_flight)
        header = f"# Generated by scripts/matrix_suites.py --max-in-flight {args.max_in_flight}.\n"
        path.write_text(header + yaml.safe_dump(body, sort_keys=False, width=100))
        print(f"wrote {path.relative_to(SUITES.parent.parent)} ({len(body['points'])} points)")
    path = SUITES / "xcheck_vllm_priority.yaml"
    body = xcheck()
    path.write_text(
        "# Generated by scripts/matrix_suites.py.\n"
        + yaml.safe_dump(body, sort_keys=False, width=100)
    )
    print(f"wrote {path.relative_to(SUITES.parent.parent)} ({len(body['points'])} points)")
    for path, body in (
        (SUITES / "prefill_threshold.yaml", prefill_threshold(args.max_in_flight)),
        (SUITES / "waas_tuning.yaml", waas_tuning(args.max_in_flight)),
        (SUITES / "waas_paired.yaml", waas_paired(args.max_in_flight)),
        (SUITES / "threshold_paired.yaml", threshold_paired(args.max_in_flight)),
    ):
        header = f"# Generated by scripts/matrix_suites.py --max-in-flight {args.max_in_flight}.\n"
        path.write_text(header + yaml.safe_dump(body, sort_keys=False, width=100))
        print(f"wrote {path.relative_to(SUITES.parent.parent)} ({len(body['points'])} points)")


if __name__ == "__main__":
    main()
