"""Policy-matrix and in-flight-cap analysis on synthetic request tables and mock runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from conftest import RunFactory, experiment_config
from test_analysis_suites import write_manifest

from llmserve.analysis.policies import (
    cells,
    choose_cap,
    class_ttft,
    matrix_requests,
    policy_of,
    slowdown_deciles,
)
from llmserve.analysis.suites import load_suite_runs
from llmserve.scheduler.estimators import CostModel, DecodeFit, PrefillFit

# Service time = prompt tokens / 1000 s: prefill only, no decode cost.
MODEL = CostModel(PrefillFit(0.0, 1e-3, 0.0, 1.0, 1), DecodeFit(0.0, 0.0, 1.0, 1))


def frame(policy: str, prompts: list[int], e2e: list[float], rho: float = 0.8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rho": rho,
            "policy": policy,
            "workload_class": "c",
            "prompt_tokens_req": prompts,
            "output_tokens_req": 1,
            "e2e_s": e2e,
            "ttft_s": e2e,
        }
    )


def test_deciles_share_edges_across_policies_and_expose_starved_long_requests() -> None:
    prompts = list(range(100, 2100, 100))  # 20 requests, two per decile
    service = [p / 1000 for p in prompts]
    fifo = frame("fifo", prompts, [s * 2 for s in service])
    # SJF serves small requests at once and makes the two largest wait 50× their service time.
    sjf = frame(
        "sjf",
        prompts,
        [s * (50 if p >= 1900 else 1) for s, p in zip(service, prompts, strict=True)],
    )
    table = slowdown_deciles(pd.concat([fifo, sjf], ignore_index=True), MODEL)
    assert sorted(table["decile"].unique()) == list(range(1, 11))
    by = table.set_index(["policy", "decile"])
    for d in range(1, 11):
        assert by.loc[("fifo", d), "service_lo_s"] == by.loc[("sjf", d), "service_lo_s"]
        assert by.loc[("fifo", d), "n"] == 2
    assert by.loc[("fifo", 10), "p50"] == 2.0
    assert by.loc[("sjf", 10), "p50"] == 50.0
    assert by.loc[("sjf", 1), "p50"] == 1.0


def test_class_ttft_reports_percentiles_per_cell_in_policy_order() -> None:
    table = class_ttft(
        pd.concat(
            [frame("spf", [1] * 100, list(np.arange(1.0, 101))), frame("fifo", [1] * 3, [1, 2, 3])]
        )
    )
    assert list(table["policy"]) == ["fifo", "spf"]
    spf = table[table["policy"] == "spf"].iloc[0]
    assert spf["n"] == 100 and spf["p50"] == 50.5 and round(spf["p99"], 2) == 99.01


def test_choose_cap_takes_the_smallest_within_five_percent_of_the_largest() -> None:
    table = pd.DataFrame({"cap": [4, 8, 16, 256], "gen_toks_s": [20.0, 28.0, 29.0, 29.0]})
    assert choose_cap(table) == 8  # 28 ≥ 0.95 × 29 = 27.55
    assert choose_cap(table.iloc[[0, 3]]) == 256
    assert choose_cap(table.iloc[[3]]) is None


def test_cells_read_policy_rho_and_seed_from_each_runs_config(
    run_factory: RunFactory, tmp_path: Path
) -> None:
    root = tmp_path / "results"
    points = []
    for policy, seed in (("spf", 43), ("fifo", 42)):
        load = {
            "mode": "open",
            "arrival": {"process": "poisson", "rho": 0.8},
            "capacity_rps": 40,
            "duration_s": 0.5,
        }
        gateway = {"enabled": True, "max_in_flight": 2, "scheduler": {"name": policy}}
        run = run_factory(
            experiment_config(
                experiment=f"m-{policy}", seed=seed, repetitions=1, load=load, gateway=gateway
            )
        )
        points.append((f"{policy}-rho080-s{seed - 42}", None, run, "done"))
    write_manifest(root, "matrix-mixed", None, points)
    suite = load_suite_runs(root, "matrix-mixed")
    assert suite is not None
    assert [(c.policy, c.rho, c.seed) for c in cells(suite)] == [
        ("fifo", 0.8, 42),
        ("spf", 0.8, 43),
    ]
    requests = matrix_requests(suite)
    assert set(requests["policy"]) == {"fifo", "spf"} and set(requests["rho"]) == {0.8}
    assert requests["ok"].all() and requests["in_window"].all()
    assert policy_of(run_factory(experiment_config(repetitions=1))) == "none"
