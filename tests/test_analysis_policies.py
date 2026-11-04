"""Policy-matrix and in-flight-cap analysis on synthetic request tables and mock runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import RunFactory, experiment_config
from test_analysis_profile import power_sample
from test_analysis_suites import write_manifest

from llmserve.analysis.policies import (
    _sorted,
    against_baseline,
    cells,
    choose_cap,
    class_ttft,
    interference,
    matrix_requests,
    measured,
    paired,
    policy_of,
    slowdown_deciles,
)
from llmserve.analysis.suites import load_suite_runs
from llmserve.runner import storage
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


def _request(
    cls: str, dispatch_s: float, first_s: float, prompt: int, **kw: object
) -> dict[str, object]:
    row = {
        "rho": 0.8,
        "run_id": "r",
        "rep": 0,
        "ok": True,
        "in_window": True,
        "workload_class": cls,
        "t_dispatch": int(dispatch_s * 1e9),
        "t_first_token": int(first_s * 1e9),
        "prompt_tokens_req": prompt,
    }
    return {**row, **kw}


def test_interference_counts_only_blocker_prefill_in_flight_at_dispatch() -> None:
    requests = pd.DataFrame(
        [
            _request("long", 0, 10, 4096),
            _request("long", 1, 12, 8192, in_window=False),  # warm-up still blocks
            _request("long", 20, 21, 2048),  # dispatched after every victim
            _request("long", 0, 5, 9999, ok=False),  # failed: ignored
            _request("short", 2, 11, 100),  # behind both: 12288 tokens
            _request("short", 11, 12, 100),  # behind the second only: 8192
            _request("short", 13, 13.5, 100),  # both have first tokens: none
            _request("short", 14, 20, 100, in_window=False),  # not a measured victim
        ]
    )
    table = interference(requests, "short", "long").set_index("bucket")
    assert list(table.index) == ["none", "4k–8k", "> 8k"]
    assert table.loc["> 8k", "p50"] == 9.0
    assert table.loc["4k–8k", "p50"] == 1.0
    assert table.loc["none", "p50"] == 0.5
    assert table["n"].sum() == 3 and table["share"].sum() == 1.0


def test_interference_splits_by_policy_when_asked() -> None:
    rows = []
    for policy, server_s in (("fifo", 9.0), ("fifo@lpt256", 1.0)):
        rows += [
            _request("long", 0, 10, 4096, policy=policy, run_id=policy),
            _request("short", 2, 2 + server_s, 100, policy=policy, run_id=policy),
        ]
    table = interference(pd.DataFrame(rows), "short", "long", by=("rho", "policy"))
    assert list(table["policy"]) == ["fifo", "fifo@lpt256"]
    assert list(table["p50"]) == [9.0, 1.0]
    assert list(table["share"]) == [1.0, 1.0]


def test_thresholds_sort_numerically_after_their_base_policy() -> None:
    policies = ["spf", "fifo@lpt1024", "fifo", "fifo@lpt256", "waas"]
    table = _sorted(pd.DataFrame({"rho": 0.8, "policy": policies}))
    assert list(table["policy"]) == ["fifo", "fifo@lpt256", "fifo@lpt1024", "spf", "waas"]


def test_policy_label_names_scheduler_params(run_factory: RunFactory) -> None:
    gateway = {
        "enabled": True,
        "scheduler": {"name": "waas", "params": {"max_bypass": 8, "cost_weight": 0.5}},
    }
    cfg = experiment_config(repetitions=1, gateway=gateway)
    assert policy_of(run_factory(cfg)) == "waas[cost_weight=0.5,max_bypass=8]"


def test_params_sort_after_their_base_policy() -> None:
    policies = ["waas[cost_weight=1]@lpt256", "fifo@lpt256", "waas[cost_weight=0.5]@lpt256"]
    table = _sorted(pd.DataFrame({"rho": 0.95, "policy": policies}))
    assert list(table["policy"]) == [
        "fifo@lpt256",
        "waas[cost_weight=0.5]@lpt256",
        "waas[cost_weight=1]@lpt256",
    ]


def test_against_baseline_compares_each_policy_at_the_same_rho() -> None:
    fast, slow = [float(x) for x in range(1, 41)], [float(x) for x in range(11, 51)]
    requests = pd.concat(
        [
            frame("fifo", [1] * 40, slow),
            frame("waas", [1] * 40, fast),
            frame("spf", [1] * 5, fast[:5]),  # too few to compare
            frame("waas", [1] * 40, fast, rho=0.5),  # no baseline at this ρ
        ],
        ignore_index=True,
    )
    table = against_baseline(requests, "fifo")
    assert list(table["policy"]) == ["waas"] and list(table["rho"]) == [0.8]
    row = table.iloc[0]
    assert row["diff"] == -10.0 and row["hi"] < 0 and row["n"] == row["n_base"] == 40


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


def test_paired_compares_each_run_with_the_baseline_at_its_seed(
    run_factory: RunFactory, tmp_path: Path
) -> None:
    root = tmp_path / "results"
    points = []
    for policy, seed in (("spf", 42), ("fifo", 42), ("spf", 43)):
        load = {
            "mode": "open",
            "arrival": {"process": "poisson", "rho": 0.8},
            "capacity_rps": 40,
            "duration_s": 0.5,
        }
        gateway = {"enabled": True, "max_in_flight": 2, "scheduler": {"name": policy}}
        run = run_factory(
            experiment_config(
                experiment=f"m-{policy}-{seed}",
                seed=seed,
                repetitions=1,
                load=load,
                gateway=gateway,
            )
        )
        points.append((f"{policy}-s{seed - 42}", None, run, "done"))
    (rep,) = points[0][2].rep_dirs()
    storage.write_power(rep / "power.parquet", [power_sample(1, 10.0), power_sample(2, 10.0)])
    write_manifest(root, "waas-paired", None, points)
    suite = load_suite_runs(root, "waas-paired")
    assert suite is not None
    table = paired(suite, "fifo", "c")
    assert list(zip(table["seed"], table["policy"], strict=True)) == [
        (42, "fifo"),
        (42, "spf"),
        (43, "spf"),
    ]
    p95 = {
        (c.seed, c.policy): float(np.percentile(measured(c.run)["ttft_s"].dropna(), 95))
        for c in cells(suite)
    }
    assert list(table["ttft"]) == pytest.approx([p95[42, "fifo"], p95[42, "spf"], p95[43, "spf"]])
    assert table["diff"].iloc[0] == 0.0
    assert table["diff"].iloc[1] == pytest.approx(p95[42, "spf"] - p95[42, "fifo"])
    assert np.isnan(table["diff"].iloc[2])  # no FIFO run at seed 43
    assert table["gpu_mhz"].iloc[1] == 1000.0 and np.isnan(table["gpu_mhz"].iloc[0])
