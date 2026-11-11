import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from tcp import served_over_tcp

from llmserve import cli
from llmserve.analysis.regress import Baseline, Change, compare, headline
from llmserve.mock import DelayModel, MockEngine, create_app
from llmserve.workload.prompts import PromptBuilder


@pytest.fixture(scope="module")
def mock_url() -> Any:
    with served_over_tcp(create_app(MockEngine(delay=DelayModel.instant()))) as base:
        yield base


def test_headline_reads_cross_rep_means_and_skips_unmeasured_metrics() -> None:
    summary = {
        "latency": {"ttft_s": {"p50": {"mean": 0.2}, "p95": {"mean": None}}},
        "throughput": {"gen_toks_s": {"mean": 31.5}},
        "success_rate": {"mean": 1.0},
    }
    assert headline(summary) == {
        "ttft p50 (s)": 0.2,
        "generated tok/s": 31.5,
        "success rate": 1.0,
    }


def test_a_move_beyond_ten_percent_either_way_is_flagged() -> None:
    assert not Change("m", 1.0, 1.09).flagged
    assert Change("m", 1.0, 1.11).flagged
    assert Change("m", 1.0, 0.85).flagged
    assert not Change("m", 0.0, 0.0).flagged
    assert Change("m", 0.0, 0.1).flagged


def test_compare_covers_only_metrics_both_runs_measured() -> None:
    changes = compare({"ttft p50 (s)": 0.2, "requests/s": 1.0}, {"ttft p50 (s)": 0.3})
    assert [c.metric for c in changes] == ["ttft p50 (s)"]
    assert changes[0].flagged


def write_cfg(tmp_path: Path, requests: int = 12) -> Path:
    path = tmp_path / "standard.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": "regress-test",
                "seed": 7,
                "repetitions": 1,
                "measurement": {"warmup": {"requests": 0}},
                "server": {"kind": "mock", "endpoint": "http://localhost:9/v1", "model": "mock"},
                "load": {"mode": "closed", "concurrency": 2, "requests": requests},
                "workload": {
                    "classes": [
                        {
                            "name": "c",
                            "prompt": {"distribution": "fixed", "tokens": 16},
                            "output": {"distribution": "fixed", "tokens": 4},
                        }
                    ]
                },
            }
        )
    )
    return path


def regress(cfg: Path, baseline: Path, out: Path, mock_url: str, *extra: str) -> Any:
    try:
        PromptBuilder.default()
    except (RuntimeError, OSError) as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    argv = ["regress", str(cfg), "--baseline", str(baseline), "-o", str(out)]
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--endpoint", f"{mock_url}/v1", "--allow-dirty", *extra])
    return exc.value.code


def test_first_run_writes_the_baseline_and_a_moved_metric_fails(
    mock_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg, baseline = write_cfg(tmp_path), tmp_path / "baseline.json"
    assert regress(cfg, baseline, tmp_path / "out-1", mock_url) == 0
    assert Baseline.load(baseline).metrics["success rate"] == 1.0

    raw = json.loads(baseline.read_text())
    raw["metrics"]["success rate"] = 0.5
    baseline.write_text(json.dumps(raw))
    capsys.readouterr()
    assert regress(cfg, baseline, tmp_path / "out-2", mock_url) == 1
    assert "success rate" in capsys.readouterr().out.splitlines()[-1]

    assert regress(cfg, baseline, tmp_path / "out-3", mock_url, "--update") == 0
    assert Baseline.load(baseline).metrics["success rate"] == 1.0


def test_a_baseline_from_another_config_is_refused_before_running(
    mock_url: str, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline.json"
    Baseline("not-this-config", "some-run", {"success rate": 1.0}).save(baseline)
    assert regress(write_cfg(tmp_path), baseline, tmp_path / "out", mock_url) == 2
    assert not (tmp_path / "out").exists()
