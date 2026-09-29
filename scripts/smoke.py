"""Phase 1 exit check: N sequential requests, 0 unexplained failures, 0 usage mismatches.

    uv run python -m llmserve.mock --port 8001          # in another shell
    uv run python scripts/smoke.py configs/dev/smoke_mock.yaml

Exit code 0 only when every request ended `ok` — that is, no timeout, HTTP error, connection
error or usage mismatch. Failures are printed; failures are data, never dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np

from llmserve.client.auth import AuthEnvError, auth_headers
from llmserve.client.capabilities import capabilities
from llmserve.client.openai_stream import stream_completion
from llmserve.config.loader import config_hash, load_config
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics import latency
from llmserve.metrics.records import RequestRecord
from llmserve.runner.clock import RunClock
from llmserve.runner.env_check import EnvCheckError, check_server
from llmserve.scheduler.base import Priority
from llmserve.workload.distributions import sample
from llmserve.workload.prompts import PromptBuilder
from llmserve.workload.spec import RequestSpec

_PRIORITY = {"high": Priority.HIGH, "medium": Priority.MEDIUM, "low": Priority.LOW}


def build_specs(cfg: ExperimentConfig, n: int, builder: PromptBuilder) -> list[RequestSpec]:
    """Sample n request specs from the config's classes, seeded by `cfg.seed`."""
    rng = np.random.default_rng(cfg.seed)
    classes = cfg.workload.classes
    weights = np.array([c.weight for c in classes], dtype=np.float64)
    weights /= weights.sum()
    specs: list[RequestSpec] = []
    for i in range(n):
        cls = classes[int(rng.choice(len(classes), p=weights))]
        prompt_tokens = int(sample(cls.prompt, rng, 1)[0])
        output_tokens = int(sample(cls.output, rng, 1)[0])
        ids = builder.build(prompt_tokens, rng)
        specs.append(
            RequestSpec(
                request_id=f"r{i:05d}",
                workload_class=cls.name,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prompt_ids=tuple(ids),
                prompt_text=builder.text(ids),
                priority=int(_PRIORITY[cls.priority]),
                slo_ttft_ms=cls.slo.ttft_ms,
                slo_tpot_ms=cls.slo.tpot_ms,
            )
        )
    return specs


async def smoke(
    cfg: ExperimentConfig,
    n: int,
    *,
    skip_env_check: bool = False,
    endpoint: str | None = None,
    quiet: bool = False,
) -> int:
    caps = capabilities(cfg.server.kind)
    target = endpoint or cfg.server.endpoint
    try:
        headers = auth_headers(cfg.server)
    except AuthEnvError as e:
        print(f"env check FAILED: {e}", file=sys.stderr)
        return 2

    if not skip_env_check:
        try:
            report = await check_server(cfg, endpoint=target)
        except EnvCheckError as e:
            print(f"env check FAILED: {e}", file=sys.stderr)
            return 2
        if not quiet:
            print(f"env check ok: {report.summary}")

    builder = PromptBuilder.default()
    specs = build_specs(cfg, n, builder)
    clock = RunClock()
    records: list[RequestRecord] = []

    def token_counter(text: str) -> int:
        return len(builder.tokenizer.encode(text))

    async with httpx.AsyncClient(headers=headers) as client:
        for i, spec in enumerate(specs, 1):
            t_arrival = clock()
            records.append(
                await stream_completion(
                    client,
                    spec,
                    clock,
                    endpoint=target,
                    model=cfg.server.model,
                    t_arrival=t_arrival,
                    caps=caps,
                    timeout_s=cfg.measurement.request_timeout_s,
                    token_counter=token_counter,
                )
            )
            if not quiet and (i % 25 == 0 or i == n):
                failed = sum(1 for r in records if not r.ok)
                print(f"  {i}/{n} requests · {failed} failed")

    return report_result(cfg, records, clock() / 1e9, quiet=quiet)


def report_result(
    cfg: ExperimentConfig, records: list[RequestRecord], wall_s: float, *, quiet: bool
) -> int:
    ok = [r for r in records if r.ok]
    failed = [r for r in records if not r.ok]
    ttft = sorted(t for t in (latency.ttft(r) for r in ok) if t is not None)
    tpot = sorted(t for t in (latency.tpot(r) for r in ok) if t is not None)
    throughput = len(ok) / wall_s if wall_s > 0 else 0.0

    def pct(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return values[idx] * 1e3

    print(
        f"\nsmoke {cfg.experiment} · config {config_hash(cfg)[:12]} · "
        f"{cfg.server.kind} · {len(records)} requests in {wall_s:.1f}s"
    )
    print(f"  ok         {len(ok)}/{len(records)}")
    print(f"  failures   {len(failed)}")
    print(f"  TTFT ms    p50 {pct(ttft, 0.50):8.1f}   p95 {pct(ttft, 0.95):8.1f}")
    print(f"  TPOT ms    p50 {pct(tpot, 0.50):8.1f}   p95 {pct(tpot, 0.95):8.1f}")
    print(f"  throughput {throughput:8.2f} req/s")

    for rec in failed[:10]:
        print(f"  FAIL {rec.request_id} [{rec.status.value}] {rec.error or ''}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")

    if failed:
        by_status: dict[str, int] = {}
        for rec in failed:
            by_status[rec.status.value] = by_status.get(rec.status.value, 0) + 1
        print(f"  exit 1: {by_status}", file=sys.stderr)
        return 1
    print("  exit 0: 0 unexplained failures, 0 usage mismatches")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="smoke")
    parser.add_argument("config", type=Path)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--endpoint", help="override server.endpoint (e.g. a different port)")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    code = asyncio.run(
        smoke(
            cfg,
            args.requests,
            skip_env_check=args.skip_env_check,
            endpoint=args.endpoint,
            quiet=args.quiet,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
