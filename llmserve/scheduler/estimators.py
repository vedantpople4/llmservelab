"""Cost model fitted from baseline runs (plan Phase 3 item 5): what a request costs to serve.

- **Prefill:** `TTFT ≈ a + b·p + c·p²` over prompt length `p`, fitted on concurrency-1 runs so
  queueing is absent. Its R² is the evidence for H3.
- **Decode:** `TPOT ≈ d0 + d1·B` over decode batch size `B` (closed-loop concurrency, or the
  server's `running` count).

SJF/SPF and WAAS (Phases 5–6) read these to estimate service time; the mock calibration reads
them to set the simulator's step-time coefficients. Fits are plain least squares, so a fitted
model is reproducible from its input points alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PrefillFit:
    a: float
    b: float
    c: float
    r2: float
    n: int

    def predict(self, prompt_tokens: float) -> float:
        """Seconds to first token for a prompt of `prompt_tokens`, with no queueing."""
        return self.a + self.b * prompt_tokens + self.c * prompt_tokens**2


@dataclass(frozen=True)
class DecodeFit:
    d0: float
    d1: float
    r2: float
    n: int

    def predict(self, batch: float) -> float:
        """Seconds per output token at decode batch size `batch`."""
        return self.d0 + self.d1 * batch


@dataclass(frozen=True)
class CostModel:
    prefill: PrefillFit
    decode: DecodeFit

    def service_time(self, prompt_tokens: float, output_tokens: float, batch: float) -> float:
        """Estimated seconds from dispatch to last token at a steady decode batch."""
        return self.prefill.predict(prompt_tokens) + max(0.0, output_tokens - 1) * (
            self.decode.predict(batch)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"prefill": asdict(self.prefill), "decode": asdict(self.decode)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostModel:
        return cls(prefill=PrefillFit(**data["prefill"]), decode=DecodeFit(**data["decode"]))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> CostModel:
        return cls.from_dict(json.loads(path.read_text()))


def fit_prefill(prompt_tokens: npt.ArrayLike, ttft_s: npt.ArrayLike) -> PrefillFit:
    """Quadratic least-squares fit of TTFT on prompt length; needs 3+ distinct lengths."""
    p, y = _points(prompt_tokens, ttft_s)
    if len(np.unique(p)) < 3:
        raise ValueError("fit_prefill needs at least 3 distinct prompt lengths")
    design = np.column_stack([np.ones_like(p), p, p**2])
    (a, b, c), r2 = _lstsq(design, y)
    return PrefillFit(a=a, b=b, c=c, r2=r2, n=len(p))


def fit_decode(batch: npt.ArrayLike, tpot_s: npt.ArrayLike) -> DecodeFit:
    """Linear least-squares fit of TPOT on decode batch size; needs 2+ distinct batch sizes."""
    x, y = _points(batch, tpot_s)
    if len(np.unique(x)) < 2:
        raise ValueError("fit_decode needs at least 2 distinct batch sizes")
    design = np.column_stack([np.ones_like(x), x])
    (d0, d1), r2 = _lstsq(design, y)
    return DecodeFit(d0=d0, d1=d1, r2=r2, n=len(x))


def _points(
    x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"x and y differ in length ({xs.size} vs {ys.size})")
    keep = np.isfinite(xs) & np.isfinite(ys)
    return xs[keep], ys[keep]


def _lstsq(
    design: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[list[float], float]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    # einsum, not @: Accelerate's matmul raises spurious divide-by-zero flags on this shape.
    residual = y - np.einsum("ij,j->i", design, coef)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
    return [float(c) for c in coef], r2
