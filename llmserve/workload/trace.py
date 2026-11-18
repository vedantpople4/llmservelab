"""Request traces: arrival offsets with each request's prompt and output length.

A trace file is CSV with the header `offset_s,prompt_tokens,output_tokens`, one request per row
in arrival order, and `#` comment lines for provenance. `workload.trace` takes lengths from it,
and a `replay` arrival naming the same file takes the offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

HEADER = ("offset_s", "prompt_tokens", "output_tokens")


@dataclass(frozen=True)
class Trace:
    offset_s: npt.NDArray[np.float64]
    prompt_tokens: npt.NDArray[np.int64]
    output_tokens: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.offset_s)


def is_trace(path: str) -> bool:
    """Whether `path` is a trace CSV rather than a plain offsets file."""
    try:
        with Path(path).open() as f:
            for line in f:
                if line.strip() and not line.lstrip().startswith("#"):
                    return line.strip().split(",")[0] == HEADER[0]
    except OSError:
        return False
    return False


@lru_cache(maxsize=8)
def load_trace(path: str) -> Trace:
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines or tuple(lines[0].split(",")) != HEADER:
        raise ValueError(f"trace {path!r}: header must be {','.join(HEADER)}")
    rows = [line.split(",") for line in lines[1:]]
    if not rows or any(len(row) != len(HEADER) for row in rows):
        raise ValueError(f"trace {path!r}: needs rows of {len(HEADER)} fields")
    try:
        offsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
        prompts = np.array([int(r[1]) for r in rows], dtype=np.int64)
        outputs = np.array([int(r[2]) for r in rows], dtype=np.int64)
    except ValueError as e:
        raise ValueError(f"trace {path!r}: {e}") from e
    if not np.all(np.isfinite(offsets)) or np.any(offsets < 0) or np.any(np.diff(offsets) < 0):
        raise ValueError(f"trace {path!r}: offsets must be finite, >= 0 and non-decreasing")
    if np.any(prompts < 1) or np.any(outputs < 1):
        raise ValueError(f"trace {path!r}: token counts must be >= 1")
    return Trace(offsets, prompts, outputs)
