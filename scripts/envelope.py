"""Back-of-envelope TTFT / TPOT predictions for Qwen2.5-7B (docs/background.md §4).

    uv run python scripts/envelope.py --preset l4
    uv run python scripts/envelope.py --tflops 30 --bandwidth-gbs 273 --weights-gb 4.3  # a Mac

Preset numbers are approximate datasheet values; check them against the vendor datasheet for the
exact card you rent. The point is to predict within ~2x *before* measuring, then explain the gap.
"""

from __future__ import annotations

import argparse

# Qwen2.5-7B architecture (from the model's config.json).
PARAMS = 7.62e9
LAYERS = 28
HIDDEN = 3584
KV_HEADS = 4
HEAD_DIM = 128
KV_BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2  # K and V, bf16: 57,344 B

# (dense bf16 tensor TFLOPS, memory bandwidth GB/s, memory GB). Approximate; verify.
PRESETS: dict[str, tuple[float, float, float]] = {
    "l4": (121, 300, 24),
    "a10": (125, 600, 24),
    "l40s": (362, 864, 48),
    "a100-80g": (312, 2039, 80),
    "h100-sxm": (989, 3350, 80),
}


def prefill_seconds(prompt: int, tflops: float, mfu: float) -> float:
    linear = 2 * PARAMS * prompt
    attention = 2 * LAYERS * prompt**2 * HIDDEN  # causal QK^T + AV
    return (linear + attention) / (tflops * 1e12 * mfu)


def decode_step_seconds(
    batch: int,
    context: int,
    weights_gb: float,
    bw_gbs: float,
    bw_eff: float,
    tflops: float,
    mfu: float,
) -> float:
    """One decode step reads all weights plus every sequence's KV cache (memory-bound at small
    batch) and does 2·params FLOPs per sequence (compute-bound at large batch)."""
    bytes_read = weights_gb * 1e9 + batch * context * KV_BYTES_PER_TOKEN
    memory = bytes_read / (bw_gbs * 1e9 * bw_eff)
    compute = 2 * PARAMS * batch / (tflops * 1e12 * mfu)
    return max(memory, compute)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--tflops", type=float)
    p.add_argument("--bandwidth-gbs", type=float)
    p.add_argument("--memory-gb", type=float)
    p.add_argument("--weights-gb", type=float, default=PARAMS * 2 / 1e9, help="bf16 by default")
    p.add_argument("--mfu", type=float, default=0.5, help="fraction of peak FLOPs achieved")
    p.add_argument("--bw-eff", type=float, default=0.8, help="fraction of peak bandwidth achieved")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    a = p.parse_args()

    tflops, bw, mem = PRESETS[a.preset] if a.preset else (0.0, 0.0, 0.0)
    tflops, bw, mem = a.tflops or tflops, a.bandwidth_gbs or bw, a.memory_gb or mem
    if not tflops or not bw:
        p.error("give --preset or both --tflops and --bandwidth-gbs")

    print(f"KV cache per token: {KV_BYTES_PER_TOKEN / 1024:.0f} KiB")
    if mem:
        kv_gb = mem * a.gpu_mem_util - a.weights_gb - 1.5  # ~1.5 GB activations/overhead
        print(f"KV budget: ~{kv_gb:.1f} GB ≈ {kv_gb * 1e9 / KV_BYTES_PER_TOKEN:,.0f} tokens")

    print(f"\nTTFT at batch 1 (MFU {a.mfu:.0%})")
    for prompt in (128, 512, 1024, 2048, 4096, 8192):
        print(f"  {prompt:>5} tokens  {prefill_seconds(prompt, tflops, a.mfu) * 1e3:8.1f} ms")

    print(f"\nTPOT (bandwidth efficiency {a.bw_eff:.0%}), context 1,024 tokens")
    for batch in (1, 8, 32, 128):
        step = decode_step_seconds(batch, 1024, a.weights_gb, bw, a.bw_eff, tflops, a.mfu)
        print(f"  batch {batch:>3}  {step * 1e3:6.1f} ms/token  {batch / step:8.0f} tok/s total")


if __name__ == "__main__":
    main()
