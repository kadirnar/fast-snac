"""Benchmark SNAC decode: baseline vs optimized (fp32, fp16)."""

import os
import time
import torch

os.environ["HF_HOME"] = "/mnt/kadirnar/huggingface"

from snac import SNAC
from snac.optimize import optimize_snac


def make_sample_codes(model, seconds=1, device="cuda"):
    """Generate sample codes matching the model's actual structure."""
    audio = torch.randn(1, 1, model.sampling_rate * seconds, device=device)
    with torch.no_grad():
        codes = model.encode(audio)
    return [c.to(device) for c in codes]


def bench(fn, codes, warmup=10, iters=100, label=""):
    """Benchmark a decode function."""
    for _ in range(warmup):
        fn(codes)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        fn(codes)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters * 1000

    print(f"  {label:40s} {elapsed:8.3f} ms")
    return elapsed


def main():
    for SECONDS in [1, 10]:
        print(f"\n{'=' * 60}")
        print(f"  SNAC decode benchmark — {SECONDS}s audio (24kHz)")
        print(f"{'=' * 60}")

        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
        codes = make_sample_codes(model, seconds=SECONDS)
        for i, c in enumerate(codes):
            print(f"  codes[{i}] shape: {c.shape}")

        # Baseline
        with torch.no_grad():
            out = model.decode(codes)
        print(f"  output shape: {out.shape}\n")

        baseline_fn = lambda c: model.decode(c)
        with torch.no_grad():
            t_base = bench(baseline_fn, codes, label="Baseline (FP32, no optimization)")

        # Optimized FP32
        model_fp32 = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
        try:
            decode_fp32 = optimize_snac(model_fp32, codes, dtype="fp32")
            t_fp32 = bench(decode_fp32, codes, label="Optimized (FP32, compile)")
            print(f"  {'→ Speedup vs baseline':40s} {t_base/t_fp32:8.2f}x")
        except Exception as e:
            print(f"  FP32 failed: {e}")

        # Optimized FP16
        model_fp16 = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
        try:
            decode_fp16 = optimize_snac(model_fp16, codes, dtype="fp16")
            t_fp16 = bench(decode_fp16, codes, label="Optimized (FP16, compile)")
            print(f"  {'→ Speedup vs baseline':40s} {t_base/t_fp16:8.2f}x")
        except Exception as e:
            print(f"  FP16 failed: {e}")

        # Optimized FP16 + CUDA Graph
        model_graph = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
        try:
            decode_graph = optimize_snac(model_graph, codes, dtype="fp16", use_cuda_graph=True)
            t_graph = bench(decode_graph, codes, label="Optimized (FP16, compile + CUDA graph)")
            print(f"  {'→ Speedup vs baseline':40s} {t_base/t_graph:8.2f}x")
        except Exception as e:
            print(f"  FP16+Graph failed: {e}")


if __name__ == "__main__":
    main()
