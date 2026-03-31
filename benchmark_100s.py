"""Benchmark SNAC decode on 100s audio — baseline vs all optimization levels."""

import os
import time
import gc
import torch

os.environ["HF_HOME"] = "/mnt/kadirnar/huggingface"


def bench(fn, codes, warmup=5, iters=50, label=""):
    for _ in range(warmup):
        fn(codes)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(codes)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    med = times[len(times) // 2]
    p10 = times[len(times) // 10]
    print(f"  {label:45s} median={med:8.2f} ms  p10={p10:8.2f} ms")
    return med


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    from snac import SNAC
    from snac.optimize import optimize_snac, optimize_snac_native

    # Load 100s codes
    codes_path = "/mnt/kadirnar/github/snac/test_codes_100s.pt"
    if os.path.exists(codes_path):
        codes = torch.load(codes_path, weights_only=True)
        codes = [c.cuda() for c in codes]
    else:
        print("Generating 100s codes...")
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
        audio = torch.randn(1, 1, 24000 * 100, device="cuda")
        with torch.no_grad():
            codes = model.encode(audio)
        torch.save(codes, codes_path)
        del model
        cleanup()

    for i, c in enumerate(codes):
        print(f"codes[{i}] shape: {c.shape}")

    print(f"\n{'=' * 65}")
    print(f"  SNAC decode benchmark — 100s audio (24kHz) — H100 PCIe")
    print(f"{'=' * 65}\n")

    # ── Baseline ──
    print("Loading baseline model...")
    model_base = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    with torch.no_grad():
        out = model_base.decode(codes)
    print(f"  Output shape: {out.shape}  duration: {out.shape[-1]/24000:.1f}s\n")

    baseline_fn = lambda c: model_base.decode(c)
    with torch.no_grad():
        t_base = bench(baseline_fn, codes, label="Baseline (FP32, no optimization)")
    del model_base
    cleanup()

    # ── Native FP32 (no Conv2d, just compile) ──
    print("\nOptimizing Native FP32...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    try:
        decode_n32 = optimize_snac_native(model, codes, dtype="fp32")
        t_n32 = bench(decode_n32, codes, label="Native FP32 + max-autotune")
        print(f"  {'→ Speedup':45s} {t_base/t_n32:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")
    del model
    cleanup()

    # ── Native FP16 ──
    print("\nOptimizing Native FP16...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    try:
        decode_n16 = optimize_snac_native(model, codes, dtype="fp16")
        t_n16 = bench(decode_n16, codes, label="Native FP16 + max-autotune")
        print(f"  {'→ Speedup':45s} {t_base/t_n16:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")
    del model
    cleanup()

    # ── Conv2d FP16 (original approach) ──
    print("\nOptimizing Conv2d FP16...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    try:
        decode_c16 = optimize_snac(model, codes, dtype="fp16")
        t_c16 = bench(decode_c16, codes, label="Conv2d FP16 + max-autotune")
        print(f"  {'→ Speedup':45s} {t_base/t_c16:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")
    del model
    cleanup()

    print(f"\n{'=' * 65}")
    print("Done.")


if __name__ == "__main__":
    main()
