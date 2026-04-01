"""Fast-SNAC benchmark: end-to-end decode + kernel-level comparison.

Usage:
    python benchmark.py                # 100s audio, all methods
    python benchmark.py --seconds 10   # 10s audio
    python benchmark.py --kernels      # kernel-level (Triton vs TileLang)
"""

import os
import gc
import time
import argparse
import torch

os.environ.setdefault("HF_HOME", "/mnt/kadirnar/huggingface")


def bench(fn, codes, warmup=10, iters=100):
    """Benchmark a decode function. Returns median latency in ms."""
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
    return times[len(times) // 2]


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_e2e(seconds=100):
    """End-to-end decode benchmark: FP32 and FP16 tables."""
    from snac import SNAC
    from snac.optimize import optimize_snac_triton

    enc = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    audio = torch.randn(1, 1, 24000 * seconds, device="cuda")
    with torch.no_grad():
        codes = enc.encode(audio)
    with torch.no_grad():
        t_base = bench(lambda c: enc.decode(c), codes)
    del enc; cleanup()

    rtf = lambda t: int(seconds * 1000 / t)

    # ── FP32 ──
    print(f"\n### Full Precision (FP32) — {seconds}s audio\n")
    print("| Method | Latency | Speedup | RTF |")
    print("|--------|:-------:|:-------:|:---:|")
    print(f"| PyTorch FP32 | {t_base:.2f} ms | 1.00x | {rtf(t_base):,}x |")

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    fn = optimize_snac_triton(model, codes, dtype="fp32", use_compile=False)
    t = bench(fn, codes)
    print(f"| + Triton kernels | {t:.2f} ms | **{t_base/t:.2f}x** | {rtf(t):,}x |")
    del model, fn; cleanup()

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    fn = optimize_snac_triton(model, codes, dtype="fp32", use_compile=True)
    t = bench(fn, codes)
    print(f"| + torch.compile | {t:.2f} ms | **{t_base/t:.2f}x** | {rtf(t):,}x |")
    sc = [c.clone() for c in codes]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        so = fn(sc)
    torch.cuda.synchronize()
    def gfn(c):
        for i in range(len(c)): sc[i].copy_(c[i])
        g.replay()
        return so
    t = bench(gfn, codes)
    print(f"| **+ CUDA graph** | **{t:.2f} ms** | **{t_base/t:.2f}x** | **{rtf(t):,}x** |")
    del model; cleanup()

    # ── FP16 ──
    print(f"\n### Half Precision (FP16) — {seconds}s audio\n")
    print("| Method | Latency | Speedup | RTF |")
    print("|--------|:-------:|:-------:|:---:|")

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval().half()
    with torch.no_grad():
        t = bench(lambda c: model.decode(c), codes)
    print(f"| PyTorch FP16 | {t:.2f} ms | {t_base/t:.2f}x | {rtf(t):,}x |")
    del model; cleanup()

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    fn = optimize_snac_triton(model, codes, dtype="fp16", use_compile=False)
    t = bench(fn, codes)
    print(f"| + Triton kernels | {t:.2f} ms | **{t_base/t:.2f}x** | {rtf(t):,}x |")
    del model, fn; cleanup()

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    fn = optimize_snac_triton(model, codes, dtype="fp16", use_compile=True)
    t = bench(fn, codes)
    print(f"| + torch.compile | {t:.2f} ms | **{t_base/t:.2f}x** | {rtf(t):,}x |")
    sc = [c.clone() for c in codes]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        so = fn(sc)
    torch.cuda.synchronize()
    def gfn2(c):
        for i in range(len(c)): sc[i].copy_(c[i])
        g.replay()
        return so
    t = bench(gfn2, codes)
    print(f"| **+ CUDA graph** | **{t:.2f} ms** | **{t_base/t:.2f}x** | **{rtf(t):,}x** |")
    del model; cleanup()


def run_kernels():
    """Kernel-level benchmark: PyTorch vs Triton vs TileLang."""
    from snac.kernels.triton_snake import snake_triton

    @torch.jit.script
    def snake_pytorch(x, alpha):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        x = x.reshape(shape)
        return x

    try:
        from snac.kernels.tilelang_snake import snake_tilelang
        has_tilelang = True
    except ImportError:
        has_tilelang = False

    def bench_snake(fn, x, alpha, warmup=10, iters=200):
        for _ in range(warmup):
            fn(x, alpha)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(x, alpha)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)
        times.sort()
        return times[len(times) // 2]

    sizes = [
        ("Encoder (1024ch)",  1024, 5442),
        ("Decoder (768ch)",   768,  38095),
        ("Decoder (384ch)",   384,  266666),
        ("Decoder (192ch)",   192,  800000),
        ("Decoder (96ch)",    96,   2400000),
    ]

    for dtype_name, torch_dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
        header = "| Layer | PyTorch | Triton |"
        sep = "|-------|:-------:|:------:|"
        if has_tilelang:
            header += " TileLang | Triton spd | TileLang spd |"
            sep += ":--------:|:----------:|:------------:|"
        else:
            header += " Triton speedup |"
            sep += ":--------------:|"

        print(f"\n### Snake Kernel — {dtype_name}\n")
        print(header)
        print(sep)

        total_pt = total_tr = total_tl = 0
        for label, C, T in sizes:
            x = torch.randn(1, C, T, device="cuda", dtype=torch_dtype)
            alpha = torch.rand(1, C, 1, device="cuda", dtype=torch_dtype) * 5.0 + 0.5
            t_pt = bench_snake(snake_pytorch, x, alpha)
            t_tr = bench_snake(snake_triton, x, alpha)
            total_pt += t_pt
            total_tr += t_tr

            row = f"| {label} | {t_pt:.0f} us | **{t_tr:.0f} us** |"
            if has_tilelang:
                try:
                    t_tl = bench_snake(snake_tilelang, x, alpha)
                    total_tl += t_tl
                    row += f" {t_tl:.0f} us | **{t_pt/t_tr:.2f}x** | {t_pt/t_tl:.2f}x |"
                except Exception:
                    row += f" FAIL | **{t_pt/t_tr:.2f}x** | N/A |"
            else:
                row += f" **{t_pt/t_tr:.2f}x** |"
            print(row)

        row = f"| **Total** | **{total_pt:.0f} us** | **{total_tr:.0f} us** |"
        if has_tilelang and total_tl > 0:
            row += f" **{total_tl:.0f} us** | **{total_pt/total_tr:.2f}x** | **{total_pt/total_tl:.2f}x** |"
        elif has_tilelang:
            row += f" — | **{total_pt/total_tr:.2f}x** | — |"
        else:
            row += f" **{total_pt/total_tr:.2f}x** |"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=100)
    parser.add_argument("--kernels", action="store_true")
    args = parser.parse_args()

    gpu = torch.cuda.get_device_name()
    print(f"{gpu} | snac_24khz | {args.seconds}s audio @ 24kHz")

    if args.kernels:
        run_kernels()
    else:
        run_e2e(args.seconds)
        print()
        run_kernels()
