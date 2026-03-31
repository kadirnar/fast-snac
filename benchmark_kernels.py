"""Benchmark Snake activation: PyTorch vs Triton vs CUDA vs TileLang vs CuTe DSL."""

import os
import time
import torch

os.environ["HF_HOME"] = "/mnt/kadirnar/huggingface"


def bench_fn(fn, warmup=20, iters=200):
    """Benchmark a function, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def main():
    print("Snake Activation Benchmark — NVIDIA H100 PCIe")
    print("=" * 70)

    # Test configs matching SNAC decoder layer sizes
    configs = [
        (96, 24576,   "Small  (final layer, ~1s audio)"),
        (192, 12288,  "Medium (decoder block 4)"),
        (384, 4096,   "Medium (decoder block 3)"),
        (768, 1024,   "Large  (decoder block 2)"),
        (1536, 512,   "XLarge (decoder block 1)"),
    ]

    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        print(f"\n{'─' * 70}")
        print(f"  Precision: {dtype_name}")
        print(f"{'─' * 70}")

        for C, T, desc in configs:
            x = torch.randn(1, C, T, device="cuda", dtype=dtype)
            alpha = torch.ones(1, C, 1, device="cuda") * 0.5

            print(f"\n  [{desc}] C={C}, T={T}")

            # ── PyTorch baseline (sin) ──
            def pytorch_sin():
                return x + (1.0 / (alpha + 1e-9)) * torch.sin(alpha * x).pow(2)

            t_pt = bench_fn(pytorch_sin)
            bw_base = 2 * C * T * x.element_size() / (t_pt / 1000) / 1e9
            print(f"    {'PyTorch (sin)':25s} {t_pt:8.3f} ms  ({bw_base:.0f} GB/s)  1.00x")

            # ── PyTorch polynomial (no sin) ──
            PI = 3.141592653589793
            INV_PI = 1.0 / PI
            a4 = alpha.unsqueeze(-1) if x.ndim == 4 else alpha
            inv_a = 1.0 / (a4 + 1e-9)

            def pytorch_poly():
                ax = a4 * x
                theta = ax - PI * torch.round(ax * INV_PI)
                t2 = theta * theta
                sin2 = t2 * (1.0 - t2 * (1.0 / 3.0 - t2 * (2.0 / 45.0)))
                return x + inv_a * sin2

            t_poly = bench_fn(pytorch_poly)
            print(f"    {'PyTorch (polynomial)':25s} {t_poly:8.3f} ms  speedup: {t_pt/t_poly:.2f}x")

            # ── Triton ──
            try:
                from snac.kernels.triton_snake import snake_triton
                # Warmup (includes JIT compile)
                snake_triton(x, alpha)
                t_triton = bench_fn(lambda: snake_triton(x, alpha))
                print(f"    {'Triton':25s} {t_triton:8.3f} ms  speedup: {t_pt/t_triton:.2f}x")

                # Verify correctness
                ref = pytorch_sin()
                out = snake_triton(x, alpha)
                err = (ref.float() - out.float()).abs().max().item()
                print(f"    {'':25s} max error: {err:.6f}")
            except Exception as e:
                print(f"    {'Triton':25s} FAILED: {e}")

            # ── CUDA C++ ──
            try:
                from snac.kernels.cuda_snake import snake_cuda
                snake_cuda(x, alpha)  # warmup + JIT compile
                t_cuda = bench_fn(lambda: snake_cuda(x, alpha))
                print(f"    {'CUDA C++':25s} {t_cuda:8.3f} ms  speedup: {t_pt/t_cuda:.2f}x")

                out = snake_cuda(x, alpha)
                err = (ref.float() - out.float()).abs().max().item()
                print(f"    {'':25s} max error: {err:.6f}")
            except Exception as e:
                print(f"    {'CUDA C++':25s} FAILED: {e}")

            # ── TileLang ──
            try:
                from snac.kernels.tilelang_snake import snake_tilelang
                snake_tilelang(x, alpha)  # warmup + compile
                t_tl = bench_fn(lambda: snake_tilelang(x, alpha))
                print(f"    {'TileLang':25s} {t_tl:8.3f} ms  speedup: {t_pt/t_tl:.2f}x")

                out = snake_tilelang(x, alpha)
                err = (ref.float() - out.float()).abs().max().item()
                print(f"    {'':25s} max error: {err:.6f}")
            except Exception as e:
                print(f"    {'TileLang':25s} FAILED: {e}")

            # ── CuTe DSL ──
            try:
                from snac.kernels.cute_snake import snake_cute, CUTE_AVAILABLE
                if not CUTE_AVAILABLE:
                    print(f"    {'CuTe DSL':25s} NOT AVAILABLE (fallback to PyTorch)")
                else:
                    snake_cute(x, alpha)  # warmup
                    t_cute = bench_fn(lambda: snake_cute(x, alpha))
                    print(f"    {'CuTe DSL':25s} {t_cute:8.3f} ms  speedup: {t_pt/t_cute:.2f}x")

                    out = snake_cute(x, alpha)
                    err = (ref.float() - out.float()).abs().max().item()
                    print(f"    {'':25s} max error: {err:.6f}")
            except Exception as e:
                print(f"    {'CuTe DSL':25s} FAILED: {e}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
