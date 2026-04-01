"""Benchmark all Snake kernel implementations: PyTorch, Triton, TileLang."""
import os
import time
import gc
import torch

os.environ['HF_HOME'] = '/mnt/kadirnar/huggingface'

from snac.kernels.triton_snake import snake_triton
from snac.kernels.tilelang_snake import snake_tilelang


@torch.jit.script
def snake_pytorch(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


def bench(fn, x, alpha, warmup=10, iters=200):
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


def run_table(dtype_name, torch_dtype):
    sizes = [
        ('Encoder (1024ch)',  1024, 5442),
        ('Decoder (768ch)',   768,  38095),
        ('Decoder (384ch)',   384,  266666),
        ('Decoder (192ch)',   192,  800000),
        ('Decoder (96ch)',    96,   2400000),
    ]

    print(f'\n### Snake Kernel — {dtype_name} (100s audio @ 24kHz)\n')
    print(f'| Layer | PyTorch | Triton | TileLang | Triton speedup | TileLang speedup |')
    print(f'|-------|:-------:|:------:|:--------:|:--------------:|:----------------:|')

    total_pt = total_tr = total_tl = 0

    for label, C, T in sizes:
        x = torch.randn(1, C, T, device='cuda', dtype=torch_dtype)
        alpha = torch.rand(1, C, 1, device='cuda', dtype=torch_dtype) * 5.0 + 0.5

        t_pt = bench(snake_pytorch, x, alpha)
        t_tr = bench(snake_triton, x, alpha)

        try:
            t_tl = bench(snake_tilelang, x, alpha)
        except Exception:
            t_tl = None

        total_pt += t_pt
        total_tr += t_tr
        if t_tl is not None:
            total_tl += t_tl

        tl_val = f'{t_tl:.0f} us' if t_tl else 'FAIL'
        tl_spd = f'**{t_pt/t_tl:.2f}x**' if t_tl else 'N/A'

        print(f'| {label} | {t_pt:.0f} us | **{t_tr:.0f} us** | {tl_val} | **{t_pt/t_tr:.2f}x** | {tl_spd} |')

    tl_tot = f'{total_tl:.0f} us' if total_tl > 0 else 'FAIL'
    tl_spd = f'**{total_pt/total_tl:.2f}x**' if total_tl > 0 else 'N/A'
    print(f'| **Total** | **{total_pt:.0f} us** | **{total_tr:.0f} us** | **{tl_tot}** | **{total_pt/total_tr:.2f}x** | {tl_spd} |')


if __name__ == '__main__':
    print('NVIDIA H100 PCIe | `hubertsiuzdak/snac_24khz` | 100s audio @ 24kHz')
    run_table('FP32', torch.float32)
    gc.collect(); torch.cuda.empty_cache()
    run_table('FP16', torch.float16)
