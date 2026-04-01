"""Benchmark polynomial sin² vs fast_sinf in fused Snake+DWConv kernel."""
import torch
import time
import triton
import triton.language as tl

_PI = 3.141592653589793
_INV_PI = 1.0 / _PI


@triton.jit
def _sinf_kernel(
    X_ptr, A_ptr, W_ptr, B_ptr, Y_ptr,
    C, T_in, T_out, stride_xb, stride_yb,
    K: tl.constexpr, conv_stride, padding, dilation,
    BLOCK_T: tl.constexpr,
):
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    b = tl.program_id(2)
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T_out
    x_base = b * stride_xb + c * T_in
    y_base = b * stride_yb + c * T_out
    alpha = tl.load(A_ptr + c).to(tl.float32)
    inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for k in range(K):
        t_in = t_offs * conv_stride - padding + k * dilation
        in_mask = mask & (t_in >= 0) & (t_in < T_in)
        x_val = tl.load(X_ptr + x_base + t_in, mask=in_mask, other=0.0).to(tl.float32)
        ax = alpha * x_val
        sin_ax = tl.extra.cuda.libdevice.fast_sinf(ax)
        x_snake = x_val + inv_alpha * sin_ax * sin_ax
        w_val = tl.load(W_ptr + c * K + k).to(tl.float32)
        acc += x_snake * w_val
    acc += tl.load(B_ptr + c).to(tl.float32)
    tl.store(Y_ptr + y_base + t_offs, acc.to(X_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _poly_kernel(
    X_ptr, A_ptr, W_ptr, B_ptr, Y_ptr,
    C, T_in, T_out, stride_xb, stride_yb,
    K: tl.constexpr, conv_stride, padding, dilation,
    BLOCK_T: tl.constexpr,
):
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    b = tl.program_id(2)
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T_out
    x_base = b * stride_xb + c * T_in
    y_base = b * stride_yb + c * T_out
    alpha = tl.load(A_ptr + c).to(tl.float32)
    inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)
    PI: tl.constexpr = 3.141592653589793
    INV_PI: tl.constexpr = 0.3183098861837907
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for k in range(K):
        t_in = t_offs * conv_stride - padding + k * dilation
        in_mask = mask & (t_in >= 0) & (t_in < T_in)
        x_val = tl.load(X_ptr + x_base + t_in, mask=in_mask, other=0.0).to(tl.float32)
        ax = alpha * x_val
        theta = ax - PI * tl.extra.cuda.libdevice.nearbyint(ax * INV_PI)
        t2 = theta * theta
        sin2 = t2 * (1.0 - t2 * (0.3333333333 - t2 * 0.0444444444))
        x_snake = x_val + inv_alpha * sin2
        w_val = tl.load(W_ptr + c * K + k).to(tl.float32)
        acc += x_snake * w_val
    acc += tl.load(B_ptr + c).to(tl.float32)
    tl.store(Y_ptr + y_base + t_offs, acc.to(X_ptr.dtype.element_ty), mask=mask)


def bench(kernel, C, T, K, dilation, label, iters=50):
    x = torch.randn(1, C, T, device='cuda', dtype=torch.float16)
    alpha = torch.rand(C, device='cuda', dtype=torch.float32) * 5.0 + 0.5
    w = torch.randn(C, K, device='cuda', dtype=torch.float16)
    b = torch.randn(C, device='cuda', dtype=torch.float16)
    out = torch.empty_like(x)
    padding = ((K - 1) * dilation) // 2
    BLOCK_T = 2048
    grid = (C, triton.cdiv(T, BLOCK_T), 1)

    for _ in range(5):
        kernel[grid](x, alpha, w, b, out, C, T, T, x.stride(0), out.stride(0),
                     K=K, conv_stride=1, padding=padding, dilation=dilation, BLOCK_T=BLOCK_T)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel[grid](x, alpha, w, b, out, C, T, T, x.stride(0), out.stride(0),
                     K=K, conv_stride=1, padding=padding, dilation=dilation, BLOCK_T=BLOCK_T)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / iters * 1000
    print(f'{label}: {t:.3f} ms')
    return t


if __name__ == '__main__':
    print('=== 64ch, T=2.4M, K=7 (bottleneck case) ===')
    for dil in [1, 3, 9]:
        print(f'\n--- dilation={dil} ---')
        t1 = bench(_sinf_kernel, 64, 2400256, 7, dil, 'fast_sinf  ')
        t2 = bench(_poly_kernel, 64, 2400256, 7, dil, 'polynomial ')
        print(f'Speedup: {t1/t2:.2f}x')

    print('\n=== 128ch, T=1.2M, K=7 ===')
    for dil in [1, 3, 9]:
        print(f'\n--- dilation={dil} ---')
        t1 = bench(_sinf_kernel, 128, 1200128, 7, dil, 'fast_sinf  ')
        t2 = bench(_poly_kernel, 128, 1200128, 7, dil, 'polynomial ')
        print(f'Speedup: {t1/t2:.2f}x')
