"""Sweep BLOCK_T and num_warps for the bottleneck fused Snake+DWConv kernel."""
import torch
import time
import triton
import triton.language as tl


@triton.jit
def _kernel(
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


def bench(C, T, K, dilation, block_t, warps, iters=50):
    x = torch.randn(1, C, T, device='cuda', dtype=torch.float16)
    alpha = torch.rand(C, device='cuda', dtype=torch.float32) * 5.0 + 0.5
    w = torch.randn(C, K, device='cuda', dtype=torch.float16)
    b = torch.randn(C, device='cuda', dtype=torch.float16)
    out = torch.empty_like(x)
    padding = ((K - 1) * dilation) // 2
    grid = (C, triton.cdiv(T, block_t), 1)

    for _ in range(5):
        _kernel[grid](x, alpha, w, b, out, C, T, T, x.stride(0), out.stride(0),
                      K=K, conv_stride=1, padding=padding, dilation=dilation,
                      BLOCK_T=block_t, num_warps=warps, num_stages=2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _kernel[grid](x, alpha, w, b, out, C, T, T, x.stride(0), out.stride(0),
                      K=K, conv_stride=1, padding=padding, dilation=dilation,
                      BLOCK_T=block_t, num_warps=warps, num_stages=2)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


if __name__ == '__main__':
    configs = [
        (256, 2), (256, 4), (512, 4), (512, 8),
        (1024, 4), (1024, 8), (1024, 16),
        (2048, 4), (2048, 8), (2048, 16),
        (4096, 8), (4096, 16), (4096, 32),
        (8192, 8), (8192, 16), (8192, 32),
    ]

    for label, C, T in [('64ch T=2.4M', 64, 2400256), ('128ch T=1.2M', 128, 1200128)]:
        print(f'\n=== {label}, K=7, dilation=1 ===')
        best_t = float('inf')
        best_cfg = None
        for bt, nw in configs:
            try:
                t = bench(C, T, 7, 1, bt, nw)
                tag = '  <-- BEST' if t < best_t else ''
                if t < best_t:
                    best_t = t
                    best_cfg = (bt, nw)
                print(f'  BLOCK_T={bt:>5d} warps={nw:>2d}  {t:.3f} ms{tag}')
            except Exception:
                print(f'  BLOCK_T={bt:>5d} warps={nw:>2d}  FAILED')
        print(f'  Best: BLOCK_T={best_cfg[0]} warps={best_cfg[1]} = {best_t:.3f} ms')
