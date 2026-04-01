"""Triton kernel for Snake activation: x + (1/α)·sin²(α·x)

Optimized with:
  - fast_sinf / fast_dividef: CUDA fast-math intrinsics
  - Batch in grid: single kernel launch for entire batch (no Python loop)
  - Autotune: BLOCK_T 512-4096, num_warps 4-16
  - torch.library.custom_op: graph-break-free with torch.compile
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_T": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 4096}, num_warps=16, num_stages=2),
    ],
    key=["T"],
)
@triton.jit
def _snake_kernel(
    X_ptr, A_ptr, Y_ptr,
    C, T,
    stride_xb, stride_yb,  # batch strides
    BLOCK_T: tl.constexpr,
):
    """Snake: y = x + (1/α)·sin²(α·x). Grid: (C, cdiv(T, BLOCK_T), B)"""
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    b = tl.program_id(2)

    alpha = tl.load(A_ptr + c).to(tl.float32)
    inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)

    offs = t_start + tl.arange(0, BLOCK_T)
    mask = offs < T
    idx = b * stride_xb + c * T + offs

    x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    ax = alpha * x
    sin_ax = tl.extra.cuda.libdevice.fast_sinf(ax)
    y = x + inv_alpha * sin_ax * sin_ax

    out_idx = b * stride_yb + c * T + offs
    tl.store(Y_ptr + out_idx, y.to(X_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("snac::snake_triton", mutates_args=())
def snake_triton(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation using Triton kernel."""
    squeeze = False
    if x.ndim == 4:
        squeeze = True
        x = x.squeeze(2)

    B, C, T = x.shape
    x = x.contiguous()
    alpha_flat = alpha.flatten().float()
    output = torch.empty_like(x)

    grid = lambda meta: (C, triton.cdiv(T, meta['BLOCK_T']), B)
    _snake_kernel[grid](
        x, alpha_flat, output,
        C, T,
        x.stride(0), output.stride(0),
    )

    if squeeze:
        output = output.unsqueeze(2)
    return output


@snake_triton.register_fake
def _snake_triton_fake(x, alpha):
    return torch.empty_like(x)
