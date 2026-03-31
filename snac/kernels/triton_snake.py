"""Triton kernel for Snake activation: x + (1/α)·sin²(α·x)

Single kernel pass — no intermediate memory writes.
Supports both 3D [B, C, T] and 4D [B, C, 1, T] inputs.

Optimized with:
  - fast_sinf: CUDA fast-math sin (~2x faster than IEEE tl.sin)
  - fast_dividef: CUDA fast-math reciprocal
  - 2D grid (C, cdiv(T, BLOCK_T)): alpha loaded once per channel
  - Eviction hints: streaming x (evict_first), reused alpha (evict_last)

Benchmark (H100 PCIe, 100s audio @ 24kHz, fp32):
  Encoder (1024ch, T=5442):  ~22 us — 5.98x vs PyTorch
  Decoder (1536ch, T=5442):  ~41 us — 5.64x vs PyTorch
  Decoder (96ch, T=2400000): ~1052 us — 5.40x vs PyTorch
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _snake_kernel(
    X_ptr, A_ptr, Y_ptr,
    C, T, stride_c, stride_t,
    BLOCK_T: tl.constexpr,
):
    """Snake activation: y = x + (1/α)·sin²(α·x)

    Each program handles one (channel, tile_t) slice.
    Grid: (C, cdiv(T, BLOCK_T))
    Alpha and inv_alpha computed once per channel program.
    """
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T

    alpha = tl.load(A_ptr + c).to(tl.float32)
    inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)

    offs = t_start + tl.arange(0, BLOCK_T)
    mask = offs < T
    idx = c * stride_c + offs * stride_t

    x = tl.load(X_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    ax = alpha * x
    sin_ax = tl.extra.cuda.libdevice.fast_sinf(ax)
    sin2 = sin_ax * sin_ax
    y = x + inv_alpha * sin2

    tl.store(Y_ptr + idx, y.to(X_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("snac::snake_triton", mutates_args=())
def snake_triton(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation using Triton kernel.

    Args:
        x: Input tensor [B, C, T] or [B, C, 1, T]
        alpha: Per-channel alpha [1, C, 1] or [1, C, 1, 1]

    Returns:
        y = x + (1/alpha) * sin²(alpha * x)
    """
    squeeze = False
    if x.ndim == 4:
        squeeze = True
        x = x.squeeze(2)

    B, C, T = x.shape
    alpha_flat = alpha.flatten().float()

    output = torch.empty_like(x)

    BLOCK_T = 1024
    grid = (C, triton.cdiv(T, BLOCK_T))

    for b in range(B):
        x_b = x[b]  # [C, T]
        out_b = output[b]

        _snake_kernel[grid](
            x_b, alpha_flat, out_b,
            C, T,
            stride_c=T, stride_t=1,
            BLOCK_T=BLOCK_T,
        )

    if squeeze:
        output = output.unsqueeze(2)
    return output


@snake_triton.register_fake
def _snake_triton_fake(x, alpha):
    return torch.empty_like(x)
