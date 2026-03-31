"""Triton kernel for Snake activation: x + (1/α)·sin²(α·x)

Single kernel pass — no intermediate memory writes.
Supports both 3D [B, C, T] and 4D [B, C, 1, T] inputs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _snake_kernel(
    X_ptr, A_ptr, IA_ptr, Y_ptr,
    C, T, stride_c, stride_t,
    BLOCK_T: tl.constexpr,
):
    """Snake activation: y = x + (1/α)·sin²(α·x)

    Each program handles one (batch, channel, tile_t) slice.
    Grid: (C, cdiv(T, BLOCK_T))
    """
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T

    alpha = tl.load(A_ptr + c)
    inv_alpha = tl.load(IA_ptr + c)

    offs = t_start + tl.arange(0, BLOCK_T)
    mask = offs < T
    idx = c * stride_c + offs * stride_t

    x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    ax = alpha * x
    sin_ax = tl.sin(ax)
    sin2 = sin_ax * sin_ax
    y = x + inv_alpha * sin2

    tl.store(Y_ptr + idx, y.to(X_ptr.dtype.element_ty), mask=mask)


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
    inv_alpha = (1.0 / (alpha_flat + 1e-9))

    output = torch.empty_like(x)

    # Handle batched input by processing as B*C channels
    for b in range(B):
        x_b = x[b]  # [C, T]
        out_b = output[b]

        BLOCK_T = 1024
        grid = (C, triton.cdiv(T, BLOCK_T))
        _snake_kernel[grid](
            x_b, alpha_flat, inv_alpha, out_b,
            C, T,
            stride_c=T, stride_t=1,
            BLOCK_T=BLOCK_T,
            num_warps=8,
        )

    if squeeze:
        output = output.unsqueeze(2)
    return output
