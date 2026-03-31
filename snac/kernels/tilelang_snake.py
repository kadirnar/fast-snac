"""TileLang kernel for Snake activation: x + (1/α)·sin²(α·x)

Uses TileLang's tiled programming model for GPU execution.
"""

import torch
import tilelang
import tilelang.language as T


def _build_snake_kernel(C, T_size, dtype="float32"):
    """Build a TileLang Snake activation kernel for given dimensions."""
    BLOCK_T = min(1024, T_size)

    @T.prim_func
    def snake_kernel(
        X: T.Buffer((C, T_size), dtype),
        Alpha: T.Buffer((C,), "float32"),
        InvAlpha: T.Buffer((C,), "float32"),
        Y: T.Buffer((C, T_size), dtype),
    ):
        with T.Kernel(C, T.ceildiv(T_size, BLOCK_T), threads=256) as (bx, by):
            for t in T.Parallel(BLOCK_T):
                idx = by * BLOCK_T + t
                if idx < T_size:
                    xv = T.cast(X[bx, idx], "float32")
                    a = Alpha[bx]
                    ia = InvAlpha[bx]
                    ax = a * xv
                    sin_ax = T.sin(ax)
                    sin2 = sin_ax * sin_ax
                    result = xv + ia * sin2
                    Y[bx, idx] = T.cast(result, dtype)

    return snake_kernel


_kernel_cache = {}


def snake_tilelang(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation using TileLang kernel.

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

    B, C, T_size = x.shape
    alpha_flat = alpha.flatten().float()
    inv_alpha = (1.0 / (alpha_flat + 1e-9))

    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    dtype_str = dtype_map[x.dtype]

    # Cache compiled kernels by (C, T, dtype)
    key = (C, T_size, dtype_str)
    if key not in _kernel_cache:
        kernel_fn = _build_snake_kernel(C, T_size, dtype_str)
        # Compile with output at index 3 (Y buffer)
        _kernel_cache[key] = tilelang.compile(kernel_fn, out_idx=[3])

    mod = _kernel_cache[key]

    output = torch.empty_like(x)
    for b in range(B):
        output[b] = mod(x[b], alpha_flat, inv_alpha)

    if squeeze:
        output = output.unsqueeze(2)
    return output
