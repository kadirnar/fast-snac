"""Fused Triton kernels: Depthwise Conv1d + Snake activation.

Depthwise conv is channel-independent, making it ideal for Triton fusion.
Each channel's convolution is a simple 1D filter applied independently.

Optimized with:
  - fast_sinf / fast_dividef: CUDA fast-math intrinsics
  - In-kernel inv_alpha: eliminates one memory load
  - Eviction hints: streaming input, cached weights
  - torch.library.custom_op: compatible with torch.compile (no graph breaks)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 4096}, num_warps=8, num_stages=2),
    ],
    key=["T_out"],
)
@triton.jit
def _depthwise_conv1d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, T_in, T_out,
    K: tl.constexpr,
    stride,
    padding,
    dilation,
    BLOCK_T: tl.constexpr,
):
    """Depthwise Conv1d: each channel filtered independently.
    Grid: (C, cdiv(T_out, BLOCK_T))
    """
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for k in range(K):
        t_in = t_offs * stride - padding + k * dilation
        in_mask = mask & (t_in >= 0) & (t_in < T_in)
        x_idx = c * T_in + t_in
        x_val = tl.load(X_ptr + x_idx, mask=in_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        w_val = tl.load(W_ptr + c * K + k).to(tl.float32)
        acc += x_val * w_val

    b_val = tl.load(B_ptr + c).to(tl.float32)
    acc += b_val

    y_idx = c * T_out + t_offs
    tl.store(Y_ptr + y_idx, acc.to(X_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 4096}, num_warps=8, num_stages=2),
    ],
    key=["T_out"],
)
@triton.jit
def _snake_depthwise_conv1d_kernel(
    X_ptr, A_ptr,
    W_ptr, B_ptr, Y_ptr,
    C, T_in, T_out,
    K: tl.constexpr,
    conv_stride, padding, dilation,
    BLOCK_T: tl.constexpr,
):
    """Fused: Snake activation -> Depthwise Conv1d in one kernel.
    K is constexpr so Triton unrolls the conv loop.
    Autotuned BLOCK_T and num_warps for best memory throughput.
    Grid: (C, cdiv(T_out, BLOCK_T))
    """
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T_out

    alpha = tl.load(A_ptr + c).to(tl.float32)
    inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for k in range(K):
        t_in = t_offs * conv_stride - padding + k * dilation
        in_mask = mask & (t_in >= 0) & (t_in < T_in)
        x_idx = c * T_in + t_in

        x_val = tl.load(X_ptr + x_idx, mask=in_mask, other=0.0).to(tl.float32)
        ax = alpha * x_val
        sin_ax = tl.extra.cuda.libdevice.fast_sinf(ax)
        x_snake = x_val + inv_alpha * sin_ax * sin_ax

        w_val = tl.load(W_ptr + c * K + k).to(tl.float32)
        acc += x_snake * w_val

    b_val = tl.load(B_ptr + c).to(tl.float32)
    acc += b_val

    y_idx = c * T_out + t_offs
    tl.store(Y_ptr + y_idx, acc.to(X_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("snac::depthwise_conv1d_triton", mutates_args=())
def depthwise_conv1d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """Depthwise Conv1d using Triton."""
    B, C, T_in = x.shape
    K = weight.shape[2]
    T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    output = torch.empty(B, C, T_out, device=x.device, dtype=x.dtype)
    w_flat = weight.squeeze(1).contiguous()
    b_flat = bias.contiguous()

    grid = lambda meta: (C, triton.cdiv(T_out, meta['BLOCK_T']))

    for b in range(B):
        _depthwise_conv1d_kernel[grid](
            x[b], w_flat, b_flat, output[b],
            C, T_in, T_out, K, stride, padding, dilation,
        )
    return output


@depthwise_conv1d_triton.register_fake
def _depthwise_conv1d_fake(x, weight, bias, stride=1, padding=0, dilation=1):
    K = weight.shape[2]
    T_out = (x.shape[2] + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    return torch.empty(x.shape[0], x.shape[1], T_out, device=x.device, dtype=x.dtype)


@torch.library.custom_op("snac::snake_depthwise_conv1d_triton", mutates_args=())
def snake_depthwise_conv1d_triton(x: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """Fused Snake + Depthwise Conv1d using Triton."""
    B, C, T_in = x.shape
    K = weight.shape[2]
    T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    alpha_flat = alpha.flatten().float()

    output = torch.empty(B, C, T_out, device=x.device, dtype=x.dtype)
    w_flat = weight.squeeze(1).contiguous()
    b_flat = bias.contiguous()

    grid = lambda meta: (C, triton.cdiv(T_out, meta['BLOCK_T']))

    for b in range(B):
        _snake_depthwise_conv1d_kernel[grid](
            x[b], alpha_flat,
            w_flat, b_flat, output[b],
            C, T_in, T_out, K, stride, padding, dilation,
        )
    return output


@snake_depthwise_conv1d_triton.register_fake
def _snake_depthwise_conv1d_fake(x, alpha, weight, bias, stride=1, padding=0, dilation=1):
    K = weight.shape[2]
    T_out = (x.shape[2] + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    return torch.empty(x.shape[0], x.shape[1], T_out, device=x.device, dtype=x.dtype)
