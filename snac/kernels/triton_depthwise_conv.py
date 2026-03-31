"""Fused Triton kernel: Depthwise Conv1d + Snake activation.

Depthwise conv is channel-independent, making it ideal for Triton fusion.
Each channel's convolution is a simple 1D filter applied independently.
This fuses Snake + DepthwiseConv into a single kernel pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _depthwise_conv1d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, T_in, T_out,
    K,  # kernel size
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

    # Convolution loop over kernel
    for k in range(K):
        t_in = t_offs * stride - padding + k * dilation
        in_mask = mask & (t_in >= 0) & (t_in < T_in)
        x_idx = c * T_in + t_in
        x_val = tl.load(X_ptr + x_idx, mask=in_mask, other=0.0).to(tl.float32)
        w_val = tl.load(W_ptr + c * K + k).to(tl.float32)
        acc += x_val * w_val

    # Add bias
    b_val = tl.load(B_ptr + c).to(tl.float32)
    acc += b_val

    # Store
    y_idx = c * T_out + t_offs
    tl.store(Y_ptr + y_idx, acc.to(X_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _snake_depthwise_conv1d_kernel(
    X_ptr, A_ptr, IA_ptr,  # Snake params
    W_ptr, B_ptr, Y_ptr,   # Conv params
    C, T_in, T_out,
    K, conv_stride, padding, dilation,
    BLOCK_T: tl.constexpr,
):
    """Fused: Snake activation → Depthwise Conv1d in one kernel.

    Eliminates the intermediate tensor between Snake and Conv.
    Grid: (C, cdiv(T_out, BLOCK_T))
    """
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T_out

    # Load Snake params for this channel
    alpha = tl.load(A_ptr + c)
    inv_alpha = tl.load(IA_ptr + c)

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for k in range(K):
        t_in = t_offs * conv_stride - padding + k * dilation
        in_mask = mask & (t_in >= 0) & (t_in < T_in)
        x_idx = c * T_in + t_in

        # Load input and apply Snake inline
        x_val = tl.load(X_ptr + x_idx, mask=in_mask, other=0.0).to(tl.float32)
        ax = alpha * x_val
        sin_ax = tl.sin(ax)
        x_snake = x_val + inv_alpha * sin_ax * sin_ax

        # Convolution
        w_val = tl.load(W_ptr + c * K + k).to(tl.float32)
        acc += x_snake * w_val

    b_val = tl.load(B_ptr + c).to(tl.float32)
    acc += b_val

    y_idx = c * T_out + t_offs
    tl.store(Y_ptr + y_idx, acc.to(X_ptr.dtype.element_ty), mask=mask)


def depthwise_conv1d_triton(x, weight, bias, stride=1, padding=0, dilation=1):
    """Depthwise Conv1d using Triton.

    Args:
        x: [B, C, T] input
        weight: [C, 1, K] depthwise weights
        bias: [C] bias
        stride, padding, dilation: conv parameters
    """
    B, C, T_in = x.shape
    K = weight.shape[2]
    T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    output = torch.empty(B, C, T_out, device=x.device, dtype=x.dtype)
    w_flat = weight.squeeze(1).contiguous()
    b_flat = bias.contiguous()

    BLOCK_T = 1024
    grid = (C, triton.cdiv(T_out, BLOCK_T))

    for b in range(B):
        _depthwise_conv1d_kernel[grid](
            x[b], w_flat, b_flat, output[b],
            C, T_in, T_out, K, stride, padding, dilation,
            BLOCK_T=BLOCK_T, num_warps=8,
        )
    return output


def snake_depthwise_conv1d_triton(x, alpha, weight, bias, stride=1, padding=0, dilation=1):
    """Fused Snake + Depthwise Conv1d using Triton.

    Args:
        x: [B, C, T] input
        alpha: [1, C, 1] Snake alpha
        weight: [C, 1, K] depthwise weights
        bias: [C] bias
    """
    B, C, T_in = x.shape
    K = weight.shape[2]
    T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    alpha_flat = alpha.flatten().float()
    inv_alpha = (1.0 / (alpha_flat + 1e-9))

    output = torch.empty(B, C, T_out, device=x.device, dtype=x.dtype)
    w_flat = weight.squeeze(1).contiguous()
    b_flat = bias.contiguous()

    BLOCK_T = 1024
    grid = (C, triton.cdiv(T_out, BLOCK_T))

    for b in range(B):
        _snake_depthwise_conv1d_kernel[grid](
            x[b], alpha_flat, inv_alpha,
            w_flat, b_flat, output[b],
            C, T_in, T_out, K, stride, padding, dilation,
            BLOCK_T=BLOCK_T, num_warps=8,
        )
    return output
