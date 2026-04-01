"""Fully fused Triton decode kernels for SNAC.

Fuses entire ResidualUnit (Snake → DWConv → Snake → PWConv → residual add)
into minimal kernel launches. Eliminates all intermediate global memory writes.
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 1. Fused Snake + Depthwise Conv1d (K=7, variable dilation)
#    Eliminates intermediate Snake output tensor.
# ============================================================

@triton.jit
def _fused_snake_dwconv_kernel(
    X_ptr, Alpha_ptr, W_ptr, Bias_ptr, Y_ptr,
    C, T_in, T_out,
    K: tl.constexpr,
    stride, padding, dilation,
    BLOCK_T: tl.constexpr,
):
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T_out

    alpha = tl.load(Alpha_ptr + c).to(tl.float32)
    inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for k in range(K):
        t_in = t_offs * stride - padding + k * dilation
        in_mask = mask & (t_in >= 0) & (t_in < T_in)
        x_val = tl.load(X_ptr + c * T_in + t_in, mask=in_mask, other=0.0).to(tl.float32)

        # Inline Snake: x + (1/alpha) * sin²(alpha * x)
        ax = alpha * x_val
        sin_ax = tl.extra.cuda.libdevice.fast_sinf(ax)
        x_snake = x_val + inv_alpha * sin_ax * sin_ax

        w_val = tl.load(W_ptr + c * K + k).to(tl.float32)
        acc += x_snake * w_val

    acc += tl.load(Bias_ptr + c).to(tl.float32)
    tl.store(Y_ptr + c * T_out + t_offs, acc.to(X_ptr.dtype.element_ty), mask=mask)


# ============================================================
# 2. Fused Snake (standalone) — for snake before ConvTranspose
# ============================================================

@triton.jit
def _snake_kernel(
    X_ptr, Alpha_ptr, Y_ptr,
    C, T,
    BLOCK_T: tl.constexpr,
):
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T

    alpha = tl.load(Alpha_ptr + c).to(tl.float32)
    inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)

    idx = c * T + t_offs
    x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    ax = alpha * x
    sin_ax = tl.extra.cuda.libdevice.fast_sinf(ax)
    result = x + inv_alpha * sin_ax * sin_ax
    tl.store(Y_ptr + idx, result.to(X_ptr.dtype.element_ty), mask=mask)


# ============================================================
# 3. Fused Snake → PointwiseConv(k=1) + Residual Add
#    Second half of ResidualUnit: snake2(y) → conv1x1(y) + x
# ============================================================

@triton.jit
def _fused_snake_pwconv_residual_kernel(
    Y_ptr,           # input from dwconv (B, C, T)
    Alpha_ptr,       # snake alpha for this layer
    W_ptr,           # pointwise weight (C, C, 1) - NOT depthwise
    Bias_ptr,        # pointwise bias (C,)
    Residual_ptr,    # original input for skip connection
    Out_ptr,         # output (B, C, T)
    C, T,
    BLOCK_T: tl.constexpr,
):
    """For each output position t, compute:
    out[c, t] = sum_over_c'(snake(y[c', t]) * W[c, c']) + bias[c] + residual[c, t]

    This is a pointwise (1x1) conv which is a per-position matrix multiply.
    For depthwise-like channel counts (96-768), this is small enough to process per-position.
    """
    # This kernel handles one channel of output across BLOCK_T time steps
    c_out = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T

    bias = tl.load(Bias_ptr + c_out).to(tl.float32)
    res = tl.load(Residual_ptr + c_out * T + t_offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Sum over input channels (pointwise conv = matmul per position)
    for c_in in range(C):
        alpha = tl.load(Alpha_ptr + c_in).to(tl.float32)
        inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)

        y_val = tl.load(Y_ptr + c_in * T + t_offs, mask=mask, other=0.0).to(tl.float32)

        # Inline Snake
        ay = alpha * y_val
        sin_ay = tl.extra.cuda.libdevice.fast_sinf(ay)
        y_snake = y_val + inv_alpha * sin_ay * sin_ay

        w = tl.load(W_ptr + c_out * C + c_in).to(tl.float32)
        acc += y_snake * w

    result = acc + bias + res
    tl.store(Out_ptr + c_out * T + t_offs, result.to(Y_ptr.dtype.element_ty), mask=mask)


# ============================================================
# 4. Fused NoiseBlock: x + randn * linear(x)
# ============================================================

@triton.jit
def _noise_block_kernel(
    X_ptr, Noise_ptr, W_ptr, Out_ptr,
    C, T,
    BLOCK_T: tl.constexpr,
):
    """NoiseBlock: out = x + noise * (W @ x) where W is pointwise (k=1, no bias)."""
    c = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T

    noise = tl.load(Noise_ptr + t_offs, mask=mask, other=0.0).to(tl.float32)
    x_val = tl.load(X_ptr + c * T + t_offs, mask=mask, other=0.0).to(tl.float32)

    # Pointwise conv: h[c,t] = sum_c'(W[c,c'] * x[c',t])
    # For NoiseBlock, W has no bias
    h = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for c_in in range(C):
        x_in = tl.load(X_ptr + c_in * T + t_offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + c * C + c_in).to(tl.float32)
        h += x_in * w

    result = x_val + noise * h
    tl.store(Out_ptr + c * T + t_offs, result.to(X_ptr.dtype.element_ty), mask=mask)


# ============================================================
# 5. Final: Snake + Conv1d(96→1, k=7) + Tanh
# ============================================================

@triton.jit
def _fused_snake_conv_tanh_kernel(
    X_ptr, Alpha_ptr, W_ptr, Bias_ptr, Out_ptr,
    C_in, T_in,
    K: tl.constexpr,
    padding,
    BLOCK_T: tl.constexpr,
):
    """Fused final layer: Snake(96ch) → Conv1d(96→1, k=7) → Tanh."""
    t_start = tl.program_id(0) * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    mask = t_offs < T_in

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for c in range(C_in):
        alpha = tl.load(Alpha_ptr + c).to(tl.float32)
        inv_alpha = tl.extra.cuda.libdevice.fast_dividef(1.0, alpha)

        for k in range(K):
            t_in = t_offs - padding + k
            in_mask = mask & (t_in >= 0) & (t_in < T_in)

            x_val = tl.load(X_ptr + c * T_in + t_in, mask=in_mask, other=0.0).to(tl.float32)
            ax = alpha * x_val
            sin_ax = tl.extra.cuda.libdevice.fast_sinf(ax)
            x_snake = x_val + inv_alpha * sin_ax * sin_ax

            w = tl.load(W_ptr + c * K + k).to(tl.float32)
            acc += x_snake * w

    acc += tl.load(Bias_ptr).to(tl.float32)

    # Tanh
    # tanh(x) = 2*sigmoid(2x) - 1, or use libdevice
    exp2x = tl.extra.cuda.libdevice.fast_expf(-2.0 * acc)
    result = (1.0 - exp2x) / (1.0 + exp2x)
    tl.store(Out_ptr + t_offs, result.to(X_ptr.dtype.element_ty), mask=mask)


# ============================================================
# Python wrappers
# ============================================================

def fused_snake_dwconv(x, alpha, weight, bias, stride=1, padding=0, dilation=1):
    """Fused Snake + Depthwise Conv1d."""
    B, C, T_in = x.shape
    K = weight.shape[2]
    T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    output = torch.empty(B, C, T_out, device=x.device, dtype=x.dtype)
    alpha_flat = alpha.flatten().float()
    w_flat = weight.squeeze(1).contiguous()
    b_flat = bias.contiguous() if bias is not None else torch.zeros(C, device=x.device, dtype=x.dtype)
    BLOCK_T = 1024
    grid = (C, triton.cdiv(T_out, BLOCK_T))
    for b in range(B):
        _fused_snake_dwconv_kernel[grid](
            x[b], alpha_flat, w_flat, b_flat, output[b],
            C, T_in, T_out, K, stride, padding, dilation,
            BLOCK_T=BLOCK_T,
        )
    return output


def snake_activation(x, alpha):
    """Standalone Snake activation."""
    B, C, T = x.shape
    output = torch.empty_like(x)
    alpha_flat = alpha.flatten().float()
    BLOCK_T = 1024
    grid = (C, triton.cdiv(T, BLOCK_T))
    for b in range(B):
        _snake_kernel[grid](x[b], alpha_flat, output[b], C, T, BLOCK_T=BLOCK_T)
    return output


def fused_snake_pwconv_residual(y, alpha, weight, bias, residual):
    """Fused Snake + pointwise Conv1d + residual add."""
    B, C, T = y.shape
    output = torch.empty_like(y)
    alpha_flat = alpha.flatten().float()
    w_flat = weight.squeeze(-1).contiguous()  # (C, C)
    b_flat = bias.contiguous() if bias is not None else torch.zeros(C, device=y.device, dtype=y.dtype)
    BLOCK_T = 256  # Smaller because inner loop over C channels
    grid = (C, triton.cdiv(T, BLOCK_T))
    for b in range(B):
        _fused_snake_pwconv_residual_kernel[grid](
            y[b], alpha_flat, w_flat, b_flat, residual[b], output[b],
            C, T, BLOCK_T=BLOCK_T,
        )
    return output


def fused_final_snake_conv_tanh(x, alpha, weight, bias):
    """Fused Snake(96) + Conv1d(96→1, k=7, pad=3) + Tanh."""
    B, C, T = x.shape
    K = weight.shape[2]
    output = torch.empty(B, 1, T, device=x.device, dtype=x.dtype)
    alpha_flat = alpha.flatten().float()
    # weight: (1, C, K) -> flatten to (C*K,) for indexed access
    w_flat = weight.squeeze(0).contiguous()  # (C, K)
    b_flat = bias.contiguous()
    BLOCK_T = 1024
    grid = (triton.cdiv(T, BLOCK_T),)
    for b in range(B):
        _fused_snake_conv_tanh_kernel[grid](
            x[b], alpha_flat, w_flat, b_flat, output[b, 0],
            C, T, K, 3,  # padding=3
            BLOCK_T=BLOCK_T,
        )
    return output
