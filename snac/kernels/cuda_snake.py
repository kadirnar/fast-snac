"""Custom CUDA C++ kernel for Snake activation: x + (1/α)·sin²(α·x)

Uses torch.utils.cpp_extension for JIT compilation.
Single fused kernel — no intermediate memory allocations.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void snake_cuda_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ alpha,
    const float* __restrict__ inv_alpha,
    scalar_t* __restrict__ y,
    int C, int T
) {
    // Grid: (C, cdiv(T, blockDim.x))
    int c = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;

    if (t >= T) return;

    float a = alpha[c];
    float ia = inv_alpha[c];

    int idx = c * T + t;
    float xv = static_cast<float>(x[idx]);

    float ax = a * xv;
    float sin_ax = sinf(ax);
    float sin2 = sin_ax * sin_ax;
    float result = xv + ia * sin2;

    y[idx] = static_cast<scalar_t>(result);
}

torch::Tensor snake_forward_cuda(
    torch::Tensor x,       // [C, T]
    torch::Tensor alpha,   // [C]
    torch::Tensor inv_alpha // [C]
) {
    auto output = torch::empty_like(x);
    int C = x.size(0);
    int T = x.size(1);

    const int threads = 256;
    dim3 blocks(C, (T + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "snake_cuda", ([&] {
            snake_cuda_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                alpha.data_ptr<float>(),
                inv_alpha.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                C, T
            );
        })
    );

    return output;
}
"""

_CPP_SRC = """
torch::Tensor snake_forward_cuda(torch::Tensor x, torch::Tensor alpha, torch::Tensor inv_alpha);
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        os.makedirs("/tmp/snake_cuda_build", exist_ok=True)
        _module = load_inline(
            name="snake_cuda",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["snake_forward_cuda"],
            build_directory="/tmp/snake_cuda_build",
            verbose=False,
        )
    return _module


def snake_cuda(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation using custom CUDA kernel.

    Args:
        x: Input tensor [B, C, T] or [B, C, 1, T]
        alpha: Per-channel alpha [1, C, 1] or [1, C, 1, 1]

    Returns:
        y = x + (1/alpha) * sin²(alpha * x)
    """
    mod = _get_module()

    squeeze = False
    if x.ndim == 4:
        squeeze = True
        x = x.squeeze(2)

    B, C, T = x.shape
    alpha_flat = alpha.flatten().float()
    inv_alpha = (1.0 / (alpha_flat + 1e-9))

    output = torch.empty_like(x)
    for b in range(B):
        output[b] = mod.snake_forward_cuda(x[b], alpha_flat, inv_alpha)

    if squeeze:
        output = output.unsqueeze(2)
    return output
