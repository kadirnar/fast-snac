"""Fully fused SNAC decode using Triton kernels.

Auto-detects decoder structure and replaces ResidualUnit forward
with fused Triton kernels. ConvTranspose stays on cuDNN.
"""

import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm

from .kernels.triton_fused_decode import (
    fused_snake_dwconv,
    snake_activation,
    fused_snake_pwconv_residual,
    fused_final_snake_conv_tanh,
)


def build_fused_decode(model, sample_codes, dtype=torch.float16):
    """Build fully fused decode. Auto-detects decoder structure."""

    # Strip weight norms
    for _, mod in model.named_modules():
        try:
            remove_weight_norm(mod)
        except (ValueError, AttributeError):
            pass

    model = model.to(dtype).eval()
    torch.backends.cudnn.benchmark = True

    decoder = model.decoder

    # Pre-cache noise tensors
    noise_cache = {}

    def _get_noise(T, device, dtype):
        if T not in noise_cache:
            noise_cache[T] = torch.randn(1, 1, T, device=device, dtype=dtype)
        return noise_cache[T]

    def fused_residual_unit(x, resunit):
        """Optimized ResidualUnit:
        Snake+DWConv (1 fused Triton) → Snake (Triton) → PWConv (cuDNN) → add
        """
        children = list(resunit.block.children())
        snake0 = children[0]
        conv_dw = children[1]
        snake1 = children[2]
        conv_pw = children[3]

        residual = x

        # Kernel 1: Snake + DepthwiseConv (fused Triton)
        y = fused_snake_dwconv(
            x, snake0.alpha,
            conv_dw.weight, conv_dw.bias,
            stride=conv_dw.stride[0],
            padding=conv_dw.padding[0],
            dilation=conv_dw.dilation[0],
        )

        # Kernel 2: Snake (Triton)
        y = snake_activation(y, snake1.alpha)

        # Kernel 3: PointwiseConv (cuDNN — tensor core matmul, much faster than Triton loop)
        y = conv_pw(y)

        # Kernel 4: Residual add
        pad = (residual.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            residual = residual[..., pad:-pad]
        return y + residual

    def fused_decoder_block(x, block):
        """Fused DecoderBlock: Snake→ConvTranspose→Noise→3xResUnit."""
        children = list(block.block.children())

        # Snake (Triton)
        x = snake_activation(x, children[0].alpha)

        # ConvTranspose (cuDNN — tensor cores, not worth rewriting)
        x = children[1](x)

        # NoiseBlock (inline)
        if type(children[2]).__name__ == 'NoiseBlock':
            noise = _get_noise(x.shape[-1], x.device, x.dtype)
            h = children[2].linear(x)
            x = x + noise * h
            resunits = children[3:]
        else:
            resunits = children[2:]

        # 3x ResidualUnit (fused Triton)
        for ru in resunits:
            if type(ru).__name__ == 'ResidualUnit':
                x = fused_residual_unit(x, ru)

        return x

    # Identify decoder structure
    layers = list(decoder.model.children())
    dec_blocks = []
    pre_layers = []
    post_snake = None
    post_conv = None

    for layer in layers:
        ctype = type(layer).__name__
        if ctype == 'DecoderBlock':
            dec_blocks.append(layer)
        elif ctype == 'Snake1d' and len(dec_blocks) > 0:
            post_snake = layer
        elif isinstance(layer, (nn.Conv1d, nn.Linear)) and post_snake is not None:
            post_conv = layer
        elif ctype == 'Tanh':
            pass  # handled in fused final
        elif len(dec_blocks) == 0:
            pre_layers.append(layer)

    def decode(codes):
        with torch.no_grad():
            z_q = model.quantizer.from_codes(codes)

            # Pre-layers (depthwise conv + pointwise + optional attention)
            x = z_q
            for layer in pre_layers:
                x = layer(x)

            # DecoderBlocks (fused Triton)
            for block in dec_blocks:
                x = fused_decoder_block(x, block)

            # Final: Snake + Conv + Tanh
            if post_snake is not None and post_conv is not None:
                x = fused_final_snake_conv_tanh(
                    x, post_snake.alpha, post_conv.weight, post_conv.bias,
                )
            else:
                # Fallback
                for layer in layers[len(pre_layers) + len(dec_blocks):]:
                    x = layer(x)

            return x

    # Warmup
    for _ in range(3):
        decode(sample_codes)
    torch.cuda.synchronize()

    return decode
