"""
SNAC inference optimization module.

Optimization levels:
  Level 1: Structural (weight norm removal, Conv2d channels_last, polynomial Snake)
  Level 2: + torch.compile with max-autotune (epilogue fusion, Triton conv templates)
  Level 3: + CUDA graph capture (zero CPU overhead)

Usage:
    from snac import SNAC
    from snac.optimize import optimize_snac

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    codes = model.encode(torch.randn(1, 1, 24000, device="cuda"))

    decode_fn = optimize_snac(model, codes, dtype="fp16")
    audio = decode_fn(codes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm


_PI = 3.141592653589793
_INV_PI = 1.0 / _PI

_DTYPE_MAP = {
    "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Structural optimizations
# ---------------------------------------------------------------------------

def _strip_weight_norm(model):
    """Remove weight_norm from all modules (fuse weight_g/weight_v)."""
    for _, mod in model.named_modules():
        try:
            remove_weight_norm(mod)
        except (ValueError, AttributeError):
            pass


def _convert_conv1d_to_conv2d(model):
    """Replace Conv1d/ConvTranspose1d with Conv2d/ConvTranspose2d in channels_last.
    Replace Snake1d with polynomial sin² approximation.
    """
    def replace_conv1d(conv1d):
        c2d = nn.Conv2d(
            conv1d.in_channels, conv1d.out_channels,
            (1, conv1d.kernel_size[0]),
            stride=(1, conv1d.stride[0]),
            padding=(0, conv1d.padding[0]),
            dilation=(1, conv1d.dilation[0]),
            groups=conv1d.groups,
            bias=conv1d.bias is not None,
        )
        c2d.weight = nn.Parameter(
            conv1d.weight.data.unsqueeze(2).to(memory_format=torch.channels_last)
        )
        if conv1d.bias is not None:
            c2d.bias = conv1d.bias
        return c2d

    def replace_ct1d(ct1d):
        ct2d = nn.ConvTranspose2d(
            ct1d.in_channels, ct1d.out_channels,
            (1, ct1d.kernel_size[0]),
            stride=(1, ct1d.stride[0]),
            padding=(0, ct1d.padding[0]),
            output_padding=(0, ct1d.output_padding[0]),
            dilation=(1, ct1d.dilation[0]),
            groups=ct1d.groups,
            bias=ct1d.bias is not None,
        )
        ct2d.weight = nn.Parameter(
            ct1d.weight.data.unsqueeze(2).to(memory_format=torch.channels_last)
        )
        if ct1d.bias is not None:
            ct2d.bias = ct1d.bias
        return ct2d

    def _replace(parent):
        for name, child in list(parent.named_children()):
            ctype = type(child).__name__
            if isinstance(child, nn.Conv1d):
                setattr(parent, name, replace_conv1d(child))
            elif isinstance(child, nn.ConvTranspose1d):
                setattr(parent, name, replace_ct1d(child))
            elif ctype == 'Snake1d':
                a4 = child.alpha.data.unsqueeze(-1)
                child.register_buffer('_a4', a4)
                child.register_buffer('_inv_a4', 1.0 / (a4 + 1e-9))
                def make_snake_poly(mod):
                    def fwd(x):
                        ax = mod._a4 * x
                        theta = ax - _PI * torch.round(ax * _INV_PI)
                        t2 = theta * theta
                        sin2 = t2 * (1.0 - t2 * (1.0 / 3.0 - t2 * (2.0 / 45.0)))
                        return x + mod._inv_a4 * sin2
                    return fwd
                child.forward = make_snake_poly(child)
            else:
                _replace(child)

    _replace(model)


def _make_noise_deterministic(model):
    """Replace NoiseBlock's random noise with static cached tensor."""
    for _, mod in model.named_modules():
        if type(mod).__name__ == 'NoiseBlock':
            def make_det(m):
                _cached = [None]
                def fwd(x):
                    B, C, H, T = x.shape
                    if _cached[0] is None or _cached[0].shape != (B, 1, H, T):
                        _cached[0] = torch.randn((B, 1, H, T), device=x.device, dtype=x.dtype)
                    return x + _cached[0] * m.linear(x)
                return fwd
            mod.forward = make_det(mod)


# ---------------------------------------------------------------------------
# Optimized decode function builder
# ---------------------------------------------------------------------------

def _run_module(mod, x):
    """Execute a module, inlining Sequential/ResidualUnit to avoid graph breaks."""
    ctype = type(mod).__name__
    if ctype == 'ResidualUnit':
        y = mod.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y
    elif ctype == 'LocalMHA':
        x_3d = x.squeeze(2)
        out_3d = mod(x_3d)
        return out_3d.unsqueeze(2).to(memory_format=torch.channels_last)
    elif isinstance(mod, nn.Tanh):
        return torch.tanh(x)
    elif isinstance(mod, nn.Sequential):
        for sub in mod:
            x = _run_module(sub, x)
        return x
    else:
        return mod(x)


def _build_decode_fn(model):
    """Build optimized decode: quantizer.from_codes → decoder in 4D channels_last."""

    def decode_optimized(codes):
        z_q = 0.0
        for i in range(model.quantizer.n_codebooks):
            q = model.quantizer.quantizers[i]
            z_p_i = F.embedding(codes[i], q.codebook.weight).transpose(1, 2)
            z_p_4d = z_p_i.unsqueeze(2).to(memory_format=torch.channels_last)
            z_q_i = q.out_proj(z_p_4d).squeeze(2)
            z_q_i = z_q_i.repeat_interleave(q.stride, dim=-1)
            z_q = z_q + z_q_i

        x = z_q.unsqueeze(2).to(memory_format=torch.channels_last)
        for layer in model.decoder.model:
            x = _run_module(layer, x)
        return x.squeeze(2)

    return decode_optimized


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def optimize_snac_native(model, sample_codes, dtype="fp32", use_cuda_graph=False):
    """Optimize SNAC decode — lightweight path (no Conv2d conversion).

    Best for large audio (10s+). Strips weight norms, compiles with
    max-autotune, keeps native Conv1d.
    """
    if isinstance(dtype, str):
        torch_dtype = _DTYPE_MAP.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"Unknown dtype '{dtype}'. Use: fp32, fp16, bf16")
    else:
        torch_dtype = dtype

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Strip weight norms only (no Conv2d conversion)
    _strip_weight_norm(model)
    model = model.to(torch_dtype).eval()

    # Compile the decode method directly
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    dynamo_config.cache_size_limit = 64
    inductor_config.freezing = True
    inductor_config.epilogue_fusion = True
    inductor_config.aggressive_fusion = True
    inductor_config.coordinate_descent_tuning = True

    @torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=False)
    def compiled_decode(codes):
        z_q = model.quantizer.from_codes(codes)
        return model.decoder(z_q)

    # Warmup
    typed_codes = [c.clone() for c in sample_codes]
    with torch.no_grad():
        for _ in range(3):
            compiled_decode(typed_codes)
    torch.cuda.synchronize()

    if not use_cuda_graph:
        def optimized_decode(codes):
            with torch.no_grad():
                return compiled_decode(codes)
        return optimized_decode

    # CUDA graph
    graph = torch.cuda.CUDAGraph()
    static_codes = [c.clone() for c in typed_codes]
    with torch.no_grad():
        with torch.cuda.graph(graph):
            static_output = compiled_decode(static_codes)
    torch.cuda.synchronize()

    def graph_decode(codes):
        for i in range(len(codes)):
            static_codes[i].copy_(codes[i])
        graph.replay()
        return static_output.clone()

    return graph_decode


def optimize_snac(model, sample_codes, dtype="fp32", use_cuda_graph=False):
    """Optimize SNAC decode with torch.compile max-autotune.

    Args:
        model: SNAC model on CUDA (.cuda().eval())
        sample_codes: Sample codes list for warmup
        dtype: "fp32" | "fp16" | "bf16"
        use_cuda_graph: Capture into CUDA graph (fixed shape only)

    Returns:
        decode_fn(codes) → audio tensor [1, 1, T]
    """
    if isinstance(dtype, str):
        torch_dtype = _DTYPE_MAP.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"Unknown dtype '{dtype}'. Use: fp32, fp16, bf16")
    else:
        torch_dtype = dtype

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # 1. Structural optimizations
    _strip_weight_norm(model)
    _convert_conv1d_to_conv2d(model)
    _make_noise_deterministic(model)

    # 2. Build decode function + cast precision
    decode_fn = _build_decode_fn(model)
    model = model.to(torch_dtype).eval()

    # 3. Aggressive Inductor configuration
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    dynamo_config.cache_size_limit = 64
    inductor_config.freezing = True
    inductor_config.epilogue_fusion = True
    inductor_config.aggressive_fusion = True
    inductor_config.conv_1x1_as_mm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.layout_optimization = True
    inductor_config.force_layout_optimization = True

    # 4. Compile with max-autotune (enables Triton conv templates + auto CUDA graphs)
    compiled = torch.compile(
        decode_fn,
        mode='max-autotune-no-cudagraphs' if not use_cuda_graph else 'max-autotune',
        fullgraph=True,
    )

    # 5. Warmup (triggers compilation + autotuning)
    typed_codes = [c.clone() for c in sample_codes]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(5):
                compiled(typed_codes)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    if not use_cuda_graph:
        def optimized_decode(codes):
            with torch.no_grad():
                return compiled(codes)
        return optimized_decode

    # 6. Manual CUDA graph capture
    graph = torch.cuda.CUDAGraph()
    static_codes = [c.clone() for c in typed_codes]
    with torch.no_grad():
        with torch.cuda.graph(graph):
            static_output = compiled(static_codes)
    torch.cuda.synchronize()

    def graph_decode(codes):
        for i in range(len(codes)):
            static_codes[i].copy_(codes[i])
        graph.replay()
        return static_output.clone()

    return graph_decode
