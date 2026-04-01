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


def _inject_triton_kernels(model):
    """Replace Snake1d with Triton fast_sinf kernel.
    Replace Snake1d + depthwise Conv1d sequences with fused Triton kernel.
    Replace standalone depthwise Conv1d with Triton kernel.
    Optimize NoiseBlock with cached noise.
    """
    from .kernels.triton_snake import snake_triton
    from .kernels.triton_depthwise_conv import (
        snake_depthwise_conv1d_triton, depthwise_conv1d_triton,
    )

    def _replace(parent):
        children = list(parent.named_children())
        skip_next = set()
        for i, (name, child) in enumerate(children):
            if name in skip_next:
                continue
            ctype = type(child).__name__

            # Fuse Snake1d + depthwise Conv1d when they're sequential
            if ctype == 'Snake1d':
                fused = False
                if isinstance(parent, nn.Sequential) and i + 1 < len(children):
                    next_name, next_child = children[i + 1]
                    if isinstance(next_child, nn.Conv1d) and next_child.groups == next_child.in_channels and next_child.groups > 1:
                        alpha_param = child.alpha
                        conv = next_child
                        class FusedSnakeDepthwiseConv(nn.Module):
                            def __init__(self, alpha, conv_mod):
                                super().__init__()
                                self.alpha = alpha
                                self.weight = conv_mod.weight
                                self.bias = conv_mod.bias
                                self.stride = conv_mod.stride[0]
                                self.padding = conv_mod.padding[0]
                                self.dilation = conv_mod.dilation[0]
                            def forward(self, x):
                                return snake_depthwise_conv1d_triton(
                                    x, self.alpha, self.weight, self.bias,
                                    stride=self.stride, padding=self.padding,
                                    dilation=self.dilation,
                                )
                        fused_mod = FusedSnakeDepthwiseConv(alpha_param, conv)
                        setattr(parent, name, fused_mod)
                        setattr(parent, next_name, nn.Identity())
                        skip_next.add(next_name)
                        fused = True

                if not fused:
                    alpha_param = child.alpha
                    def _make_fwd(a):
                        def fwd(x):
                            return snake_triton(x, a)
                        return fwd
                    child.forward = _make_fwd(alpha_param)

            # Replace standalone depthwise Conv1d with Triton kernel
            elif isinstance(child, nn.Conv1d) and child.groups == child.in_channels and child.groups > 1:
                conv = child
                def _make_dw_fwd(c):
                    w = c.weight
                    b = c.bias
                    s, p, d = c.stride[0], c.padding[0], c.dilation[0]
                    def fwd(x):
                        return depthwise_conv1d_triton(x, w, b, stride=s, padding=p, dilation=d)
                    return fwd
                child.forward = _make_dw_fwd(conv)

            # Optimize NoiseBlock: cache noise tensor
            elif ctype == 'NoiseBlock':
                def _make_noise_fwd(mod):
                    _cached = [None]
                    def fwd(x):
                        B, C, T = x.shape
                        if _cached[0] is None or _cached[0].shape[-1] != T:
                            _cached[0] = torch.randn(B, 1, T, device=x.device, dtype=x.dtype)
                        h = mod.linear(x)
                        return x + _cached[0] * h
                    return fwd
                child.forward = _make_noise_fwd(child)

            else:
                _replace(child)

    _replace(model)


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

def optimize_snac_triton(model, sample_codes, dtype="fp32", use_compile=False):
    """Optimize SNAC decode with Triton kernels.

    Replaces Snake1d with fast_sinf Triton kernel.
    Fuses Snake1d + depthwise Conv1d into single Triton kernel.
    Replaces standalone depthwise Conv1d with Triton kernel.
    Caches NoiseBlock random tensors.
    Strips weight norms. Optionally applies torch.compile on top.
    """
    if isinstance(dtype, str):
        torch_dtype = _DTYPE_MAP.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"Unknown dtype '{dtype}'. Use: fp32, fp16, bf16")
    else:
        torch_dtype = dtype

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    _strip_weight_norm(model)
    _inject_triton_kernels(model)
    model = model.to(torch_dtype).eval()

    if not use_compile:
        # Warmup Triton kernels
        typed_codes = [c.clone() for c in sample_codes]
        with torch.no_grad():
            for _ in range(3):
                model.decode(typed_codes)
        torch.cuda.synchronize()

        def optimized_decode(codes):
            with torch.no_grad():
                return model.decode(codes)
        return optimized_decode

    # Triton + torch.compile (custom_ops prevent graph breaks)
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    dynamo_config.cache_size_limit = 64
    inductor_config.freezing = True
    inductor_config.epilogue_fusion = True
    inductor_config.aggressive_fusion = True
    inductor_config.coordinate_descent_tuning = True

    @torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=False)
    @torch.no_grad()
    def compiled_decode(codes):
        z_q = model.quantizer.from_codes(codes)
        return model.decoder(z_q)

    typed_codes = [c.clone() for c in sample_codes]
    for _ in range(3):
        compiled_decode(typed_codes)
    torch.cuda.synchronize()

    return compiled_decode


def optimize_snac_cudnn(model, sample_codes, dtype="fp16"):
    """Optimize SNAC decode with cuDNN v9 Runtime Fusion.

    Fuses depthwise Conv + Snake into single cuDNN kernels.
    Fuses pointwise Conv + residual add.
    Uses Triton for standalone Snake activations.
    """
    from .kernels.cudnn_fused import CudnnDWConvSnakeGraph, CudnnConvResGraph
    from .kernels.triton_snake import snake_triton

    if isinstance(dtype, str):
        torch_dtype = _DTYPE_MAP.get(dtype.lower())
    else:
        torch_dtype = dtype

    import cudnn
    cudnn_dtype = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
    }.get(torch_dtype, cudnn.data_type.HALF)

    torch.backends.cudnn.benchmark = True
    _strip_weight_norm(model)

    # Convert Conv1d → Conv2d for cuDNN
    def _to_conv2d(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv1d):
                c2d = nn.Conv2d(
                    child.in_channels, child.out_channels,
                    (1, child.kernel_size[0]),
                    stride=(1, child.stride[0]),
                    padding=(0, child.padding[0]),
                    dilation=(1, child.dilation[0]),
                    groups=child.groups,
                    bias=child.bias is not None,
                )
                c2d.weight = nn.Parameter(child.weight.data.unsqueeze(2).to(memory_format=torch.channels_last))
                if child.bias is not None:
                    c2d.bias = child.bias
                setattr(parent, name, c2d)
            elif isinstance(child, nn.ConvTranspose1d):
                ct2d = nn.ConvTranspose2d(
                    child.in_channels, child.out_channels,
                    (1, child.kernel_size[0]),
                    stride=(1, child.stride[0]),
                    padding=(0, child.padding[0]),
                    output_padding=(0, child.output_padding[0]),
                    groups=child.groups,
                    bias=child.bias is not None,
                )
                ct2d.weight = nn.Parameter(child.weight.data.unsqueeze(2).to(memory_format=torch.channels_last))
                if child.bias is not None:
                    ct2d.bias = child.bias
                setattr(parent, name, ct2d)
            else:
                _to_conv2d(child)
    _to_conv2d(model.decoder)  # Only decoder, not quantizer

    # Static noise
    for _, mod in model.named_modules():
        if type(mod).__name__ == 'NoiseBlock':
            def _make_det(m):
                _cached = [None]
                def fwd(x):
                    if _cached[0] is None or _cached[0].shape != (x.shape[0], 1, *x.shape[2:]):
                        _cached[0] = torch.randn(x.shape[0], 1, *x.shape[2:], device=x.device, dtype=x.dtype)
                    return x + _cached[0] * m.linear(x)
                return fwd
            mod.forward = _make_det(mod)

    model = model.to(torch_dtype).eval()

    # Build cuDNN graphs by tracing decode
    fused_graphs = {}  # id(resunit) -> {graphs, params, buffers}

    def _trace_build(mod, x):
        """Trace forward pass and build cuDNN graphs for ResidualUnits."""
        ctype = type(mod).__name__
        if ctype == 'ResidualUnit':
            children = list(mod.block.children())
            # Pattern: Snake → Conv2d(depthwise) → Snake → Conv2d(k=1) → add
            snake0 = children[0]
            conv_dw = children[1]
            snake1 = children[2]
            conv_pw = children[3]

            C = conv_dw.in_channels
            T = x.shape[-1]
            K = conv_dw.kernel_size[1]
            D = conv_dw.dilation[1]
            pad = D * (K - 1) // 2
            is_depthwise = (conv_dw.groups == C)

            if is_depthwise and type(snake0).__name__ == 'Snake1d':
                try:
                    # Snake0 params
                    a0 = snake0.alpha.data.squeeze().float().contiguous()
                    ia0 = (1.0 / (a0 + 1e-9)).contiguous()

                    # Fused DWConv + Snake1 params
                    a1 = snake1.alpha.data.squeeze().float().contiguous()
                    ia1 = (1.0 / (a1 + 1e-9)).contiguous()
                    w_dw = conv_dw.weight.data.to(memory_format=torch.channels_last)
                    b_dw = conv_dw.bias.data.reshape(1, C, 1, 1).to(torch_dtype) if conv_dw.bias is not None else torch.zeros(1, C, 1, 1, device='cuda', dtype=torch_dtype)

                    # Conv(k=1) + residual params
                    w_pw = conv_pw.weight.data.to(memory_format=torch.channels_last)
                    b_pw = conv_pw.bias.data.reshape(1, C, 1, 1).to(torch_dtype) if conv_pw.bias is not None else torch.zeros(1, C, 1, 1, device='cuda', dtype=torch_dtype)

                    g_cs = CudnnDWConvSnakeGraph(C, T, K, D, pad, dtype=cudnn_dtype)
                    g_cr = CudnnConvResGraph(C, T, dtype=cudnn_dtype)

                    buf_sn = torch.empty_like(x)
                    buf_cs = torch.empty_like(x)
                    buf_cr = torch.empty_like(x)

                    fused_graphs[id(mod)] = {
                        'a0': a0, 'ia0': ia0,
                        'a1': a1, 'ia1': ia1,
                        'w_dw': w_dw, 'b_dw': b_dw,
                        'w_pw': w_pw, 'b_pw': b_pw,
                        'g_cs': g_cs, 'g_cr': g_cr,
                        'buf_sn': buf_sn, 'buf_cs': buf_cs, 'buf_cr': buf_cr,
                    }
                except Exception as e:
                    pass  # Fallback to standard forward

            y = x
            for sub in mod.block:
                y = _trace_build(sub, y)
            pad_len = (x.shape[-1] - y.shape[-1]) // 2
            if pad_len > 0:
                x = x[..., pad_len:-pad_len]
            return x + y

        elif ctype == 'Snake1d':
            # Standalone snake: store shape for Triton
            return mod(x)
        elif isinstance(mod, nn.Sequential):
            for sub in mod:
                x = _trace_build(sub, x)
            return x
        else:
            return mod(x)

    # Trace to build graphs
    with torch.no_grad():
        z_q = model.quantizer.from_codes(sample_codes)
        x = z_q.unsqueeze(2).to(memory_format=torch.channels_last)
        for layer in model.decoder.model:
            x = _trace_build(layer, x)

    print(f"Built {len(fused_graphs)} cuDNN fused ResidualUnits")

    # Replace Snake1d with Triton for standalone activations
    for _, mod in model.named_modules():
        if type(mod).__name__ == 'Snake1d':
            a = mod.alpha
            def _make_fwd(a):
                def fwd(x):
                    x3d = x.squeeze(2)
                    out = snake_triton(x3d, a)
                    return out.unsqueeze(2).to(memory_format=torch.channels_last)
                return fwd
            mod.forward = _make_fwd(a)

    # Fused forward
    def _run_fused(mod, x):
        ctype = type(mod).__name__
        if ctype == 'ResidualUnit' and id(mod) in fused_graphs:
            fg = fused_graphs[id(mod)]
            residual = x

            # Snake0 via Triton (3D conversion)
            x3d = x.squeeze(2)
            a0_3d = fg['a0'].unsqueeze(0).unsqueeze(-1)
            sn0 = snake_triton(x3d, a0_3d)
            fg['buf_sn'].copy_(sn0.unsqueeze(2).to(memory_format=torch.channels_last))

            # DWConv + Snake1 via cuDNN fused
            fg['g_cs'](fg['buf_sn'], fg['w_dw'], fg['b_dw'],
                       fg['a1'], fg['ia1'], fg['buf_cs'])

            # Conv(k=1) + residual via cuDNN fused
            fg['g_cr'](fg['buf_cs'], fg['w_pw'], fg['b_pw'],
                       residual, fg['buf_cr'])

            return fg['buf_cr']

        elif isinstance(mod, nn.Sequential):
            for sub in mod:
                x = _run_fused(sub, x)
            return x
        elif ctype == 'ResidualUnit':
            y = x
            for sub in mod.block:
                y = _run_fused(sub, y)
            pad_len = (x.shape[-1] - y.shape[-1]) // 2
            if pad_len > 0:
                x = x[..., pad_len:-pad_len]
            return x + y
        elif isinstance(mod, nn.Tanh):
            return torch.tanh(x)
        else:
            return mod(x)

    def decode_cudnn(codes):
        with torch.no_grad():
            z_q = model.quantizer.from_codes(codes)
            x = z_q.unsqueeze(2).to(memory_format=torch.channels_last)
            for layer in model.decoder.model:
                x = _run_fused(layer, x)
            return x.squeeze(2).squeeze(1)

    # Warmup
    for _ in range(5):
        decode_cudnn(sample_codes)
    torch.cuda.synchronize()

    return decode_cudnn


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
