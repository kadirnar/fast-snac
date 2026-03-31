# Fast-SNAC

Fast inference engine for [SNAC](https://github.com/hubertsiuzdak/snac), a hierarchical neural audio codec that compresses audio using multi-scale residual vector quantization. This library accelerates SNAC decode up to **3.7x** on NVIDIA GPUs through graph-level optimizations — with no quality loss at FP32 and no changes to model weights.

## Benchmark

NVIDIA H100 PCIe | `hubertsiuzdak/snac_24khz` | 100s audio @ 24kHz

| Method | Layer | Latency | Speedup | Optimizations |
|--------|-------|:-------:|:-------:|:--------------|
| FP16 + compile | decode (1s) | **0.93 ms** | **3.72x** | `torch.compile`, FP16 autocast |
| FP16 + compile | decode (6s) | **2.91 ms** | **1.96x** | `torch.compile`, FP16 autocast |
| FP16 + compile | decode (100s) | **42.72 ms** | **1.67x** | `torch.compile`, FP16 autocast |
| Triton kernel | Snake Encoder (1024ch) | **21 us** | **6.16x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Triton kernel | Snake Decoder (1536ch) | **41 us** | **5.64x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |

## Quick Start

```bash
pip install git+https://github.com/kadirnar/fast-snac.git
```

```python
from snac import SNAC
from snac.optimize import optimize_snac_native
import torch

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
audio = torch.randn(1, 1, 24000, device="cuda")
codes = model.encode(audio)

# FP32 — zero quality loss
decode_fn = optimize_snac_native(model, codes, dtype="fp32")
audio_hat = decode_fn(codes)

# FP16 — fastest
decode_fn = optimize_snac_native(model, codes, dtype="fp16")
audio_hat = decode_fn(codes)

# Triton Snake kernel — drop-in replacement
from snac.kernels.triton_snake import snake_triton
y = snake_triton(x, alpha)
```

## Requirements

- PyTorch 2.6+
- NVIDIA GPU (Hopper/Ampere)

## License

MIT
