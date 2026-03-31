# Fast-SNAC

Fast inference engine for [SNAC](https://github.com/hubertsiuzdak/snac), a hierarchical neural audio codec that compresses audio using multi-scale residual vector quantization. This library accelerates SNAC decode up to **3.7x** on NVIDIA GPUs through graph-level optimizations — with no quality loss at FP32 and no changes to model weights.

## Benchmark

NVIDIA H100 PCIe | `hubertsiuzdak/snac_24khz` | decode

| Duration | Baseline (FP32) | FP16 + compile | Speedup | Real-time Factor |
|----------|:---------------:|:--------------:|:-------:|:----------------:|
| 1s | 3.45 ms | **0.93 ms** | **3.72x** | **25,806x** |
| 6s | 5.70 ms | **2.91 ms** | **1.96x** | **49,485x** |
| 100s | 71.40 ms | **42.72 ms** | **1.67x** | **56,180x** |

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
```

## Triton Snake Kernel

Custom Triton kernel for the Snake activation (`x + (1/α)·sin²(α·x)`) used throughout SNAC's encoder and decoder.

| Duration | Layer | PyTorch | Triton | Speedup | Optimizations |
|----------|-------|:-------:|:------:|:-------:|:--------------|
| 100s | Encoder (1024ch) | 131 us | **21 us** | **6.16x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| 100s | Decoder (1536ch) | 233 us | **41 us** | **5.64x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| 100s | Decoder (96ch) | 5680 us | **1045 us** | **5.44x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |

```python
from snac.kernels.triton_snake import snake_triton

y = snake_triton(x, alpha)  # drop-in replacement for Snake1d
```

## Requirements

- PyTorch 2.6+
- NVIDIA GPU (Hopper/Ampere)

## License

MIT
