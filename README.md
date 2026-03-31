# Fast-SNAC

Fast inference engine for [SNAC](https://github.com/hubertsiuzdak/snac), a hierarchical neural audio codec that compresses audio using multi-scale residual vector quantization. This library accelerates SNAC decode up to **3.7x** on NVIDIA GPUs through graph-level optimizations — with no quality loss at FP32 and no changes to model weights.

## Benchmark

NVIDIA H100 PCIe | `hubertsiuzdak/snac_24khz` | 100s audio @ 24kHz

| Layer | PyTorch | Triton | Speedup | Optimizations |
|-------|:-------:|:------:|:-------:|:--------------|
| Encoder Block1 (64ch) | 1291 us | **238 us** | **5.44x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Encoder Block2 (128ch) | 874 us | **169 us** | **5.17x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Encoder Block3 (256ch) | 275 us | **50 us** | **5.51x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Encoder Block4 (512ch) | 55 us | **21 us** | **2.67x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Encoder Output (1024ch) | 132 us | **22 us** | **6.10x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Decoder Input (1536ch) | 235 us | **42 us** | **5.54x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Decoder Block1 (768ch) | 756 us | **144 us** | **5.27x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Decoder Block2 (384ch) | 2548 us | **484 us** | **5.27x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Decoder Block3 (192ch) | 3800 us | **701 us** | **5.42x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| Decoder Block4 (96ch) | 5679 us | **1045 us** | **5.44x** | `fast_sinf`, `fast_dividef`, L2 eviction hints |
| **Total** | **15646 us** | **2914 us** | **5.37x** | |

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
