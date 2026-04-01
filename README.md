# Fast-SNAC

Fast inference engine for [SNAC](https://github.com/hubertsiuzdak/snac), a hierarchical neural audio codec. Accelerates SNAC decode up to **2.88x** end-to-end on NVIDIA GPUs through Triton kernels + torch.compile — with no changes to model weights.

## Benchmark

NVIDIA H100 PCIe | `hubertsiuzdak/snac_24khz` | 100s audio @ 24kHz

### Full Precision (FP32) — Zero Quality Loss

| Method | Latency | Speedup | RTF |
|--------|:-------:|:-------:|:---:|
| PyTorch FP32 | 71.46 ms | 1.00x | 1,399x |
| + Triton kernels | 50.85 ms | **1.41x** | 1,966x |
| + torch.compile | 42.85 ms | **1.67x** | 2,333x |
| **+ CUDA graph** | **42.15 ms** | **1.70x** | **2,372x** |

### Half Precision (FP16)

| Method | Latency | Speedup | RTF |
|--------|:-------:|:-------:|:---:|
| PyTorch FP16 | 58.44 ms | 1.22x | 1,711x |
| + Triton kernels | 31.83 ms | **2.24x** | 3,141x |
| + torch.compile | 25.41 ms | **2.81x** | 3,935x |
| **+ CUDA graph** | **24.78 ms** | **2.88x** | **4,035x** |

## Quick Start

```bash
pip install git+https://github.com/kadirnar/fast-snac.git
```

```python
from snac import SNAC
from snac.optimize import optimize_snac_triton
import torch

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
audio = torch.randn(1, 1, 24000, device="cuda")
codes = model.encode(audio)

# FP16 + Triton + compile — fastest (2.88x)
decode_fn = optimize_snac_triton(model, codes, dtype="fp16", use_compile=True)
audio_hat = decode_fn(codes)

# FP32 + Triton + compile — zero quality loss (1.70x)
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
decode_fn = optimize_snac_triton(model, codes, dtype="fp32", use_compile=True)
audio_hat = decode_fn(codes)
```

## Requirements

- PyTorch 2.6+
- Triton 3.0+
- NVIDIA GPU (Hopper/Ampere)

## License

MIT
