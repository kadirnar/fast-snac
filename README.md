# Fast-SNAC

Fast inference engine for [SNAC](https://github.com/hubertsiuzdak/snac), a hierarchical neural audio codec. Accelerates SNAC decode up to **2.70x** end-to-end on NVIDIA GPUs through Triton kernels + torch.compile — with no changes to model weights.

## Benchmark

NVIDIA H100 PCIe | `hubertsiuzdak/snac_24khz` | 100s audio @ 24kHz

### Full Precision (FP32) — Zero Quality Loss

| Method | Latency | Speedup |
|--------|:-------:|:-------:|
| PyTorch FP32 | 70.91 ms | 1.0x |
| + torch.compile | **54.68 ms** | **1.30x** |

### Half Precision (FP16)

| Method | Latency | Speedup |
|--------|:-------:|:-------:|
| PyTorch FP16 | 58.67 ms | 1.21x |
| + Triton Snake (`fast_sinf`, fused depthwise) | 33.30 ms | 2.13x |
| + torch.compile (elementwise fusion) | 26.90 ms | 2.64x |
| **+ CUDA graph** | **26.29 ms** | **2.70x** |

### Snake Activation Kernel (isolated)

| Method | Latency | Speedup |
|--------|:-------:|:-------:|
| PyTorch Snake (all layers, 100s) | 15,646 us | 1.0x |
| **Triton Snake** | **2,914 us** | **5.37x** |

## Optimizations

- **Triton Snake kernel** — `fast_sinf` + `fast_dividef` CUDA intrinsics, L2 eviction hints
- **Fused Snake + Depthwise Conv1d** — single Triton kernel, eliminates intermediate tensor
- **Standalone Depthwise Conv1d** — Triton kernel with unrolled loop (`K` constexpr)
- **torch.compile** — Inductor epilogue fusion for elementwise ops (residual add, sigmoid, noise)
- **torch.library.custom_op** — graph-break-free Triton integration with torch.compile
- **Weight norm removal** — fuses `weight_g * weight_v / ||weight_v||` at load time
- **NoiseBlock caching** — static noise tensor, enables CUDA graph capture
- **CUDA graph** — zero CPU overhead kernel replay

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

# FP16 + Triton + compile — fastest (2.70x)
decode_fn = optimize_snac_triton(model, codes, dtype="fp16", use_compile=True)
audio_hat = decode_fn(codes)

# FP32 + compile — zero quality loss (1.30x)
from snac.optimize import optimize_snac_native
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
decode_fn = optimize_snac_native(model, codes, dtype="fp32")
audio_hat = decode_fn(codes)
```

## Requirements

- PyTorch 2.6+
- Triton 3.0+
- NVIDIA GPU (Hopper/Ampere)

## License

MIT
