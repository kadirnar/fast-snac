# Fast-SNAC

Fast inference engine for [SNAC](https://github.com/hubertsiuzdak/snac), a hierarchical neural audio codec. Accelerates SNAC decode up to **2.71x** end-to-end on NVIDIA GPUs through Triton kernels + torch.compile — with no changes to model weights.

## Benchmark

NVIDIA H100 PCIe | `hubertsiuzdak/snac_24khz` | 100s audio @ 24kHz

### End-to-End Decode

| Method | Latency | Speedup |
|--------|:-------:|:-------:|
| PyTorch FP32 | 70.98 ms | 1.00x |
| + torch.compile | 54.67 ms | **1.30x** |
| FP16 + Triton kernels | 33.36 ms | **2.13x** |
| FP16 + Triton + compile | 26.94 ms | **2.63x** |
| **FP16 + Triton + compile + graph** | **26.16 ms** | **2.71x** |

### Snake Kernel — FP32

| Layer | PyTorch | Triton | TileLang | Triton speedup | TileLang speedup |
|-------|:-------:|:------:|:--------:|:--------------:|:----------------:|
| Encoder (1024ch) | 45 us | **53 us** | 91 us | 0.84x | 0.49x |
| Decoder (768ch) | 178 us | **171 us** | 349 us | **1.04x** | 0.51x |
| Decoder (384ch) | 608 us | **517 us** | 1008 us | **1.17x** | 0.60x |
| Decoder (192ch) | 896 us | **740 us** | 1461 us | **1.21x** | 0.61x |
| Decoder (96ch) | 1333 us | **1088 us** | 2159 us | **1.23x** | 0.62x |
| **Total** | **3060 us** | **2570 us** | **5068 us** | **1.19x** | 0.60x |

### Snake Kernel — FP16

| Layer | PyTorch | Triton | TileLang | Triton speedup | TileLang speedup |
|-------|:-------:|:------:|:--------:|:--------------:|:----------------:|
| Encoder (1024ch) | 47 us | **56 us** | 68 us | 0.85x | 0.70x |
| Decoder (768ch) | 204 us | **115 us** | 235 us | **1.77x** | 0.87x |
| Decoder (384ch) | 663 us | **280 us** | 590 us | **2.37x** | 1.12x |
| Decoder (192ch) | 992 us | **401 us** | 780 us | **2.47x** | 1.27x |
| Decoder (96ch) | 1475 us | **577 us** | 1138 us | **2.56x** | 1.30x |
| **Total** | **3382 us** | **1430 us** | **2810 us** | **2.37x** | **1.20x** |

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

# FP16 + Triton + compile — fastest (2.71x)
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
