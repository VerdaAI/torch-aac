# `torch_aac/` — Core Package

The main encoder package. Two execution modes:

| Module | Role |
|--------|------|
| `encoder.py` | **Encode mode orchestrator.** Coordinates GPU stages (MDCT, quantization, Huffman) with CPU stages (bitstream packing) to produce AAC-LC ADTS bytes. |
| `differentiable.py` | **Differentiable mode.** `DifferentiableAAC` PyTorch module that simulates encode→decode as a differentiable operation. Gradients flow through STE or noise injection quantization. |
| `config.py` | Encoder configuration, AAC-LC constants, sample rate tables, PNS constants. |
| `__init__.py` | Public API: `encode()`, `encode_batch()`, `encode_file()`, `DifferentiableAAC`. |
| `__main__.py` | CLI entry point: `python -m torch_aac -i input.wav -o output.aac`. |

## Data Flow (Encode Mode)

```
PCM float32
 → gpu/filterbank.py    (frame, window, MDCT)
 → gpu/rate_control.py  (binary search for optimal gain)
 → gpu/quantizer.py     (non-uniform AAC quantization)
 → gpu/huffman_select.py (codebook selection per band)
 → gpu/huffman_encode.py (batch GPU Huffman lookup)
 → cpu/bitstream.py     (ADTS frame assembly)
 → cpu/_bitwriter.c     (C bit-packing via ctypes)
 → .aac bytes
```

## Data Flow (Differentiable Mode)

```
PCM tensor (requires_grad=True)
 → gpu/filterbank.py    (MDCT — differentiable matmul)
 → gpu/quantizer.py     (STE or noise quantize — differentiable)
 → gpu/quantizer.py     (dequantize)
 → gpu/filterbank.py    (IMDCT + overlap-add)
 → reconstructed PCM (gradients flow back)
```
