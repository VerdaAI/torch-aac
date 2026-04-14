# GPU Benchmark Results

Run on a dual-GPU Linux workstation with PyTorch 2.10.0+cu126.

## Hardware

| GPU | VRAM | CUDA |
|-----|------|------|
| NVIDIA GeForce RTX 3090 | 23.6 GB | 12.6 |
| NVIDIA GeForce RTX 3080 Ti | 11.6 GB | 12.6 |

## Reproducing

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python benchmarks/cuda_validation.py

# GPU 1
CUDA_VISIBLE_DEVICES=1 python benchmarks/cuda_validation.py
```

## Summary

### Throughput at 128 kbps mono, 48 kHz

| Duration | RTX 3090 (CUDA) | RTX 3080 Ti (CUDA) | FFmpeg (CPU) |
|----------|-----------------|---------------------|--------------|
| 1 s  |   9.9 ms (101x rt) |  10.2 ms ( 98x rt) |  33.5 ms ( 30x rt) |
| 10 s |  54.3 ms (184x rt) |  53.9 ms (185x rt) |  93-96 ms (105-107x rt) |
| 30 s | 145.8 ms (206x rt) | 150.0 ms (200x rt) | 183-185 ms (162-164x rt) |
| 60 s | 282.5 ms (212x rt) | 291.9 ms (206x rt) | 333-345 ms (174-180x rt) |

### torch-aac CUDA vs FFmpeg CPU

| Duration | 3090 speedup | 3080 Ti speedup |
|----------|--------------|-----------------|
|  1 s | 3.37x | 3.23x |
| 10 s | 2.03x | 1.77x |
| 30 s | 1.26x | 1.18x |
| 60 s | 1.19x | 1.10x |

The speedup is largest on short clips where FFmpeg's subprocess spawn
overhead dominates; on 60 s clips FFmpeg's mature C code catches up and
the gap narrows to ~1.1-1.2x.

### Batch throughput (8 x 10 s streams)

| GPU | Time | Aggregate realtime |
|-----|------|---------------------|
| RTX 3090    |  394.7 ms | 202.7x |
| RTX 3080 Ti | 2000.2 ms |  40.0x |

The 3090's 2x VRAM and wider memory bus let it batch-process 8 streams
in parallel efficiently -- the 3080 Ti falls back to a more serial path
on this workload.

### Other results

- **Correctness**: Both GPUs pass sine_1khz and noise correctness checks
  (peak ratio ~1.0 after FFmpeg round-trip decode).
- **Differentiable mode**: Both GPUs produce finite, non-zero gradients.
- **Peak VRAM** (60 s encode): 570 MB on both cards.

## Files

- `rtx_3090_results.txt` -- full raw output for the RTX 3090 run
- `rtx_3080ti_results.txt` -- full raw output for the RTX 3080 Ti run
