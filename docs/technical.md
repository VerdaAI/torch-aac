# Technical Deep Dive

How torch-aac achieves high quality at high speed — and the bugs we had to find along the way.

## The Challenge

AAC-LC encoding involves a pipeline of signal processing stages (MDCT, quantization, Huffman coding) that traditionally runs on CPU. Making this GPU-accelerated AND differentiable requires solving several hard problems:

1. **Huffman coding is inherently serial** — each codeword's bit position depends on the cumulative length of all preceding codewords.
2. **AAC's non-uniform quantizer** has a specific sweet spot (q ≈ 5-50) where bit efficiency is highest. Outside this range, escape codes waste bits.
3. **FFmpeg's AAC decoder** has undocumented normalization quirks that must be matched exactly for correct amplitude reconstruction.
4. **The bit cost estimator** drives rate control — if it's wrong, the encoder wastes 30-50% of its bit budget.

## Key Technical Decisions

### 1. Unnormalized MDCT with Empirical SF Calibration

Standard AAC encoders normalize the MDCT output by `2/N`. We use an unnormalized matmul-based MDCT (`X = frames @ basis`), which produces coefficients ~512× larger than standard.

This means the AAC scalefactor "unity point" (where decoder output amplitude matches input) is NOT at the spec's `SF=100` but at an empirically determined `SF_OFFSET=164`.

**How we found it**: We wrote a synthetic bitstream with `q=1` at a known MDCT bin and swept the scalefactor value. At `SF=164`, the decoded peak amplitude was exactly 1.0. At `SF=100` (the spec value), it was 1/32768 — off by 2^15.

The 164 = 200 - 36 reflects our MDCT being 2^9 × larger than FFmpeg's expected range, where 200 is FFmpeg's `POW_SF2_ZERO` constant and 36 = 4 × log2(512).

**Why not just normalize?** We tested normalizing by `2/N` and re-calibrating. The math is self-consistent either way — normalizing doesn't change the quantized values. Keeping the unnormalized path avoids an extra multiply per coefficient and matches our established test baselines.

> See `gpu/quantizer.py:SF_OFFSET` and `docs/internal/assumptions.md:A22`.

### 2. Accurate Huffman Bit Cost Estimator

Rate control uses a binary search to find the optimal quantization gain: the finest quantization (best quality) that fits the bit budget. The search depends on a bit cost estimator that predicts how many bits the Huffman encoder will produce for a given set of quantized values.

**The original estimator** used a rough formula:
```
cost(q) = 0.25       if q == 0
         log2(q) + 4  if q ≤ 16
         2·log2(q) + 8 if q > 16
```

This overestimated actual Huffman costs by **1.2-2.5×**, causing rate control to converge to gains that used only 25-50% of the bit budget. The encoder was leaving massive quality on the table.

**The fix**: We built a 8193-entry lookup table from the ACTUAL Huffman code lengths in each AAC codebook (cb5-11). For each `|q|` in `[0, 8192]`, the table stores the real bits for encoding a pair `(±q, 0)` in the optimal codebook, including sign bits and escape codes.

**Impact**: +6.8 dB on speech, +7.3 dB on noise at 128 kbps. This single change was the highest-leverage optimization in the entire project.

> See `gpu/quantizer.py:_build_bits_table` and `gpu/quantizer.py:estimate_bit_count`.

### 3. MAX_QUANTIZED_VALUE = 8191 (Not 4095)

AAC's escape code format uses N leading 1-bits followed by a (N+4)-bit mantissa. FFmpeg's decoder limits N to 8, giving a maximum encoded value of `2^13 - 1 = 8191`.

We originally set `MAX_QUANTIZED_VALUE = 4095` (which is `2^12 - 1` — the maximum *mantissa*, not the maximum *value*). This meant rate control stopped refining at 50% of the available precision headroom.

**Impact**: Combined with the accurate estimator, raising the limit from 4095 to 8191 let rate control use 92% of the bit budget (up from 71%), with proportional quality improvements across all signals.

> See `gpu/rate_control.py:MAX_QUANTIZED_VALUE` and `cpu/huffman.py:MAX_ESCAPE_VAL`.

### 4. Three Compounding Amplitude Bugs

When we first enabled spectral encoding, the decoded output was ~1/60000 of the expected amplitude — despite perfect signal structure (correlation 0.95). Three independent bugs compounded:

**Bug 1 — Huffman escape code off-by-one**: The escape encoder computed `N = bit_length(val) - 4` instead of `- 5`, and stored the raw value as the mantissa instead of `val - 2^(N+4)`. Every escape-coded coefficient decoded to ~3× its intended magnitude and misaligned all subsequent bits.

**Bug 2 — Window shape flag inverted**: We wrote `window_shape=1` (KBD) while using a sine window for the forward MDCT. The decoder applied KBD synthesis to our sine-windowed data.

**Bug 3 — SF_OFFSET**: The spec-derived value of 100 was wrong for our MDCT normalization (should be 164, as described above).

Each bug independently produced a multiplicative error. Together: 3× (escape) × ~4× (window) × ~5000× (SF) ≈ 60000×.

**How we found them**: We wrote a bitstream parser that traced every field, compared our output byte-by-byte against FFmpeg's, and tested with trivially simple inputs (q=1 at one bin, everything else zero).

> See `cpu/huffman.py:_encode_escape`, `cpu/bitstream.py:window_shape`, and `gpu/quantizer.py:SF_OFFSET`.

### 5. Batch-Flattened GPU Huffman Encoding

The initial CPU Huffman encoder had a Python `for` loop that iterated per-group (pair or quad of coefficients), doing a dictionary lookup and bit-packing per group. For 10 seconds of audio: ~300,000 Python function calls.

**Optimization 1 — C BitWriter**: Replaced the per-bit Python `for` loop with a uint64 accumulator that packs whole codewords in one operation. JIT-compiled via `ctypes` on first import. **2.3× speedup.**

**Optimization 2 — Batch GPU Huffman**: All groups from all frames are flattened into one `(N, G, 2)` tensor. A single GPU gather computes Huffman codes and lengths for all 240,000 groups at once. Sign bits and escape codes are computed vectorized. The CPU just does fast numpy extraction + C byte-packing. **6× total speedup** (15.5× → 97× realtime).

The `pairs_only` codebook mode restricts selection to pair codebooks (cb5-11), making group dimension uniformly 2 and enabling the fixed-layout tensor.

> See `gpu/huffman_encode.py:encode_spectral_batched`, `cpu/_bitwriter.c`, and `cpu/_bitwriter_native.py`.

### 6. Encoder-Decoder Parity for Differentiable Mode

`DifferentiableAAC` simulates the full encode→decode pipeline as a differentiable PyTorch operation. For this to be useful for training, it MUST produce the same output as the real encoder.

We verified parity by encoding the same signal through both paths and comparing:

| Signal | Correlation | Max Error |
|--------|------------|-----------|
| Sine 1kHz | 1.000000 | < 2e-4 |
| Chord | 1.000000 | < 2e-4 |
| Music-like | 1.000000 | < 2e-4 |
| Noise | 1.000000 | < 2e-4 |
| Speech-like | 1.000000 | < 2e-4 |

The tiny residual (< 2e-4) comes from float32 accumulation differences between the two paths.

> See `differentiable.py:DifferentiableAAC`.

## Performance Profile

For 10s mono audio at 128 kbps (CPU):

| Stage | Time | % |
|-------|-----:|--:|
| MDCT + windowing | 16 ms | 15% |
| Rate control (binary search) | 17 ms | 16% |
| GPU Huffman gather | 8 ms | 8% |
| Per-frame assembly (Python) | 31 ms | 30% |
| C BitWriter packing | 33 ms | 32% |
| **Total** | **~105 ms** | |

The remaining bottleneck is per-frame Python assembly of section data and scalefactor deltas. Moving this to C or fully to GPU would push toward 200× realtime.
