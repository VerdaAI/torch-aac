# Internal Assumptions Log

Track every assumption made during development. Each must be validated or challenged before the relevant phase ships.

---

## Phase 1 Assumptions (Core Pipeline)

### A1: MDCT via Matrix Multiply is Fast Enough
**Assumption**: Using `torch.matmul(frames, basis_matrix)` for MDCT is efficient enough for batch GPU encoding.
**Why**: The basis matrix is `(2048, 1024)` — 8MB of float32. For B frames, the matmul is `(B, 2048) @ (2048, 1024)`. On GPU this is a single cuBLAS call.
**Risk**: For very large batches (B > 8192), the basis matrix is replicated per-batch and memory could be an issue. Also, an FFT-based MDCT would be O(N log N) vs O(N²) per frame.
**Status**: ✅ Works correctly. Performance untested on GPU.
**Action for Phase 3**: Benchmark matmul vs FFT-based MDCT. If matmul is a bottleneck, switch to the FFT approach (the folding + DCT-IV method attempted earlier — needs correct implementation).

### A2: Basis Matrix Caching is Safe
**Assumption**: `_basis_cache` (module-level dict) caching MDCT basis matrices by `(N, device)` is safe for multi-threaded/multi-GPU use.
**Risk**: Not thread-safe. Multiple threads encoding on different devices could race on the cache dict. Also, cached tensors hold GPU memory indefinitely.
**Action for Phase 3**: Replace with `torch.nn.Module` buffer or `functools.lru_cache` with proper cleanup.

### A3: Silent Frames (max_sfb=0) Produce Valid AAC
**Assumption**: Writing max_sfb=0 with no section/scalefactor/spectral data produces spec-compliant AAC-LC frames.
**Status**: ✅ FFmpeg 7.1.1 decodes successfully. Produces silence output.
**Risk**: Other decoders (Apple CoreAudio, Android MediaCodec, VLC's internal AAC) might reject these frames. Not tested.
**Action for Phase 2**: Test with `afplay` (macOS), VLC, and Android's MediaCodec if possible.

### A4: 3 Zero Bits After ICS Header with max_sfb=0
**Assumption**: FFmpeg's decoder expects 3 zero bits between `predictor_data_present=0` and `ID_END` when max_sfb=0.
**Why**: Empirically determined by byte-comparing with FFmpeg's own output. The ISO spec doesn't explicitly document this padding.
**Risk**: This might be specific to FFmpeg 7.1.1 and break on older/newer versions. The 3 bits might actually be from a spec-mandated empty structure we're not aware of.
**Action for Phase 2**: Read ISO 14496-3 section on `individual_channel_stream()` parsing for max_sfb=0 case. Determine what these 3 bits actually represent. CRITICAL: if they are part of a real syntax element, we need to understand it before enabling spectral encoding.

### A5: ID_SCE = 0, ID_CPE = 1
**Assumption**: AAC syntax element IDs are: SCE=0, CPE=1, END=7.
**Status**: ✅ Confirmed by FFmpeg reference output.
**Previous bug**: Initially wrote SCE=1, CPE=2 (off by one). Fixed.

### A6: ADTS ID=0 (MPEG-4) Works
**Assumption**: Using MPEG-4 AAC (ID=0 in ADTS header) rather than MPEG-2 (ID=1).
**Risk**: Some older players/devices may only support MPEG-2 AAC in ADTS. MPEG-4 AAC in ADTS is technically non-standard (MPEG-4 should use MP4 container), though widely supported.
**Status**: ✅ FFmpeg decodes it. FFmpeg's own encoder also uses ID=0.
**Action**: Consider adding an option for MPEG-2 (ID=1) for broader compatibility.

### A7: Sine Window Only
**Assumption**: V1 uses only the sine window (`window_shape=1`). KBD window is deferred.
**Risk**: None for correctness. Quality impact is minor — KBD has slightly better stopband rejection but sine is the AAC default.
**Status**: ✅ Correct.

### A8: Long Blocks Only
**Assumption**: V1 uses only ONLY_LONG_SEQUENCE (window_sequence=0). Short blocks, start/stop transitions deferred.
**Risk**: Quality degradation on transient signals (drums, speech plosives). The MDCT will smear transients across the full 1024-sample window.
**Status**: ✅ Correct for V1 scope.
**Action for future**: Implement block switching when quality benchmarks show it matters.

### A9: SFB Tables are Correct
**Assumption**: The scalefactor band widths in `sfb_tables.py` match ISO 14496-3 Table 4.85.
**Risk**: Transcription errors. The tables were generated from memory, not mechanically extracted.
**Status**: ⚠️ Not verified against a reference implementation.
**Action for Phase 2**: Cross-check against FFmpeg's `aactab.c` or FDK-AAC's tables. A single wrong width produces incorrect quantization and undecodable frames.

### A10: Huffman Tables Are Stubs
**Assumption**: V1 Huffman tables are empty stubs. The fallback encoding in `cpu/huffman.py` is not spec-compliant and produces invalid spectral data.
**Status**: ✅ Acknowledged. This is why V1 uses max_sfb=0 (no spectral data).
**Action for Phase 2**: Populate exact Huffman tables from ISO spec or extract from FFmpeg/FDK-AAC source. This is the #1 blocker for producing audible (non-silent) AAC output.

### A11: Rate Control Binary Search Converges
**Assumption**: Binary search on global_gain (0-255) with 12 iterations always converges to a valid bit budget.
**Risk**: For near-silent frames, the optimal gain might be at the boundary. For very loud frames, even gain=255 might not produce few enough bits.
**Status**: ⚠️ Not validated with real spectral data (since V1 is silent).
**Action for Phase 2**: Add bounds checking and fallback (e.g., set all coefficients to zero if gain=255 still overflows).

### A12: Bit Estimation Approximation is Adequate
**Assumption**: `estimate_bit_count()` in `gpu/quantizer.py` uses a rough approximation (`log2(|q|+1) + 2` per coefficient) instead of actual Huffman code lengths.
**Risk**: The rate control loop may converge to gains that produce frames significantly over/under the target bitrate.
**Status**: ⚠️ Adequate for V1 (silent frames). Must be validated when real spectral encoding is enabled.
**Action for Phase 2**: Compare estimated vs actual bit counts after Huffman encoding. Adjust the cost model if needed.

### A13: Channel Layout Inference
**Assumption**: When PCM input is 2-D, we infer channel layout from `min(shape)`. E.g., `(48000, 2)` → channels=2, `(2, 48000)` → channels=2.
**Risk**: Ambiguous for short stereo audio where num_samples < num_channels (e.g., 1 sample of stereo). In practice this shouldn't happen.
**Status**: ✅ Reasonable heuristic.

### A14: CPU Fallback Not Yet Implemented
**Assumption**: `cpu_fallback/` directory exists but modules are empty stubs. All encoding currently runs through the `gpu/` modules (which work on CPU via PyTorch).
**Risk**: None — PyTorch CPU mode works fine. The fallback is for numpy-only environments (no PyTorch installed), which is a Phase 4 deliverable.
**Status**: ✅ Acknowledged.

### A15: No M4A/MP4 Container Support
**Assumption**: V1 only outputs ADTS (.aac) container format. M4A/MP4 containers require additional muxing (moov/mdat atoms) not implemented.
**Risk**: Users expecting `.m4a` output will be disappointed.
**Action for Phase 4**: Add MP4 container support or document that users should use FFmpeg to remux: `ffmpeg -i output.aac -c copy output.m4a`.

---

## Conflicts / Issues Found

### C1: RESOLVED — ID_SCE was wrong
Initially wrote `ID_SCE = 0b001 (1)` but correct value is `0b000 (0)`. Fixed in bitstream.py.

### C2: RESOLVED — MDCT normalization
First MDCT implementation using FFT twiddle factors had a ~512x energy scaling error. Replaced with correct matrix multiply approach.

### C3: OPEN — Codebook 1-2 signed/unsigned flag
The Huffman tables agent identified that `gpu/huffman_select.py` and `cpu/huffman.py` mark codebooks 1-2 as unsigned (`True`), but per ISO spec they are **signed** (values -1, 0, +1 encoded directly without separate sign bits). This needs to be fixed when real Huffman encoding is enabled.
**Impact**: None in V1 (max_sfb=0, no spectral data). Will cause incorrect encoding in Phase 2.

### C4: PARTIALLY RESOLVED — 3 mystery bits after ICS header
For max_sfb=0, these 3 bits are byte-alignment padding that FFmpeg's bit reader
consumes. For max_sfb > 0, section_data fills the corresponding bit positions.

### C5: OPEN — FFmpeg requires fully correct spectral encoding to init channels
FFmpeg only allocates channel elements after the first frame is fully decoded
with valid spectral data. We cannot use "max_sfb > 0 with all zeros" as a
stepping stone — we must go directly to fully correct spectral encoding.

**Root cause found**: The section_data + scalefactor_data + spectral_data bit
layout is structurally correct (verified by bit-level parsing), but FFmpeg's
decoder rejects frames that don't contain properly encoded spectral content.

**Debug approach for next session**:
1. Generate a 1-frame AAC with FFmpeg's encoder (reference)
2. Parse the reference frame completely — every bit of spectral data
3. Generate our frame with identical MDCT coefficients and gain
4. Compare bit-by-bit to find the first divergence
5. Fix the divergence and iterate

The encoding infrastructure (Huffman tables, section writer, scalefactor coder,
spectral data writer) is COMPLETE. The remaining work is ensuring the bit-level
output exactly matches what FFmpeg's decoder expects.

---

## Phase 2 Assumptions (Differentiable Mode)

### A16: STE Gradients are Useful for Training
**Assumption**: The straight-through estimator (round in forward, identity in backward) produces gradients useful enough to train codec-robust models.
**Status**: ✅ Verified — optimization step updates parameters, gradients are non-zero and finite.
**Risk**: STE gradients are biased. For large quantization steps (low bitrate), the gradient estimate may be poor, causing training instability.
**Action**: Monitor gradient variance in the benchmark. If STE fails at low bitrates, noise injection mode may work better.

### A17: Detached Rate Control in Differentiable Mode
**Assumption**: In `DifferentiableAAC.forward()`, `find_global_gain` runs with `torch.no_grad()` (detached). The gain selection is NOT differentiable.
**Why**: The binary search on global_gain is inherently discrete. Making it differentiable would require a continuous relaxation.
**Risk**: The model cannot learn to influence the rate control loop. It can only learn to produce audio that quantizes well at the FIXED gain the rate controller selects.
**Status**: ✅ Works for V1. May need revisiting if users want end-to-end rate-distortion optimization.

### A18: MDCT Basis Matrix is Differentiable
**Assumption**: The matmul-based MDCT (`frames @ basis`) and IMDCT (`coeffs @ basis.T`) have correct gradients via PyTorch autograd.
**Status**: ✅ Verified — gradients flow through correctly.

### A19: Overlap-Add Reconstruction Quality
**Assumption**: MDCT → quantize → dequantize → IMDCT → window → overlap-add produces a reasonable approximation of the input signal (ignoring quantization artifacts).
**Status**: ✅ Verified in `test_mdct_imdct_roundtrip` (correlation > 0.9 for clean signals).
**Risk**: The reconstruction quality depends on correct window normalization. The `2/M` scaling in IMDCT combined with the sine window overlap-add should give perfect reconstruction for the unquantized path. Quantization degrades this.

### A20: ATH Dominates High-Frequency Masking Thresholds
**Assumption**: The absolute threshold of hearing (ATH) at high frequencies (>16kHz) produces extremely large values (up to 1e27) which dominate the mean masking threshold.
**Risk**: This is physically correct (humans can't hear above ~20kHz, so ATH is very high), but it means the masking threshold is essentially infinite for the top ~15 SFB bands at 48kHz. The quantizer should allocate zero bits to these bands.
**Status**: ✅ Correct behavior. Test adjusted to only compare audible bands (<30 SFB).
**Action**: When spectral encoding is enabled, verify that high-frequency bands get codebook 0 (zero bits) automatically.

### A21: Very Short Audio Padding
**Assumption**: Audio shorter than one window (2048 samples = 42ms at 48kHz) is zero-padded to fill one complete window.
**Status**: ✅ Produces valid output. The padding means the encoded audio will be slightly longer than the input.
**Risk**: Padding introduces a discontinuity that could cause artifacts. Acceptable for V1.

---

## Cross-Phase Dependencies

| Phase 2 depends on | Description |
|---|---|
| A10 resolved | Need real Huffman tables to enable spectral encoding |
| A9 verified | SFB tables must be correct for quantization to work |
| C3 fixed | Codebook unsigned flags must be correct |
| C4 understood | Must know what the 3 post-ICS bits are |
| A4 validated | Silent frame compatibility with other decoders |

| Phase 3 depends on | Description |
|---|---|
| A1 benchmarked | MDCT performance on GPU determines if FFT approach is needed |
| A2 addressed | Basis cache needs to be thread-safe for multi-GPU |
| A11 validated | Rate control convergence with real spectral data |
| A12 validated | Bit estimation accuracy with real Huffman encoding |
