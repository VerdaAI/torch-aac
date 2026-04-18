# Stereo M/S Test Reference Files

3 reference stereo audio files extracted from musdb18 (10 seconds each,
48 kHz, stereo FLAC). Use these to reproduce the M/S stereo coding
issues reported in `benchmarks/results/stereo_test_output.txt`.

| File | Source | Genre |
|------|--------|-------|
| `AM_Contra___Heart_Peripheral_10s.flac` | musdb18/test | electronic |
| `BKS___Bulldozer_10s.flac` | musdb18/test | rock |
| `Carlos_Gonzalez___A_Place_For_Us_10s.flac` | musdb18/test | singer-songwriter |

## Reproducing

```bash
python benchmarks/stereo_test.py --data-root benchmarks/test_audio/stereo --max-dur 10
```

## Observed issues on v0.6.0 (commit 9e4dc43, M/S stereo added)

Tested on RTX 3090, CUDA. torch-aac v0.6.0 with the newly-added
channel pair element / M/S coding path.

| File | Target BR | Actual kbps | SNR | Issue |
|------|-----------|-------------|-----|-------|
| AM Contra | 64k | **4.9k** | **0.0 dB** | Output is zero; bitrate 1/13x target |
| AM Contra | 128k | 47k | 9.8 dB | Bitrate 1/2.7x target, low SNR |
| BKS Bulldozer | 64k | **4.9k** | **0.0 dB** | Same -- zero output |
| BKS Bulldozer | 128k | 37k | 8.4 dB | Bitrate 1/3.5x target |
| Carlos Gonzalez | 64k | **4.9k** | **0.0 dB** | Same -- zero output |
| Carlos Gonzalez | 128k | 36k | 15.5 dB | Bitrate 1/3.6x target |

### Summary

1. **Severe bitrate under-allocation**: all files produce far fewer
   bits than the target. At 64k target, only ~4.9 kbps is actually
   produced. At 128k target, only 36-47 kbps. The rate control seems
   not to account for the CPE (channel pair element) structure.
2. **Zero-signal output at 64k**: exactly 0.0 dB SNR on all three
   files means the decoded output is all zeros (the ADTS bitstream
   is syntactically valid but contains no spectral data). FFmpeg
   accepts the bitstream with `-xerror` but outputs silence.
3. **No decode errors**: every bitstream passes strict FFmpeg
   decode -- the problem is the encoder output, not bitstream
   corruption.

### Comparison with mono benchmark (v5, same musdb18 files)

At 128k mono (downmixed), torch-aac achieved ~132 kbps actual and
27.6 dB SNR on these same tracks. The stereo path is regressed.
