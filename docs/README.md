# `docs/` — Documentation

| Document | Audience | Contents |
|----------|----------|----------|
| [`technical.md`](technical.md) | Contributors, researchers | Deep dive into engineering decisions: MDCT normalization, bit estimator accuracy, escape code bugs, GPU Huffman architecture, encoder-decoder parity. The story behind the numbers. |
| [`internal/assumptions.md`](internal/assumptions.md) | Maintainers | Living log of every assumption made during development — what was assumed, why, whether it was validated, and what to do if it breaks. 27 tracked assumptions across 3 development phases. |

## Where to Find What

| Question | Where to look |
|----------|---------------|
| How do I use the library? | [README.md](../README.md), [examples/](../examples/) |
| How does the encoder work? | [technical.md](technical.md), [torch_aac/README.md](../torch_aac/README.md) |
| Why is SF_OFFSET 164? | [technical.md](technical.md) § "Unnormalized MDCT" |
| What bugs were found? | [technical.md](technical.md) § "Three Compounding Amplitude Bugs" |
| How fast is it and why? | [technical.md](technical.md) § "Performance Profile" |
| What's experimental? | [internal/assumptions.md](internal/assumptions.md) § "Post-spectral-encoding additions" |
| How do I contribute? | [CONTRIBUTING.md](../CONTRIBUTING.md) |
