# `integrations/` — Framework Integrations

Drop-in adapters for popular audio ML frameworks.

## torchaudio

```python
from torch_aac.integrations.torchaudio import AACSimulation, save_aac, load_aac
```

| Class / Function | Description |
|-----------------|-------------|
| `AACSimulation` | Differentiable AAC transform for training pipelines. Compatible with `torch.nn.Sequential` and torchaudio transforms. Gradients flow through. |
| `AACEncode` | Real AAC encode → FFmpeg decode. Non-differentiable. For evaluation pipelines. |
| `save_aac()` | Save waveform to `.aac` file (torchaudio.save-like API). |
| `load_aac()` | Load `.aac` file to tensor (torchaudio.load-like API). |

### Training Example

```python
import torchaudio.transforms as T
from torch_aac.integrations.torchaudio import AACSimulation

# Augmentation pipeline: resample → AAC compress → resample back
augment = torch.nn.Sequential(
    T.Resample(16000, 48000),
    AACSimulation(sample_rate=48000, bitrate=128000),
    T.Resample(48000, 16000),
)

for batch in dataloader:
    clean = batch["audio"]
    compressed = augment(clean)  # differentiable!
    loss = model(compressed, clean)
    loss.backward()
```
