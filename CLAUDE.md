# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **two distinct diffusion model implementations** for image generation on the MNIST dataset:

1. **`stabel_diffusion/`** - U-Net based diffusion model (CNN architecture)
2. **`diffusion_transformer/`** - DiT (Diffusion Transformer) based model (Transformer architecture)

Both implement denoising diffusion probabilistic models (DDPM) but with different neural network architectures.

## Development Commands

### U-Net Diffusion (`stabel_diffusion/`)

```bash
cd stabel_diffusion

# Train the model (200 epochs, batch size 100, saves to model.pt)
python train.py

# Generate images using the trained model
python inference.py

# View TensorBoard logs
tensorboard --logdir=../runs
```

### DiT Diffusion (`diffusion_transformer/`)

```bash
cd diffusion_transformer

# Train the DiT model (500 epochs, batch size 100, saves to model.pth)
python train.py

# Generate images using the trained model
python inference.py

# Test individual components
python dit.py
python diffusion.py
```

## Architecture Comparison

| Feature | U-Net (`stabel_diffusion/`) | DiT (`diffusion_transformer/`) |
|---------|----------------------------|-------------------------------|
| Architecture Type | CNN (U-Net) | Transformer |
| Image Size | 48×48 (upscaled) | 28×28 (native) |
| Patch/Block Size | N/A | 4×4 patches |
| Training Epochs | 200 | 500 |
| Conditional Generation | No | Yes (digit labels 0-9) |
| Time Embedding Dim | 256 | 64 |
| Channel/Hidden Dim | [1,64,128,256,512,1024] | 64 |
| TensorBoard | Yes | No |
| Model Checkpoint | `model.pt` | `model.pth` |

### Shared Diffusion Process

Both implementations use the **same diffusion process** defined in their respective `diffusion.py`:

- **Timesteps**: T = 1000
- **Beta schedule**: Linear from 0.0001 to 0.02
- **Noise addition**: `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`
- **Device auto-detection**: CUDA if available, else CPU

## U-Net Implementation (`stabel_diffusion/`)

### Core Components

- **`unet.py`**: U-Net architecture with encoder-decoder structure and skip connections
  - Channel progression: [1, 64, 128, 256, 512, 1024]
  - Time embedding dimension: 256
  - Uses max pooling for downsampling, transpose convolutions for upsampling

- **`conv_block.py`**: Convolutional blocks with time conditioning
  - 3×3 convolutions with batch normalization
  - Time embeddings added to feature maps before final convolution

- **`time_position_emb.py`**: Sinusoidal positional embeddings for timesteps (dim: 256)

- **`diffusion.py`**: Forward diffusion process
  - Pre-computed alpha and beta schedules
  - Note: Function name has typo `diffusion_foward` instead of `diffusion_forward`

- **`dataset.py`**: MNIST dataset loader with 48×48 image resizing

- **`train.py`**: Training loop with L1 loss, Adam optimizer (lr=0.001), TensorBoard logging

- **`inference.py`**: Reverse diffusion for generation

### Configuration

```python
IMG_SIZE = 48  # Upscaled from native 28×28
T = 1000
DEVICE = CUDA/CPU auto-detection
```

## DiT Implementation (`diffusion_transformer/`)

### Core Components

- **`dit.py`**: Main DiT model with patch embeddings and transformer blocks
  - Patch size: 4×4 → 7×7=49 patches for 28×28 images
  - Learnable positional embeddings: `[1, patch_count², emb_size]`
  - Supports label conditioning (digit classes 0-9)
  - **Bug Alert**: Line 77 has `x = self.patch_pos_emb(x)` which should be `x = x + self.patch_pos_emb`

- **`dit_block.py`**: Custom transformer block with adaptive layer norm
  - Time and label conditioning integrated via adaLN
  - Multi-head self-attention (4 heads)

- **`time_embedding.py`**: Sinusoidal positional embeddings for timesteps (dim: 64)

- **`diffusion.py`**: Forward diffusion process (same as U-Net version)
  - Note: Function name has typo `diffusion_foward` instead of `diffusion_forward`

- **`dataset.py`**: MNIST dataset loader (native 28×28)

- **`train.py`**: Training with L1 loss, 500 epochs, batch size 100
  - **Import Note**: When running directly, use absolute imports (not relative imports with `.`)

- **`inference.py`**: Reverse diffusion for generation

### Configuration

```python
T = 1000
DEVICE = CUDA/CPU auto-detection
```

### DiT Model Parameters

```python
img_size=28
patch_size=4
channel=1
emb_size=64
label_num=10
dit_num=3  # Number of transformer blocks
head=4     # Number of attention heads
```

## Design Patterns

- **Modular Architecture**: Each component separated into its own module
- **Time Conditioning**: All network components conditioned on diffusion timesteps via sinusoidal embeddings
- **Skip Connections**: U-Net uses skip connections to preserve fine-grained details
- **Patch Embeddings**: DiT uses patch-based tokenization like ViT
- **Conditional Generation**: DiT supports label-conditional generation via label embeddings
- **Device Management**: All tensors explicitly moved to configured device (CUDA/CPU)

## Training Details

### U-Net
- **Loss**: L1 loss for noise prediction
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 100
- **Epochs**: 200
- **Checkpoint**: `model.pt`
- **Monitoring**: TensorBoard logging enabled

### DiT
- **Loss**: L1 loss for noise prediction
- **Optimizer**: Adam (lr=1e-3)
- **Batch Size**: 100
- **Epochs**: 500
- **Checkpoint**: `model.pth`
- **Monitoring**: No TensorBoard

## Code Quirks and Known Issues

### Naming Typos
- **Function name**: `diffusion_foward` instead of `diffusion_forward` (used in both implementations)
- **Directory name**: `stabel_diffusion` instead of `stable_diffusion`
- **Inconsistent naming**: `time_embedding.py` vs `time_position_emb.py`

### Known Bugs
1. **`diffusion_transformer/dit.py:77`**: Positional embedding incorrectly used as function call
   ```python
   # Current (incorrect)
   x = self.patch_pos_emb(x)
   # Should be
   x = x + self.patch_pos_emb
   ```

2. **Positional embedding initialization**: Uses `torch.rand()` uniform [0,1) instead of standard truncated normal

### Import Considerations
- When running `diffusion_transformer/train.py` directly, use absolute imports (without `.` prefix)
- Relative imports only work when the module is run as part of a package

## Code Notes

- Code contains Chinese comments (中文注释)
- No formal test suite; modules can be tested by running them directly
- Both implementations share the same mathematical diffusion process but differ in network architecture
