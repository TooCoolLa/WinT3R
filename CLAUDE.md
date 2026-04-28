# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WinT3R (Window-Based Streaming Reconstruction with Camera Token Pool) is a feed-forward model that infers precise camera poses and high-quality point maps from image streams in an online manner. It builds on DUSt3R, MASt3R, CUT3R, and VGGT.

## Running Inference

```bash
# Default example images
python recon.py

# Custom data (image directory or video file)
python recon.py --data_path <path/to/images_or_video> --inference_mode online

# Key arguments:
#   --data_path       Input image directory or video file (default: examples/001)
#   --inference_mode  "online" or "offline" (default: online)
#   --interval        Frame sampling interval for video input (default: 10)
#   --ckpt            Model checkpoint path (default: checkpoints/pytorch_model.bin)
#   --save_dir        Output directory (default: output)
#   --device          "cuda" or "cpu" (default: cuda)
```

Checkpoint must be downloaded from [HuggingFace](https://huggingface.co/lizizun/WinT3R/resolve/main/pytorch_model.bin) and placed at `checkpoints/pytorch_model.bin`.

## Installation

```bash
conda create -n WinT3R python=3.10
conda activate WinT3R
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Architecture

### Core Model: `dust3r/wint3r.py` — `WinT3R(CroCoNet)`

The main model class inherits from CroCoNet (CroCo backbone). The forward pass has three modes controlled by `mode` parameter:

- **`train`** (`_forward_impl`): Encodes all views at once, then processes in sliding windows of `window_size` (default 4) with stride `window_size//2`.
- **`online`** (`online_inference`): Encodes views per-window on the fly (lower memory, suitable for streaming).
- **`offline`** (`offline_inference`): Encodes all views upfront, then processes windows (higher quality).

**Key pipeline stages:**
1. **Patch embedding + ViT encoder** (`_encode_image` → `_encode_views`): Shared encoder processes all images into token features.
2. **State initialization** (`_init_state`): Creates a learnable state token pool (`register_tokens`, size `state_size=1024`) with 2D positional encoding.
3. **Windowed decoding** (`_decoder`): For each window, prepends a `cam_token` to image tokens, then runs dual-stream decoding:
   - `dec_blocks_state` (DecoderBlock): Updates global state tokens via self-attention + cross-attention with image tokens.
   - `dec_blocks` (GlobalLocalDecoderBlock): Updates image tokens via cross-attention with state + global self-attention + local per-view self-attention.
4. **Heads** (run in fp32 via `torch.amp.autocast(enabled=False)`):
   - `PtsHead`: Predicts local 3D points (xy*z, z via exp) + confidence from decoder features.
   - `CameraHead`: Iterative refinement (4 iterations) with adaptive layer normalization to predict camera extrinsics (absT_quaR encoding: 3-translation + 4-quaternion).

**Confidence-based view selection:** When a view appears in multiple overlapping windows, the prediction with highest total confidence is selected.

**Coordinate system:** Local points are transformed to world coordinates using predicted extrinsics. Relative poses are computed w.r.t. the first frame via `compute_relative_poses`.

### Key Module Map

| Path | Purpose |
|------|---------|
| `recon.py` | Inference entry point: loads images/video, runs model, saves .ply point cloud |
| `dust3r/wint3r.py` | Main `WinT3R` model class |
| `dust3r/blocks.py` | Decoder blocks: `DecoderBlock`, `GlobalLocalDecoderBlock`, attention modules |
| `dust3r/patch_embed.py` | Patch embedding (supports `ManyAR_PatchEmbed`) |
| `layers/camera_head.py` | `CameraHead` — iterative camera pose prediction with adaLN modulation |
| `layers/depth_head.py` | `PtsHead` — point map prediction with UV-aware upsampling |
| `layers/geometry.py` | Coordinate transforms, relative pose computation, quaternion math |
| `layers/pose_enc.py` | Quaternion↔rotation matrix conversions, pose encoding↔extrinsics |
| `layers/head_act.py` | Activation functions for pose and point predictions |
| `layers/block.py` | Transformer encoder block (from VGGT/DINO) |
| `layers/attention.py` | Attention with RoPE support |
| `dust3r/utils/image.py` | Image loading, preprocessing (resize/crop/normalize), depth edge detection |
| `dust3r/utils/vis_utils.py` | PLY file writing for point cloud output |
| `croco/` | CroCo backbone (ViT encoder, positional encoding, pretraining utilities) |

### Data Flow

```
Images → load_images_for_eval() → list of dicts {img, true_shape, idx}
  → WinT3R.forward(views, mode="online"|"offline")
    → _encode_views() → patch_embed + ViT encoder → (feat, pos, shape)
    → _init_state() → state tokens + 2D position encoding
    → For each window:
        → prepend cam_token to image tokens
        → _recurrent_rollout() → dual-stream decoder (state + image)
        → collect camera tokens + decoder features
    → PtsHead (fp32) → local 3D points + confidence
    → CameraHead (fp32) → camera extrinsics
    → confidence-based view selection
    → compute_relative_poses() → world coordinate points
  → write_ply() → output .ply file
```

## Conventions

- Quaternion order is **XYZW** (scalar-last) throughout the codebase.
- Pose encoding type is `absT_quaR` (7-dim: 3 translation + 4 quaternion).
- Image normalization uses ImageNet mean/std: `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`.
- Supported image resolutions are constrained to multiples of 16 (patch_size=16); default is 512×384.
- The `croco/` directory is a vendored dependency (CroCo model) with its own license (CC BY-NC-SA 4.0).
- Training and evaluation code are not yet released (TODO in README).
