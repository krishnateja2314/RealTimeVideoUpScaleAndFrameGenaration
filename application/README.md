# Real-Time Video Frame Interpolation & 4x Upscaling

Complete pipeline for generating intermediate video frames and upscaling to 4x resolution in real-time.

## 🎯 Quick Start

### 1. Install Dependencies

```bash
cd application
pip3 install --user -r requirements.txt
```

### 2. Test the Pipeline

```bash
python3 test_pipeline.py
```

### 3. Use the Application

#### Detect Hardware

```bash
python3 main.py --detect-hardware
```

#### Estimate Performance

```bash
python3 main.py --estimate \
  --video your_video.mp4 \
  --target-fps 60
```

#### Process Video

```bash
python3 main.py --process \
  --video input.mp4 \
  --output output.mp4 \
  --fps 60 \
  --interpolation 1 \
  --codec mp4v
```

## 🚀 Features

### Hardware Auto-Detection

- **Linux**: ROCm (AMD) → CUDA (NVIDIA) → CPU
- **Windows**: CUDA → DirectML → CPU
- **macOS**: CoreML → CPU

### Performance Estimation

Tests first 5 frames to estimate:

- Achievable FPS
- Whether real-time preview is possible
- Processing time per frame

### Video Processing

- **Two-stage pipeline**: Interpolation → Upscaling
- **Batch processing**: Optimal batch size auto-detected
- **Multiple codecs**: MP4V, H.264, H.265, VP9
- **Flexible FPS**: Any target FPS achievable

## 📋 Architecture

### Core Modules

| Module                     | Purpose                                 |
| -------------------------- | --------------------------------------- |
| `hardware_detector.py`     | GPU/provider detection                  |
| `onnx_loader.py`           | Model loading + batch size optimization |
| `performance_estimator.py` | Test frames, estimate FPS               |
| `video_processor.py`       | Main interpolation + upscaling pipeline |

### Utilities

| Module             | Purpose                                 |
| ------------------ | --------------------------------------- |
| `normalization.py` | Frame normalization [0-255] ↔ [0.0-1.0] |
| `padding.py`       | Reflection padding to multiple of 32    |

## 🔧 Configuration

Edit `config.py` to customize:

```python
PERF_ESTIMATION_FRAMES = 5        # Test frames for estimation
PERF_OVERHEAD_MARGIN = 1.2         # 20% safety margin
DEFAULT_TARGET_FPS = 60            # Default output FPS
FRAME_PADDING_MULTIPLE = 32        # ONNX requirement
```

## 🐛 GPU Setup

### AMD RDNA4 (9070XT) on Fedora

```bash
# Run the setup script
bash installer/SETUP_ROCm.sh

# Or manually:
sudo dnf install -y rocm-runtime rocm-devel
sudo usermod -aG video $USER
newgrp video
```

Then verify:

```bash
rocm-smi
python3 main.py --detect-hardware
```

### NVIDIA GPU Setup

Ensure CUDA Toolkit 11.8+ is installed:

```bash
nvidia-smi
python3 main.py --detect-hardware
```

## 📊 Test Results

Current test on **Fedora 9070XT (CPU fallback)**:

- Test video: 480x270 @ 30 FPS (output_lowres.mp4)
- Frame processing: ~210ms per frame (CPU)
- CPU Mode: Fallback only, GPU recommended for real-time
- Batch size detected: 8

## 🎬 Pipeline Specifications

### Frame Interpolation (ONNX Model)

```
Input:  [B, 3, H, W] - Normalized [0.0, 1.0]
Output: [B, 3, H, W] - Single interpolated frame
Constraint: H, W must be multiples of 32
```

### 4x Upscaling (ONNX Model)

```
Input:  [B, 3, H, W] - Normalized [0.0, 1.0]
Output: [B, 3, 4H, 4W] - 4x spatial resolution
Process: Early upsampling strategy + pixel shuffling
```

## 📝 CLI Options

```
--detect-hardware     Only detect GPU and exit
--estimate           Estimate performance on video
--process            Process entire video
--video PATH         Input video path (required for estimate/process)
--output PATH        Output video path (default: output.mp4)
--target-fps FPS     Target output FPS (default: 60)
--interpolation N    Intermediate frames per pair (default: 1)
--codec CODEC        Output codec: mp4v, h264, h265, vp9 (default: mp4v)
--batch-size N       Batch size (auto-detect if not specified)
--debug              Enable debug logging
```

## ⚡ Performance Tips

1. **Use GPU**: Install ROCm/CUDA for real-time processing
2. **Monitor Memory**: Use batch size auto-detection
3. **Adjust FPS**: Lower target FPS increases feasibility
4. **Test First**: Always run `--estimate` before `--process`

## 📦 Models Used

- **Frame Interpolator**: Deep Flow-Based Interpolation (ONNX, opset_version=16)
- **Upscaler**: ESPCN 4x Spatial Super-Resolution (ONNX, opset_version=16)

Both models are optimized for:

- Dynamic batch sizes
- Dynamic resolutions
- CPU/GPU inference

## 🔐 Next Steps

1. **Install ROCm** for GPU acceleration
2. **Build installer** using PyInstaller
3. **Create GUI** (PyQt6 recommended)
4. **Package distribution** for Windows/macOS

## 📞 Troubleshooting

### GPU not detected

```bash
# Verify GPU is available
rocm-smi          # AMD
nvidia-smi        # NVIDIA

# Re-install ONNX Runtime for GPU
pip3 install --user --upgrade onnxruntime-rocm
```

### Memory errors

```bash
# Lower batch size
python3 main.py --process --video input.mp4 --batch-size 2
```

### Slow processing

```bash
# Check which provider is being used
python3 main.py --detect-hardware

# Lower target FPS to check feasibility
python3 main.py --estimate --video input.mp4 --target-fps 30
```

## 📄 License

[Your License Here]
