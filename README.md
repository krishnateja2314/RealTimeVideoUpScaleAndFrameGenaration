# RealTimeVideoUpScaleAndFrameGenaration

## Overview

This repository contains a full real-time video processing solution with:

- `application/`: complete pipeline for frame interpolation + 4x upscaling
- `Frame genaration on amd/`: AMD-targeted model training for frame interpolation
- `image upscaling nvidia/`: NVIDIA-targeted model training for video super-resolution

The root `README.md` provides:

- install instructions
- how to run the pipeline on AMD and NVIDIA
- how to train each model for its specific GPU
- links to official AMD and NVIDIA documentation for GPU setup

---

## Repository Structure

- `application/`
  - `main.py`, `main_improved.py`, `main_optimized.py` - pipeline entrypoints
  - `core/` - hardware detection, ONNX loading, performance estimation, video processing
  - `models/` - ONNX model files used by the pipeline
  - `requirements.txt` - pipeline Python dependencies
- `Frame genaration on amd/`
  - AMD-focused training and export code
  - `requirements.txt` - training dependencies
  - `test.py`, `train.py`, `export_onnx.py`
- `image upscaling nvidia/`
  - NVIDIA-focused super-resolution training and inference
  - `requirements.txt` - training dependencies
  - `train.py`, `inference.py`

---

## 1. Common Python Setup

1. Open a terminal in the repo root.
2. Create and activate a Python virtual environment.

```bash
cd /home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration
python3 -m venv venv
source venv/bin/activate
```

3. Install pipeline dependencies.

```bash
pip install --upgrade pip
pip install --user -r application/requirements.txt
```

> Note: `application/requirements.txt` currently includes `numpy` and `opencv-python`. GPU support requires ONNX Runtime GPU packages installed separately.

---

## 2. Run the Pipeline (`application`)

The pipeline uses:

- frame interpolation model: `frame_interpolator.onnx`
- upscaler model: `espcn_4x_dynamic.onnx`

### 2.1 Detect Hardware

```bash
cd application
python3 main.py --detect-hardware
```

### 2.2 Estimate Performance

```bash
python3 main.py --estimate --video /path/to/input.mp4 --target-fps 60
```

### 2.3 Process Video

```bash
python3 main.py --process --video /path/to/input.mp4 --output /path/to/output.mp4 --target-fps 60 --interpolation 1 --codec mp4v
```

### 2.4 GPU Options

- Use AMD/ROCm GPU inference:
  ```bash
  python3 main.py --process --video input.mp4 --output output.mp4 --use-gpu
  ```
- Force CPU inference:
  ```bash
  python3 main.py --process --video input.mp4 --output output.mp4 --cpu
  ```

> By default, `application/main.py` chooses CPU for stability. Pass `--use-gpu` to try GPU providers.

### 2.5 Pipeline Notes

- `--interpolation` sets the number of intermediate frames generated between source frames.
- `--codec` supports `mp4v`, `h264`, `h265`, `vp9`.
- `--batch-size` is optional; if omitted the code auto-detects a safe batch size.
- `--duration` can limit processing to the first N seconds of video.

---

## 3. GPU Installation Links

Use official vendor documentation so these links can stay accurate over time.

- AMD ROCm: https://rocmdocs.amd.com/
- AMD DirectML: https://learn.microsoft.com/windows/ai/directml/
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- PyTorch GPU install guide: https://pytorch.org/get-started/locally/
- ONNX Runtime execution providers: https://onnxruntime.ai/

---

## 4. Train the AMD Frame Interpolation Model

This model is intended for AMD GPU training on the AMD-specific training folder.

### 4.1 Install AMD training dependencies

```bash
cd "Frame genaration on amd"
pip install -r requirements.txt
pip install torch-directml
```

### 4.2 Optional AMD GPU package

If you need AMD GPU acceleration for ONNX or model export, install ROCm and then:

```bash
pip install --user --upgrade onnxruntime-rocm
```

### 4.3 Train the model

```bash
python3 train.py
```

### 4.4 Test / infer

Place `test/im1.png` and `test/im3.png`, then run:

```bash
python3 test.py
```

### 4.5 Notes for AMD model training

- This folder uses the AMD/DirectML GPU training path.
- The training workflow is best on AMD hardware.
- Official AMD docs are the stable source for future GPU setup changes.

---

## 5. Train the NVIDIA Upscaling Model

This model is intended for NVIDIA GPU training in the `image upscaling nvidia/` folder.

### 5.1 Install NVIDIA training dependencies

```bash
cd "image upscaling nvidia"
pip install -r requirements.txt
```

### 5.2 Install CUDA-aware PyTorch

Follow the PyTorch official install instructions for your CUDA version:

https://pytorch.org/get-started/locally/

For example, a supporting installation might be:

```bash
pip install torch==2.1.0 torchvision==0.16.0
```

### 5.3 Train the upscaler

```bash
python3 train.py --dataset_path ./data --epochs 200 --batch_size 16 --learning_rate 1e-4
```

### 5.4 Inference / evaluation

Use the `inference.py` module for single-image, batch, or video upscaling.

```bash
python3 inference.py
```

### 5.5 Notes for NVIDIA model training

- The training setup is designed for CUDA-enabled NVIDIA GPUs.
- Use the official NVIDIA CUDA Toolkit installation guide when updating drivers or CUDA versions.

---

## 6. Recommended Software Setup for Pipeline Execution

### 6.1 AMD GPU (Linux)

- Install ROCm from: https://rocmdocs.amd.com/
- Install ONNX Runtime ROCm package:
  ```bash
  pip install --user --upgrade onnxruntime-rocm
  ```
- Verify GPU:
  ```bash
  rocm-smi
  python3 application/main.py --detect-hardware
  ```

### 6.2 NVIDIA GPU

- Install CUDA from: https://developer.nvidia.com/cuda-toolkit
- Install ONNX Runtime GPU package:
  ```bash
  pip install --user --upgrade onnxruntime-gpu
  ```
- Verify GPU:
  ```bash
  nvidia-smi
  python3 application/main.py --detect-hardware
  ```

### 6.3 CPU-only fallback

If GPU is unavailable, the pipeline still runs on CPU with `--cpu`.

---

## 7. Troubleshooting

### `main.py` cannot find ONNX models

Ensure the files exist in the repository:

- `**/frame_interpolator.onnx`
- `**/espcn_4x*.onnx`

### GPU not detected

- AMD: check `rocm-smi`
- NVIDIA: check `nvidia-smi`

### ONNX provider missing

Install the GPU-specific ONNX Runtime package for your hardware.

### Slow performance

- Use `--use-gpu` when a compatible GPU is present
- Lower `--batch-size`
- Lower `--target-fps`

---

## 8. Additional Notes

- The `application/` pipeline is the main end-to-end runtime.
- `Frame genaration on amd/` is the AMD-specific model training folder.
- `image upscaling nvidia/` is the NVIDIA-specific super-resolution training folder.
- Official docs are linked for AMD and NVIDIA so this README remains stable as GPU installation details evolve.
