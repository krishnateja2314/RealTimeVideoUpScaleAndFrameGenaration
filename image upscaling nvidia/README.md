# Real-Time Video Super-Resolution (Upscaling) Framework

A PyTorch-based super-resolution framework designed for real-time video upscaling (4x) using the Vimeo-90K dataset. This module integrates with a frame generation pipeline to enable low-framerate video playback in smooth slow-motion without pre-processing.

## Project Overview

**Problem Statement:**
Convert low framerate, low-resolution video to high framerate, high-resolution video in real-time, allowing smooth slow-motion playback while storing videos at smaller sizes.

**Components:**

- **Upscaler (this project):** CNN-based model for 4x image upscaling
- **Frame Generator:** GANs-based model for frame interpolation (already implemented)

## Dataset

**Vimeo-90K Dataset Structure:**

```
dataset/
├── input/                    # Bicubically upscaled frames (4x)
│   └── %05d/%04d/           # Two-level folder structure
│       ├── im1.png
│       ├── im2.png
│       └── ...im7.png
├── target/                   # Ground truth high-resolution frames
│   └── %05d/%04d/
│       ├── im1.png
│       └── ...im7.png
└── low_resolution/          # Original low-resolution frames
    └── %05d/%04d/
        └── ...im7.png
```

Each sequence contains 7 frames with the same naming convention.

## Installation

### 1. Clone/Setup the Project

```bash
cd image\ upscaling\ nvidia
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Available Models

### 1. **ESPCN (Efficient Sub-Pixel Convolutional Neural Network)** - Recommended

- **Advantages:** Fast inference, efficient, suitable for real-time processing
- **Architecture:** Sub-pixel convolution for efficient upsampling
- **Use Case:** Real-time video upscaling

### 2. **SRCNN (Super-Resolution Convolutional Neural Network)**

- **Advantages:** Lightweight, good quality
- **Architecture:** 3 convolutional layers
- **Use Case:** Quick training, testing

### 3. **Residual Super-Resolution**

- **Advantages:** Better quality, deeper network
- **Architecture:** Residual blocks for better gradient flow
- **Use Case:** High-quality output (slower than ESPCN)

## Training

### Basic Training

```bash
python train.py \
  --dataset_path ./data \
  --model_type espcn \
  --epochs 200 \
  --batch_size 16 \
  --learning_rate 1e-4
```

### Advanced Training Options

```bash
python train.py \
  --dataset_path ./data \
  --model_type residual \
  --epochs 300 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --scheduler cosine \
  --num_workers 4 \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
```

### Resume Training

```bash
python train.py \
  --dataset_path ./data \
  --resume ./checkpoints/checkpoint_epoch_50.pth
```

### Training Arguments

| Argument           | Default         | Description                         |
| ------------------ | --------------- | ----------------------------------- |
| `--dataset_path`   | `./data`        | Path to dataset root                |
| `--model_type`     | `espcn`         | Model type (espcn, srcnn, residual) |
| `--epochs`         | 200             | Number of training epochs           |
| `--batch_size`     | 16              | Batch size                          |
| `--learning_rate`  | 1e-4            | Initial learning rate               |
| `--weight_decay`   | 0               | L2 regularization                   |
| `--scheduler`      | step            | LR scheduler (step or cosine)       |
| `--step_size`      | 50              | Decay step size                     |
| `--gamma`          | 0.5             | LR decay factor                     |
| `--min_lr`         | 1e-6            | Minimum LR for cosine scheduler     |
| `--num_workers`    | 4               | Data loading workers                |
| `--save_every`     | 10              | Save checkpoint frequency           |
| `--checkpoint_dir` | `./checkpoints` | Checkpoint save directory           |
| `--log_dir`        | `./logs`        | TensorBoard log directory           |
| `--resume`         | None            | Resume from checkpoint              |

## Inference

### 1. Single Image Upscaling

```python
from inference import SuperResolutionInference
import cv2

upscaler = SuperResolutionInference(
    model_path='./checkpoints/best_model.pth',
    model_type='espcn',
    upscale_factor=4,
    device='cuda'
)

# Load and upscale image
image = cv2.imread('low_res_image.png')
upscaled = upscaler.upscale_image(image)
cv2.imwrite('upscaled_image.png', upscaled)
```

### 2. Video Upscaling

```python
upscaler = SuperResolutionInference(
    model_path='./checkpoints/best_model.pth',
    model_type='espcn',
    device='cuda'
)

upscaler.upscale_video('input_video.mp4', 'output_4x.mp4', fps=30)
```

### 3. Real-Time Frame Processing (Integration with Frame Generator)

```python
from inference import RealTimeVideoUpscaler
import cv2

# Initialize upscaler
upscaler = RealTimeVideoUpscaler(
    model_path='./checkpoints/best_model.pth',
    model_type='espcn',
    device='cuda'
)

# Process frames from video or stream
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Upscale frame
    upscaled_frame = upscaler.process_frame(frame)

    # Pass to frame generator or display
    cv2.imshow('Upscaled Frame', upscaled_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. Batch Processing

```python
from inference import SuperResolutionInference

upscaler = SuperResolutionInference(
    model_path='./checkpoints/best_model.pth'
)

# Process multiple frames at once (faster on GPU)
frames = [cv2.imread(f'frame_{i}.png') for i in range(10)]
upscaled_frames = upscaler.upscale_batch(frames)
```

## Performance Optimization

### GPU Memory Usage

- **ESPCN:** ~1GB for batch_size=16
- **SRCNN:** ~500MB for batch_size=16
- **Residual:** ~2GB for batch_size=16

### Inference Speed (on NVIDIA GPU)

- **ESPCN:** ~50-100 fps for 256x256 input
- **SRCNN:** ~60-120 fps for 256x256 input
- **Residual:** ~20-40 fps for 256x256 input

### Tips for Faster Inference

1. Use ESPCN model for real-time applications
2. Enable half-precision (FP16) for 2x speedup:

```python
upscaler = SuperResolutionInference(
    model_path='./checkpoints/best_model.pth',
    use_half_precision=True
)
```

3. Process frames in batches when possible
4. Reduce input resolution if needed

## Pipeline Integration

### Combining with Frame Generation

```python
from inference import RealTimeVideoUpscaler
from frame_generator import FrameGenerator  # Your GANs module

def process_video_pipeline(input_video, output_video, upscale_factor=4, fps_multiplier=4):
    """
    Complete pipeline: Upscale + Generate Frames + Output
    """

    # Initialize components
    upscaler = RealTimeVideoUpscaler(
        model_path='./checkpoints/best_model.pth',
        upscale_factor=upscale_factor,
        device='cuda'
    )

    frame_generator = FrameGenerator(model_path='frame_gen_model.pth')

    # Process video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = w * upscale_factor, h * upscale_factor

    out = cv2.VideoWriter(output_video,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps * fps_multiplier,
                          (out_w, out_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Upscale
        upscaled = upscaler.process_frame(frame)
        out.write(upscaled)

        # Step 2: Generate intermediate frames (fps_multiplier-1 frames)
        # This would use your GANs frame generator
        for _ in range(fps_multiplier - 1):
            generated_frame = frame_generator.generate(upscaled)
            out.write(generated_frame)

    cap.release()
    out.release()

# Usage
process_video_pipeline('input_240p_30fps.mp4',
                       'output_960p_120fps.mp4',
                       upscale_factor=4,
                       fps_multiplier=4)
```

## File Structure

```
image upscaling nvidia/
├── model.py              # Super-resolution models (ESPCN, SRCNN, Residual)
├── dataset.py            # Vimeo-90K dataset loader
├── train.py              # Training script
├── inference.py          # Inference and real-time processing
├── requirements.txt      # Python dependencies
└── README.md            # This file

checkpoints/
├── best_model.pth       # Best trained model
└── checkpoint_epoch_*.pth # Interim checkpoints

logs/                      # TensorBoard logs
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size: `--batch_size 8`
- Use SRCNN instead of Residual model
- Enable gradient checkpointing

### Slow Training

- Increase number of workers: `--num_workers 8`
- Use a machine with GPU
- Reduce validation frequency

### Poor Quality Output

- Ensure dataset is correctly structured
- Train for more epochs
- Use Residual model for better quality
- Check that input/target images are properly aligned

## References

- ESPCN: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
- SRCNN: Image Super-Resolution Using Deep Convolutional Networks
- Vimeo-90K: A Large-Scale Database for Video Restoration
- https://github.com/jiny2/dcnew-vimeo90k

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:

1. Check existing solutions in troubleshooting section
2. Verify dataset structure matches expected format
3. Ensure all dependencies are installed correctly
4. Check GPU VRAM is sufficient

---

**Note:** This is a super-resolution upscaling module designed to integrate with your frame generation pipeline. Combine this with your existing GANs frame generator for complete real-time 4K smooth slow-motion video playback capability.
