# AMD 9070 XT ROCm Setup Guide for Fedora

Date: April 26, 2026
GPU: AMD Radeon RX 9070 XT (RDNA4)
CPU: Ryzen 9 9600X
OS: Fedora Linux

## Prerequisites

```bash
# Check Fedora version
cat /etc/os-release

# Expected: Fedora 39+
```

## Step 1: Install ROCm Runtime & Development Tools

```bash
# Update system packages
sudo dnf update -y

# Add ROCm repository (if not already added)
sudo dnf install -y rocm-runtime rocm-devel

# Verify installation
rocm-smi

# Expected output:
# ROCk module is loaded
# GPU Model   GFX Version
# 0x740d      90a (RDNA4)
```

### If rocm-smi fails:

```bash
# Check if driver is loaded
lsmod | grep amdgpu

# If not loaded, load the module
sudo modprobe amdgpu

# Add to startup (optional)
echo "amdgpu" | sudo tee /etc/modules-load.d/amdgpu.conf
```

## Step 2: Configure User Permissions

```bash
# Add current user to video group
sudo usermod -aG video $USER

# Apply group changes
newgrp video

# Verify (should show "video" in output)
groups
```

## Step 3: Install ONNX Runtime with ROCm

```bash
# Install with pip
pip3 install --user --upgrade onnxruntime-rocm

# Verify installation
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Expected output:
# ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
```

### If onnxruntime-rocm is not available:

```bash
# Try alternative installation method
pip3 install --user --no-cache-dir onnxruntime-rocm --index-url https://pypi.org/simple/

# Or use specific version
pip3 install --user onnxruntime-rocm==1.18.0
```

## Step 4: Configure ROCm Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# Source the file
source ~/.bashrc

# Verify
rocm-smi --showid

# Expected: Shows your 9070 XT GPU
```

## Step 5: Optimize for RDNA4 (9070 XT)

The application automatically configures for RDNA4, but you can manually optimize:

```bash
# Check GFX version
rocm-smi

# For 9070 XT, expect: GFX Version 90a (RDNA4)

# Set optimization flags (optional, already in application)
export HSA_OVERRIDE_GFX_VERSION=90a
```

## Step 6: Verify GPU Setup

```bash
# Run hardware detection
python main_optimized.py --detect-hardware

# Expected output:
# ✓ GPU Available | GPU: AMD | Provider: MIGraphXExecutionProvider
```

## Step 7: Test Performance

```bash
# Performance estimation (if you have a test video)
python main_optimized.py --estimate --video test.mp4 --fps 60

# Expected: Should show FPS >= 60 if GPU is working
```

## Performance Tuning

### Memory Settings

```bash
# Check available VRAM
rocm-smi --showmeminfo

# The 9070 XT has 16GB VRAM
# Application pre-allocates 90% for processing
```

### Batch Size Optimization

```bash
# The application auto-detects optimal batch size
# For 9070 XT: typically 8-16 frames per batch

# Manual override in code (if needed):
loader = GPUOptimizedONNXLoader()
loader.optimal_batch_size = 16  # For 9070 XT
```

## Troubleshooting

### Problem: "rocm-smi" command not found

```bash
# Solution: Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# Permanent fix: Add to ~/.bashrc
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
```

### Problem: GPU not detected in Python

```bash
# Check ONNX Runtime providers
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# If only shows CPU, reinstall ROCm ONNX Runtime:
pip3 uninstall onnxruntime -y
pip3 install --user --no-cache-dir onnxruntime-rocm
```

### Problem: "Permission denied" when accessing GPU

```bash
# Solution: Add user to video group
sudo usermod -aG video $USER
newgrp video

# Verify permission
groups  # Should show "video"
```

### Problem: Out of Memory (OOM) errors

```bash
# Check GPU memory usage
rocm-smi --showuse

# Solutions:
# 1. Reduce cache memory: --cache-memory-mb 1024
# 2. Reduce batch size: auto-detection will handle this
# 3. Reduce interpolation: --interpolation 1
# 4. Process at lower resolution
```

### Problem: Very slow processing (still using CPU)

```bash
# Verify GPU is being used:
rocm-smi --showuse
# Should show >0% GPU usage during processing

# Check for errors:
python main_optimized.py --detect-hardware --debug

# May need to reinstall MIGraphX:
sudo dnf reinstall rocm-core rocm-runtime rocm-devel -y
```

## Advanced Configuration

### Enable GPU Memory Pooling

```bash
# The application enables this automatically
# To verify it's working, check logs for:
# "enable_memory_pooling: True"
```

### Multi-GPU Support (Future)

```bash
# Check if you have multiple GPUs:
rocm-smi

# The application currently supports single GPU
# Multi-GPU support planned for future release
```

### Performance Profiling

```bash
# Enable detailed profiling (advanced users):
python main_optimized.py --process --video input.mp4 --profile --debug

# Generates profiling data in:
# ~/.upscaler_config/logs/
```

## Verification Checklist

- [ ] `rocm-smi` shows AMD GPU with gfx90a
- [ ] User is in video group (`groups` shows "video")
- [ ] `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` shows MIGraphX
- [ ] `python main_optimized.py --detect-hardware` shows "GPU Available"
- [ ] Performance estimation shows FPS >= 60 for your resolution
- [ ] First test video processes without corruption

## Performance Expectations

### AMD Radeon RX 9070 XT

- **VRAM**: 16GB
- **Stream Processors**: 2560
- **Memory Bandwidth**: 512 GB/s
- **Batch Size**: 8-16 frames
- **Throughput**:
  - 720p → 2880p: 50-70 FPS
  - 1080p → 4320p: 30-45 FPS
  - With 2x interpolation: ~60% throughput

## Next Steps

1. **Verify GPU Works**: `python main_optimized.py --detect-hardware`
2. **Test Performance**: `python main_optimized.py --estimate --video test.mp4`
3. **Process Video**: `python main_optimized.py --process --video input.mp4 --output output.mp4 --preview`

## Support

If you encounter issues:

1. Enable debug logging: `--debug` flag
2. Check logs: `~/.upscaler_config/logs/`
3. Verify GPU: `rocm-smi --showuse`
4. Check driver: `lsmod | grep amdgpu`

## References

- [ROCm Documentation](https://rocmdocs.amd.com/)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/)
- [AMD RDNA4 Architecture](https://www.amd.com/en/processors/radeon-9000)
