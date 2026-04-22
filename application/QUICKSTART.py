#!/usr/bin/env python3
"""
Quick Start Guide - Copy & Paste Commands
"""

import subprocess
import sys

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")

def run_command(cmd, description):
    print(f"▶ {description}")
    print(f"  Command: {cmd}\n")

def main():
    print_section("REAL-TIME UPSCALER - QUICK START GUIDE")
    
    print("📍 Your setup:")
    print("  • OS: Fedora")
    print("  • CPU: 9600X")
    print("  • GPU: 9070XT (RDNA4)")
    print("  • Test video: /run/media/krishnateja/Coding/ca2030/Ca2030/output_lowres.mp4")
    
    print_section("STEP 1: INSTALL & TEST (Already Done ✓)")
    print("Dependencies installed:")
    run_command(
        "pip3 install --user -r application/requirements.txt",
        "Install required packages"
    )
    
    print("Pipeline tested successfully ✓")
    print("  • Hardware detection: Working")
    print("  • Model loading: Working")
    print("  • Batch size optimization: Working (optimal: 8)")
    print("  • Performance estimation: Working (using CPU fallback)")
    
    print_section("STEP 2: ENABLE GPU (RECOMMENDED)")
    print("⚠️  GPU not detected - ROCm not installed")
    print("\nTo enable GPU support on your 9070XT:")
    run_command(
        "bash application/installer/SETUP_ROCm.sh",
        "View ROCm installation guide"
    )
    
    print("Then manually:")
    print("  1. sudo dnf install -y rocm-runtime rocm-devel")
    print("  2. sudo usermod -aG video $USER")
    print("  3. newgrp video  (or restart session)")
    print("  4. pip3 install --user --upgrade onnxruntime-rocm")
    print("  5. python3 application/main.py --detect-hardware")
    
    print_section("STEP 3: TEST HARDWARE DETECTION")
    run_command(
        "cd /home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration",
        "Navigate to project directory"
    )
    
    run_command(
        "python3 application/main.py --detect-hardware",
        "Check if GPU is detected"
    )
    
    print_section("STEP 4: ESTIMATE PERFORMANCE")
    print("Before processing, test on first few frames:")
    run_command(
        "python3 application/main.py --estimate \\",
        "Estimate FPS on your test video"
    )
    print("  --video /run/media/krishnateja/Coding/ca2030/Ca2030/output_lowres.mp4 \\")
    print("  --target-fps 60")
    
    print("\nExpected output:")
    print("  ✅ PREVIEW MODE if FPS >= 60")
    print("  ⚠️  EXPORT MODE if FPS < 60")
    
    print_section("STEP 5: PROCESS VIDEO")
    print("If performance is good, process entire video:")
    run_command(
        "python3 application/main.py --process \\",
        "Full video processing"
    )
    print("  --video input.mp4 \\")
    print("  --output output.mp4 \\")
    print("  --fps 60 \\")
    print("  --interpolation 1 \\")
    print("  --codec mp4v")
    
    print("\nOptions:")
    print("  --fps RATE          Target FPS (default: 60)")
    print("  --interpolation N   Frames to generate per pair (1=2x FPS, 2=3x FPS, etc.)")
    print("  --codec CODEC       mp4v (default), h264, h265, vp9")
    print("  --batch-size N      Leave blank for auto-detection")
    
    print_section("CLI QUICK REFERENCE")
    
    commands = [
        ("Hardware Detection", "python3 application/main.py --detect-hardware"),
        ("Performance Estimate", "python3 application/main.py --estimate --video video.mp4 --target-fps 60"),
        ("Process Video (30→60fps)", "python3 application/main.py --process --video in.mp4 --output out.mp4 --fps 60 --interpolation 1"),
        ("Process Video (30→90fps)", "python3 application/main.py --process --video in.mp4 --output out.mp4 --fps 90 --interpolation 2"),
        ("Process with H.265", "python3 application/main.py --process --video in.mp4 --output out.mp4 --fps 60 --codec h265"),
        ("Debug Mode", "python3 application/main.py --process --video in.mp4 --debug"),
        ("Custom Batch Size", "python3 application/main.py --process --video in.mp4 --batch-size 4"),
    ]
    
    for description, command in commands:
        print(f"\n{description}")
        print(f"  {command}")
    
    print_section("EXPECTED PERFORMANCE")
    
    print("CPU (Current):")
    print("  • Input: 480x270 @ 30 FPS")
    print("  • Processing: ~210ms/frame")
    print("  • Output: ~4 FPS (export mode only)")
    print("  • Use for: Testing, small videos")
    
    print("\nGPU (After ROCm Setup):")
    print("  • Input: 480x270 @ 30 FPS")
    print("  • Processing: ~15-30ms/frame (estimated)")
    print("  • Output: 30-60+ FPS (real-time preview)")
    print("  • Use for: Production, real-time processing")
    
    print_section("FILE STRUCTURE")
    print("""
application/
├── main.py                          # Entry point
├── config.py                        # Configuration
├── test_pipeline.py                 # Test suite
├── requirements.txt                 # Dependencies
├── README.md                        # Full documentation
├── core/
│   ├── hardware_detector.py         # GPU detection
│   ├── onnx_loader.py              # Model loading
│   ├── performance_estimator.py    # FPS estimation
│   └── video_processor.py          # Main pipeline
├── utils/
│   ├── normalization.py            # Frame normalization
│   └── padding.py                  # Padding utilities
├── ui/                             # (UI files to be added)
├── models/                         # (Copy ONNX files here)
└── installer/
    ├── SETUP_ROCm.sh               # GPU setup guide
    └── build.py                    # (PyInstaller script to be added)
    """)
    
    print_section("NEXT STEPS")
    print("""
1. ✅ Core pipeline built and tested
2. ⏳ Install ROCm for GPU support
3. ⏳ Create PyQt6 GUI interface
4. ⏳ Build PyInstaller packages (Windows/macOS/Linux)
5. ⏳ Create proper installers
6. ⏳ Package and distribute
    """)
    
    print_section("TROUBLESHOOTING")
    
    issues = {
        "GPU not detected": "Run: bash application/installer/SETUP_ROCm.sh",
        "Models not found": "Ensure ONNX files are in correct directories",
        "Memory errors": "Try: python3 application/main.py --process --video in.mp4 --batch-size 2",
        "Slow processing": "Check GPU with: python3 application/main.py --detect-hardware",
        "Import errors": "Reinstall: pip3 install --user -r application/requirements.txt",
    }
    
    for issue, solution in issues.items():
        print(f"\n❌ {issue}")
        print(f"   ✓ {solution}")
    
    print_section("DOCUMENTATION")
    print("Full docs available in: application/README.md")
    print("Test results logged in: ~/.upscaler_config/logs/")
    
    print_section("YOU'RE READY!")
    print("""
The complete pipeline is working! ✨

Next action:
  1. Install ROCm to enable GPU
  2. Run: python3 application/main.py --detect-hardware
  3. Start processing videos!
    """)

if __name__ == '__main__':
    main()
