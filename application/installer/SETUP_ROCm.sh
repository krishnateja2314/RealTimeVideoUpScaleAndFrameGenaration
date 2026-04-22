#!/bin/bash
# ROCm Installation Guide for Fedora with RDNA4 GPU (9070XT)

echo "================================"
echo "ROCm Setup for AMD RDNA4 9070XT"
echo "================================"

# Check current system
echo ""
echo "Current System Info:"
uname -a
echo "Fedora version:"
cat /etc/fedora-release

echo ""
echo "Checking for existing ROCm installation..."
which rocm-smi 2>/dev/null && echo "✓ ROCm found" || echo "✗ ROCm not found"

echo ""
echo "Installation steps for Fedora:"
echo ""
echo "1. Add AMD ROCm repository:"
echo "   sudo dnf install -y 'dnf-command(config-manager)'"
echo "   sudo dnf config-manager --add-repo https://repo.radeon.com/rocm/rhel/rpm/"
echo ""
echo "2. Install ROCm runtime:"
echo "   sudo dnf install -y rocm-runtime rocm-devel"
echo ""
echo "3. Add user to video group:"
echo "   sudo usermod -aG video \$USER"
echo "   # Restart session or run: newgrp video"
echo ""
echo "4. Verify installation:"
echo "   rocm-smi"
echo ""
echo "5. Optional: Update ONNX Runtime for ROCm:"
echo "   pip3 install --user --upgrade onnxruntime-rocm"
echo ""
echo "================================"
echo "For more info: https://rocmdocs.amd.com/en/docs-5.7.0/deploy/linux/index.html"
echo "================================"
