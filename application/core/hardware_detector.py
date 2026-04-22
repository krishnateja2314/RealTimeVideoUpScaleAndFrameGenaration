"""
Detect hardware capabilities and select appropriate ONNX Runtime providers.
Supports: ROCm (AMD), CUDA (NVIDIA), DirectML (Windows), CoreML (macOS), CPU
"""

import platform
import subprocess
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect GPU and CPU capabilities, select optimal ONNX providers."""
    
    PROVIDER_PRIORITY = {
        'Linux': ['ROCmExecutionProvider', 'CPUExecutionProvider'],
        'Windows': ['CUDAExecutionProvider', 'DirectmlExecutionProvider', 'CPUExecutionProvider'],
        'Darwin': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
    }
    
    def __init__(self):
        self.os_name = platform.system()
        self.gpu_info = {}
        self.detected_providers = []
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available GPUs based on OS."""
        if self.os_name == 'Linux':
            self._detect_linux_gpu()
        elif self.os_name == 'Windows':
            self._detect_windows_gpu()
        elif self.os_name == 'Darwin':
            self._detect_macos_gpu()
        
        logger.info(f"Detected OS: {self.os_name}")
        logger.info(f"GPU Info: {self.gpu_info}")
    
    def _detect_linux_gpu(self):
        """Detect GPU on Linux using rocm-smi or nvidia-smi."""
        try:
            # Check for AMD/ROCm
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.gpu_info['type'] = 'AMD'
                self.gpu_info['provider'] = 'ROCmExecutionProvider'
                self._parse_rocm_output(result.stdout)
                logger.info("✓ AMD ROCm GPU detected")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # Check for NVIDIA/CUDA
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.gpu_info['type'] = 'NVIDIA'
                self.gpu_info['provider'] = 'CUDAExecutionProvider'
                self._parse_nvidia_output(result.stdout)
                logger.info("✓ NVIDIA CUDA GPU detected")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Fallback to CPU
        self.gpu_info['type'] = 'CPU'
        self.gpu_info['provider'] = 'CPUExecutionProvider'
        logger.warning("⚠ No GPU detected, falling back to CPU")
    
    def _detect_windows_gpu(self):
        """Detect GPU on Windows using nvidia-smi or DirectML."""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.gpu_info['type'] = 'NVIDIA'
                self.gpu_info['provider'] = 'CUDAExecutionProvider'
                self._parse_nvidia_output(result.stdout)
                logger.info("✓ NVIDIA CUDA GPU detected on Windows")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Default to DirectML on Windows
        self.gpu_info['type'] = 'DirectML'
        self.gpu_info['provider'] = 'DirectmlExecutionProvider'
        logger.info("ℹ DirectML will be used on Windows")
    
    def _detect_macos_gpu(self):
        """Detect GPU on macOS."""
        self.gpu_info['type'] = 'Apple Silicon / Intel'
        self.gpu_info['provider'] = 'CoreMLExecutionProvider'
        logger.info("ℹ CoreML provider set for macOS")
    
    def _parse_rocm_output(self, output: str):
        """Parse rocm-smi output to extract GPU info."""
        try:
            # Parse GPU name and memory
            for line in output.split('\n'):
                if 'GPU' in line and 'Memory' in line:
                    self.gpu_info['memory_info'] = line.strip()
                    break
        except Exception as e:
            logger.debug(f"Could not parse ROCm output: {e}")
    
    def _parse_nvidia_output(self, output: str):
        """Parse nvidia-smi output to extract GPU info."""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'GPU' in line:
                    self.gpu_info['name'] = line.strip()
                    break
        except Exception as e:
            logger.debug(f"Could not parse NVIDIA output: {e}")
    
    def get_providers(self) -> List[str]:
        """
        Get ONNX Runtime providers in priority order.
        
        Returns:
            List of provider names to pass to ONNX Session
        """
        providers = self.PROVIDER_PRIORITY.get(self.os_name, ['CPUExecutionProvider'])
        logger.info(f"ONNX Provider priority: {providers}")
        return providers
    
    def get_gpu_type(self) -> str:
        """Get detected GPU type (AMD, NVIDIA, CPU, etc.)"""
        return self.gpu_info.get('type', 'CPU')
    
    def get_gpu_provider(self) -> str:
        """Get the selected ONNX provider."""
        return self.gpu_info.get('provider', 'CPUExecutionProvider')
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_info.get('type') != 'CPU'
    
    def get_summary(self) -> str:
        """Get a human-readable summary of detected hardware."""
        gpu_type = self.get_gpu_type()
        provider = self.get_gpu_provider()
        status = "✓ GPU Available" if self.is_gpu_available() else "⚠ CPU Only"
        return f"{status} | GPU: {gpu_type} | Provider: {provider}"
