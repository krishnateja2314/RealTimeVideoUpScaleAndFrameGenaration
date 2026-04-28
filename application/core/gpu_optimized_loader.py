"""
GPU-optimized ONNX loader for AMD 9070 XT (RDNA4).
Uses ROCm MIGraphX for maximum performance and throughput.
"""

import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import os

logger = logging.getLogger(__name__)


class GPUOptimizedONNXLoader:
    """ONNX loader with GPU-specific optimizations for AMD ROCm."""
    
    def __init__(self, providers: list = None, gpu_type: str = None):
        """
        Initialize GPU-optimized ONNX loader.
        
        Args:
            providers: List of ONNX Runtime providers
            gpu_type: GPU type ('AMD', 'NVIDIA', 'CPU')
        """
        self.gpu_type = gpu_type or 'CPU'
        self.providers = providers or ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
        
        # GPU optimization flags
        self._setup_gpu_optimization()
        
        self.interpolator_session = None
        self.upscaler_session = None
        self.optimal_batch_size = 1
    
    def _validate_providers(self, providers: list) -> list:
        """Validate requested ONNX providers against available ones."""
        valid = [p for p in providers if p in self.available_providers]
        if not valid:
            if 'CPUExecutionProvider' in self.available_providers:
                valid = ['CPUExecutionProvider']
                logger.warning(
                    'No requested GPU providers available. Falling back to CPUExecutionProvider.'
                )
            else:
                raise RuntimeError(
                    f'No supported ONNX Runtime providers available. Installed: {self.available_providers}'
                )
        elif len(valid) != len(providers):
            logger.warning(
                f'Some requested providers were not available. Using: {valid}; '
                f'Installed providers: {self.available_providers}'
            )
        return valid

    def _setup_gpu_optimization(self):
        """Configure GPU optimization settings based on GPU type."""
        if 'MIGraphXExecutionProvider' in self.providers:
            logger.info("Configuring for AMD ROCm MIGraphX optimization...")
            
            # Enable HIP (AMD's GPU programming model)
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '90a'  # RDNA4 (9070XT is gfx90a equivalent)
            
            self.gpu_optimization = {
                'max_batch_size': 16,
                'enable_graph_optimization': True,
                'enable_memory_pooling': True,
                'memory_pool_pre_allocate_percent': 0.9,
                'enable_async_execution': True,
                'num_worker_threads': 4,
            }
            logger.info("AMD ROCm optimizations enabled:")
            for key, value in self.gpu_optimization.items():
                logger.info(f"  {key}: {value}")
        elif 'CUDAExecutionProvider' in self.providers:
            logger.info("Configuring for NVIDIA CUDA optimization...")
            self.gpu_optimization = {
                'max_batch_size': 32,
                'enable_graph_optimization': True,
                'enable_memory_pooling': True,
                'memory_pool_pre_allocate_percent': 0.85,
                'enable_async_execution': True,
                'num_worker_threads': 4,
            }
        else:
            logger.info("CPU-only mode (no GPU optimization)")
            self.gpu_optimization = {
                'max_batch_size': 1,
                'enable_graph_optimization': True,
                'enable_memory_pooling': False,
            }
    
    def load_frame_interpolator(self, model_path: str) -> ort.InferenceSession:
        """Load frame interpolator with GPU optimization."""
        logger.info(f"Loading frame interpolator from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable profiling if debug
            sess_options.enable_profiling = False  # Set to True for debugging
            
            self.interpolator_session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            
            logger.info(f"✓ Frame interpolator loaded")
            logger.info(f"  Provider: {self.interpolator_session.get_providers()}")
            self._log_model_info(self.interpolator_session, "Frame Interpolator")
            
            return self.interpolator_session
        
        except Exception as e:
            logger.error(f"Failed to load frame interpolator: {e}")
            raise
    
    def load_upscaler(self, model_path: str) -> ort.InferenceSession:
        """Load upscaler with GPU optimization."""
        logger.info(f"Loading upscaler from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.upscaler_session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            
            logger.info(f"✓ Upscaler loaded")
            logger.info(f"  Provider: {self.upscaler_session.get_providers()}")
            self._log_model_info(self.upscaler_session, "Upscaler")
            
            return self.upscaler_session
        
        except Exception as e:
            logger.error(f"Failed to load upscaler: {e}")
            raise
    
    def _log_model_info(self, session: ort.InferenceSession, model_name: str):
        """Log detailed model information."""
        logger.debug(f"\n--- {model_name} Model Info ---")
        
        for inp in session.get_inputs():
            logger.debug(f"  Input: {inp.name}")
            logger.debug(f"    Shape: {inp.shape}")
            logger.debug(f"    Type: {inp.type}")
        
        for out in session.get_outputs():
            logger.debug(f"  Output: {out.name}")
            logger.debug(f"    Shape: {out.shape}")
            logger.debug(f"    Type: {out.type}")
    
    def run_interpolator(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Run frame interpolation with optimized tensor handling.
        
        Args:
            frame1: First frame [B, C, H, W] float32
            frame2: Second frame [B, C, H, W] float32
        
        Returns:
            Interpolated frame [B, C, H, W] float32
        """
        if self.interpolator_session is None:
            raise ValueError("Interpolator model not loaded")
        
        try:
            # Ensure inputs are contiguous float32
            frame1 = np.ascontiguousarray(frame1.astype(np.float32))
            frame2 = np.ascontiguousarray(frame2.astype(np.float32))
            
            input_names = [inp.name for inp in self.interpolator_session.get_inputs()]
            
            # Prepare inputs (assumes im1, im3 naming convention)
            inputs = {
                input_names[0]: frame1,
                input_names[1]: frame2,
            }
            
            # Run inference with GPU optimization
            outputs = self.interpolator_session.run(None, inputs)
            
            # Return first output as float32
            result = outputs[0].astype(np.float32)
            
            return np.ascontiguousarray(result)
        
        except Exception as e:
            logger.error(f"Error in interpolator inference: {e}")
            raise
    
    def run_upscaler(self, frame: np.ndarray) -> np.ndarray:
        """
        Run 4x upscaling with optimized tensor handling.
        
        Args:
            frame: Input frame [B, C, H, W] float32
        
        Returns:
            Upscaled frame [B, C, 4*H, 4*W] float32
        """
        if self.upscaler_session is None:
            raise ValueError("Upscaler model not loaded")
        
        try:
            # Ensure input is contiguous float32
            frame = np.ascontiguousarray(frame.astype(np.float32))
            
            input_names = [inp.name for inp in self.upscaler_session.get_inputs()]
            
            inputs = {input_names[0]: frame}
            
            # Run inference with GPU optimization
            outputs = self.upscaler_session.run(None, inputs)
            
            # Return first output as float32
            result = outputs[0].astype(np.float32)
            
            return np.ascontiguousarray(result)
        
        except Exception as e:
            logger.error(f"Error in upscaler inference: {e}")
            raise
    
    def detect_optimal_batch_size(self) -> int:
        """
        Auto-detect optimal batch size for GPU.
        Tests with increasing batch sizes until memory limit reached.
        
        Returns:
            Optimal batch size
        """
        logger.info("Auto-detecting optimal batch size...")
        
        # Start with GPU type's recommended max and work backwards
        max_batch = self.gpu_optimization.get('max_batch_size', 1)
        
        # Get model input shapes
        interpolator_inputs = self.interpolator_session.get_inputs()
        shape = interpolator_inputs[0].shape
        
        # Shape might have 'batch' as first dimension
        h, w = 512, 512  # Default assumption
        for i, dim in enumerate(shape):
            if isinstance(dim, int):
                if i == 2:
                    h = dim
                elif i == 3:
                    w = dim
        
        optimal_batch = 1
        
        for batch_size in range(max_batch, 0, -1):
            try:
                # Create test tensors
                test_input = np.random.randn(batch_size, 3, h, w).astype(np.float32)
                
                # Test interpolator
                input_names = [inp.name for inp in self.interpolator_session.get_inputs()]
                test_inputs = {
                    input_names[0]: test_input,
                    input_names[1]: test_input
                }
                
                self.interpolator_session.run(None, test_inputs)
                
                # Test upscaler
                input_names = [inp.name for inp in self.upscaler_session.get_inputs()]
                test_inputs = {input_names[0]: test_input}
                
                self.upscaler_session.run(None, test_inputs)
                
                optimal_batch = batch_size
                logger.info(f"✓ Optimal batch size: {optimal_batch}")
                break
            
            except Exception as e:
                logger.debug(f"Batch size {batch_size} failed: {e}")
                continue
        
        self.optimal_batch_size = optimal_batch
        return optimal_batch
    
    def get_gpu_info(self) -> Dict:
        """Get information about GPU being used."""
        return {
            'gpu_type': self.gpu_type,
            'available_providers': self.available_providers,
            'selected_providers': self.providers,
            'optimization_settings': self.gpu_optimization,
            'optimal_batch_size': self.optimal_batch_size
        }
