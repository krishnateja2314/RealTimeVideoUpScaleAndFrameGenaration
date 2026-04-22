"""
ONNX Model Loader with automatic batch size detection.
Loads frame interpolator and upscaler models with optimal batch sizing.
"""

import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """Load and manage ONNX models with dynamic batch size optimization."""
    
    def __init__(self, providers: list = None):
        """
        Initialize ONNX model loader.
        
        Args:
            providers: List of ONNX Runtime providers in priority order
        """
        self.providers = providers or ['CPUExecutionProvider']
        self.interpolator_session = None
        self.upscaler_session = None
        self.optimal_batch_size = None
    
    def load_frame_interpolator(self, model_path: str) -> ort.InferenceSession:
        """
        Load frame interpolator model.
        
        Args:
            model_path: Path to frame_interpolator.onnx
        
        Returns:
            ONNX InferenceSession
        """
        logger.info(f"Loading frame interpolator from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            self.interpolator_session = ort.InferenceSession(
                model_path,
                providers=self.providers
            )
            logger.info(f"✓ Frame interpolator loaded with provider: {self.interpolator_session.get_providers()}")
            
            # Log input/output info
            self._log_model_info(self.interpolator_session, "Frame Interpolator")
            
            return self.interpolator_session
        
        except Exception as e:
            logger.error(f"Failed to load frame interpolator: {e}")
            raise
    
    def load_upscaler(self, model_path: str) -> ort.InferenceSession:
        """
        Load 4x upscaler model.
        
        Args:
            model_path: Path to espcn_4x.onnx
        
        Returns:
            ONNX InferenceSession
        """
        logger.info(f"Loading upscaler from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            self.upscaler_session = ort.InferenceSession(
                model_path,
                providers=self.providers
            )
            logger.info(f"✓ Upscaler loaded with provider: {self.upscaler_session.get_providers()}")
            
            # Log input/output info
            self._log_model_info(self.upscaler_session, "Upscaler")
            
            return self.upscaler_session
        
        except Exception as e:
            logger.error(f"Failed to load upscaler: {e}")
            raise
    
    def _log_model_info(self, session: ort.InferenceSession, model_name: str):
        """Log input/output information for debugging."""
        logger.info(f"\n--- {model_name} Model Info ---")
        
        for inp in session.get_inputs():
            logger.info(f"  Input: {inp.name}")
            logger.info(f"    Shape: {inp.shape}")
            logger.info(f"    Type: {inp.type}")
        
        for out in session.get_outputs():
            logger.info(f"  Output: {out.name}")
            logger.info(f"    Shape: {out.shape}")
            logger.info(f"    Type: {out.type}")
    
    def auto_detect_batch_size(
        self,
        session: ort.InferenceSession,
        max_attempts: int = 5,
        start_batch: int = 1,
        memory_limit_gb: float = 2.0
    ) -> int:
        """
        Auto-detect optimal batch size by testing incrementally.
        Starts from batch_size=1 and increases until GPU memory exhausts.
        
        Args:
            session: ONNX InferenceSession to test
            max_attempts: Maximum number of batch size tests
            start_batch: Starting batch size
            memory_limit_gb: Stop when approaching this GPU memory limit
        
        Returns:
            Optimal batch size
        """
        logger.info(f"Auto-detecting optimal batch size (starting from {start_batch})...")
        
        # Get input shape info
        input_info = []
        for inp in session.get_inputs():
            input_info.append({
                'name': inp.name,
                'shape': inp.shape,
                'type': inp.type
            })
        
        optimal_batch = start_batch
        
        for attempt in range(max_attempts):
            batch_size = start_batch * (2 ** attempt)
            
            try:
                # Create dummy input tensors
                test_inputs = self._create_test_inputs(input_info, batch_size)
                
                # Run inference
                logger.debug(f"Testing batch size: {batch_size}")
                session.run(None, test_inputs)
                
                optimal_batch = batch_size
                logger.info(f"  ✓ Batch size {batch_size} successful")
                
                # Clean up
                del test_inputs
            
            except RuntimeError as e:
                if "Out of memory" in str(e) or "CUDA" in str(e):
                    logger.warning(f"  ✗ Batch size {batch_size} exceeded memory limit")
                    break
                raise
        
        self.optimal_batch_size = optimal_batch
        logger.info(f"✓ Optimal batch size detected: {optimal_batch}")
        
        return optimal_batch
    
    def _create_test_inputs(self, input_info: list, batch_size: int) -> dict:
        """
        Create dummy input tensors for batch size testing.
        Handles dynamic shape dimensions (represented as strings).
        """
        test_inputs = {}
        
        for inp_info in input_info:
            name = inp_info['name']
            shape = inp_info['shape']
            
            # Convert dynamic shape to concrete shape
            concrete_shape = []
            for dim in shape:
                if isinstance(dim, str):
                    # Dynamic dimension represented as string
                    # Map common ONNX shape names to concrete values
                    if 'batch' in dim.lower() or dim == 'N' or dim == 'B':
                        concrete_shape.append(batch_size)
                    elif 'height' in dim.lower() or dim == 'H':
                        concrete_shape.append(256)  # Arbitrary test height
                    elif 'width' in dim.lower() or dim == 'W':
                        concrete_shape.append(256)  # Arbitrary test width
                    elif 'channel' in dim.lower() or dim == 'C':
                        concrete_shape.append(3)  # RGB
                    else:
                        # Unknown dynamic dimension, use a default
                        concrete_shape.append(1)
                elif dim is None:
                    # None means dynamic in ONNX
                    concrete_shape.append(batch_size if len(concrete_shape) == 0 else 256)
                else:
                    # Fixed dimension
                    concrete_shape.append(int(dim))
            
            # Create random float32 tensor
            test_inputs[name] = np.random.randn(*concrete_shape).astype(np.float32)
            logger.debug(f"Created test input {name}: {concrete_shape}")
        
        return test_inputs
    
    def get_optimal_batch_size(self) -> int:
        """Get previously detected optimal batch size."""
        if self.optimal_batch_size is None:
            logger.warning("Batch size not detected yet, using default of 1")
            return 1
        return self.optimal_batch_size
    
    def run_interpolator(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Run frame interpolation inference.
        
        Args:
            frame1: First frame [B, C, H, W]
            frame2: Second frame (third frame) [B, C, H, W]
        
        Returns:
            Interpolated frame [B, C, H, W]
        """
        if self.interpolator_session is None:
            raise ValueError("Interpolator model not loaded")
        
        # Get input names from model
        input_names = [inp.name for inp in self.interpolator_session.get_inputs()]
        
        # Prepare inputs (assumes im1, im3 naming from spec)
        inputs = {
            input_names[0]: frame1.astype(np.float32),
            input_names[1]: frame2.astype(np.float32),
        }
        
        # Run inference
        outputs = self.interpolator_session.run(None, inputs)
        
        return outputs[0].astype(np.float32)
    
    def run_upscaler(self, frame: np.ndarray) -> np.ndarray:
        """
        Run 4x upscaling inference.
        
        Args:
            frame: Input frame [B, C, H, W]
        
        Returns:
            Upscaled frame [B, C, 4*H, 4*W]
        """
        if self.upscaler_session is None:
            raise ValueError("Upscaler model not loaded")
        
        # Get input name from model
        input_names = [inp.name for inp in self.upscaler_session.get_inputs()]
        
        # Prepare input
        inputs = {input_names[0]: frame.astype(np.float32)}
        
        # Run inference
        outputs = self.upscaler_session.run(None, inputs)
        
        return outputs[0].astype(np.float32)
