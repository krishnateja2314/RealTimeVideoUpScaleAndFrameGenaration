#!/usr/bin/env python3
"""
Comprehensive test suite for optimized video processing pipeline.
Tests tensor handling, GPU optimization, and preview system.
"""

import sys
import logging
import numpy as np
from pathlib import Path
import cv2
import time

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestSuite:
    """Comprehensive testing suite."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def test(self, name):
        """Decorator for test functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"TEST: {name}")
                    logger.info(f"{'='*60}")
                    func(*args, **kwargs)
                    logger.info(f"✓ PASSED: {name}")
                    self.passed += 1
                except AssertionError as e:
                    logger.error(f"✗ FAILED: {name}")
                    logger.error(f"  {e}")
                    self.failed += 1
                except Exception as e:
                    logger.error(f"✗ ERROR: {name}")
                    logger.error(f"  {type(e).__name__}: {e}")
                    self.failed += 1
            return wrapper
        return decorator
    
    def assert_equal(self, actual, expected, msg=""):
        """Assert equality."""
        if actual != expected:
            raise AssertionError(f"{msg}\nExpected: {expected}\nActual: {actual}")
    
    def assert_true(self, condition, msg=""):
        """Assert condition is true."""
        if not condition:
            raise AssertionError(f"Assertion failed: {msg}")
    
    def assert_shape(self, array, expected_shape, msg=""):
        """Assert array shape."""
        if array.shape != expected_shape:
            raise AssertionError(f"Shape mismatch {msg}\nExpected: {expected_shape}\nActual: {array.shape}")
    
    def assert_dtype(self, array, expected_dtype, msg=""):
        """Assert array dtype."""
        if array.dtype != expected_dtype:
            raise AssertionError(f"Dtype mismatch {msg}\nExpected: {expected_dtype}\nActual: {array.dtype}")
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Passed: {self.passed}/{total}")
        logger.info(f"Failed: {self.failed}/{total}")
        logger.info(f"{'='*60}\n")
        return self.failed == 0


def main():
    """Run all tests."""
    suite = TestSuite()
    
    # ========================================
    # TEST 1: HARDWARE DETECTION
    # ========================================
    @suite.test("Hardware Detection")
    def test_hardware_detection():
        from core import HardwareDetector
        
        detector = HardwareDetector()
        gpu_type = detector.get_gpu_type()
        is_gpu = detector.is_gpu_available()
        provider = detector.get_gpu_provider()
        
        logger.info(f"  GPU Type: {gpu_type}")
        logger.info(f"  GPU Available: {is_gpu}")
        logger.info(f"  Provider: {provider}")
        
        suite.assert_true(gpu_type in ['AMD', 'NVIDIA', 'CPU', 'DirectML', 'Apple Silicon / Intel'],
                         "Invalid GPU type detected")
        suite.assert_true(provider in ['MIGraphXExecutionProvider', 'CUDAExecutionProvider',
                                       'DirectmlExecutionProvider', 'CoreMLExecutionProvider',
                                       'CPUExecutionProvider'],
                         "Invalid provider detected")
    
    test_hardware_detection()
    
    # ========================================
    # TEST 2: NORMALIZATION UTILITIES
    # ========================================
    @suite.test("Frame Normalization")
    def test_normalization():
        from utils.normalization import normalize_frame, denormalize_frame, to_nchw, to_hwc
        
        # Create test frame [H, W, C] uint8
        test_frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Test normalize
        normalized = normalize_frame(test_frame)
        suite.assert_dtype(normalized, np.float32, "Normalization should return float32")
        suite.assert_true(normalized.min() >= 0.0 and normalized.max() <= 1.0,
                         "Normalized values should be in [0, 1]")
        
        # Test denormalize
        denormalized = denormalize_frame(normalized)
        suite.assert_dtype(denormalized, np.uint8, "Denormalization should return uint8")
        suite.assert_true(np.allclose(denormalized, test_frame, atol=1),
                         "Denormalized should match original within tolerance")
        
        # Test NCHW conversion
        nchw = to_nchw(normalized)
        suite.assert_shape(nchw, (1, 3, 256, 256), "NCHW should be [1, 3, H, W]")
        
        # Test HWC conversion
        hwc = to_hwc(nchw)
        suite.assert_shape(hwc, (256, 256, 3), "HWC should be [H, W, C]")
        
        logger.info("  ✓ Normalization pipeline working correctly")
    
    test_normalization()
    
    # ========================================
    # TEST 3: PADDING UTILITIES
    # ========================================
    @suite.test("Frame Padding")
    def test_padding():
        from utils.padding import reflect_pad, crop_padding, get_padding_size
        
        # Create test frame
        test_frame = np.random.randn(1, 3, 100, 100).astype(np.float32)
        
        # Test reflection padding
        padded, pad_info = reflect_pad(test_frame)
        
        # Height and width should be multiples of 32
        suite.assert_true(padded.shape[2] % 32 == 0, "Padded height should be multiple of 32")
        suite.assert_true(padded.shape[3] % 32 == 0, "Padded width should be multiple of 32")
        
        # Test crop
        cropped = crop_padding(padded, pad_info)
        suite.assert_shape(cropped, test_frame.shape, "Cropped should match original shape")
        
        logger.info(f"  Original shape: {test_frame.shape}")
        logger.info(f"  Padded shape: {padded.shape}")
        logger.info(f"  Cropped shape: {cropped.shape}")
    
    test_padding()
    
    # ========================================
    # TEST 4: TENSOR FINALIZATION (KEY FIX)
    # ========================================
    @suite.test("Tensor Finalization (Corruption Fix)")
    def test_tensor_finalization():
        from core.improved_video_processor import ImprovedVideoProcessor
        from utils.padding import reflect_pad, crop_padding
        from utils.normalization import to_nchw
        
        # Create mock processor
        class MockModelLoader:
            pass
        
        processor = ImprovedVideoProcessor(MockModelLoader())
        
        # Test various tensor formats
        test_cases = [
            (np.random.randn(1, 3, 512, 512).astype(np.float32), "Standard NCHW [0,1]"),
            (np.random.randn(1, 3, 512, 512).astype(np.float32) * 2 - 1, "NCHW [-1,1]"),
            (np.random.randn(3, 512, 512).astype(np.float32), "CHW [0,1]"),
        ]
        
        for tensor, desc in test_cases:
            # Create padding info
            if tensor.ndim == 4:
                _, pad_info = reflect_pad(tensor)
            else:
                tensor_nchw = to_nchw(tensor)
                _, pad_info = reflect_pad(tensor_nchw)
            
            try:
                result = processor._finalize_frame(tensor.reshape(1, 3, 512, 512), pad_info)
                
                suite.assert_dtype(result, np.uint8, f"Output should be uint8 for {desc}")
                suite.assert_true(result.min() >= 0 and result.max() <= 255,
                                 f"Output should be [0,255] for {desc}")
                suite.assert_true(result.ndim == 3 and result.shape[2] == 3,
                                 f"Output should be HWC RGB for {desc}")
                
                logger.info(f"  ✓ {desc}: {result.shape} uint8")
            
            except Exception as e:
                logger.error(f"  ✗ {desc}: {e}")
                raise
    
    test_tensor_finalization()
    
    # ========================================
    # TEST 5: FRAME CACHE
    # ========================================
    @suite.test("Frame Cache System")
    def test_frame_cache():
        from core import FrameCache
        
        cache = FrameCache(max_memory_mb=512)
        
        # Add frames
        for i in range(10):
            frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            success = cache.add_frame(i, frame)
            suite.assert_true(success, f"Should be able to add frame {i}")
        
        # Test retrieval
        frame = cache.get_frame(0)
        suite.assert_true(frame is not None, "Should retrieve cached frame")
        
        # Test cache stats
        stats = cache.get_stats()
        logger.info(f"  Cache size: {stats['num_frames']} frames")
        logger.info(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB / {stats['max_memory_mb']:.1f} MB")
    
    test_frame_cache()
    
    # ========================================
    # TEST 6: GPU OPTIMIZATION
    # ========================================
    @suite.test("GPU Optimization Configuration")
    def test_gpu_optimization():
        from core import HardwareDetector, GPUOptimizedONNXLoader
        
        detector = HardwareDetector()
        loader = GPUOptimizedONNXLoader(
            providers=detector.get_providers(),
            gpu_type=detector.get_gpu_type()
        )
        
        info = loader.get_gpu_info()
        logger.info(f"  GPU Type: {info['gpu_type']}")
        logger.info(f"  Providers: {info['providers']}")
        logger.info(f"  Batch Size: {info['optimal_batch_size']}")
        logger.info(f"  Settings: {info['optimization_settings']}")
        
        suite.assert_true(info['optimal_batch_size'] >= 1, "Batch size must be at least 1")
    
    test_gpu_optimization()
    
    # ========================================
    # TEST 7: END-TO-END PROCESSING
    # ========================================
    @suite.test("End-to-End Frame Processing")
    def test_e2e_processing():
        from core.improved_video_processor import ImprovedVideoProcessor
        
        class MockModelLoader:
            def run_interpolator(self, f1, f2):
                # Return average of inputs
                return (f1 + f2) / 2
            
            def run_upscaler(self, frame):
                # Return 4x upscaled frame
                b, c, h, w = frame.shape
                return np.repeat(np.repeat(frame, 4, axis=2), 4, axis=3)
        
        processor = ImprovedVideoProcessor(MockModelLoader())
        
        # Create test frames
        f1 = np.random.uniform(0, 1, (256, 256, 3)).astype(np.float32)
        f2 = np.random.uniform(0, 1, (256, 256, 3)).astype(np.float32)
        
        # Process
        output_frames = processor._process_frame_pair(f1, f2, interpolation_factor=1, is_last_pair=False)
        
        logger.info(f"  Output frames: {len(output_frames)}")
        for i, frame in enumerate(output_frames):
            logger.info(f"    Frame {i}: {frame.shape} {frame.dtype}")
            suite.assert_dtype(frame, np.uint8, f"Output frame {i} should be uint8")
            suite.assert_true(frame.min() >= 0 and frame.max() <= 255,
                             f"Output frame {i} should be [0,255]")
    
    test_e2e_processing()
    
    # ========================================
    # PRINT SUMMARY
    # ========================================
    if suite.summary():
        logger.info("✓ ALL TESTS PASSED")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
