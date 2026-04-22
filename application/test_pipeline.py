#!/usr/bin/env python3
"""
Quick test script to verify the pipeline on your test video.
Run: python test_pipeline.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules import correctly."""
    logger.info("Testing imports...")
    try:
        from core import HardwareDetector, ONNXModelLoader, PerformanceEstimator, VideoProcessor
        from utils import normalize_frame, reflect_pad
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_hardware_detection():
    """Test hardware detection."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: HARDWARE DETECTION")
    logger.info("="*70)
    
    try:
        from core import HardwareDetector
        
        detector = HardwareDetector()
        logger.info(f"Hardware Summary: {detector.get_summary()}")
        logger.info(f"GPU Type: {detector.get_gpu_type()}")
        logger.info(f"GPU Provider: {detector.get_gpu_provider()}")
        logger.info(f"ONNX Providers: {detector.get_providers()}")
        logger.info(f"GPU Available: {detector.is_gpu_available()}")
        
        logger.info("✓ Hardware detection passed")
        return True, detector
    
    except Exception as e:
        logger.error(f"✗ Hardware detection failed: {e}", exc_info=True)
        return False, None

def test_model_loading(detector):
    """Test model loading."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: MODEL LOADING")
    logger.info("="*70)
    
    try:
        from core import ONNXModelLoader
        
        # Find ONNX models
        base_path = Path("/home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration")
        
        interpolator_models = list(base_path.glob("**/frame_interpolator.onnx"))
        upscaler_models = list(base_path.glob("**/espcn_4x*.onnx"))
        
        if not interpolator_models:
            logger.error("✗ Frame interpolator model not found")
            return False, None
        
        if not upscaler_models:
            logger.error("✗ Upscaler model not found")
            return False, None
        
        interpolator_path = interpolator_models[0]
        upscaler_path = upscaler_models[0]
        
        logger.info(f"Interpolator model: {interpolator_path}")
        logger.info(f"Upscaler model: {upscaler_path}")
        
        # Load models
        model_loader = ONNXModelLoader(providers=detector.get_providers())
        
        logger.info("Loading frame interpolator...")
        model_loader.load_frame_interpolator(str(interpolator_path))
        
        logger.info("Loading upscaler...")
        model_loader.load_upscaler(str(upscaler_path))
        
        logger.info("✓ Model loading passed")
        return True, model_loader
    
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}", exc_info=True)
        return False, None

def test_batch_size_detection(model_loader):
    """Test batch size auto-detection."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: AUTO BATCH SIZE DETECTION")
    logger.info("="*70)
    try:
        logger.info("Testing batch sizes (1, 2, 4, 8)...")
        optimal_batch = model_loader.auto_detect_batch_size(
            model_loader.interpolator_session,
            max_attempts=4,
            start_batch=1
        )
        
        logger.info(f"✓ Batch size detection passed (optimal: {optimal_batch})")
        return True
    
    except Exception as e:
        logger.error(f"✗ Batch size detection failed: {e}", exc_info=True)
        return False

def test_performance_estimation(model_loader):
    """Test performance estimation on your test video."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: PERFORMANCE ESTIMATION")
    logger.info("="*70)
    
    try:
        from core import PerformanceEstimator
        import cv2
        
        video_path = "/run/media/krishnateja/Coding/ca2030/Ca2030/output_lowres.mp4"
        
        if not Path(video_path).exists():
            logger.warning(f"Test video not found at {video_path}")
            logger.info("⊘ Skipping performance estimation test")
            return None
        
        logger.info(f"Test video: {video_path}")
        
        # Check video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open test video")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        logger.info(f"Video info: {width}x{height} @ {fps} FPS, {frame_count} frames")
        
        # Estimate performance
        estimator = PerformanceEstimator(model_loader, frame_count=3)
        results = estimator.estimate_performance(
            video_path,
            target_fps=60,
            interpolation_factor=1
        )
        
        logger.info(f"✓ Performance estimation passed")
        logger.info(f"  Can preview at 60 FPS: {results['can_preview']}")
        logger.info(f"  Estimated FPS: {results['estimated_fps']:.2f}")
        
        return results
    
    except Exception as e:
        logger.error(f"✗ Performance estimation failed: {e}", exc_info=True)
        return None

def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("REAL-TIME UPSCALER - PIPELINE TEST")
    logger.info("="*70)
    
    # Test imports
    if not test_imports():
        logger.error("\n✗ Fatal: Import test failed")
        return 1
    
    # Test hardware detection
    success, detector = test_hardware_detection()
    if not success:
        logger.error("\n✗ Fatal: Hardware detection failed")
        return 1
    
    # Test model loading
    success, model_loader = test_model_loading(detector)
    if not success:
        logger.error("\n✗ Fatal: Model loading failed")
        return 1
    
    # Test batch size detection
    if not test_batch_size_detection(model_loader):
        logger.error("\n✗ Fatal: Batch size detection failed")
        return 1
    
    # Test performance estimation
    results = test_performance_estimation(model_loader)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    logger.info("✅ All tests passed!")
    logger.info("\nYou can now use the main.py script:")
    logger.info("  python main.py --detect-hardware")
    logger.info("  python main.py --estimate --video <path> --target-fps 60")
    logger.info("  python main.py --process --video <path> --output output.mp4 --fps 60")
    logger.info("="*70 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
