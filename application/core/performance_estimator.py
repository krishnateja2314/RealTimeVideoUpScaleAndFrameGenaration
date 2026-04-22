"""
Performance estimator to predict real-time feasibility.
Tests first N frames to estimate processing speed and determine preview vs export mode.
"""

import time
import logging
import numpy as np
import cv2
from typing import Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceEstimator:
    """Estimate video processing performance before full rendering."""
    
    def __init__(
        self,
        model_loader,
        frame_count: int = 5,
        fps_overhead_margin: float = 1.2
    ):
        """
        Initialize performance estimator.
        
        Args:
            model_loader: ONNXModelLoader instance
            frame_count: Number of frames to test
            fps_overhead_margin: Multiply estimated time by this factor for safety (1.2 = 20% overhead)
        """
        self.model_loader = model_loader
        self.test_frame_count = frame_count
        self.overhead_margin = fps_overhead_margin
        self.results = {}
    
    def estimate_performance(
        self,
        video_path: str,
        target_fps: int,
        interpolation_factor: int = 1
    ) -> Dict:
        """
        Test first N frames to estimate performance.
        
        Args:
            video_path: Path to input video
            target_fps: Desired output FPS
            interpolation_factor: Number of intermediate frames to generate (1 = no interpolation)
        
        Returns:
            Dict with keys:
                - 'can_preview': bool - Whether real-time preview is possible
                - 'estimated_frame_time_ms': float - Average time per output frame
                - 'estimated_fps': float - Achievable FPS
                - 'interpolation_factor': int - How many intermediate frames needed
                - 'is_gpu_available': bool - Whether GPU was used
                - 'total_test_time_ms': float - Total test duration
        """
        logger.info(f"Starting performance estimation on {self.test_frame_count} frames")
        logger.info(f"Target FPS: {target_fps}, Interpolation factor: {interpolation_factor}")
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        source_fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {frame_width}x{frame_height} @ {source_fps} FPS")
        
        timing_results = []
        frames_read = 0
        
        try:
            prev_frame = None
            total_test_start = time.time()
            
            while frames_read < self.test_frame_count:
                ret, frame = video.read()
                if not ret:
                    logger.warning(f"Only {frames_read} frames available, using fewer for test")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if prev_frame is not None:
                    # Test one frame pair
                    frame_time_ms = self._test_frame_pair(
                        prev_frame,
                        frame_rgb,
                        interpolation_factor
                    )
                    timing_results.append(frame_time_ms)
                
                prev_frame = frame_rgb
                frames_read += 1
            
            total_test_time = (time.time() - total_test_start) * 1000  # Convert to ms
            
        finally:
            video.release()
        
        # Analyze results
        if not timing_results:
            logger.error("No timing data collected")
            return {
                'can_preview': False,
                'estimated_frame_time_ms': float('inf'),
                'estimated_fps': 0,
                'interpolation_factor': interpolation_factor,
                'is_gpu_available': False,
                'total_test_time_ms': total_test_time,
                'error': 'Failed to collect timing data'
            }
        
        avg_frame_time_ms = np.mean(timing_results)
        std_frame_time_ms = np.std(timing_results)
        
        # Apply overhead margin for safety
        estimated_frame_time_ms = avg_frame_time_ms * self.overhead_margin
        estimated_fps = 1000.0 / estimated_frame_time_ms if estimated_frame_time_ms > 0 else 0
        
        # Determine if preview is possible
        can_preview = estimated_fps >= target_fps * 0.95  # Allow 5% variance
        
        result = {
            'can_preview': can_preview,
            'estimated_frame_time_ms': estimated_frame_time_ms,
            'estimated_fps': estimated_fps,
            'avg_frame_time_ms': avg_frame_time_ms,
            'std_frame_time_ms': std_frame_time_ms,
            'target_fps': target_fps,
            'interpolation_factor': interpolation_factor,
            'is_gpu_available': self.model_loader.interpolator_session is not None,
            'total_test_time_ms': total_test_time,
            'frames_tested': len(timing_results),
            'video_info': {
                'width': frame_width,
                'height': frame_height,
                'source_fps': source_fps,
                'total_frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            }
        }
        
        self.results = result
        self._log_results(result)
        
        return result
    
    def _test_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        interpolation_factor: int
    ) -> float:
        """
        Test processing time for one frame pair.
        
        Args:
            frame1: First frame [H, W, 3]
            frame2: Second frame [H, W, 3]
            interpolation_factor: Number of intermediate frames
        
        Returns:
            Total processing time in milliseconds
        """
        from utils.normalization import normalize_frame, to_nchw
        from utils.padding import reflect_pad, crop_padding
        
        # Normalize and convert to NCHW
        f1_norm = normalize_frame(frame1)
        f2_norm = normalize_frame(frame2)
        
        f1_nchw, pad_info = reflect_pad(to_nchw(f1_norm))
        f2_nchw = reflect_pad(to_nchw(f2_norm))[0]
        
        start_time = time.time()
        
        # Test interpolation
        interpolated = self.model_loader.run_interpolator(f1_nchw, f2_nchw)
        
        # Test upscaling on interpolated frame
        upscaled = self.model_loader.run_upscaler(interpolated)
        
        # Account for multiple interpolation passes if needed
        if interpolation_factor > 1:
            for _ in range(interpolation_factor - 1):
                interpolated = self.model_loader.run_interpolator(f1_nchw, f2_nchw)
                upscaled = self.model_loader.run_upscaler(interpolated)
        
        elapsed_time_ms = (time.time() - start_time) * 1000
        
        return elapsed_time_ms
    
    def _log_results(self, results: Dict):
        """Log performance estimation results."""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE ESTIMATION RESULTS")
        logger.info("="*60)
        
        if 'error' in results:
            logger.error(f"  Error: {results['error']}")
            return
        
        logger.info(f"  Video: {results['video_info']['width']}x{results['video_info']['height']} @ {results['video_info']['source_fps']} FPS")
        logger.info(f"  Frames tested: {results['frames_tested']}")
        logger.info(f"  Test duration: {results['total_test_time_ms']:.2f} ms")
        logger.info(f"\n  Frame Processing Times:")
        logger.info(f"    Average: {results['avg_frame_time_ms']:.2f} ms")
        logger.info(f"    Std Dev: {results['std_frame_time_ms']:.2f} ms")
        logger.info(f"    With Overhead ({self.overhead_margin}x): {results['estimated_frame_time_ms']:.2f} ms")
        
        logger.info(f"\n  Performance:")
        logger.info(f"    Target FPS: {results['target_fps']}")
        logger.info(f"    Estimated FPS: {results['estimated_fps']:.2f}")
        logger.info(f"    Interpolation Factor: {results['interpolation_factor']}x")
        
        if results['can_preview']:
            logger.info(f"\n  ✅ PREVIEW MODE AVAILABLE - GPU is fast enough!")
        else:
            logger.warning(f"\n  ⚠️  EXPORT MODE ONLY - GPU cannot keep up with target FPS")
        
        logger.info("="*60 + "\n")
    
    def get_results(self) -> Dict:
        """Get last estimation results."""
        return self.results
    
    def can_preview(self) -> bool:
        """Check if real-time preview is possible based on last estimation."""
        return self.results.get('can_preview', False)
    
    def get_estimated_fps(self) -> float:
        """Get estimated achievable FPS."""
        return self.results.get('estimated_fps', 0)
