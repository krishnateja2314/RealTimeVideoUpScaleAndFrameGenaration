"""
Improved main application with preview, caching, and GPU optimization.
Handles hardware detection, performance estimation, and video processing with preview.
"""

import sys
import logging
import argparse
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point with improved features."""
    parser = argparse.ArgumentParser(
        description='Real-Time Video Frame Interpolation & 4x Upscaling with Preview',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test hardware detection
  python main_improved.py --detect-hardware
  
  # Estimate performance
  python main_improved.py --estimate --video video.mp4 --target-fps 60
  
  # Process with preview
  python main_improved.py --process --video input.mp4 --output output.mp4 --fps 60 --preview
  
  # Export only (no preview)
  python main_improved.py --process --video input.mp4 --output output.mp4 --fps 60
        """
    )
    
    parser.add_argument('--detect-hardware', action='store_true',
                       help='Only detect hardware and exit')
    parser.add_argument('--estimate', action='store_true',
                       help='Estimate performance on video frames')
    parser.add_argument('--process', action='store_true',
                       help='Process entire video')
    parser.add_argument('--video', type=str, required=False,
                       help='Input video path')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Output video path (default: output.mp4)')
    parser.add_argument('--target-fps', '--fps', type=int, default=60,
                       dest='target_fps',
                       help='Target output FPS (default: 60)')
    parser.add_argument('--interpolation', type=int, default=1,
                       help='Interpolation factor (default: 1)')
    parser.add_argument('--codec', choices=['mp4v', 'h264', 'h265', 'vp9'],
                       default='mp4v',
                       help='Output video codec (default: mp4v)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto-detect if not specified)')
    parser.add_argument('--preview', action='store_true',
                       help='Enable preview window and frame caching')
    parser.add_argument('--cache-memory-mb', type=int, default=2048,
                       help='Maximum cache memory in MB (default: 2048)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import modules
    from core import (HardwareDetector, ONNXModelLoader, PerformanceEstimator,
                      ImprovedVideoProcessor, PreviewPlayer, FrameCache)
    
    # ==========================================
    # STEP 1: HARDWARE DETECTION
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: HARDWARE DETECTION & CONFIGURATION")
    logger.info("="*70)
    
    detector = HardwareDetector()
    logger.info(detector.get_summary())
    
    gpu_type = detector.get_gpu_type()
    is_gpu_available = detector.is_gpu_available()
    
    if is_gpu_available:
        logger.info(f"✓ GPU Detected: {gpu_type}")
        logger.info(f"✓ Provider: {detector.get_gpu_provider()}")
    else:
        logger.warning("⚠ GPU Not Available - Using CPU (Slow!)")
    
    if args.detect_hardware:
        logger.info("\nHardware detection complete. Exiting.")
        return 0
    
    # ==========================================
    # STEP 2: MODEL LOADING
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: LOADING ONNX MODELS")
    logger.info("="*70)
    
    try:
        # Find ONNX models
        model_dir = Path(__file__).parent
        
        # Search for models in common locations
        search_paths = [
            Path(__file__).parent / "models",
            Path.home() / "RealTimeVideoUpScaleAndFrameGenaration" / "application" / "models",
            Path("/home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration").glob("**/frame_interpolator.onnx"),
            Path("/home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration").glob("**/espcn_4x*.onnx"),
        ]
        
        interpolator_path = None
        upscaler_path = None
        
        # Try to find interpolator
        for pattern_or_path in search_paths:
            if isinstance(pattern_or_path, Path):
                if pattern_or_path.is_file() and 'interpolator' in str(pattern_or_path):
                    interpolator_path = pattern_or_path
                    break
                elif pattern_or_path.is_dir():
                    candidates = list(pattern_or_path.glob("*interpolator*.onnx"))
                    if candidates:
                        interpolator_path = candidates[0]
                        break
        
        # Try to find upscaler
        for pattern_or_path in search_paths:
            if isinstance(pattern_or_path, Path):
                if pattern_or_path.is_file() and 'espcn' in str(pattern_or_path):
                    upscaler_path = pattern_or_path
                    break
                elif pattern_or_path.is_dir():
                    candidates = list(pattern_or_path.glob("*espcn*.onnx"))
                    if candidates:
                        upscaler_path = candidates[0]
                        break
        
        if not interpolator_path or not upscaler_path:
            logger.error("Could not find ONNX model files")
            logger.error(f"  Interpolator: {interpolator_path}")
            logger.error(f"  Upscaler: {upscaler_path}")
            return 1
        
        logger.info(f"✓ Found interpolator: {interpolator_path}")
        logger.info(f"✓ Found upscaler: {upscaler_path}")
        
        # Load models with auto-selected providers
        model_loader = ONNXModelLoader(providers=detector.get_providers())
        model_loader.load_frame_interpolator(str(interpolator_path))
        model_loader.load_upscaler(str(upscaler_path))
        
        logger.info("✓ Models loaded successfully")
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return 1
    
    # ==========================================
    # STEP 3: PERFORMANCE ESTIMATION
    # ==========================================
    if args.estimate:
        logger.info("\n" + "="*70)
        logger.info("STEP 3: PERFORMANCE ESTIMATION")
        logger.info("="*70)
        
        if not args.video:
            logger.error("--video is required for estimation")
            return 1
        
        try:
            estimator = PerformanceEstimator(model_loader)
            results = estimator.estimate_performance(
                args.video,
                args.target_fps,
                args.interpolation
            )
            
            logger.info("\n" + "="*70)
            logger.info("ESTIMATION RESULTS")
            logger.info("="*70)
            logger.info(f"Target FPS: {results['target_fps']}")
            logger.info(f"Estimated FPS: {results['estimated_fps']:.1f}")
            logger.info(f"Frame Time: {results['estimated_frame_time_ms']:.2f} ms")
            logger.info(f"GPU Available: {results['is_gpu_available']}")
            
            if results['can_preview']:
                logger.info("✓ PREVIEW MODE - Real-time playback possible!")
            else:
                logger.info("⚠ EXPORT MODE - Will process at lower speed")
            
        except Exception as e:
            logger.error(f"Estimation failed: {e}")
            return 1
        
        return 0
    
    # ==========================================
    # STEP 4: VIDEO PROCESSING
    # ==========================================
    if args.process:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: VIDEO PROCESSING")
        logger.info("="*70)
        
        if not args.video:
            logger.error("--video is required for processing")
            return 1
        
        try:
            processor = ImprovedVideoProcessor(model_loader)
            
            if args.preview:
                # ========== PREVIEW MODE ==========
                logger.info("Mode: PREVIEW with caching and real-time frame saving")
                
                cache = FrameCache(max_memory_mb=args.cache_memory_mb)
                player = PreviewPlayer(
                    target_fps=args.target_fps,
                    cache_max_memory_mb=args.cache_memory_mb,
                    preload_buffer=30
                )
                
                # Create output directory for frames
                output_dir = Path(args.output).parent / f"{Path(args.output).stem}_frames"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Frame processor callback
                def frame_processor(frame_idx):
                    """Load and process frame."""
                    # This would load and process the frame
                    # For now, return a placeholder
                    return None
                
                # Progress callback
                def progress_callback(info):
                    cached = info.get('cached_frames', 0)
                    mem = info.get('cache_memory_mb', 0)
                    logger.info(f"Frame {info['frame_index']}/{info['total_frames']} | "
                               f"Cached: {cached} | Memory: {mem:.1f} MB")
                
                # Start preview
                player.start_preview(
                    frame_processor=frame_processor,
                    total_frames=0,  # Will be set by estimator
                    output_path=str(output_dir / "frame.png"),
                    progress_callback=progress_callback
                )
                
                # Wait for preview completion
                while player.is_playing:
                    time.sleep(0.1)
                
                player.stop_preview()
                
                logger.info(f"Preview cached {cache.get_cached_frame_count()} frames")
            
            else:
                # ========== EXPORT MODE ==========
                logger.info("Mode: DIRECT EXPORT (no preview)")
                
                def progress_callback(info):
                    logger.info(f"Processing frame {info['frame']}/{info['total_frames']}")
                
                stats = processor.process_video_to_file(
                    input_path=args.video,
                    output_path=args.output,
                    target_fps=args.target_fps,
                    interpolation_factor=args.interpolation,
                    codec=args.codec,
                    progress_callback=progress_callback,
                    max_frames=args.max_frames
                )
                
                logger.info("\n" + "="*70)
                logger.info("PROCESSING COMPLETE")
                logger.info("="*70)
                logger.info(f"Input frames: {stats['frames_processed']}")
                logger.info(f"Output frames: {stats['frames_written']}")
                logger.info(f"Time: {stats['total_time_seconds']:.1f}s")
                logger.info(f"Output: {args.output}")
                
                if stats['errors']:
                    logger.warning(f"Errors encountered: {len(stats['errors'])}")
        
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=args.debug)
            return 1
        
        return 0
    
    # ==========================================
    # NO ACTION SPECIFIED
    # ==========================================
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
