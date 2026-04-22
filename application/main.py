"""
Main application entry point.
Demonstrates the complete pipeline: hardware detection → model loading → performance estimation → processing.
"""

import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Real-Time Video Frame Interpolation & 4x Upscaling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test hardware detection
  python main.py --detect-hardware
  
  # Estimate performance on video
  python main.py --estimate --video video.mp4 --target-fps 60
  
  # Process video
  python main.py --process --video input.mp4 --output output.mp4 --fps 60 --interpolation 1
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import after logging setup
    from core import HardwareDetector, ONNXModelLoader, PerformanceEstimator, VideoProcessor
    
    # Step 1: Hardware Detection
    logger.info("\n" + "="*70)
    logger.info("STEP 1: HARDWARE DETECTION")
    logger.info("="*70)
    
    detector = HardwareDetector()
    logger.info(detector.get_summary())
    
    if args.detect_hardware:
        logger.info("\nHardware detection complete. Exiting.")
        return 0
    
    # Step 2: Load Models
    logger.info("\n" + "="*70)
    logger.info("STEP 2: LOADING MODELS")
    logger.info("="*70)
    
    try:
        # Try to find ONNX models
        model_dir = Path(__file__).parent / "models"
        interpolator_path = list(Path("/home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration").glob("**/frame_interpolator.onnx"))[0]
        upscaler_path = list(Path("/home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration").glob("**/espcn_4x*.onnx"))[0]
        
        logger.info(f"Found interpolator: {interpolator_path}")
        logger.info(f"Found upscaler: {upscaler_path}")
        
    except IndexError:
        logger.error("Could not find ONNX model files. Please ensure models are in the correct location.")
        return 1
    
    model_loader = ONNXModelLoader(providers=detector.get_providers())
    model_loader.load_frame_interpolator(str(interpolator_path))
    model_loader.load_upscaler(str(upscaler_path))
    
    # Step 3: Auto-detect batch size
    logger.info("\n" + "="*70)
    logger.info("STEP 3: AUTO-DETECTING BATCH SIZE")
    logger.info("="*70)
    
    optimal_batch = model_loader.auto_detect_batch_size(
        model_loader.interpolator_session,
        max_attempts=4,
        start_batch=1
    )
    logger.info(f"Optimal batch size: {optimal_batch}")
    
    # If no video provided, just do hardware/model testing
    if args.video is None:
        logger.info("\nNo video provided. Completed hardware and model testing. Exiting.")
        return 0
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {args.video}")
        return 1
    
    # Step 4: Performance Estimation
    if args.estimate or args.process:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: PERFORMANCE ESTIMATION")
        logger.info("="*70)
        
        estimator = PerformanceEstimator(model_loader, frame_count=5)
        results = estimator.estimate_performance(
            str(video_path),
            target_fps=args.target_fps,
            interpolation_factor=args.interpolation
        )
        
        if not results.get('can_preview'):
            logger.warning("Preview mode not available. Use export mode instead.")
            if args.process:
                logger.warning("Proceeding with export (batch processing)...")
            else:
                return 0
    
    # Step 5: Process Video
    if args.process:
        logger.info("\n" + "="*70)
        logger.info("STEP 5: PROCESSING VIDEO")
        logger.info("="*70)
        
        processor = VideoProcessor(model_loader, batch_size=optimal_batch)
        
        def progress_callback(current, total, fps):
            percent = (current / total) * 100 if total > 0 else 0
            logger.info(f"Progress: {current}/{total} ({percent:.1f}%) - {fps:.2f} FPS")
        
        try:
            stats = processor.process_video(
                input_path=str(video_path),
                output_path=args.output,
                target_fps=args.target_fps,
                interpolation_factor=args.interpolation,
                codec=args.codec,
                progress_callback=progress_callback
            )
            
            logger.info(f"\n✅ Video processing complete!")
            logger.info(f"Output saved to: {args.output}")
            
        except Exception as e:
            logger.error(f"Error during video processing: {e}", exc_info=True)
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
