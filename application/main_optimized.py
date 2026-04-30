"""
Optimized main application with GPU and preview support.
Uses the same ONNX loader path as main.py for stable GPU execution.
"""

import sys
import logging
import argparse
from pathlib import Path
import time
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Optimized main entry point with GPU and preview support."""
    parser = argparse.ArgumentParser(
        description='Real-Time Video Frame Interpolation & 4x Upscaling (GPU-Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Start:
  1. Check GPU:     python main.py --detect-hardware
  2. Test perf:     python main.py --estimate --video input.mp4 --fps 60
  3. Process:       python main.py --process --video input.mp4 --output output.mp4 --preview

Examples:
  # Hardware detection
  python main.py --detect-hardware
  
  # Performance estimation
  python main.py --estimate --video video.mp4 --target-fps 60
  
  # Process with preview and caching
  python main.py --process --video input.mp4 --output output.mp4 --fps 60 --preview
  
  # Process multiple frames with interpolation
  python main.py --process --video input.mp4 --output output.mp4 --fps 60 --interpolation 2 --preview
  
  # Direct export without preview
  python main.py --process --video input.mp4 --output output.mp4 --fps 60 --codec h265
        """
    )
    
    # Detect mode
    detect_group = parser.add_argument_group('Detection')
    detect_group.add_argument('--detect-hardware', action='store_true',
                              help='Detect hardware and exit')
    
    # Estimation mode
    estimate_group = parser.add_argument_group('Estimation')
    estimate_group.add_argument('--estimate', action='store_true',
                                help='Estimate FPS before processing')
    
    # Processing mode
    process_group = parser.add_argument_group('Processing')
    process_group.add_argument('--process', action='store_true',
                               help='Process video')
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--video', type=str, required=False,
                          help='Input video path')
    io_group.add_argument('--output', type=str, default='output.mp4',
                          help='Output video path (default: output.mp4)')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--fps', '--target-fps', type=int, default=60,
                            dest='target_fps',
                            help='Target output FPS (default: 60)')
    proc_group.add_argument('--interpolation', type=int, default=1,
                            help='Interpolation factor: 1=no interp, 2=3x FPS (default: 1)')
    proc_group.add_argument('--codec', choices=['mp4v', 'h264', 'h265', 'vp9'],
                            default='mp4v',
                            help='Output codec (default: mp4v)')
    proc_group.add_argument('--batch-size', type=int, default=None,
                            help='Batch size (auto-detect if not specified)')
    proc_group.add_argument('--duration', type=float, default=None,
                            help='Process only the first N seconds of the video')
    provider_group = proc_group.add_mutually_exclusive_group()
    provider_group.add_argument('--use-gpu', action='store_true',
                            help='Use detected GPU provider when available')
    provider_group.add_argument('--cpu', action='store_true',
                            help='Force CPU ONNX inference even if GPU is available')
    
    # Preview and caching
    preview_group = parser.add_argument_group('Preview & Caching')
    preview_group.add_argument('--preview', action='store_true',
                               help='Enable preview window with frame caching')
    preview_group.add_argument('--cache-memory-mb', type=int, default=2048,
                               help='Max cache memory in MB (default: 2048)')
    preview_group.add_argument('--preload-buffer', type=int, default=30,
                               help='Frames to pre-load ahead (default: 30)')
    
    # Testing and debugging
    debug_group = parser.add_argument_group('Debug & Testing')
    debug_group.add_argument('--max-frames', type=int, default=None,
                             help='Max frames to process (for testing)')
    debug_group.add_argument('--debug', action='store_true',
                             help='Enable debug logging')
    debug_group.add_argument('--profile', action='store_true',
                             help='Enable performance profiling')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import core modules
    from core import (HardwareDetector, ONNXModelLoader, PerformanceEstimator,
                      VideoProcessor, PreviewPlayer, FrameCache)
    
    # ========================================
    # STEP 1: HARDWARE DETECTION
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: HARDWARE DETECTION & GPU OPTIMIZATION")
    logger.info("="*80)
    
    try:
        detector = HardwareDetector()
        logger.info(detector.get_summary())
        
        gpu_type = detector.get_gpu_type()
        is_gpu = detector.is_gpu_available()
        
        if is_gpu:
            logger.info(f"✓ GPU: {gpu_type}")
            logger.info(f"✓ Provider: {detector.get_gpu_provider()}")
        else:
            logger.warning("⚠ GPU Not Available - Using CPU (Slow!)")
    
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        return 1
    
    if args.detect_hardware:
        return 0
    
    # ========================================
    # STEP 2: MODEL LOADING
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOADING ONNX MODELS")
    logger.info("="*80)
    
    try:
        # Find models
        search_paths = [
            Path(__file__).parent / "models",
            Path.home() / "RealTimeVideoUpScaleAndFrameGenaration" / "application" / "models",
        ]
        
        interpolator_path = None
        upscaler_path = None
        
        for search_dir in search_paths:
            if not search_dir.exists():
                continue
            
            candidates = list(search_dir.glob("*interpolator*.onnx"))
            if candidates and not interpolator_path:
                interpolator_path = candidates[0]
            
            candidates = list(search_dir.glob("*espcn*.onnx"))
            if candidates and not upscaler_path:
                upscaler_path = candidates[0]
        
        # Search workspace root
        if not interpolator_path or not upscaler_path:
            workspace = Path("/home/krishnateja/RealTimeVideoUpScaleAndFrameGenaration")
            all_files = list(workspace.rglob("*.onnx"))
            
            for f in all_files:
                if 'interpolator' in f.name and not interpolator_path:
                    interpolator_path = f
                if 'espcn' in f.name and not upscaler_path:
                    upscaler_path = f
        
        if not interpolator_path or not upscaler_path:
            logger.error(f"❌ Could not find ONNX models")
            logger.error(f"   Looking for: *interpolator*.onnx and *espcn*.onnx")
            return 1
        
        logger.info(f"✓ Interpolator: {interpolator_path}")
        logger.info(f"✓ Upscaler: {upscaler_path}")
        
        # Default to CPU for stability; GPU requires explicit flag
        if args.cpu:
            providers = ['CPUExecutionProvider']
            logger.info("Forcing CPU ONNX inference")
        elif args.use_gpu:
            available = detector.available_providers
            selected = [p for p in detector.get_providers() if p in available]
            if selected and selected[0] != 'CPUExecutionProvider':
                providers = selected
                logger.info(f"Using detected GPU provider(s): {providers}")
            else:
                providers = ['CPUExecutionProvider']
                logger.warning(
                    'GPU provider requested but not available. Falling back to CPUExecutionProvider.'
                )
        else:
            providers = ['CPUExecutionProvider']
            logger.info("Using CPUExecutionProvider for stability")

        model_loader = ONNXModelLoader(providers=providers)
        logger.info(f"Selected ONNX providers: {model_loader.providers}")

        model_loader.load_frame_interpolator(str(interpolator_path))
        model_loader.load_upscaler(str(upscaler_path))

        # Detect optimal batch size using the proven ONNXModelLoader path
        if is_gpu:
            logger.info("Detecting optimal batch size...")
            optimal_batch = model_loader.auto_detect_batch_size(
                model_loader.interpolator_session,
                max_attempts=1,
                start_batch=1
            )
            logger.info(f"✓ Optimal batch size: {optimal_batch}")

        logger.info("✓ Models loaded successfully")
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return 1
    
    # ========================================
    # STEP 3: PERFORMANCE ESTIMATION
    # ========================================
    if args.estimate:
        logger.info("\n" + "="*80)
        logger.info("STEP 3: PERFORMANCE ESTIMATION")
        logger.info("="*80)
        
        if not args.video:
            logger.error("❌ --video is required for estimation")
            return 1
        
        try:
            estimator = PerformanceEstimator(model_loader)
            results = estimator.estimate_performance(
                args.video,
                args.target_fps,
                args.interpolation
            )
            
            logger.info("\n" + "="*80)
            logger.info("ESTIMATION RESULTS")
            logger.info("="*80)
            logger.info(f"Target FPS: {results['target_fps']}")
            logger.info(f"Estimated FPS: {results['estimated_fps']:.1f}")
            logger.info(f"Frame Time: {results['estimated_frame_time_ms']:.2f} ms")
            logger.info(f"Frames Tested: {results['frames_tested']}")
            logger.info(f"GPU Available: {results['is_gpu_available']}")
            
            if results['can_preview']:
                logger.info("✓ PREVIEW MODE POSSIBLE - Real-time playback achievable!")
            else:
                logger.info("⚠ EXPORT MODE - Processing slower than target FPS")
        
        except Exception as e:
            logger.error(f"Estimation failed: {e}")
            return 1
        
        return 0
    
    # ========================================
    # STEP 4: VIDEO PROCESSING
    # ========================================
    if args.process:
        logger.info("\n" + "="*80)
        logger.info("STEP 4: VIDEO PROCESSING")
        logger.info("="*80)
        
        if not args.video:
            logger.error("❌ --video is required for processing")
            return 1
        
        try:
            processor = VideoProcessor(model_loader, batch_size=optimal_batch if 'optimal_batch' in locals() else 1)
            
            max_frames = args.max_frames
            if args.duration is not None:
                capture = cv2.VideoCapture(args.video)
                if capture.isOpened():
                    source_fps = capture.get(cv2.CAP_PROP_FPS)
                    capture.release()
                else:
                    source_fps = 0

                if source_fps > 0:
                    max_frames = int(round(source_fps * args.duration))
                    logger.info(f"Limiting processing to first {args.duration:.2f} seconds ({max_frames} frames) based on source FPS {source_fps:.2f}")
                else:
                    logger.warning("Could not determine source FPS. Duration limit will be ignored.")

            if args.preview:
                # ========== PREVIEW MODE ==========
                logger.info("Mode: PREVIEW with intelligent caching")
                logger.info(f"Cache Memory: {args.cache_memory_mb} MB")
                logger.info(f"Preload Buffer: {args.preload_buffer} frames")
                logger.info("Preview feature requires integration with playback system")
                logger.info("Using export mode instead...")
            
            # ========== EXPORT MODE (or fallback) ==========
            logger.info("Starting video processing...")
            start_time = time.time()
            
            def progress_callback(current, total, fps):
                percent = (current / total) * 100 if total > 0 else 0
                logger.info(f"Progress: {current}/{total} ({percent:.1f}%) - {fps:.2f} FPS")
            
            stats = processor.process_video(
                input_path=args.video,
                output_path=args.output,
                target_fps=args.target_fps,
                interpolation_factor=args.interpolation,
                codec=args.codec,
                progress_callback=progress_callback,
                max_frames=max_frames
            )
            
            # ========== RESULTS ==========
            logger.info("\n" + "="*80)
            logger.info("PROCESSING COMPLETE")
            logger.info("="*80)
            logger.info(f"Input Frames: {stats['frames_processed']}")
            logger.info(f"Output Frames: {stats['frames_written']}")
            logger.info(f"Processing Time: {stats['total_time_seconds']:.1f} seconds")
            
            if stats['total_time_seconds'] > 0:
                fps = stats['frames_written'] / stats['total_time_seconds']
                logger.info(f"Processing Speed: {fps:.1f} fps")
            
            logger.info(f"✓ Output saved to: {args.output}")
            
            if stats.get('errors'):
                logger.warning(f"⚠ {len(stats['errors'])} errors encountered")
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

            # Retry on CPU if the GPU provider failed
            gpu_providers = [p for p in model_loader.providers if p != 'CPUExecutionProvider']
            if gpu_providers and 'CPUExecutionProvider' in detector.available_providers:
                logger.warning("GPU processing failed. Retrying with CPUExecutionProvider...")
                try:
                    cpu_loader = ONNXModelLoader(providers=['CPUExecutionProvider'])
                    cpu_loader.load_frame_interpolator(str(interpolator_path))
                    cpu_loader.load_upscaler(str(upscaler_path))
                    cpu_processor = VideoProcessor(cpu_loader, batch_size=1)

                    stats = cpu_processor.process_video(
                        input_path=args.video,
                        output_path=args.output,
                        target_fps=args.target_fps,
                        interpolation_factor=args.interpolation,
                        codec=args.codec,
                        progress_callback=progress_callback,
                        max_frames=max_frames
                    )

                    logger.info("\n" + "="*80)
                    logger.info("CPU retry processing complete")
                    logger.info("="*80)
                    logger.info(f"Input Frames: {stats['frames_processed']}")
                    logger.info(f"Output Frames: {stats['frames_written']}")
                    logger.info(f"Processing Time: {stats['total_time_seconds']:.1f} seconds")
                    logger.info(f"✓ Output saved to: {args.output}")
                    return 0

                except Exception as e2:
                    logger.error(f"CPU retry also failed: {e2}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
            return 1
        
        return 0
    
    # ========================================
    # HELP
    # ========================================
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
