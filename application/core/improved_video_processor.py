"""
Improved video processing pipeline with fixed tensor handling and preview support.
Handles frame interpolation, upscaling, and export with proper normalization.
"""

import cv2
import numpy as np
import logging
import time
import threading
import queue
from typing import Callable, Optional, List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ImprovedVideoProcessor:
    """Fixed video processing with proper tensor handling and preview support."""
    
    def __init__(self, model_loader, batch_size: int = 1):
        self.model_loader = model_loader
        self.batch_size = batch_size
        self.frame_cache = {}  # Cache for frame pairs
    
    def process_video_with_preview(
        self,
        input_path: str,
        target_fps: int = 60,
        interpolation_factor: int = 1,
        preview_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
        max_frames: int = None
    ) -> Dict:
        """
        Process video and provide preview-ready frames.
        
        Args:
            input_path: Path to input video
            target_fps: Target output FPS
            interpolation_factor: Number of intermediate frames to generate
            preview_callback: Callback for preview frames
            progress_callback: Progress callback
            max_frames: Maximum frames to process (None = all)
        
        Returns:
            Statistics about processing
        """
        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")
        
        source_fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"Input: {frame_width}x{frame_height} @ {source_fps} FPS ({total_frames} frames)")
        
        # Output resolution (4x upscale)
        output_height = frame_height * 4
        output_width = frame_width * 4
        
        stats = {
            'frames_processed': 0,
            'frames_output': 0,
            'total_time_seconds': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            prev_frame = None
            frame_count = 0
            output_frame_count = 0
            
            while frame_count < total_frames:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if prev_frame is not None:
                    try:
                        # Process frame pair
                        output_frames = self._process_frame_pair(
                            prev_frame,
                            frame_rgb,
                            interpolation_factor,
                            is_last_pair=(frame_count == total_frames - 1)
                        )
                        
                        # Call preview callback for each output frame
                        if preview_callback:
                            for out_frame in output_frames:
                                preview_callback(out_frame, output_frame_count)
                                output_frame_count += 1
                        
                        stats['frames_output'] += len(output_frames)
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback({
                                'input_frame': frame_count,
                                'output_frame': output_frame_count,
                                'total_frames': total_frames,
                                'elapsed_time_s': time.time() - start_time
                            })
                    
                    except Exception as e:
                        logger.error(f"Error processing frame pair {frame_count}: {e}")
                        stats['errors'].append(str(e))
                
                prev_frame = frame_rgb
                frame_count += 1
                stats['frames_processed'] += 1
        
        finally:
            video.release()
            stats['total_time_seconds'] = time.time() - start_time
        
        return stats
    
    def _process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        interpolation_factor: int,
        is_last_pair: bool = False
    ) -> List[np.ndarray]:
        """
        Process a pair of frames through interpolation and upscaling.
        
        Args:
            frame1: First frame [H, W, 3] RGB
            frame2: Second frame [H, W, 3] RGB
            interpolation_factor: Number of intermediate frames to generate
            is_last_pair: Whether this is the last frame pair
        
        Returns:
            List of output frames [H*4, W*4, 3] uint8 RGB
        """
        from utils.normalization import normalize_frame, to_nchw
        from utils.padding import reflect_pad, crop_padding
        
        # Step 1: Normalize frames [0,255] uint8 -> [0,1] float32
        f1_norm = normalize_frame(frame1)
        f2_norm = normalize_frame(frame2)
        
        # Step 2: Convert to NCHW and pad
        f1_nchw, pad_info = reflect_pad(to_nchw(f1_norm))
        f2_nchw, _ = reflect_pad(to_nchw(f2_norm))
        
        output_frames = []
        
        try:
            # Step 3: Process first frame through upscaler
            f1_up = self.model_loader.run_upscaler(f1_nchw)
            f1_final = self._finalize_frame(f1_up, pad_info, scale=4)
            output_frames.append(f1_final)
            
            # Step 4: Interpolation frames
            for inter_idx in range(interpolation_factor):
                # Run interpolator
                inter = self.model_loader.run_interpolator(f1_nchw, f2_nchw)
                
                # Upscale interpolated frame
                inter_up = self.model_loader.run_upscaler(inter)
                inter_final = self._finalize_frame(inter_up, pad_info, scale=4)
                output_frames.append(inter_final)
            
            # Step 5: Process second frame (only for last pair)
            if is_last_pair:
                f2_up = self.model_loader.run_upscaler(f2_nchw)
                f2_final = self._finalize_frame(f2_up, pad_info, scale=4)
                output_frames.append(f2_final)
        
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            raise
        
        return output_frames
    
    def _finalize_frame(
        self,
        tensor: np.ndarray,
        pad_info: Tuple,
        scale: int = 1
    ) -> np.ndarray:
        """
        Convert ONNX output tensor to displayable frame.
        
        Args:
            tensor: Output tensor from ONNX [B, C, H, W] float32
            pad_info: Padding information for cropping
        
        Returns:
            Frame [H, W, 3] uint8 RGB
        """
        from utils.padding import crop_padding
        
        # Step 1: Ensure tensor is float32
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        
        # Step 2: Crop padding
        tensor = crop_padding(tensor, pad_info, scale=scale)
        
        # Step 3: Squeeze batch dimension [1, C, H, W] -> [C, H, W]
        tensor = np.squeeze(tensor, axis=0)
        
        # Step 4: Handle different tensor shapes
        if tensor.ndim == 3:
            # Check if it's CHW format
            if tensor.shape[0] in [1, 3, 4]:
                # CHW -> HWC
                tensor = np.transpose(tensor, (1, 2, 0))
        
        # Step 5: Handle single channel
        if tensor.ndim == 2:
            tensor = np.stack([tensor, tensor, tensor], axis=2)
        elif tensor.shape[2] == 1:
            tensor = np.repeat(tensor, 3, axis=2)
        
        # Step 6: Detect and fix range
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Handle [-1, 1] range
        if min_val < -0.1:
            tensor = (tensor + 1.0) / 2.0
        
        # Ensure [0, 1] range
        tensor = np.clip(tensor, 0.0, 1.0)
        
        # Step 7: Convert to uint8 [0, 255]
        frame_uint8 = (tensor * 255.0).round().astype(np.uint8)
        
        # Step 8: Ensure contiguous for OpenCV
        return np.ascontiguousarray(frame_uint8)
    
    def process_video_to_file(
        self,
        input_path: str,
        output_path: str,
        target_fps: int = 60,
        interpolation_factor: int = 1,
        codec: str = 'mp4v',
        progress_callback: Optional[Callable] = None,
        max_frames: int = None
    ) -> Dict:
        """
        Process video and write to file.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_fps: Target FPS
            interpolation_factor: Interpolation frames per pair
            codec: Video codec
            progress_callback: Progress callback
            max_frames: Max frames to process
        
        Returns:
            Processing statistics
        """
        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")
        
        source_fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        output_height = frame_height * 4
        output_width = frame_width * 4
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, target_fps, (output_width, output_height))
        
        if not writer.isOpened():
            raise ValueError(f"Cannot create video writer for {output_path}")
        
        stats = {
            'frames_processed': 0,
            'frames_written': 0,
            'total_time_seconds': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            prev_frame = None
            frame_count = 0
            
            while frame_count < total_frames:
                ret, frame = video.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if prev_frame is not None:
                    try:
                        output_frames = self._process_frame_pair(
                            prev_frame,
                            frame_rgb,
                            interpolation_factor,
                            is_last_pair=(frame_count == total_frames - 1)
                        )
                        
                        for out_frame in output_frames:
                            # Convert RGB to BGR for OpenCV
                            out_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                            writer.write(out_bgr)
                            stats['frames_written'] += 1
                        
                        if progress_callback:
                            progress_callback({
                                'frame': frame_count,
                                'total_frames': total_frames
                            })
                    
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {e}")
                        stats['errors'].append(str(e))
                
                prev_frame = frame_rgb
                frame_count += 1
                stats['frames_processed'] += 1
        
        finally:
            video.release()
            writer.release()
            stats['total_time_seconds'] = time.time() - start_time
        
        return stats
