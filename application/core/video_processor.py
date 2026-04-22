"""
Main video processing pipeline.
Chains frame interpolation and 4x upscaling with asynchronous threading support.
"""

import cv2
import numpy as np
import logging
import time
import threading
import queue
from typing import Callable, Optional, Tuple, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process video through interpolation and upscaling pipeline with Async I/O."""
    
    def __init__(self, model_loader, batch_size: int = 2):
        self.model_loader = model_loader
        self.batch_size = batch_size
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        target_fps: int = 60,
        interpolation_factor: int = 1,
        codec: str = 'mp4v',
        quality: str = 'high',
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        logger.info(f"Starting video processing")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        
        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")
        
        source_fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_height = frame_height * 4
        output_width = frame_width * 4
        
        writer = self._setup_video_writer(
            output_path, output_width, output_height, target_fps, codec
        )
        
        if writer is None:
            raise RuntimeError("Failed to create video writer")
        
        stats = {
            'frames_processed': 0,
            'frames_written': 0,
            'total_time_seconds': 0,
            'input_frames': total_frames,
            'output_frames': 0,
            'errors': []
        }

        input_queue = queue.Queue(maxsize=16)
        output_queue = queue.Queue(maxsize=32)
        stop_event = threading.Event()

        def reader_thread():
            frame_buffer = []
            while not stop_event.is_set():
                ret, frame = video.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame_rgb)
                
                if len(frame_buffer) >= 2:
                    for i in range(len(frame_buffer) - 1):
                        pair = (frame_buffer[i], frame_buffer[i + 1])
                        while not stop_event.is_set():
                            try:
                                input_queue.put(pair, timeout=0.5)
                                break
                            except queue.Full:
                                continue
                    frame_buffer = [frame_buffer[-1]]
            
            while not stop_event.is_set():
                try:
                    input_queue.put(None, timeout=0.5)
                    break
                except queue.Full:
                    continue

        def writer_thread():
            while not stop_event.is_set():
                try:
                    frames_to_write = output_queue.get(timeout=0.5)
                    if frames_to_write is None:
                        break
                    
                    for out_frame in frames_to_write:
                        out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                        writer.write(out_frame_bgr)
                        stats['frames_written'] += 1
                        
                    output_queue.task_done()
                except queue.Empty:
                    continue

        t_read = threading.Thread(target=reader_thread, daemon=True)
        t_write = threading.Thread(target=writer_thread, daemon=True)
        t_read.start()
        t_write.start()

        start_time = time.time()
        try:
            while True:
                try:
                    pair = input_queue.get(timeout=0.5)
                except queue.Empty:
                    if not t_read.is_alive() and input_queue.empty():
                        break
                    continue
                
                if pair is None:
                    break
                    
                frame1, frame2 = pair
                stats['frames_processed'] += 1  # Fixed: Advance timeline by 1 per pair
                
                # Check if this is the very last pair in the video
                is_last_pair = (stats['frames_processed'] >= total_frames - 1)
                
                try:
                    output_frames = self._process_frame_pair(
                        frame1, frame2, interpolation_factor, is_last_pair
                    )
                    
                    while not stop_event.is_set():
                        try:
                            output_queue.put(output_frames, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                            
                    if progress_callback and stats['frames_processed'] % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = stats['frames_processed'] / elapsed if elapsed > 0 else 0
                        progress_callback(stats['frames_processed'], total_frames, fps)
                        
                except Exception as e:
                    logger.error(f"Error processing frame pair: {e}")
                    stats['errors'].append(str(e))
                
                input_queue.task_done()

        except KeyboardInterrupt:
            logger.warning("\n✋ Interrupted by user! Flushing queues and saving processed frames...")
            stop_event.set()
        
        finally:
            stop_event.set()
            
            while not input_queue.empty():
                try: input_queue.get_nowait()
                except queue.Empty: break
                
            try: output_queue.put(None, timeout=1)
            except queue.Full: pass
            
            t_write.join(timeout=3.0)
            
            video.release()
            writer.release()
            
            stats['total_time_seconds'] = time.time() - start_time
            stats['output_frames'] = stats['frames_written']
            
            self._log_stats(stats)
            
        return stats
    
    def _process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        interpolation_factor: int = 1,
        is_last_pair: bool = False
    ) -> List[np.ndarray]:
        """Process one frame pair through ONNX models."""
        from utils.normalization import normalize_frame, to_nchw, to_hwc, denormalize_frame
        from utils.padding import reflect_pad, crop_padding
        
        f1_norm = normalize_frame(frame1)
        f2_norm = normalize_frame(frame2)
        
        f1_nchw, pad_info = reflect_pad(to_nchw(f1_norm))
        f2_nchw = reflect_pad(to_nchw(f2_norm))[0]
        
        output_frames = []
        
        # Frame 1
        f1_upscaled = self.model_loader.run_upscaler(f1_nchw)
        f1_upscaled = np.clip(f1_upscaled, 0.0, 1.0)
        output_frames.append(denormalize_frame(to_hwc(f1_upscaled)))
        
        # Interpolated frames
        for i in range(interpolation_factor):
            interpolated = self.model_loader.run_interpolator(f1_nchw, f2_nchw)
            interpolated = np.clip(interpolated, 0.0, 1.0)
            
            upscaled = self.model_loader.run_upscaler(interpolated)
            upscaled = np.clip(upscaled, 0.0, 1.0)
            
            upscaled_cropped = crop_padding(upscaled, pad_info)
            output_frames.append(denormalize_frame(to_hwc(upscaled_cropped)))
        
        # Fixed: ONLY append Frame 2 if it's the end of the video to prevent duplicates
        if is_last_pair:
            f2_upscaled = self.model_loader.run_upscaler(f2_nchw)
            f2_upscaled = np.clip(f2_upscaled, 0.0, 1.0)
            output_frames.append(denormalize_frame(to_hwc(f2_upscaled)))
        
        return output_frames
    
    def _setup_video_writer(self, output_path: str, width: int, height: int, fps: float, codec: str) -> cv2.VideoWriter:
        codec_map = {
            'mp4v': cv2.VideoWriter_fourcc(*'mp4v'),
            'h264': cv2.VideoWriter_fourcc(*'avc1'),
            'h265': cv2.VideoWriter_fourcc(*'hev1'),
            'vp9': cv2.VideoWriter_fourcc(*'vp90'),
        }
        fourcc = codec_map.get(codec, cv2.VideoWriter_fourcc(*'mp4v'))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error(f"Failed to create video writer with codec {codec}")
            return None
        return writer
    
    def _log_stats(self, stats: dict):
        logger.info("\n" + "="*60)
        logger.info("VIDEO PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"  Frames Processed: {stats['frames_processed']}")
        logger.info(f"  Frames Written: {stats['frames_written']}")
        logger.info(f"  Total Time: {stats['total_time_seconds']:.2f} seconds")
        if stats['total_time_seconds'] > 0:
            fps = stats['frames_processed'] / stats['total_time_seconds']
            logger.info(f"  Processing Speed: {fps:.2f} FPS")
        logger.info("="*60 + "\n")