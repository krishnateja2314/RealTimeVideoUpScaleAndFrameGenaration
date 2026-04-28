"""
Preview system for real-time video playback with intelligent frame pre-loading.
Handles preview window, caching, and export-ready frame saving.
"""

import cv2
import numpy as np
import threading
import queue
import logging
import time
from typing import Callable, Optional, Dict, Tuple
from pathlib import Path
from core.frame_cache import FrameCache

logger = logging.getLogger(__name__)


class PreviewPlayer:
    """Handles real-time preview playback with intelligent frame pre-loading."""
    
    def __init__(
        self,
        target_fps: int = 60,
        cache_max_memory_mb: int = 2048,
        preload_buffer: int = 30
    ):
        """
        Initialize preview player.
        
        Args:
            target_fps: Target playback FPS
            cache_max_memory_mb: Maximum cache size
            preload_buffer: Number of frames to pre-load ahead
        """
        self.target_fps = target_fps
        self.frame_time_ms = 1000.0 / target_fps
        self.cache = FrameCache(max_memory_mb=cache_max_memory_mb)
        self.preload_buffer = preload_buffer
        
        # Playback state
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.lock = threading.RLock()
        
        # Queues for communication
        self.frame_queue = queue.Queue(maxsize=preload_buffer * 2)
        self.render_queue = queue.Queue(maxsize=preload_buffer)
        
        # Threads
        self.preload_thread = None
        self.render_thread = None
        self.stop_event = threading.Event()
    
    def start_preview(
        self,
        frame_processor: Callable,
        total_frames: int,
        output_path: str = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        Start preview playback and rendering.
        
        Args:
            frame_processor: Function that takes frame_index and returns processed frame
            total_frames: Total number of frames to process
            output_path: Path to save preview frames for export
            progress_callback: Callback function for progress updates
        """
        self.total_frames = total_frames
        self.is_playing = True
        self.stop_event.clear()
        self.current_frame = 0
        
        # Create output directory if specified
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start preload thread
        self.preload_thread = threading.Thread(
            target=self._preload_worker,
            args=(frame_processor, output_path, progress_callback),
            daemon=True
        )
        self.preload_thread.start()
        
        # Start render thread
        self.render_thread = threading.Thread(
            target=self._render_worker,
            daemon=True
        )
        self.render_thread.start()
        
        logger.info(f"Preview started: {total_frames} frames at {self.target_fps} FPS")
    
    def _preload_worker(
        self,
        frame_processor: Callable,
        output_path: str,
        progress_callback: Callable
    ):
        """
        Pre-load frames ahead of current playback position.
        Processes frames and saves them for export.
        """
        frame_index = 0
        
        while not self.stop_event.is_set() and frame_index < self.total_frames:
            with self.lock:
                playback_pos = self.current_frame
            
            # Pre-load frames ahead of current position
            target_preload = playback_pos + self.preload_buffer
            
            while frame_index < target_preload and frame_index < self.total_frames:
                try:
                    # Check if frame already in cache
                    if self.cache.has_frame(frame_index):
                        frame_index += 1
                        continue
                    
                    # Process frame
                    start_time = time.time()
                    processed_frame = frame_processor(frame_index)
                    process_time = (time.time() - start_time) * 1000
                    
                    if processed_frame is not None:
                        # Add to cache
                        metadata = {
                            'index': frame_index,
                            'process_time_ms': process_time
                        }
                        self.cache.add_frame(frame_index, processed_frame, metadata)
                        
                        # Save to export path if specified
                        if output_path:
                            self._save_frame(processed_frame, frame_index, output_path)
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback({
                                'frame_index': frame_index,
                                'total_frames': self.total_frames,
                                'cached_frames': self.cache.get_cached_frame_count(),
                                'cache_memory_mb': self.cache.get_memory_usage_mb()
                            })
                    
                    frame_index += 1
                    
                except Exception as e:
                    logger.error(f"Error preloading frame {frame_index}: {e}")
                    frame_index += 1
                    continue
            
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.001)
        
        self.cache.export_ready = True
        logger.info("Preload worker finished")
    
    def _render_worker(self):
        """Handle frame rendering and display."""
        # Create OpenCV window
        window_name = "Preview - Press SPACE to pause, RIGHT to skip, Q to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        try:
            while not self.stop_event.is_set() and self.is_playing:
                with self.lock:
                    frame_idx = self.current_frame
                
                # Get frame from cache
                frame = self.cache.get_frame(frame_idx)
                
                if frame is not None:
                    # Ensure frame is in correct format for display
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    
                    # Handle different shapes
                    if len(frame.shape) == 2:
                        # Grayscale
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 3:
                        # RGB or BGR
                        if frame.shape[2] == 3:
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            display_frame = frame
                    else:
                        display_frame = frame[:, :, :3]
                    
                    # Add info overlay
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(display_frame, f"Frame: {frame_idx}/{self.total_frames}",
                               (20, 40), font, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"FPS: {self.target_fps}",
                               (20, 70), font, 1, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(max(1, int(self.frame_time_ms))) & 0xFF
                    
                    if key == ord('q'):
                        self.stop_event.set()
                        break
                    elif key == ord(' '):
                        # Pause/resume
                        self.is_playing = not self.is_playing
                        logger.info(f"Playback {'resumed' if self.is_playing else 'paused'}")
                    elif key == 83:  # Right arrow
                        # Skip forward 10 frames
                        with self.lock:
                            self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                        logger.info(f"Skipped to frame {self.current_frame}")
                    elif key == 81:  # Left arrow
                        # Skip backward 10 frames
                        with self.lock:
                            self.current_frame = max(self.current_frame - 10, 0)
                        logger.info(f"Skipped to frame {self.current_frame}")
                    
                    if self.is_playing:
                        with self.lock:
                            self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                
                else:
                    # Frame not yet loaded, wait
                    time.sleep(0.01)
        
        finally:
            cv2.destroyAllWindows()
            logger.info("Preview window closed")
    
    def _save_frame(self, frame: np.ndarray, frame_idx: int, output_path: str):
        """Save frame for export."""
        try:
            output_file = Path(output_path).parent / f"{Path(output_path).stem}_{frame_idx:06d}.png"
            
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                save_frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                save_frame = frame
            
            # Handle RGB to BGR conversion if needed
            if len(save_frame.shape) == 3 and save_frame.shape[2] == 3:
                save_frame = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(output_file), save_frame)
            logger.debug(f"Saved frame {frame_idx} to {output_file}")
        
        except Exception as e:
            logger.error(f"Error saving frame {frame_idx}: {e}")
    
    def stop_preview(self):
        """Stop preview and cleanup."""
        self.is_playing = False
        self.stop_event.set()
        
        if self.preload_thread:
            self.preload_thread.join(timeout=5)
        
        if self.render_thread:
            self.render_thread.join(timeout=5)
        
        logger.info("Preview stopped")
    
    def get_preview_stats(self) -> Dict:
        """Get current preview statistics."""
        with self.lock:
            return {
                'current_frame': self.current_frame,
                'total_frames': self.total_frames,
                'is_playing': self.is_playing,
                'cache_stats': self.cache.get_stats(),
                'frame_time_ms': self.frame_time_ms
            }
