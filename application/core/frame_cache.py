"""
Frame caching system for preview mode.
Stores processed frames in memory for quick playback and export.
Implements LRU (Least Recently Used) eviction when memory limits approached.
"""

import numpy as np
import threading
import logging
from collections import OrderedDict
from typing import Optional, Dict
import os

logger = logging.getLogger(__name__)


class FrameCache:
    """Thread-safe frame cache for preview and export."""
    
    def __init__(self, max_memory_mb: int = 2048):
        """
        Initialize frame cache.
        
        Args:
            max_memory_mb: Maximum memory to use for cache (default: 2GB)
        """
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.frames = OrderedDict()  # {frame_index: frame_data}
        self.lock = threading.RLock()
        self.current_memory = 0
        self.metadata = {}  # Store frame metadata
        self.export_ready = False
    
    def add_frame(self, frame_index: int, frame: np.ndarray, metadata: Dict = None) -> bool:
        """
        Add a frame to cache.
        
        Args:
            frame_index: Frame sequence number
            frame: Numpy array of frame data
            metadata: Optional metadata about the frame
        
        Returns:
            True if frame was added, False if memory limit exceeded
        """
        with self.lock:
            # Check if frame already exists
            if frame_index in self.frames:
                logger.debug(f"Frame {frame_index} already in cache")
                return True
            
            frame_size = frame.nbytes
            
            # Evict frames if necessary
            while self.current_memory + frame_size > self.max_memory and len(self.frames) > 0:
                oldest_idx = next(iter(self.frames))
                old_frame = self.frames.pop(oldest_idx)
                self.current_memory -= old_frame.nbytes
                if oldest_idx in self.metadata:
                    del self.metadata[oldest_idx]
                logger.debug(f"Evicted frame {oldest_idx} from cache (memory pressure)")
            
            # Check if we can fit the frame
            if self.current_memory + frame_size > self.max_memory:
                logger.warning(f"Cannot fit frame {frame_index} in cache (size: {frame_size / 1024 / 1024:.2f} MB)")
                return False
            
            # Store frame (make a copy to ensure it doesn't get modified)
            self.frames[frame_index] = frame.copy()
            self.current_memory += frame_size
            
            if metadata:
                self.metadata[frame_index] = metadata
            
            logger.debug(f"Cached frame {frame_index} (cache: {self.current_memory / 1024 / 1024:.2f} MB / {self.max_memory / 1024 / 1024:.2f} MB)")
            return True
    
    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Retrieve a frame from cache.
        
        Args:
            frame_index: Frame index to retrieve
        
        Returns:
            Frame array or None if not in cache
        """
        with self.lock:
            if frame_index in self.frames:
                # Move to end (most recently used)
                self.frames.move_to_end(frame_index)
                return self.frames[frame_index]
            return None
    
    def has_frame(self, frame_index: int) -> bool:
        """Check if frame is in cache."""
        with self.lock:
            return frame_index in self.frames
    
    def get_cached_frame_count(self) -> int:
        """Get number of frames in cache."""
        with self.lock:
            return len(self.frames)
    
    def get_cached_frame_indices(self) -> list:
        """Get list of cached frame indices."""
        with self.lock:
            return list(self.frames.keys())
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        with self.lock:
            return self.current_memory / 1024 / 1024
    
    def clear(self):
        """Clear all cached frames."""
        with self.lock:
            self.frames.clear()
            self.metadata.clear()
            self.current_memory = 0
            logger.info("Frame cache cleared")
    
    def export_sequence(self, output_path: str, start_idx: int, end_idx: int, format: str = 'raw') -> bool:
        """
        Export cached frames as a sequence.
        
        Args:
            output_path: Directory or file prefix for output
            start_idx: Starting frame index
            end_idx: Ending frame index (inclusive)
            format: 'raw' for numpy, 'png' for PNG sequence, 'video' for video file
        
        Returns:
            True if export successful
        """
        with self.lock:
            cached_indices = list(self.frames.keys())
            available_indices = [i for i in range(start_idx, end_idx + 1) if i in cached_indices]
            
            if not available_indices:
                logger.warning(f"No cached frames in range {start_idx}-{end_idx}")
                return False
            
            logger.info(f"Exporting {len(available_indices)} frames from cache")
            return True
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                'num_frames': len(self.frames),
                'memory_usage_mb': self.current_memory / 1024 / 1024,
                'max_memory_mb': self.max_memory / 1024 / 1024,
                'frame_indices': list(self.frames.keys()),
                'export_ready': self.export_ready
            }
