"""
Normalization utilities for frame processing.
Handles conversion between 8-bit [0, 255] and float32 [0.0, 1.0] ranges.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize frame from [0, 255] uint8 to [0.0, 1.0] float32.
    
    Args:
        frame: Input frame with shape [H, W, C] or [B, C, H, W], dtype uint8
    
    Returns:
        Normalized frame as float32 in range [0.0, 1.0]
    """
    if frame.dtype == np.uint8:
        return frame.astype(np.float32) / 255.0
    elif frame.dtype == np.float32:
        # Already normalized
        if frame.max() > 1.1:  # Probably in [0, 255] range
            return frame / 255.0
        return frame
    else:
        logger.warning(f"Unexpected frame dtype: {frame.dtype}, attempting conversion")
        return frame.astype(np.float32) / 255.0


def denormalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Denormalize frame from [0.0, 1.0] float32 to [0, 255] uint8.
    
    Args:
        frame: Input frame with shape [H, W, C] or [B, C, H, W], dtype float32 in [0.0, 1.0]
    
    Returns:
        Denormalized frame as uint8 in range [0, 255]
    """
    # Ensure frame is in [0, 1] range
    frame = np.clip(frame, 0.0, 1.0)
    
    # Convert to uint8
    return (frame * 255.0).astype(np.uint8)


def to_nchw(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame from HWC to NCHW format (add batch dimension if needed).
    
    Args:
        frame: Input frame with shape [H, W, C] (3D) or [B, H, W, C] (4D)
    
    Returns:
        Frame with shape [B, C, H, W] (4D)
    """
    if frame.ndim == 3:  # [H, W, C]
        frame = np.expand_dims(frame, 0)  # [1, H, W, C]
    
    if frame.shape[-1] == 3 or frame.shape[-1] == 1:  # Last dimension is channels
        # BHWC -> BCHW
        return np.transpose(frame, (0, 3, 1, 2))
    
    return frame


def to_hwc(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame from NCHW to HWC format (remove batch dimension if batch=1).
    
    Args:
        frame: Input frame with shape [B, C, H, W] (4D)
    
    Returns:
        Frame with shape [H, W, C] (3D) if B=1
    """
    if frame.ndim == 4:  # [B, C, H, W]
        # BCHW -> BHWC
        frame = np.transpose(frame, (0, 2, 3, 1))
        
        # Remove batch dimension if B=1
        if frame.shape[0] == 1:
            frame = frame[0]
    
    return frame
