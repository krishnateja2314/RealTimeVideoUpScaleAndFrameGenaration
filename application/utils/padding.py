"""
Padding utilities for handling arbitrary video resolutions.
Models require height/width to be multiples of 32.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_padding_size(height: int, width: int, multiple: int = 32) -> tuple:
    """
    Calculate padding needed to make dimensions multiples of 'multiple'.
    
    Args:
        height: Current frame height
        width: Current frame width
        multiple: Target multiple (default 32 for our models)
    
    Returns:
        Tuple of (padded_height, padded_width, pad_h, pad_w)
        where pad_h and pad_w are (top/left, bottom/right) tuples
    """
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    
    padded_height = height + pad_h
    padded_width = width + pad_w
    
    # Distribute padding (top-left and bottom-right for reflection)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    return padded_height, padded_width, (pad_top, pad_bottom), (pad_left, pad_right)


def reflect_pad(frame: np.ndarray, target_height: int = None, target_width: int = None) -> tuple:
    """
    Apply reflection padding to frame to make dimensions multiples of 32.
    
    Args:
        frame: Input frame with shape [H, W, C] or [B, C, H, W]
        target_height: Target height (if None, compute from frame)
        target_width: Target width (if None, compute from frame)
    
    Returns:
        Tuple of (padded_frame, (orig_h, orig_w, pad_h, pad_w))
    """
    is_nchw = frame.ndim == 4 and frame.shape[1] in [1, 3, 4]
    
    if is_nchw:
        # [B, C, H, W] format
        orig_h, orig_w = frame.shape[2], frame.shape[3]
    else:
        # [H, W, C] format
        orig_h, orig_w = frame.shape[0], frame.shape[1]
    
    if target_height is None:
        target_height = ((orig_h + 31) // 32) * 32  # Round up to nearest multiple of 32
    if target_width is None:
        target_width = ((orig_w + 31) // 32) * 32
    
    pad_h = target_height - orig_h
    pad_w = target_width - orig_w
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if is_nchw:
        # Pad in format [B, C, H, W] - pad height and width dimensions
        padded = np.pad(
            frame,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='reflect'
        )
    else:
        # Pad in format [H, W, C] - pad height and width dimensions
        padded = np.pad(
            frame,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='reflect'
        )
    
    padding_info = (orig_h, orig_w, (pad_top, pad_bottom), (pad_left, pad_right))
    
    logger.debug(f"Applied reflection padding: {frame.shape} -> {padded.shape}")
    
    return padded, padding_info


def crop_padding(frame: np.ndarray, padding_info: tuple) -> np.ndarray:
    """
    Remove padding from frame to restore original dimensions.
    
    Args:
        frame: Padded frame with shape [H', W', C] or [B, C, H', W']
        padding_info: Tuple of (orig_h, orig_w, (pad_top, pad_bottom), (pad_left, pad_right))
    
    Returns:
        Cropped frame with original dimensions
    """
    orig_h, orig_w, (pad_top, pad_bottom), (pad_left, pad_right) = padding_info
    
    is_nchw = frame.ndim == 4 and frame.shape[1] in [1, 3, 4]
    
    if is_nchw:
        # [B, C, H, W] format
        if pad_top > 0 or pad_bottom > 0:
            frame = frame[:, :, pad_top:frame.shape[2]-pad_bottom, :]
        if pad_left > 0 or pad_right > 0:
            frame = frame[:, :, :, pad_left:frame.shape[3]-pad_right]
    else:
        # [H, W, C] format
        if pad_top > 0 or pad_bottom > 0:
            frame = frame[pad_top:frame.shape[0]-pad_bottom, :, :]
        if pad_left > 0 or pad_right > 0:
            frame = frame[:, pad_left:frame.shape[1]-pad_right, :]
    
    logger.debug(f"Cropped padding: restored to {orig_h}x{orig_w}")
    
    return frame


def verify_dimensions(height: int, width: int, multiple: int = 32) -> bool:
    """Check if dimensions are multiples of 'multiple'."""
    return height % multiple == 0 and width % multiple == 0
