"""
Image utility functions for memory optimization
"""
import cv2
import numpy as np

MAX_IMAGE_DIMENSION = 2048  # Maximum width or height for processing

def resize_if_large(image, max_dim=MAX_IMAGE_DIMENSION):
    """
    Resize image if it's too large to save memory
    Returns resized image and scale factor
    """
    h, w = image.shape[:2]
    
    if h <= max_dim and w <= max_dim:
        return image, 1.0
    
    # Calculate scale factor
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def optimize_for_processing(image):
    """Optimize image for processing to reduce memory usage"""
    # Resize if too large
    optimized, _ = resize_if_large(image)
    return optimized

