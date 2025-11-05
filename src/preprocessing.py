import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple

def preprocess_signature(image_path: str, target_size: Tuple[int, int] = (300, 150)) -> np.ndarray:
    # Step 1: Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Unable to load image from: {image_path}")
    
    # Step 2: Invert the image (white signature on black -> black on white)
    inverted = cv2.bitwise_not(image)
    
    # Step 3: Normalize pixel values to [0, 1]
    normalized = inverted.astype(np.float32) / 255.0
    
    # Step 4: Crop tightly around the signature region
    # Find non-zero pixels (signature pixels)
    # Convert back to uint8 for findNonZero
    binary = (normalized > 0.1).astype(np.uint8) * 255
    coords = cv2.findNonZero(binary)
    
    # Get bounding box around signature
    x, y, w, h = cv2.boundingRect(coords)
    cropped = normalized[y:y+h, x:x+w]
    
    # Step 5: Resize the cropped image to target size while maintaining aspect ratio
    # Calculate scaling factor to fit within target size
    scale_w = target_size[0] / w
    scale_h = target_size[1] / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the cropped signature
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Step 6: Center the resized signature in a white frame
    # Create a white canvas of target size
    canvas = np.ones((*target_size[::-1], ), dtype=np.float32)
    
    # Calculate position to center the signature
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    
    # Place the signature on the canvas
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    return canvas