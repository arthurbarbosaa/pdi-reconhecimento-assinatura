"""
Preprocessing module for offline handwritten signature verification.
Handles signature image preprocessing including inversion, normalization,
cropping, and resizing.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple

def preprocess_signature(image_path: str, target_size: Tuple[int, int] = (300, 150)) -> np.ndarray:
    """
    Preprocess a single signature image.
    
    Steps:
    1. Load image in grayscale
    2. Invert colors (white signature on black -> black on white)
    3. Normalize pixel values to [0, 1]
    4. Crop tightly around signature region
    5. Resize to fixed dimensions
    6. Center in a white frame
    
    Args:
        image_path: Path to the signature image file
        target_size: Desired output size as (width, height). Default: (300, 150)
    
    Returns:
        Preprocessed signature as a NumPy array with values in [0, 1]
    """
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
    
    if coords is None:
        # If no signature found, return a white image
        print(f"Warning: No signature detected in {image_path}")
        return np.ones((*target_size[::-1], ), dtype=np.float32)
    
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


def preprocess_dataset(base_path: str, output_path: str, target_size: Tuple[int, int] = (300, 150)) -> None:
    """
    Preprocess all signature images in a dataset directory.
    
    The function expects the following structure:
    base_path/
        person1/
            signature1.jpg
            signature2.png
            ...
        person2/
            signature1.jpg
            ...
    
    And creates:
    output_path/
        person1/
            signature1.jpg
            signature2.png
            ...
        person2/
            signature1.jpg
            ...
    
    Args:
        base_path: Root directory containing signature folders
        output_path: Directory where preprocessed images will be saved
        target_size: Desired output size as (width, height). Default: (300, 150)
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    
    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    total_processed = 0
    total_folders = 0
    
    print(f"Starting preprocessing...")
    print(f"Source: {base_path}")
    print(f"Output: {output_path}")
    print(f"Target size: {target_size}")
    print("-" * 60)
    
    # Iterate through all subfolders (each represents a person)
    for person_folder in sorted(base_path.iterdir()):
        if not person_folder.is_dir():
            continue
        
        total_folders += 1
        person_name = person_folder.name
        
        # Create corresponding output folder
        output_person_folder = output_path / person_name
        output_person_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all image files in the person's folder
        image_files = [
            f for f in person_folder.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        print(f"\nProcessing folder: {person_name} ({len(image_files)} images)")
        
        folder_count = 0
        for image_file in sorted(image_files):
            try:
                # Preprocess the signature
                preprocessed = preprocess_signature(str(image_file), target_size)
                
                # Convert back to uint8 for saving
                output_image = (preprocessed * 255).astype(np.uint8)
                
                # Save with the same filename
                output_file = output_person_folder / image_file.name
                cv2.imwrite(str(output_file), output_image)
                
                folder_count += 1
                total_processed += 1
                
                print(f"  ✓ {image_file.name}")
                
            except Exception as e:
                print(f"  ✗ Error processing {image_file.name}: {str(e)}")
        
        print(f"  → Processed {folder_count}/{len(image_files)} images from {person_name}")
    
    print("-" * 60)
    print(f"Preprocessing complete!")
    print(f"Total folders: {total_folders}")
    print(f"Total images processed: {total_processed}")
    print(f"Output saved to: {output_path}")
