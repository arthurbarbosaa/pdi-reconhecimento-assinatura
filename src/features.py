import cv2
import numpy as np
from typing import Dict

def extract_hu_moments(image: np.ndarray) -> np.ndarray:
    """Extracts 7 Hu invariant moments from a preprocessed signature image."""
    # Convert to 8-bit for cv2.moments
    img_uint8 = (image * 255).astype(np.uint8)
    moments = cv2.moments(img_uint8)
    hu = cv2.HuMoments(moments).flatten()
    
    # Log scale transform to compress range (common practice)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def extract_hog_features(image: np.ndarray,
                         cell_size=(16, 16),
                         block_size=(2, 2),
                         nbins=9) -> np.ndarray:
    """Extracts Histogram of Oriented Gradients (HOG) features."""
    img_uint8 = (image * 255).astype(np.uint8)
    hog = cv2.HOGDescriptor(
        _winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                  image.shape[0] // cell_size[0] * cell_size[0]),
        _blockSize=(block_size[1] * cell_size[1],
                    block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=nbins
    )
    h = hog.compute(img_uint8)
    return h.flatten()


def extract_signature_features(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Combines multiple feature extraction methods."""
    hu = extract_hu_moments(image)
    hog = extract_hog_features(image)
    
    features = np.concatenate([hu, hog])
    return {
        "hu": hu,
        "hog": hog,
        "combined": features
    }
