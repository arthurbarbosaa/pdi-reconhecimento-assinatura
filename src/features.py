import cv2
import numpy as np

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


def stroke_thickness(image: np.ndarray) -> np.ndarray:
    """
    Calcula a espessura média do traço na assinatura.
    image: imagem pré-processada, com valores entre 0 e 1.
    Retorna um array 1D com um único valor.
    """
    # binariza: traço = 1, fundo = 0
    binary = (image > 0.5).astype(np.uint8)

    # distance transform mede a distância de cada pixel de traço até o fundo
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    if np.count_nonzero(dist) == 0:
        mean_thickness = 0.0
    else:
        # média das distâncias > 0, vezes 2 ≈ diâmetro médio
        mean_thickness = float(np.mean(dist[dist > 0]) * 2.0)

    return np.array([mean_thickness], dtype=np.float32)


def filled_area_ratio(image: np.ndarray) -> np.ndarray:
    """
    Calcula a proporção de pixels de traço em relação ao total da imagem.
    image: imagem pré-processada, com valores entre 0 e 1.
    Retorna um array 1D com um único valor.
    """
    binary = (image > 0.5).astype(np.uint8)
    total_pixels = image.size
    filled_pixels = int(np.sum(binary))
    ratio = filled_pixels / total_pixels if total_pixels > 0 else 0.0
    return np.array([ratio], dtype=np.float32)


def build_feature_vector(image: np.ndarray, mode: str = "full") -> np.ndarray:
    """
    Gera o vetor final de características a partir de uma imagem pré-processada.
    mode:
      - "full": Hu + HOG + extras
      - "hu": apenas Hu Moments
      - "hog": apenas HOG
      - "hu_extra": Hu + extras simples
    """
    hu = extract_hu_moments(image)
    hog = extract_hog_features(image)
    thick = stroke_thickness(image)
    area = filled_area_ratio(image)
    extra = np.concatenate([thick, area], axis=0)

    if mode == "full":
        return np.concatenate([hu, hog, extra], axis=0).astype(np.float32)
    elif mode == "hu":
        return hu.astype(np.float32)
    elif mode == "hog":
        return hog.astype(np.float32)
    elif mode == "hu_extra":
        return np.concatenate([hu, extra], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Modo de features desconhecido: {mode}")
