from typing import List, Tuple
import numpy as np
from .preprocessing import preprocess_signature
from .features import build_feature_vector


def load_signature_image(path: str, target_size=(300, 150)) -> np.ndarray:
    """
    Carrega e pré-processa a imagem de assinatura.
    """
    return preprocess_signature(path, target_size=target_size)


def build_dataset(samples: List[Tuple[str, int]], mode: str = "full") -> tuple[np.ndarray, np.ndarray]:
    """
    Gera X (features) e y (labels) a partir de uma lista de caminhos de imagem + rótulos.
    """
    X_list = []
    y_list = []

    for img_path, label in samples:
        try:
            image = load_signature_image(img_path)
            features = build_feature_vector(image, mode=mode)
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {img_path}: {e}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    return X, y

