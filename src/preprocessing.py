import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple

def preprocess_signature(image_path: str, target_size: Tuple[int, int] = (300, 150)) -> np.ndarray:
    # Etapa 1: Carregar imagem em escala de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem de: {image_path}")
    
    # Etapa 2: Inverter a imagem (assinatura branca em fundo preto -> preta em fundo branco)
    inverted = cv2.bitwise_not(image)
    
    # Etapa 3: Normalizar valores de pixel para [0, 1]
    normalized = inverted.astype(np.float32) / 255.0
    
    # Etapa 4: Recortar ao redor da região da assinatura
    # Encontrar pixels não-zero (pixels da assinatura)
    # Converter de volta para uint8 para findNonZero
    binary = (normalized > 0.1).astype(np.uint8) * 255
    coords = cv2.findNonZero(binary)
    
    # Obter caixa delimitadora ao redor da assinatura
    x, y, w, h = cv2.boundingRect(coords)
    cropped = normalized[y:y+h, x:x+w]
    
    # Etapa 5: Redimensionar a imagem recortada para o tamanho alvo mantendo a proporção
    # Calcular fator de escala para caber no tamanho alvo
    scale_w = target_size[0] / w
    scale_h = target_size[1] / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Redimensionar a assinatura recortada
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Etapa 6: Centralizar a assinatura redimensionada em um quadro branco
    # Criar um canvas branco do tamanho alvo
    canvas = np.ones((*target_size[::-1], ), dtype=np.float32)
    
    # Calcular posição para centralizar a assinatura
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    
    # Posicionar a assinatura no canvas
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    return canvas