from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


def train_signature_classifier(X, y, test_size=0.3, random_state=42):
    """
    Treina um classificador SVM para verificação de assinaturas.
    """
    n_samples = len(y)
    
    if n_samples < 2:
        raise ValueError(
            f"É necessário pelo menos 2 amostras para treinar o modelo. "
            f"Encontradas apenas {n_samples} amostra(s)."
        )
    
    # Se houver poucas amostras, ajusta o test_size
    min_train_samples = 1
    if n_samples * (1 - test_size) < min_train_samples:
        test_size = max(0.1, 1 - min_train_samples / n_samples)
        print(f"[AVISO] Ajustando test_size para {test_size:.2f} devido ao número limitado de amostras")
    
    # Verifica se há classes suficientes para stratify
    # stratify requer pelo menos 2 amostras em cada classe
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts) if len(class_counts) > 0 else 0
    use_stratify = len(unique_classes) >= 2 and min_class_count >= 2
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if use_stratify else None
    )

    clf = svm.SVC(kernel="rbf", C=10.0, gamma=0.001)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Acurácia: {acc * 100:.2f}%")
    print("Matriz de confusão:")
    print(cm)

    return clf, (y_test, y_pred)


def compute_far_frr(y_true, y_pred):
    """
    Calcula FAR e FRR considerando:
      - classe 1 = genuína
      - classe 0 = forjada
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # genuína = 1
    genuine_mask = (y_true == 1)
    forged_mask = (y_true == 0)

    # FRR: genuína classificada como forjada
    frr = (y_pred[genuine_mask] == 0).mean() if genuine_mask.any() else 0.0

    # FAR: forjada classificada como genuína
    far = (y_pred[forged_mask] == 1).mean() if forged_mask.any() else 0.0

    return far, frr

