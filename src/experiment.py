from .dataset import build_dataset
from .train_model import train_signature_classifier, compute_far_frr
from typing import List, Tuple


def build_samples_from_persons(person_ids: List[int], samples_per_person: int = 24) -> List[Tuple[str, int]]:
    samples = []
    
    for person_id in person_ids:
        # Adiciona assinaturas genuínas (label = 1)
        for i in range(1, samples_per_person + 1):
            samples.append(
                (f"signatures/full_org/original_{person_id}_{i}.png", 1)
            )
        
        # Adiciona assinaturas forjadas (label = 0)
        for i in range(1, samples_per_person + 1):
            samples.append(
                (f"signatures/full_forg/forgeries_{person_id}_{i}.png", 0)
            )
    
    return samples


def main():
    # ============================================
    # CONFIGURAÇÃO DO DATASET
    # ============================================
    # Escolha quais pessoas incluir no treinamento
    person_ids = [1]
    
    # Quantas amostras usar de cada pessoa (padrão: 24 = todas)
    samples_per_person = 24
    
    # Gera a lista de amostras automaticamente
    samples = build_samples_from_persons(person_ids, samples_per_person)

    X, y = build_dataset(samples, mode="full")
    
    print(f"\nDataset montado: {len(y)} amostra(s) processada(s) com sucesso")
    
    if len(y) < 2:
        print("[ERRO] É necessário pelo menos 2 amostras para treinar o modelo.")
        print("Verifique se os caminhos das imagens estão corretos.")
        return
    
    model, (y_test, y_pred) = train_signature_classifier(X, y)
    far, frr = compute_far_frr(y_test, y_pred)

    print(f"\nFAR: {far * 100:.2f}%")
    print(f"FRR: {frr * 100:.2f}%")


if __name__ == "__main__":
    main()

