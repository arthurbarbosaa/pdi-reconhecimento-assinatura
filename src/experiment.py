from .dataset import build_dataset
from .train_model import train_signature_classifier, compute_far_frr


def main():
    # ============================================
    # ADICIONE SUAS AMOSTRAS AQUI
    # ============================================
    # Formato: (caminho_da_imagem, label)
    # - label = 1 para assinaturas GENUÍNAS
    # - label = 0 para assinaturas FORJADAS
    # ============================================
    
    samples = [
        # Classe 1: Assinaturas GENUÍNAS
        ("signatures/arthur/original/signature-1.jpeg", 1),
        ("signatures/arthur/original/signature-2.jpeg", 1),
        ("signatures/arthur/original/signature-3.jpeg", 1),
        ("signatures/arthur/original/signature-4.jpeg", 1),
        ("signatures/arthur/original/signature-5.jpeg", 1),
        
        # Classe 0: Assinaturas FORJADAS
        ("signatures/joao/signature-1.jpeg", 0),
    ]

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

