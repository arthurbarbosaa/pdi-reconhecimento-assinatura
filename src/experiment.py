from dataset import build_dataset
from train_model import train_signature_classifier, compute_far_frr


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
        ("signatures/full_org/original_2_1.png", 1),
        ("signatures/full_org/original_2_2.png", 1),
        ("signatures/full_org/original_2_3.png", 1),
        ("signatures/full_org/original_2_4.png", 1),
        ("signatures/full_org/original_2_5.png", 1),
        ("signatures/full_org/original_2_6.png", 1),
        ("signatures/full_org/original_2_7.png", 1),
        ("signatures/full_org/original_2_8.png", 1),
        ("signatures/full_org/original_2_9.png", 1),    
        ("signatures/full_org/original_2_10.png", 1),
        ("signatures/full_org/original_2_11.png", 1),
        ("signatures/full_org/original_2_12.png", 1),
        ("signatures/full_org/original_2_13.png", 1),
        ("signatures/full_org/original_2_14.png", 1),
        ("signatures/full_org/original_2_15.png", 1),
        ("signatures/full_org/original_2_16.png", 1),   
        ("signatures/full_org/original_2_17.png", 1),
        ("signatures/full_org/original_2_18.png", 1),
        ("signatures/full_org/original_2_19.png", 1),    
        ("signatures/full_org/original_2_20.png", 1),
        ("signatures/full_org/original_2_21.png", 1),
        ("signatures/full_org/original_2_22.png", 1),
        ("signatures/full_org/original_2_23.png", 1),
        ("signatures/full_org/original_2_24.png", 1),

        # Classe 0: Assinaturas FORJADAS
        ("signatures/full_forg/forgeries_2_1.png", 0),
        ("signatures/full_forg/forgeries_2_2.png", 0),
        ("signatures/full_forg/forgeries_2_3.png", 0),
        ("signatures/full_forg/forgeries_2_4.png", 0),
        ("signatures/full_forg/forgeries_2_5.png", 0),
        ("signatures/full_forg/forgeries_2_6.png", 0),
        ("signatures/full_forg/forgeries_2_7.png", 0),
        ("signatures/full_forg/forgeries_2_8.png", 0),
        ("signatures/full_forg/forgeries_2_9.png", 0),    
        ("signatures/full_forg/forgeries_2_10.png", 0),
        ("signatures/full_forg/forgeries_2_11.png", 0),
        ("signatures/full_forg/forgeries_2_12.png", 0),
        ("signatures/full_forg/forgeries_2_13.png", 0),
        ("signatures/full_forg/forgeries_2_14.png", 0),
        ("signatures/full_forg/forgeries_2_15.png", 0),
        ("signatures/full_forg/forgeries_2_16.png", 0),   
        ("signatures/full_forg/forgeries_2_17.png", 0),
        ("signatures/full_forg/forgeries_2_18.png", 0),
        ("signatures/full_forg/forgeries_2_19.png", 0),    
        ("signatures/full_forg/forgeries_2_20.png", 0),
        ("signatures/full_forg/forgeries_2_21.png", 0),
        ("signatures/full_forg/forgeries_2_22.png", 0),
        ("signatures/full_forg/forgeries_2_23.png", 0),
        ("signatures/full_forg/forgeries_2_24.png", 0),
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

