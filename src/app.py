from preprocessing import preprocess_signature
from features import extract_signature_features
import cv2
import numpy as np
import os


def main():
    # Example usage - process a single signature for testing
    print("=" * 60)
    print("Signature Preprocessing & Feature Extraction System")
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Path to a test signature
    test_image_path = os.path.join(project_root, "signatures", "arthur", "original", "signature-1.jpeg")
    
    print("-" * 60)
    print(f"\nProcessing single signature: {test_image_path}")
    
    # Step 1: Preprocess the signature
    preprocessed = preprocess_signature(test_image_path, target_size=(300, 150))
    
    print(f"✓ Signature preprocessed successfully!")
    print(f"  Output shape: {preprocessed.shape}")
    print(f"  Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Step 2: Save the preprocessed signature
    output_dir = os.path.join(project_root, "signatures", "arthur", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "signature.png")
    output_image = (preprocessed * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_image)
    
    print(f"✓ Saved preprocessed signature to: {output_path}")
    
    # Step 3: Extract features from the preprocessed signature
    print("-" * 60)
    print("Extracting features...")
    
    features = extract_signature_features(preprocessed)
    
    print(f"✓ Features extracted successfully!")
    print("\n=== Hu Moments ===")
    print(features["hu"])
    print("Shape:", features["hu"].shape)

    print("\n=== HOG Features (first 20 values) ===")
    print(features["hog"][:20])  # Mostra só os primeiros 20 valores
    print("Shape:", features["hog"].shape)
    print(f"  Hu Moments shape: {features['hu'].shape}")
    print(f"  HOG Features shape: {features['hog'].shape}")
    print(f"  Combined feature vector length: {features['combined'].shape[0]}")
    
    # Step 4: Save the combined feature vector
    features_dir = os.path.join(project_root, "signatures", "arthur", "features")
    os.makedirs(features_dir, exist_ok=True)
    feature_path = os.path.join(features_dir, "signature_features.npy")
    
    np.save(feature_path, features["combined"])
    
    print(f"✓ Saved feature vector to: {feature_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
