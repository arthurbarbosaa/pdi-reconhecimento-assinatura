from preprocessing import preprocess_signature
import cv2
import numpy as np
import os


def main():
    # Example usage - process a single signature for testing
    print("=" * 60)
    print("Signature Preprocessing System")
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Path to a test signature
    test_image_path = os.path.join(project_root, "signatures", "arthur", "original", "signature-1.jpeg")
    
    print("-" * 60)
    print(f"\nProcessing single signature: {test_image_path}")
    
    # Preprocess the signature
    preprocessed = preprocess_signature(test_image_path, target_size=(300, 150))
    
    print(f"✓ Signature preprocessed successfully!")
    print(f"  Output shape: {preprocessed.shape}")
    print(f"  Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Save the preprocessed signature
    output_dir = os.path.join(project_root, "signatures", "arthur", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "signature.png")
    output_image = (preprocessed * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_image)
    
    print(f"✓ Saved preprocessed signature to: {output_path}")
    print("-" * 60)

if __name__ == "__main__":
    main()