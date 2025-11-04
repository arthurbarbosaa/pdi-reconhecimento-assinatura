from preprocessing import preprocess_signature
import cv2
import numpy as np
import os


def main():
    """Main function to run signature preprocessing."""
    
    # Example usage - process a single signature for testing
    print("=" * 60)
    print("Signature Preprocessing System")
    print("=" * 60)
    
    # Get the project root directory (parent of src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Path to a test signature
    test_image_path = os.path.join(project_root, "signatures", "arthur", "signature-1.jpeg")
    
    print(f"\nProcessing single signature: {test_image_path}")
    print("-" * 60)
    
    # Preprocess the signature
    preprocessed = preprocess_signature(test_image_path, target_size=(300, 150))
    
    print(f"✓ Signature preprocessed successfully!")
    print(f"  Output shape: {preprocessed.shape}")
    print(f"  Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Save the preprocessed signature
    output_path = os.path.join(project_root, "test_preprocessed_signature.png")
    output_image = (preprocessed * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_image)
    
    print(f"✓ Saved preprocessed signature to: {output_path}")
    print("-" * 60)
    
    # Uncomment below to process the entire dataset later:
    # print("\n" + "=" * 60)
    # print("Processing entire dataset...")
    # print("=" * 60)
    # preprocess_dataset(
    #     base_path="../signatures",
    #     output_path="../signatures_preprocessed",
    #     target_size=(300, 150)
    # )


if __name__ == "__main__":
    main()