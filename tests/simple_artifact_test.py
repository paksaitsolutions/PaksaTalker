"""Simple test script for artifact reduction."""
import cv2
import numpy as np
import torch
import os

def main():
    print("Testing artifact reduction...")
    
    # Create test directories
    os.makedirs('test_images', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Create a simple test image with artifacts
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.putText(img, "Test Image", (50, 128), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some noise/artifacts
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    
    # Save the noisy image
    noisy_path = 'test_images/noisy_test.png'
    cv2.imwrite(noisy_path, noisy_img)
    print(f"Created noisy test image at: {noisy_path}")
    
    # Try to import and use the artifact reducer
    try:
        sys.path.append(str(Path(__file__).parent))
        from models.artifact_reduction import create_artifact_reducer
        
        print("Artifact reducer imported successfully")
        
        # Initialize the reducer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        reducer = create_artifact_reducer(device=device)
        
        # Process the image
        print("Processing image...")
        processed = reducer.process_frame(noisy_img, use_deep=True)
        
        # Save the result
        output_path = 'output/processed_test.png'
        cv2.imwrite(output_path, processed)
        print(f"Saved processed image to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    from pathlib import Path
    main()
