import cv2
import numpy as np
from pathlib import Path
from models.super_resolution import FaceSuperResolution

def test_face_super_resolution(input_path: str, output_path: str, upscale_factor: int = 2):
    """
    Test the face super-resolution on an input image.
    
    Args:
        input_path: Path to the input image
        output_path: Path to save the output image
        upscale_factor: Upscaling factor (2 or 4)
    """
    # Initialize the super-resolution model
    sr = FaceSuperResolution(upscale_factor=upscale_factor)
    
    # Read the input image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    print(f"Input image shape: {image.shape}")
    
    # Process the image
    enhanced_image = sr.process_frame(image)
    
    # Save the result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), enhanced_image)
    print(f"Enhanced image saved to: {output_path}")
    
    # Show before/after comparison
    cv2.imshow("Original", cv2.resize(image, (400, 400)))
    cv2.imshow("Enhanced", cv2.resize(enhanced_image, (800, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    input_image = "test_images/face.jpg"  # Replace with your test image path
    output_dir = "output"
    
    # Test with 2x upscaling
    test_face_super_resolution(
        input_path=input_image,
        output_path=f"{output_dir}/enhanced_2x.jpg",
        upscale_factor=2
    )
    
    # Test with 4x upscaling (more computationally intensive)
    test_face_super_resolution(
        input_path=input_image,
        output_path=f"{output_dir}/enhanced_4x.jpg",
        upscale_factor=4
    )
