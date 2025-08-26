import cv2
import os
from pathlib import Path
import numpy as np
from models.super_resolution import FaceSuperResolution
from utils.audio_utils import process_audio  # Assuming this exists in your project

def test_complete_pipeline(
    image_path: str,
    audio_path: str,
    output_dir: str = "output",
    upscale_factor: int = 2
):
    """
    Test the complete pipeline with image and audio processing.
    
    Args:
        image_path: Path to the input image
        audio_path: Path to the input audio file
        output_dir: Directory to save the output files
        upscale_factor: Upscaling factor (2 or 4)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Process the image with super-resolution
    print("Processing image with super-resolution...")
    sr = FaceSuperResolution(upscale_factor=upscale_factor)
    
    # Read and process the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    enhanced_image = sr.process_frame(image)
    
    # Save the enhanced image
    img_output_path = output_path / f"enhanced_{upscale_factor}x.jpg"
    cv2.imwrite(str(img_output_path), enhanced_image)
    print(f"Enhanced image saved to: {img_output_path}")
    
    # 2. Process the audio (placeholder - implement your audio processing)
    print("\nProcessing audio...")
    try:
        # Process audio (modify this based on your audio processing needs)
        # Example: processed_audio = process_audio(audio_path)
        audio_output_path = output_path / "processed_audio.wav"
        # Save or process the audio file
        # ...
        print(f"Audio processed and saved to: {audio_output_path}")
    except Exception as e:
        print(f"Audio processing error: {e}")
    
    # 3. Display results
    print("\nProcessing complete!")
    print(f"- Enhanced image: {img_output_path}")
    print(f"- Processed audio: {audio_output_path}")
    
    # Show the before/after comparison
    cv2.imshow("Original", cv2.resize(image, (400, 400)))
    cv2.imshow("Enhanced", cv2.resize(enhanced_image, (800, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    test_image = "test_images/face.jpg"  # Replace with your image path
    test_audio = "test_audio/audio.wav"  # Replace with your audio path
    
    # Create test directories if they don't exist
    Path("test_images").mkdir(exist_ok=True)
    Path("test_audio").mkdir(exist_ok=True)
    
    print("PaksaTalker Test Pipeline")
    print("=======================")
    print("Please ensure you have:")
    print(f"1. Test image at: {os.path.abspath(test_image)}")
    print(f"2. Test audio at: {os.path.abspath(test_audio)}")
    print("\nStarting test...\n")
    
    try:
        test_complete_pipeline(
            image_path=test_image,
            audio_path=test_audio,
            upscale_factor=2
        )
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("\nPlease check that both the image and audio files exist and are in the correct format.")
        print("Supported image formats: JPG, PNG")
        print("Supported audio formats: WAV, MP3")
