#!/usr/bin/env python3
"""Test script to verify video generation works."""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import wave

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from models.sadtalker import SadTalkerModel

def create_test_image(width=640, height=480):
    """Create a simple test image."""
    # Create a simple face-like image
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add a simple face shape (circle)
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Face color
    img_array[mask] = [220, 180, 140]  # Skin tone
    
    # Add simple eyes
    eye_y = center_y - radius // 3
    eye_radius = radius // 8
    
    # Left eye
    left_eye_x = center_x - radius // 3
    eye_mask = (x - left_eye_x)**2 + (y - eye_y)**2 <= eye_radius**2
    img_array[eye_mask] = [0, 0, 0]  # Black
    
    # Right eye
    right_eye_x = center_x + radius // 3
    eye_mask = (x - right_eye_x)**2 + (y - eye_y)**2 <= eye_radius**2
    img_array[eye_mask] = [0, 0, 0]  # Black
    
    # Add simple mouth
    mouth_y = center_y + radius // 3
    mouth_width = radius // 2
    mouth_height = radius // 8
    
    mouth_mask = (
        (x >= center_x - mouth_width // 2) & 
        (x <= center_x + mouth_width // 2) &
        (y >= mouth_y - mouth_height // 2) & 
        (y <= mouth_y + mouth_height // 2)
    )
    img_array[mouth_mask] = [100, 50, 50]  # Dark red
    
    return Image.fromarray(img_array)

def create_test_audio(duration=3.0, sample_rate=22050):
    """Create a simple test audio file."""
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    return audio_data, sample_rate

def test_video_generation():
    """Test the video generation pipeline."""
    print("Testing PaksaTalker video generation...")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image
        print("Creating test image...")
        test_image = create_test_image()
        image_path = temp_path / "test_face.jpg"
        test_image.save(image_path)
        print(f"Test image saved: {image_path}")
        
        # Create test audio
        print("Creating test audio...")
        audio_data, sample_rate = create_test_audio()
        audio_path = temp_path / "test_audio.wav"
        
        with wave.open(str(audio_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"Test audio saved: {audio_path}")
        
        # Initialize SadTalker model
        print("Initializing SadTalker model...")
        model = SadTalkerModel(device="cpu")  # Use CPU for testing
        
        # Generate video
        print("Generating video...")
        output_path = temp_path / "test_output.mp4"
        
        try:
            result_path = model.generate(
                image_path=str(image_path),
                audio_path=str(audio_path),
                output_path=str(output_path),
                resolution="480p"
            )
            
            print(f"[DEBUG] Checking for video at: {result_path}")
            
            # Check if video was created
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path)
                print("[SUCCESS] Video generated successfully!")
                print(f"   Output: {result_path}")
                print(f"   Size: {file_size:,} bytes")
                
                # Copy to a permanent location for inspection
                permanent_output = Path("test_output.mp4")
                import shutil
                shutil.copy2(result_path, permanent_output)
                print(f"   Copied to: {permanent_output.absolute()}")
                
                return True
            else:
                print(f"[ERROR] Video file was not created at: {result_path}")
                print(f"[DEBUG] Directory contents: {list(os.listdir(os.path.dirname(result_path)))}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Video generation failed: {e}")
            return False

if __name__ == "__main__":
    success = test_video_generation()
    if success:
        print("\n[SUCCESS] Test completed successfully!")
        print("The video generation system is working.")
    else:
        print("\n[FAILED] Test failed!")
        print("There are issues with the video generation system.")
    
    sys.exit(0 if success else 1)