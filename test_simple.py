#!/usr/bin/env python3
"""Simple test to verify video generation works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.sadtalker import SadTalkerModel
import tempfile
import numpy as np
from PIL import Image
import wave

def test_video_generation():
    """Test video generation directly."""
    print("Testing video generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image
        img_array = np.ones((480, 640, 3), dtype=np.uint8) * 200
        img = Image.fromarray(img_array)
        image_path = temp_path / "test.jpg"
        img.save(image_path)
        
        # Create test audio
        duration = 3.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        audio_path = temp_path / "test.wav"
        with wave.open(str(audio_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Generate video
        model = SadTalkerModel(device="cpu")
        output_path = temp_path / "output.mp4"
        
        try:
            result = model.generate(
                image_path=str(image_path),
                audio_path=str(audio_path),
                output_path=str(output_path),
                resolution="480p"
            )
            
            if Path(result).exists():
                size = Path(result).stat().st_size
                print(f"SUCCESS: Video created at {result} ({size:,} bytes)")
                
                # Copy to current directory
                import shutil
                shutil.copy2(result, "test_output.mp4")
                print("Video copied to test_output.mp4")
                return True
            else:
                print(f"FAILED: Video not created at {result}")
                return False
                
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_video_generation()
    sys.exit(0 if success else 1)