#!/usr/bin/env python3
"""Test perfect video generation."""

import tempfile
import numpy as np
from PIL import Image
import wave
from pathlib import Path
from perfect_sadtalker import PerfectSadTalker

def test_perfect_generation():
    """Test perfect video generation."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create high-quality test image (1080p)
        img_array = np.random.randint(100, 200, (1080, 1920, 3), dtype=np.uint8)
        # Add face-like features
        center_y, center_x = 540, 960
        cv2.circle(img_array, (center_x, center_y), 200, (220, 180, 140), -1)  # Face
        cv2.circle(img_array, (center_x-60, center_y-40), 20, (0, 0, 0), -1)   # Left eye
        cv2.circle(img_array, (center_x+60, center_y-40), 20, (0, 0, 0), -1)   # Right eye
        cv2.ellipse(img_array, (center_x, center_y+40), (40, 20), 0, 0, 180, (100, 50, 50), -1)  # Mouth
        
        img = Image.fromarray(img_array)
        image_path = temp_path / "test_hd.jpg"
        img.save(image_path, quality=95)
        
        # Create longer audio (10 seconds for testing)
        audio_path = temp_path / "test_long.wav"
        sample_rate = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create speech-like audio with varying frequency
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
        audio_data += np.sin(2 * np.pi * 880 * t) * 0.1  # Harmonics
        audio_data = (audio_data * 16383).astype(np.int16)
        
        with wave.open(str(audio_path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_data.tobytes())
        
        # Generate perfect video
        output_path = temp_path / "perfect_output.mp4"
        
        sadtalker = PerfectSadTalker()
        result = sadtalker.generate_perfect_video(
            str(image_path),
            str(audio_path),
            str(output_path),
            "1080p"
        )
        
        if Path(result).exists():
            size = Path(result).stat().st_size
            print(f"SUCCESS: Generated {size:,} byte HD video")
            
            # Copy to main directory
            import shutil
            shutil.copy2(result, "perfect_video.mp4")
            return True
        
        return False

if __name__ == "__main__":
    import cv2
    test_perfect_generation()