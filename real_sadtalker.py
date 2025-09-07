#!/usr/bin/env python3
"""Real SadTalker implementation for high-quality video generation."""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil

class RealSadTalker:
    """Real SadTalker implementation using actual AI models."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sadtalker_path = Path("SadTalker")
        
    def generate_video(self, image_path, audio_path, output_path, resolution="1080p"):
        """Generate high-quality talking head video using real SadTalker."""
        
        if not self.sadtalker_path.exists():
            raise Exception("SadTalker not found. Please install it first.")
        
        # Prepare command for SadTalker inference
        cmd = [
            sys.executable, "inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", str(Path(output_path).parent),
            "--preprocess", "full",
            "--still"
        ]
        # Optional enhancers via env if installed
        face_enhancer = os.getenv("PAKSA_FACE_ENHANCER")
        if face_enhancer:
            cmd.extend(["--enhancer", face_enhancer])
        bg_enhancer = os.getenv("PAKSA_BG_ENHANCER")
        if bg_enhancer:
            cmd.extend(["--background_enhancer", bg_enhancer])
        
        # Add resolution settings
        size_map = {
            "480p": 512,
            "720p": 720, 
            "1080p": 1024,
            "1440p": 1440,
            "4k": 2048
        }
        
        if resolution in size_map:
            cmd.extend(["--size", str(size_map[resolution])])
        
        # Execute SadTalker
        print(f"Running SadTalker with command: {' '.join(cmd)}")
        
        # No timeout: let long CPU runs finish
        result = subprocess.run(
            cmd,
            cwd=str(self.sadtalker_path),
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise Exception(f"SadTalker failed: {result.stderr}")
        
        # Find generated video
        result_dir = Path(output_path).parent
        video_files = list(result_dir.glob("*.mp4"))
        
        if not video_files:
            raise Exception("No video file generated")
        
        # Get the most recent video file
        latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
        
        # Move to final output path
        if str(latest_video) != output_path:
            shutil.move(str(latest_video), output_path)
        
        return output_path

def test_real_sadtalker():
    """Test real SadTalker video generation."""
    
    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image
        img = np.ones((512, 512, 3), dtype=np.uint8) * 200
        image_path = temp_path / "test.jpg"
        cv2.imwrite(str(image_path), img)
        
        # Create test audio (3 seconds)
        import wave
        audio_path = temp_path / "test.wav"
        with wave.open(str(audio_path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2) 
            wav.setframerate(22050)
            # 3 seconds of sine wave
            t = np.linspace(0, 3, 22050 * 3)
            audio_data = (np.sin(2 * np.pi * 440 * t) * 16383).astype(np.int16)
            wav.writeframes(audio_data.tobytes())
        
        # Generate video
        output_path = temp_path / "output.mp4"
        
        sadtalker = RealSadTalker()
        result = sadtalker.generate_video(
            str(image_path),
            str(audio_path), 
            str(output_path),
            "1080p"
        )
        
        if Path(result).exists():
            size = Path(result).stat().st_size
            print(f"SUCCESS: Generated {size:,} byte video at {result}")
            
            # Copy to main directory
            shutil.copy2(result, "real_output.mp4")
            return True
        else:
            print("FAILED: No video generated")
            return False

if __name__ == "__main__":
    test_real_sadtalker()
