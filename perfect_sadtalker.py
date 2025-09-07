#!/usr/bin/env python3
"""Perfect SadTalker implementation for production-quality video generation."""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import subprocess
import time

class PerfectSadTalker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_sadtalker()
    
    def setup_sadtalker(self):
        """Setup SadTalker with all optimizations."""
        if not Path("SadTalker").exists():
            # Clone and setup SadTalker
            subprocess.run(["git", "clone", "https://github.com/OpenTalker/SadTalker.git"], check=True)
            
            # Install requirements
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "SadTalker/requirements.txt"], check=True)
            
            # Download models
            os.chdir("SadTalker")
            if os.name == 'nt':
                subprocess.run(["bash", "scripts/download_models.sh"], shell=True)
            else:
                subprocess.run(["bash", "scripts/download_models.sh"])
            os.chdir("..")
    
    def generate_perfect_video(self, image_path, audio_path, output_path, resolution="1080p", duration_minutes=5):
        """Generate perfect quality talking head video."""
        
        # Resolution settings for camera quality
        size_settings = {
            "480p": {"size": 512, "fps": 25},
            "720p": {"size": 720, "fps": 30}, 
            "1080p": {"size": 1024, "fps": 30},
            "1440p": {"size": 1440, "fps": 30},
            "4k": {"size": 2048, "fps": 24}
        }
        
        settings = size_settings.get(resolution, size_settings["1080p"])
        
        # Perfect SadTalker command for maximum quality
        cmd = [
            sys.executable, "inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", str(Path(output_path).parent),
            "--preprocess", "full",
            "--size", str(settings["size"]),
            "--expression_scale", "1.3",
            "--still",
            "--cpu" if self.device == "cpu" else ""
        ]
        # Optional enhancers via env if installed
        face_enhancer = os.getenv("PAKSA_FACE_ENHANCER")
        if face_enhancer:
            cmd.extend(["--enhancer", face_enhancer])
        bg_enhancer = os.getenv("PAKSA_BG_ENHANCER")
        if bg_enhancer:
            cmd.extend(["--background_enhancer", bg_enhancer])
        
        # Remove empty strings
        cmd = [x for x in cmd if x]
        
        print(f"Generating {duration_minutes}-minute video at {resolution}...")
        start_time = time.time()
        
        # Execute SadTalker
        # No timeout: allow long CPU runs
        result = subprocess.run(
            cmd,
            cwd="SadTalker",
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise Exception(f"SadTalker failed: {result.stderr}")
        
        # Find and move output
        result_dir = Path(output_path).parent
        video_files = list(result_dir.glob("*.mp4"))
        
        if video_files:
            latest = max(video_files, key=lambda x: x.stat().st_mtime)
            if str(latest) != output_path:
                latest.rename(output_path)
            
            # Enhance video quality with ffmpeg
            self.enhance_video_quality(output_path, settings["fps"])
            
            elapsed = time.time() - start_time
            print(f"Video generated in {elapsed:.1f}s: {Path(output_path).stat().st_size:,} bytes")
            return output_path
        
        raise Exception("No video generated")
    
    def enhance_video_quality(self, video_path, fps):
        """Enhance video quality using ffmpeg."""
        temp_path = str(video_path).replace(".mp4", "_temp.mp4")
        
        # High-quality encoding
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            temp_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            Path(video_path).unlink()
            Path(temp_path).rename(video_path)
        except:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

# Integration with API
def generate_production_video(image_path, audio_path, output_path, resolution="1080p"):
    """Generate production-quality video."""
    sadtalker = PerfectSadTalker()
    return sadtalker.generate_perfect_video(image_path, audio_path, output_path, resolution)
