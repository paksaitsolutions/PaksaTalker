#!/usr/bin/env python3
"""Setup real SadTalker for high-quality video generation."""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download file with progress."""
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def setup_sadtalker():
    """Setup real SadTalker with all required models."""
    
    # Create SadTalker directory
    sadtalker_dir = Path("SadTalker")
    sadtalker_dir.mkdir(exist_ok=True)
    
    # Download SadTalker source
    if not (sadtalker_dir / "inference.py").exists():
        print("Downloading SadTalker source...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/OpenTalker/SadTalker.git", 
            str(sadtalker_dir)
        ], check=True)
    
    # Create checkpoints directory
    checkpoints_dir = sadtalker_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Download required model checkpoints
    models = {
        "auido2exp_00300-model.pth": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00300-model.pth",
        "auido2pose_00140-model.pth": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2pose_00140-model.pth", 
        "mapping_00109-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
        "facevid2vid_00189-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar"
    }
    
    for filename, url in models.items():
        filepath = checkpoints_dir / filename
        if not filepath.exists():
            download_file(url, str(filepath))
    
    # Download GFPGAN models for face enhancement
    gfpgan_dir = sadtalker_dir / "gfpgan" / "weights"
    gfpgan_dir.mkdir(parents=True, exist_ok=True)
    
    gfpgan_model = gfpgan_dir / "GFPGANv1.4.pth"
    if not gfpgan_model.exists():
        download_file(
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            str(gfpgan_model)
        )
    
    # Install requirements
    requirements_file = sadtalker_dir / "requirements.txt"
    if requirements_file.exists():
        print("Installing SadTalker requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file)
        ], check=True)
    
    print("SadTalker setup complete!")
    return True

if __name__ == "__main__":
    setup_sadtalker()