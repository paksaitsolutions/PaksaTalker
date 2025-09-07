#!/usr/bin/env python3
"""Install real SadTalker using official method."""

import os
import subprocess
import sys
from pathlib import Path

def install_sadtalker():
    """Install SadTalker using official repository."""
    
    # Clone official SadTalker
    if not Path("SadTalker").exists():
        print("Cloning SadTalker repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/OpenTalker/SadTalker.git"
        ], check=True)
    
    os.chdir("SadTalker")
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True)
    
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "-r", "requirements.txt"
    ], check=True)
    
    # Download models using official script
    print("Downloading models...")
    if os.name == 'nt':  # Windows
        subprocess.run(["bash", "scripts/download_models.sh"], shell=True, check=True)
    else:
        subprocess.run(["bash", "scripts/download_models.sh"], check=True)
    
    print("SadTalker installation complete!")

if __name__ == "__main__":
    install_sadtalker()