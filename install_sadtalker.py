#!/usr/bin/env python3
"""Install real SadTalker model."""

import os
import subprocess
import sys
from pathlib import Path

def install_sadtalker():
    """Install SadTalker from GitHub."""
    
    # Clone SadTalker repository
    if not os.path.exists("SadTalker"):
        print("Cloning SadTalker repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/OpenTalker/SadTalker.git"
        ], check=True)
    
    # Install requirements
    print("Installing SadTalker requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "-r", "SadTalker/requirements.txt"
    ], check=True)
    
    # Download checkpoints
    print("Downloading SadTalker checkpoints...")
    os.chdir("SadTalker")
    
    # Download models
    subprocess.run([
        "bash", "scripts/download_models.sh"
    ], check=True)
    
    print("SadTalker installation complete!")

if __name__ == "__main__":
    install_sadtalker()