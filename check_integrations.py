#!/usr/bin/env python3
"""Check all installed AI model integrations and paths."""

import os
import sys
from pathlib import Path

def check_sadtalker():
    """Check SadTalker installation."""
    print("=== SadTalker ===")
    sadtalker_path = Path("SadTalker")
    print(f"Path: {sadtalker_path.absolute()}")
    print(f"Exists: {sadtalker_path.exists()}")
    
    if sadtalker_path.exists():
        inference_py = sadtalker_path / "inference.py"
        checkpoints = sadtalker_path / "checkpoints"
        print(f"inference.py: {inference_py.exists()}")
        print(f"checkpoints/: {checkpoints.exists()}")
        
        if checkpoints.exists():
            models = list(checkpoints.glob("*.pth*")) + list(checkpoints.glob("*.safetensors"))
            print(f"Model files: {len(models)}")
            for model in models[:3]:  # Show first 3
                print(f"  - {model.name}")

def check_emage():
    """Check EMAGE installation."""
    print("\n=== EMAGE ===")
    
    # Check multiple possible locations
    locations = [
        Path("EMAGE"),
        Path("SadTalker/EMAGE"), 
        Path("SadTalker/EMAGE/SadTalker/EMAGE")
    ]
    
    emage_path = None
    for loc in locations:
        if loc.exists():
            emage_path = loc
            break
    
    print(f"Found at: {emage_path.absolute() if emage_path else 'Not found'}")
    
    if emage_path:
        models_dir = emage_path / "models"
        checkpoints_dir = emage_path / "checkpoints"
        
        print(f"models/: {models_dir.exists()}")
        print(f"checkpoints/: {checkpoints_dir.exists()}")
        
        if models_dir.exists():
            py_files = list(models_dir.glob("*.py"))
            print(f"Python files: {len(py_files)}")
            
        if checkpoints_dir.exists():
            pth_files = list(checkpoints_dir.glob("*.pth"))
            print(f"Checkpoint files: {len(pth_files)}")
    
    # Check environment variables
    env_vars = ["PAKSA_EMAGE_ROOT", "EMAGE_ROOT"]
    for var in env_vars:
        val = os.environ.get(var)
        print(f"{var}: {val}")

def check_openseeface():
    """Check OpenSeeFace installation."""
    print("\n=== OpenSeeFace ===")
    osf_path = Path("OpenSeeFace")
    print(f"Path: {osf_path.absolute()}")
    print(f"Exists: {osf_path.exists()}")
    
    if osf_path.exists():
        models_dir = osf_path / "models"
        tracker_py = osf_path / "tracker.py"
        
        print(f"models/: {models_dir.exists()}")
        print(f"tracker.py: {tracker_py.exists()}")
        
        if models_dir.exists():
            onnx_files = list(models_dir.glob("*.onnx"))
            print(f"ONNX models: {len(onnx_files)}")

def check_wav2lip():
    """Check Wav2Lip installation."""
    print("\n=== Wav2Lip ===")
    wav2lip_paths = [
        Path("wav2lip2-aoti"),
        Path("models/wav2lip"),
        Path("src/wav2lip")
    ]
    
    for path in wav2lip_paths:
        print(f"{path}: {path.exists()}")
        if path.exists() and (path / "checkpoints").exists():
            checkpoints = list((path / "checkpoints").glob("*.pth*"))
            print(f"  Checkpoints: {len(checkpoints)}")

def check_python_imports():
    """Check if modules can be imported."""
    print("\n=== Python Imports ===")
    
    modules = [
        "torch",
        "cv2", 
        "numpy",
        "librosa",
        "gradio",
        "fastapi"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"OK {module}")
        except ImportError as e:
            print(f"FAIL {module}: {e}")

def main():
    print("PaksaTalker Integration Check")
    print("=" * 40)
    
    check_sadtalker()
    check_emage() 
    check_openseeface()
    check_wav2lip()
    check_python_imports()
    
    print("\n=== Environment Variables ===")
    env_vars = [
        "PAKSA_EMAGE_ROOT",
        "EMAGE_ROOT", 
        "PAKSA_OSF_ROOT",
        "OPENSEEFACE_ROOT",
        "SADTALKER_WEIGHTS"
    ]
    
    for var in env_vars:
        val = os.environ.get(var, "Not set")
        print(f"{var}: {val}")

if __name__ == "__main__":
    main()