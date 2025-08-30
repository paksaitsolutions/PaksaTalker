import sys
import os
from pathlib import Path

def test_import(module_name, path=None):
    try:
        if path:
            sys.path.insert(0, str(path))
        __import__(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}")
        print(f"   Error: {str(e)}")
        return False

def main():
    print("Testing Imports Step by Step")
    print("=" * 50)
    
    # Add SadTalker to path
    sadtalker_path = Path("D:/SadTalker")
    if not sadtalker_path.exists():
        print("❌ SadTalker directory not found at D:/SadTalker")
        return
    
    # Test basic imports
    print("\n1. Testing basic imports:")
    test_import("numpy")
    test_import("torch")
    test_import("cv2")
    test_import("yaml")
    
    # Test SadTalker imports
    print("\n2. Testing SadTalker imports:")
    test_import("src", sadtalker_path)
    test_import("src.facerender", sadtalker_path)
    test_import("src.facerender.animate", sadtalker_path)
    
    # Test other required modules
    print("\n3. Testing other required modules:")
    test_import("scipy")
    test_import("librosa")
    test_import("soundfile")

if __name__ == "__main__":
    main()
