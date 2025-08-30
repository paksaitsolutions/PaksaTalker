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
    except ImportError as e:
        print(f"❌ Failed to import {module_name}")
        print(f"   Error: {str(e)}")
        return False

def main():
    print("Testing SadTalker Imports")
    print("=" * 50)
    
    # Add SadTalker to path
    sadtalker_path = Path("D:/SadTalker")
    if not sadtalker_path.exists():
        print("❌ SadTalker directory not found at D:/SadTalker")
        return
    
    # Test basic imports
    print("\nTesting basic imports:")
    test_import("numpy")
    test_import("torch")
    
    # Test SadTalker imports
    print("\nTesting SadTalker imports:")
    test_import("src.facerender.animate", sadtalker_path)
    test_import("src.generate_batch", sadtalker_path)
    test_import("src.utils.preprocess", sadtalker_path)

if __name__ == "__main__":
    main()
