import sys
import os
from pathlib import Path

def main():
    print("Testing SadTalker Imports")
    print("=" * 50)
    
    # Add SadTalker to path
    sadtalker_path = Path("D:/SadTalker")
    if not sadtalker_path.exists():
        print("❌ SadTalker directory not found at D:/SadTalker")
        return
    
    sys.path.append(str(sadtalker_path))
    
    # Test basic imports
    try:
        from src.facerender.animate import AnimateFromCoeff
        print("✅ Successfully imported AnimateFromCoeff")
    except Exception as e:
        print(f"❌ Error importing AnimateFromCoeff: {str(e)}")
    
    try:
        from src.generate_batch import get_data
        print("✅ Successfully imported get_data")
    except Exception as e:
        print(f"❌ Error importing get_data: {str(e)}")
    
    try:
        from src.utils.preprocess import CropAndExtract
        print("✅ Successfully imported CropAndExtract")
    except Exception as e:
        print(f"❌ Error importing CropAndExtract: {str(e)}")

if __name__ == "__main__":
    main()
