import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("Testing Gesture Integration Setup")
    print("=" * 50)
    
    # Test basic imports
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   Device: {'GPU available' if torch.cuda.is_available() else 'Using CPU'}")
    except ImportError:
        print("❌ PyTorch not installed")
        return
    
    # Test config loading
    try:
        import yaml
        config_path = os.path.join("config", "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config file loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return
    
    # Test SadTalker import
    try:
        sadtalker_path = Path("D:/SadTalker")
        if sadtalker_path.exists():
            sys.path.append(str(sadtalker_path))
        from src.facerender.animate import AnimateFromCoeff
        print("✅ SadTalker imports working")
    except Exception as e:
        print(f"❌ Error importing SadTalker: {str(e)}")
    
    print("\nTest completed. Check for any ❌ errors above.")

if __name__ == "__main__":
    main()
