""Test script to verify all imports and basic functionality."""

def test_imports():
    """Test importing all core modules."""
    print("Testing imports...")
    
    # Core imports
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    
    # Config
    from config import config
    
    # Models
    from models.base import BaseModel
    from models.sadtalker import SadTalkerModel
    from models.wav2lip import Wav2LipModel
    from models.gesture import GestureModel
    from models.qwen import QwenModel
    
    # Utils
    from utils import audio_utils, video_utils
    
    print("All imports successful!")
    
    # Verify config
    print("\nConfig:")
    print(f"Device: {config.device}")
    print(f"Models directory: {config.get('models_dir', 'Not set')}")
    
    # Verify CUDA
    print("\nCUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\nBasic functionality test completed successfully!")

if __name__ == "__main__":
    test_imports()
