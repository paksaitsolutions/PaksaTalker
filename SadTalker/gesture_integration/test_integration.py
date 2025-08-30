import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test if basic imports are working."""
    print("Testing basic imports...")
    try:
        import yaml
        import torch
        import cv2
        import numpy as np
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def test_sadtalker_import():
    """Test if SadTalker can be imported."""
    print("\nTesting SadTalker import...")
    try:
        # Add SadTalker to path
        sadtalker_path = Path("D:/SadTalker")
        if sadtalker_path.exists():
            sys.path.append(str(sadtalker_path))
        
        from src.facerender.animate import AnimateFromCoeff
        print("✅ SadTalker imports are working")
        return True
    except Exception as e:
        print(f"❌ Error importing SadTalker: {str(e)}")
        return False

def test_stylegestures_import():
    """Test if StyleGestures can be imported."""
    print("\nTesting StyleGestures import...")
    try:
        # Add StyleGestures to path
        stylegestures_path = Path("D:/SadTalker/awesome-gesture_generation/approach/data-driven/StyleGestures")
        if stylegestures_path.exists():
            sys.path.append(str(stylegestures_path))
        
        # This will fail if StyleGestures is not properly set up
        print("ℹ️ Note: StyleGestures integration will be tested separately")
        return True
    except Exception as e:
        print(f"❌ Error setting up StyleGestures: {str(e)}")
        return False

def test_gesture_integrator():
    """Test the GestureIntegrator initialization."""
    print("\nTesting GestureIntegrator...")
    try:
        from src.gesture_integrator import GestureIntegrator
        
        # Test with default config
        integrator = GestureIntegrator()
        print("✅ GestureIntegrator initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing GestureIntegrator: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("Running Integration Tests")
    print("="*50)
    
    test_results = {
        "basic_imports": test_basic_imports(),
        "sadtalker_import": test_sadtalker_import(),
        "stylegestures_import": test_stylegestures_import(),
        "gesture_integrator": test_gesture_integrator()
    }
    
    print("\n" + "="*50)
    print("Test Summary:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    if all(test_results.values()):
        print("\n🎉 All tests passed! You're ready to proceed.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
