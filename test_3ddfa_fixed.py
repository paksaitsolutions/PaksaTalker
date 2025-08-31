import os
import sys
import cv2
import torch
import numpy as np

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add 3DDFA_V2 to Python path
sys.path.insert(0, os.path.join(project_root, '3DDFA_V2'))

# Import with absolute paths
try:
    # Import FaceBoxes
    sys.path.insert(0, os.path.join(project_root, '3DDFA_V2', 'FaceBoxes'))
    from FaceBoxes import FaceBoxes
    print("✅ Successfully imported FaceBoxes")
    
    # Import TDDFA
    sys.path.insert(0, os.path.join(project_root, '3DDFA_V2'))
    from TDDFA import TDDFA
    print("✅ Successfully imported TDDFA")
    
    # Import utils
    sys.path.insert(0, os.path.join(project_root, '3DDFA_V2', 'utils'))
    from functions import cv_draw_landmark
    print("✅ Successfully imported cv_draw_landmark")
    
    # Test initialization
    print("\nTesting model initialization...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tddfa = TDDFA(gpu_mode=True if device == 'cuda' else False)
    face_boxes = FaceBoxes()
    print("✅ Successfully initialized models")
    
    print("\n🎉 All tests passed! 3DDFA_V2 is ready to use!")
    
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you're in the project root directory")
    print("2. Run 'pip install -e .' in the 3DDFA_V2 directory")
    print("3. Check that all required dependencies are installed")
    print("4. Verify the directory structure matches the expected layout")
    print("\nCurrent Python path:")
    for p in sys.path:
        print(f"- {p}")
    
    # Print directory structure for debugging
    print("\n3DDFA_V2 directory structure:")
    for root, dirs, files in os.walk(os.path.join(project_root, '3DDFA_V2')):
        level = root.replace(project_root, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f.endswith('.py') or f.endswith('.pth') or f.endswith('.pkl') or f.endswith('.tar'):
                print(f"{subindent}{f}")
