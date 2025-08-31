import os
import sys
import cv2
import numpy as np

def test_imports():
    print("Testing imports...")
    
    # Add 3DDFA_V2 to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    try:
        # Test FaceBoxes import
        sys.path.insert(0, os.path.join(project_root, '3DDFA_V2'))
        from FaceBoxes.FaceBoxes import FaceBoxes
        print("✅ Successfully imported FaceBoxes")
        
        # Test TDDFA import
        from TDDFA import TDDFA
        print("✅ Successfully imported TDDFA")
        
        # Test utils import
        from utils.functions import cv_draw_landmark
        print("✅ Successfully imported cv_draw_landmark")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        
        # Create a dummy image
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Test drawing landmarks
        pts = np.array([[100, 100], [150, 100], [125, 150]], dtype=np.float32)
        img_with_landmarks = cv_draw_landmark(img, pts)
        
        # Save the result
        output_path = os.path.join(project_root, 'test_output.jpg')
        cv2.imwrite(output_path, img_with_landmarks)
        print(f"✅ Successfully saved test output to {output_path}")
        
        print("\n🎉 All tests passed! 3DDFA_V2 is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPython path:")
        for p in sys.path:
            print(f"- {p}")
        return False

if __name__ == "__main__":
    test_imports()
