import os
import sys
import cv2
import torch
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Try to import from the package
try:
    from TDDFA_V2 import TDDFA, FaceBoxes, cv_draw_landmark
    print("Successfully imported 3DDFA_V2 package!")
except ImportError as e:
    print(f"Error importing 3DDFA_V2 package: {e}")
    print("Trying alternative import method...")
    try:
        # Fallback to direct imports
        sys.path.insert(0, os.path.join(project_root, '3DDFA_V2'))
        from TDDFA import TDDFA
        from FaceBoxes.FaceBoxes import FaceBoxes
        from utils.functions import cv_draw_landmark
        print("Successfully imported using direct imports!")
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        print("Please make sure you've run 'pip install -e .' in the 3DDFA_V2 directory")
        sys.exit(1)

def main():
    # Initialize TDDFA
    cfg = {
        'arch': 'mobilenet',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'tddfa_ckpt_fp': '3DDFA_V2/checkpoints/phase1_wpdc_vdc.pth.tar',
        'bfm_fp': 'configs/bfm_noneck_v3.pkl',
        'size': 120,
        'num_params': 62
    }
    
    tddfa = TDDFA(**cfg)
    face_boxes = FaceBoxes()
    
    # Load a test image
    img_fp = 'samples/1.jpg'  # You may need to provide a test image
    if not os.path.exists(img_fp):
        print(f"Test image not found at {img_fp}")
        return
        
    img = cv2.imread(img_fp)
    
    # Face detection
    boxes = face_boxes(img)
    
    # 3D face alignment
    param_lst, roi_box_lst = tddfa(img, boxes)
    
    # Visualize landmarks
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    for ver in ver_lst:
        img = cv_draw_landmark(img, ver)
    
    # Save the result
    output_fp = 'output.jpg'
    cv2.imwrite(output_fp, img)
    print(f"Result saved to {output_fp}")

if __name__ == '__main__':
    # Add 3DDFA_V2 to Python path
    sys.path.append('3DDFA_V2')
    main()
