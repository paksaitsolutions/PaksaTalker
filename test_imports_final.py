import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Try to import the package
try:
    # Try importing directly from the package
    print("Trying to import from 3ddfa_v2 package...")
    from TDDFA_V2 import TDDFA, FaceBoxes, cv_draw_landmark
    print("Successfully imported from 3ddfa_v2 package!")
    print(f"TDDFA module: {TDDFA.__module__}")
    print(f"FaceBoxes module: {FaceBoxes.__module__}")
    print(f"cv_draw_landmark module: {cv_draw_landmark.__module__}")
except ImportError as e:
    print(f"Failed to import from 3ddfa_v2 package: {e}")
    print("\nTrying direct imports from source files...")
    
    # Add 3DDFA_V2 to Python path for direct imports
    sys.path.insert(0, os.path.join(project_root, '3DDFA_V2'))
    
    try:
        from TDDFA import TDDFA as TDDFA_direct
        from FaceBoxes.FaceBoxes import FaceBoxes as FaceBoxes_direct
        from utils.functions import cv_draw_landmark as cv_draw_landmark_direct
        print("\nSuccessfully imported using direct imports!")
        print(f"TDDFA module: {TDDFA_direct.__module__}")
        print(f"FaceBoxes module: {FaceBoxes_direct.__module__}")
        print(f"cv_draw_landmark module: {cv_draw_landmark_direct.__module__}")
        
        # If direct imports work, create a simple test function
        def test_imports():
            print("\nTesting imports by creating instances...")
            try:
                # Test TDDFA
                tddfa = TDDFA_direct()
                print("Successfully created TDDFA instance!")
                
                # Test FaceBoxes
                face_boxes = FaceBoxes_direct()
                print("Successfully created FaceBoxes instance!")
                
                return True
            except Exception as e:
                print(f"Error creating instances: {e}")
                return False
        
        if test_imports():
            print("\n✅ All tests passed! You can now use 3DDFA_V2 in your project.")
            print("\nExample usage in your code:")
            print("""
            import os
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from TDDFA_V2 import TDDFA, FaceBoxes
            
            # Initialize models
            tddfa = TDDFA()
            face_boxes = FaceBoxes()
            """)
        
    except ImportError as e:
        print(f"\n❌ Failed to import required modules: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you're in the project root directory")
        print("2. Run 'pip install -e .' in the 3DDFA_V2 directory")
        print("3. Check that all required dependencies are installed")
        print("4. Verify the directory structure matches the expected layout")
