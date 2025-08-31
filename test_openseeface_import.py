import sys
import os

# Add OpenSeeFace to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
open_see_face_path = os.path.join(project_root, 'OpenSeeFace')
sys.path.insert(0, open_see_face_path)

# Try importing OpenSeeFace
try:
    import openseeface
    print("✅ Successfully imported openseeface")
    print(f"OpenSeeFace version: {openseeface.__version__ if hasattr(openseeface, '__version__') else 'version not found'}")
except ImportError as e:
    print(f"❌ Failed to import openseeface: {e}")
    print("\nPython path:")
    for p in sys.path:
        print(f"- {p}")
    
    # Check if files exist
    print("\nChecking for OpenSeeFace files:")
    required_files = [
        os.path.join(open_see_face_path, 'facetracker.py'),
        os.path.join(open_see_face_path, 'tracker.py')
    ]
    
    for file_path in required_files:
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"{exists} {file_path}")
