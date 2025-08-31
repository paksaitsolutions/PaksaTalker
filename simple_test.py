import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
print(f"Project root: {project_root}")

# Add 3DDFA_V2 to Python path
df_path = os.path.join(project_root, '3DDFA_V2')
print(f"3DDFA_V2 path: {df_path}")
sys.path.insert(0, df_path)

# List files in 3DDFA_V2
try:
    print("\nFiles in 3DDFA_V2:")
    for f in os.listdir(df_path):
        if os.path.isdir(os.path.join(df_path, f)):
            print(f"- {f}/")
        else:
            print(f"- {f}")
except Exception as e:
    print(f"Error listing directory: {e}")

# Try importing FaceBoxes
try:
    print("\nTrying to import FaceBoxes...")
    from FaceBoxes.FaceBoxes import FaceBoxes
    print("✅ Successfully imported FaceBoxes!")
    print(f"FaceBoxes module: {FaceBoxes.__module__}")
except Exception as e:
    print(f"❌ Failed to import FaceBoxes: {e}")

# Try importing TDDFA
try:
    print("\nTrying to import TDDFA...")
    from TDDFA import TDDFA
    print("✅ Successfully imported TDDFA!")
    print(f"TDDFA module: {TDDFA.__module__}")
except Exception as e:
    print(f"❌ Failed to import TDDFA: {e}")

# Try importing cv_draw_landmark
try:
    print("\nTrying to import cv_draw_landmark...")
    from utils.functions import cv_draw_landmark
    print("✅ Successfully imported cv_draw_landmark!")
    print(f"cv_draw_landmark module: {cv_draw_landmark.__module__}")
except Exception as e:
    print(f"❌ Failed to import cv_draw_landmark: {e}")

print("\nPython path:")
for p in sys.path:
    print(f"- {p}")
