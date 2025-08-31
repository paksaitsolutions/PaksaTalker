import os
import sys

print("Python version:", sys.version)
print("\nPython paths:")
for path in sys.path:
    print(f"- {path}")

print("\nCurrent working directory:", os.getcwd())
print("\nFiles in 3DDFA_V2/FaceBoxes:")
try:
    print(os.listdir(os.path.join('3DDFA_V2', 'FaceBoxes')))
except Exception as e:
    print(f"Error listing directory: {e}")

print("\nTrying to import FaceBoxes...")
try:
    from FaceBoxes.FaceBoxes import FaceBoxes
    print("Successfully imported FaceBoxes!")
    print("FaceBoxes module location:", FaceBoxes.__module__)
except Exception as e:
    print(f"Error importing FaceBoxes: {e}")

print("\nTrying to import TDDFA...")
try:
    from TDDFA import TDDFA
    print("Successfully imported TDDFA!")
    print("TDDFA module location:", TDDFA.__module__)
except Exception as e:
    print(f"Error importing TDDFA: {e}")
