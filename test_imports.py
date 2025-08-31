import sys
import os

# Add 3DDFA_V2 to Python path
sys.path.append(os.path.abspath('3DDFA_V2'))

# Try importing required modules
try:
    from FaceBoxes import FaceBoxes
    print("Successfully imported FaceBoxes")
    from TDDFA import TDDFA
    print("Successfully imported TDDFA")
    from utils.functions import cv_draw_landmark
    print("Successfully imported cv_draw_landmark")
    print("\nAll required modules imported successfully!")
except ImportError as e:
    print(f"Error importing module: {e}")
    print("\nPlease make sure you're running this script from the project root directory.")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
