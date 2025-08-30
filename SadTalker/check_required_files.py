import os
import sys
from pathlib import Path

def check_required_files():
    """Check for required model files and return a list of missing files."""
    base_dir = Path("D:/SadTalker")
    checkpoints_dir = base_dir / "checkpoints"
    
    required_files = [
        "auido2exp_00300-model.pth",
        "auido2pose_00140-model.pth",
        "facevid2vid_00189-model.pth.tar",
        "mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar",
        "SadTalker_V0.0.2_256.safetensors",
        "shape_predictor_68_face_landmarks.dat"
    ]
    
    missing_files = []
    for file in required_files:
        if not (checkpoints_dir / file).exists():
            missing_files.append(file)
    
    return missing_files

def main():
    print("Checking for required model files...\n")
    
    missing_files = check_required_files()
    
    if not missing_files:
        print("✅ All required model files are present!")
        print("You can now proceed with the gesture integration.")
        return
    
    print(f"❌ Missing {len(missing_files)} required model files:")
    for file in missing_files:
        print(f"- {file}")
    
    print("\nPlease download the missing files from the SadTalker repository:")
    print("https://github.com/OpenTalker/SadTalker#getting-started")
    print("\nPlace the downloaded files in the 'D:\\SadTalker\\checkpoints' directory.")

if __name__ == "__main__":
    main()
