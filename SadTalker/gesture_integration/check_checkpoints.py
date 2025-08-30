from pathlib import Path
import os

def main():
    checkpoints_dir = Path("D:/SadTalker/checkpoints")
    
    if not checkpoints_dir.exists():
        print("❌ Checkpoints directory does not exist")
        return
    
    print(f"📁 Checkpoints directory: {checkpoints_dir}")
    print("\nContents:")
    
    files = list(checkpoints_dir.glob("*"))
    
    if not files:
        print("  (empty)")
    else:
        for file in files:
            print(f"  - {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
    
    # Check for required files
    required_files = [
        "auido2exp_00300-model.pth",
        "auido2pose_00140-model.pth",
        "facevid2vid_00189-model.pth.tar",
        "mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar",
        "SadTalker_V0.0.2_256.safetensors",
        "shape_predictor_68_face_landmarks.dat"
    ]
    
    print("\nChecking for required files:")
    missing_files = []
    for file in required_files:
        file_path = checkpoints_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files")
        print("Please download the missing files from the SadTalker repository")
    else:
        print("\n✅ All required files are present!")

if __name__ == "__main__":
    main()
