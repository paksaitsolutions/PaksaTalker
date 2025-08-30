import os
import urllib.request
from pathlib import Path

def download_file(url, destination):
    try:
        print(f"Downloading {os.path.basename(url)}...")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, reporthook=reporthook)
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {os.path.basename(url)}: {e}")
        return False

# Base URL for the models
base_url = "https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/checkpoints"
checkpoints_dir = Path(r"D:\SadTalker\checkpoints")

# Files to download
files = [
    "auido2exp_00300-model.pth",
    "auido2pose_00140-model.pth",
    "facevid2vid_00189-model.pth.tar",
    "shape_predictor_68_face_landmarks.dat"
]

print("Starting downloads...\n" + "="*50)

for file in files:
    url = f"{base_url}/{file}"
    dest = checkpoints_dir / file
    
    # Skip if file already exists and has size > 0
    if dest.exists() and dest.stat().st_size > 0:
        print(f"\n✅ {file} already exists, skipping...")
        continue
        
    print(f"\n--- Downloading {file} ---")
    success = download_file(url, dest)
    
    if not success:
        print(f"Failed to download {file}")

# List all files in checkpoints directory
print("\nCurrent contents of checkpoints directory:" + "\n" + "-"*50)
for f in checkpoints_dir.glob("*"):
    if f.is_file():
        print(f"- {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
