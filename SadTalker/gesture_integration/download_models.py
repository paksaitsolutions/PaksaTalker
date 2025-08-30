import os
import sys
import requests
import zipfile
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination."""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download the file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Downloaded {os.path.basename(destination)}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False

def main():
    print("SadTalker Model Downloader")
    print("=" * 50)
    
    # Define paths
    base_dir = Path("D:/SadTalker")
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Model files to download
    model_files = [
        {
            'url': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00300-model.pth',
            'path': checkpoints_dir / 'auido2exp_00300-model.pth'
        },
        {
            'url': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2pose_00140-model.pth',
            'path': checkpoints_dir / 'auido2pose_00140-model.pth'
        },
        {
            'url': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar',
            'path': checkpoints_dir / 'facevid2vid_00189-model.pth.tar'
        },
        {
            'url': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar',
            'path': checkpoints_dir / 'mapping_00109-model.pth.tar'
        },
        {
            'url': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar',
            'path': checkpoints_dir / 'mapping_00229-model.pth.tar'
        },
        {
            'url': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors',
            'path': checkpoints_dir / 'SadTalker_V0.0.2_256.safetensors'
        },
        {
            'url': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/shape_predictor_68_face_landmarks.dat',
            'path': checkpoints_dir / 'shape_predictor_68_face_landmarks.dat'
        }
    ]
    
    # Download model files
    print("\nDownloading model files...")
    for model in model_files:
        if not model['path'].exists():
            download_file(model['url'], model['path'])
        else:
            print(f"✅ {model['path'].name} already exists")
    
    print("\nAll required model files are ready!")

if __name__ == "__main__":
    main()
