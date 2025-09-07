#!/usr/bin/env python3
"""
Download EMAGE weights for PaksaTalker
"""
import os
import requests
from pathlib import Path

def download_emage_weights():
    """Download EMAGE model weights"""
    
    # EMAGE checkpoint directory
    emage_root = Path(os.getenv('PAKSA_EMAGE_ROOT', 'd:/PaksaTalker/SadTalker/EMAGE'))
    checkpoints_dir = emage_root / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # EMAGE weights URL (using HuggingFace mirror)
    weights_url = "https://huggingface.co/PantoMatrix/EMAGE/resolve/main/emage_best.pth"
    weights_path = checkpoints_dir / "emage_best.pth"
    
    if weights_path.exists():
        print(f"✓ EMAGE weights already exist: {weights_path}")
        return str(weights_path)
    
    print(f"Downloading EMAGE weights to {weights_path}...")
    
    try:
        response = requests.get(weights_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ EMAGE weights downloaded successfully: {weights_path}")
        return str(weights_path)
        
    except Exception as e:
        print(f"✗ Failed to download EMAGE weights: {e}")
        
        # Try alternative download method
        try:
            import urllib.request
            print("Trying alternative download method...")
            urllib.request.urlretrieve(weights_url, weights_path)
            print(f"✓ EMAGE weights downloaded via urllib: {weights_path}")
            return str(weights_path)
        except Exception as e2:
            print(f"✗ Alternative download also failed: {e2}")
            return None

if __name__ == "__main__":
    download_emage_weights()