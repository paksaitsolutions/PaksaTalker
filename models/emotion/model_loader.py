import os
import gdown
import requests
from tqdm import tqdm

MODEL_URLS = {
    'mini_xception_weights.h5': 'https://drive.google.com/uc?id=1M3qZA0d44s8eWXr6E0pEZ3B5yfKX3U9Y'
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'emotion')

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def ensure_model_downloaded():
    """Ensure the emotion recognition model is downloaded"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                if 'drive.google.com' in url:
                    gdown.download(url, filepath, quiet=False)
                else:
                    download_file(url, filepath)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return False
    return True

def get_model_path():
    """Get the path to the pre-trained model weights"""
    weights_path = os.path.join(MODEL_DIR, 'mini_xception_weights.h5')
    if not os.path.exists(weights_path):
        if not ensure_model_downloaded():
            return None
    return weights_path
