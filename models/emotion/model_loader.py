import os
import requests
from tqdm import tqdm

# gdown is optional; we handle absence gracefully
try:
    import gdown  # type: ignore
except Exception:  # pragma: no cover
    gdown = None  # type: ignore

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
    """Ensure the emotion recognition (mini_XCEPTION) weights are downloaded."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                if 'drive.google.com' in url:
                    if gdown is None:
                        raise RuntimeError("gdown not available to fetch Google Drive file")
                    gdown.download(url, filepath, quiet=False)  # type: ignore
                else:
                    download_file(url, filepath)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return False
    return True

def get_model_path():
    """Get the path to the pre-trained mini_XCEPTION model weights."""
    weights_path = os.path.join(MODEL_DIR, 'mini_xception_weights.h5')
    if not os.path.exists(weights_path):
        if not ensure_model_downloaded():
            return None
    return weights_path


# ------------------
# EMAGE helpers
# ------------------

def _emage_root() -> str:
    """Resolve EMAGE repo root via env override or default ./EMAGE."""
    override = os.environ.get('PAKSA_EMAGE_ROOT') or os.environ.get('EMAGE_ROOT')
    return override if override else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'EMAGE')


def ensure_emage_weights() -> str | None:
    """Ensure EMAGE checkpoint emage_best.pth is present. Downloads from Hugging Face if needed.

    Returns the local checkpoint path, or None if it cannot be ensured.
    """
    try:
        emage_root = _emage_root()
        ck_dir = os.path.join(emage_root, 'checkpoints')
        os.makedirs(ck_dir, exist_ok=True)
        ck_path = os.path.join(ck_dir, 'emage_best.pth')
        if os.path.exists(ck_path):
            return ck_path
        # Download via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:
            print(f"huggingface_hub not available to fetch EMAGE weights: {e}")
            return None
        try:
            downloaded = hf_hub_download(
                repo_id="PantoMatrix/EMAGE",
                filename="emage_best.pth",
                cache_dir=ck_dir,
            )
            # Move into checkpoints (hf_hub_download may store in cache dir)
            if downloaded != ck_path:
                import shutil
                shutil.copyfile(downloaded, ck_path)
            return ck_path if os.path.exists(ck_path) else None
        except Exception as e:
            print(f"Failed to download EMAGE weights: {e}")
            return None
    except Exception as e:
        print(f"Error ensuring EMAGE weights: {e}")
        return None


# ------------------
# OpenSeeFace helpers
# ------------------

_OSF_REQUIRED = [
    'lm_model0_opt.onnx', 'lm_model1_opt.onnx', 'lm_model2_opt.onnx', 'lm_model3_opt.onnx',
    'lm_model4_opt.onnx', 'lm_modelT_opt.onnx', 'lm_modelU_opt.onnx', 'lm_modelV_opt.onnx',
    'mnv3_detection_opt.onnx', 'retinaface_640x640_opt.onnx', 'priorbox_640x640.json', 'benchmark.bin'
]


def _osf_root() -> str:
    override = os.environ.get('PAKSA_OSF_ROOT') or os.environ.get('OPENSEEFACE_ROOT')
    if override:
        return override
    # default: project_root/OpenSeeFace
    proj = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(proj, 'OpenSeeFace')


def ensure_openseeface_models() -> str | None:
    """Ensure OpenSeeFace models directory exists with required files.

    Returns the absolute path to the models directory or None if unavailable.
    """
    try:
        root = _osf_root()
        models_dir = os.path.join(root, 'models')
        os.makedirs(models_dir, exist_ok=True)

        def _ok() -> bool:
            return all(os.path.exists(os.path.join(models_dir, fn)) for fn in _OSF_REQUIRED)

        if _ok():
            return models_dir

        # Attempt to fetch from GitHub as a single zip of the repo and extract /models
        zip_url = 'https://github.com/emilianavt/OpenSeeFace/archive/refs/heads/master.zip'
        try:
            import io, zipfile
            print('Downloading OpenSeeFace repository (to extract models)...')
            r = requests.get(zip_url, timeout=60)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            # Find paths under OpenSeeFace-<branch>/models/
            prefix = None
            for name in z.namelist():
                if name.endswith('/models/') and name.count('/') >= 2:
                    prefix = name.split('/models/')[0] + '/models/'
                    break
            if prefix is None:
                # fallback: look for any models dir
                for name in z.namelist():
                    if name.endswith('/models/'):
                        prefix = name
                        break
            if prefix is None:
                print('Could not locate models directory in downloaded OpenSeeFace archive')
                return models_dir if _ok() else None
            # Extract only files from that models dir
            for member in z.namelist():
                if member.startswith(prefix) and not member.endswith('/'):
                    rel = member[len(prefix):]
                    dest = os.path.join(models_dir, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with z.open(member) as src, open(dest, 'wb') as dst:
                        dst.write(src.read())
        except Exception as e:
            print(f"Failed to download OpenSeeFace models: {e}")
            # As a fallback, just return current models_dir if anything exists
            return models_dir if _ok() else None

        return models_dir if _ok() else None
    except Exception as e:
        print(f"Error ensuring OpenSeeFace models: {e}")
        return None
