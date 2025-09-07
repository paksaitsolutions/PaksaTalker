"""
Model loader utilities
"""
import os
from pathlib import Path

def ensure_emage_weights():
    """Ensure EMAGE weights are available"""
    emage_root = os.getenv('PAKSA_EMAGE_ROOT', 'd:/PaksaTalker/SadTalker/EMAGE')
    weights_path = Path(emage_root) / "checkpoints" / "emage_best.pth"
    return weights_path.exists()