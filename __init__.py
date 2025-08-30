"""
PaksaTalker - Advanced Talking Head Generation Framework

Integrates:
- SadTalker for face animation
- Wav2Lip for lip-sync
- Awesome Gesture Generation for body movements
- Qwen for natural language understanding
"""

__version__ = "0.1.0"

# Import core components
from .config import PaksaConfig
from .core import PaksaTalker
from .integrations import (
    SadTalkerIntegration,
    Wav2LipIntegration,
    GestureGenerator,
    QwenIntegration
)

__all__ = [
    'PaksaConfig',
    'PaksaTalker',
    'SadTalkerIntegration',
    'Wav2LipIntegration',
    'GestureGenerator',
    'QwenIntegration'
]
