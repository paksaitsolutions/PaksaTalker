"""
Integration modules for PaksaTalker

This package contains all the integration modules for different models
and services used by PaksaTalker.
"""

from .base import BaseIntegration
from .sadtalker import SadTalkerIntegration
from .wav2lip import Wav2LipIntegration
from .gesture import GestureGenerator
from .qwen import QwenIntegration

__all__ = [
    'BaseIntegration',
    'SadTalkerIntegration',
    'Wav2LipIntegration',
    'GestureGenerator',
    'QwenIntegration'
]
