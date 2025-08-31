"""
Lightweight placeholder emotion recognizer.

This module satisfies optional imports in the server. It does not perform
real inference unless you replace the predict method with a real model.
"""
from typing import Dict


class EmotionRecognizer:
    """Stub EmotionRecognizer that always returns neutral."""

    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path

    def predict(self, image_path: str) -> Dict[str, float]:
        # Always return neutral as a safe default
        return {
            "neutral": 1.0,
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprise": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
        }


def predict_emotions_from_image(image_path: str) -> Dict[str, float]:
    """Convenience function for callers expecting a module-level API."""
    return EmotionRecognizer().predict(image_path)

