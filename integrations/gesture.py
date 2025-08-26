"""Gesture Generator integration for PaksaTalker"""
import os
import torch
import numpy as np
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import cv2

from .base import BaseIntegration
from ...config import config

# Import Gesture Generator with error handling
try:
    from awesome_gesture_generation import GestureGenerator as _GestureGenerator
    GESTURE_GEN_AVAILABLE = True
except ImportError:
    GESTURE_GEN_AVAILABLE = False
    _GestureGenerator = object

class GestureGenerator(BaseIntegration):
    """Integration with Gesture Generation for body movements"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """Initialize Gesture Generator integration
        
        Args:
            model_path: Path to gesture generator model files
            device: Device to run the model on
        """
        super().__init__(device)
        self.model_path = model_path or config.get('models.gesture.model_path')
        self.model = None
        self.initialized = False
    
    def load_model(self, **kwargs) -> None:
        """Load the Gesture Generator model"""
        if not GESTURE_GEN_AVAILABLE:
            raise ImportError("Gesture Generator is not available. Please install the required dependencies.")
        
        if not self.initialized:
            try:
                self.model = _GestureGenerator(
                    model_path=self.model_path,
                    device=self.device,
                    **{
                        'style': config.get('models.gesture.style', 'casual'),
                        'intensity': config.get('models.gesture.intensity', 0.7),
                        **kwargs
                    }
                )
                self.initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to load Gesture Generator: {e}")
    
    def generate_gestures(
        self,
        text: str,
        duration: Optional[float] = None,
        style: Optional[str] = None,
        intensity: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate gesture parameters from text
        
        Args:
            text: Input text to generate gestures for
            duration: Duration of the generated gestures in seconds
            style: Gesture style (e.g., 'casual', 'excited')
            intensity: Intensity of the gestures (0.0 to 1.0)
            **kwargs: Additional parameters for generation
            
        Returns:
            Numpy array of gesture parameters
        """
        if not self.initialized:
            self.load_model()
        
        try:
            # Generate gestures using the model
            gestures = self.model.generate(
                text=text,
                duration=duration,
                style=style or config.get('models.gesture.style', 'casual'),
                intensity=intensity or config.get('models.gesture.intensity', 0.7),
                **kwargs
            )
            
            return gestures
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate gestures: {e}")
    
    def add_gestures(
        self,
        video_path: str,
        text: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Add gestures to a video based on text
        
        Args:
            video_path: Path to input video file
            text: Text to generate gestures from
            output_path: Path to save output video with gestures
            **kwargs: Additional parameters for generation
            
        Returns:
            Path to video with added gestures
        """
        if not self.initialized:
            self.load_model()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if output_path else config['paths.output']
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(output_dir, 'gesture_output.mp4')
        
        try:
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if frame_count and fps > 0 else 5.0  # Default to 5 seconds
            cap.release()
            
            # Generate gestures
            gestures = self.generate_gestures(
                text=text,
                duration=duration,
                **kwargs
            )
            
            # TODO: Apply gestures to video
            # This is a placeholder - actual implementation would depend on the gesture model
            # and how it interfaces with the video
            
            # For now, just return the input path
            # In a real implementation, you would process the video and save the result
            return video_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to add gestures to video: {e}")
    
    def is_loaded(self) -> bool:
        """Check if Gesture Generator is loaded"""
        return self.initialized and self.model is not None
    
    def unload(self) -> None:
        """Unload Gesture Generator and free resources"""
        if hasattr(self, 'model') and self.model is not None:
            # Cleanup any resources if needed
            pass
        super().unload()
    
    def __del__(self):
        """Cleanup on object deletion"""
        self.unload()
