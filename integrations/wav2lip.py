"""Wav2Lip integration for PaksaTalker"""
import os
import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path
import cv2

from .base import BaseIntegration
from ...config import config

# Import Wav2Lip with error handling
try:
    from src.wav2lip.inference import Wav2Lip as _Wav2Lip
    WAV2LIP_AVAILABLE = True
except ImportError:
    WAV2LIP_AVAILABLE = False
    _Wav2Lip = object

class Wav2LipIntegration(BaseIntegration):
    """Integration with Wav2Lip for lip-sync enhancement"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """Initialize Wav2Lip integration
        
        Args:
            model_path: Path to Wav2Lip model files
            device: Device to run the model on
        """
        super().__init__(device)
        self.model_path = model_path or config.get('models.wav2lip.model_path')
        self.model = None
        self.initialized = False
    
    def load_model(self, **kwargs) -> None:
        """Load the Wav2Lip model"""
        if not WAV2LIP_AVAILABLE:
            raise ImportError("Wav2Lip is not available. Please install the required dependencies.")
        
        if not self.initialized:
            try:
                checkpoint_path = os.path.join(self.model_path, 'wav2lip.pth')
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Wav2Lip model not found at {checkpoint_path}")
                
                self.model = _Wav2Lip(
                    checkpoint_path=checkpoint_path,
                    device=self.device,
                    **{
                        'pads': config.get('models.wav2lip.pads', [0, 10, 0, 0]),
                        'resize_factor': config.get('models.wav2lip.resize_factor', 1),
                        **kwargs
                    }
                )
                self.initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to load Wav2Lip: {e}")
    
    def enhance(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Enhance lip-sync in a video
        
        Args:
            video_path: Path to input video file
            audio_path: Path to input audio file
            output_path: Path to save output video
            **kwargs: Additional parameters for enhancement
            
        Returns:
            Path to enhanced video file
        """
        if not self.initialized:
            self.load_model()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if output_path else config['paths.output']
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(output_dir, 'enhanced_output.mp4')
        
        try:
            # Call Wav2Lip's inference method
            enhanced_video = self.model.inference(
                face=video_path,
                audio=audio_path,
                outfile=output_path,
                **kwargs
            )
            
            return output_path if os.path.exists(output_path) else enhanced_video
            
        except Exception as e:
            raise RuntimeError(f"Failed to enhance video: {e}")
    
    def is_loaded(self) -> bool:
        """Check if Wav2Lip is loaded"""
        return self.initialized and self.model is not None
    
    def unload(self) -> None:
        """Unload Wav2Lip and free resources"""
        if hasattr(self, 'model') and self.model is not None:
            # Cleanup any resources if needed
            pass
        super().unload()
    
    def __del__(self):
        """Cleanup on object deletion"""
        self.unload()
