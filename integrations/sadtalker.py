"""SadTalker integration for PaksaTalker"""
import os
import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path
import cv2

from .base import BaseIntegration
from config import config

# Import SadTalker with error handling
try:
    from src.gradio_demo import SadTalker as _SadTalker
    SADTALKER_AVAILABLE = True
except ImportError:
    SADTALKER_AVAILABLE = False
    _SadTalker = object

class SadTalkerIntegration(BaseIntegration):
    """Integration with SadTalker for face animation"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """Initialize SadTalker integration
        
        Args:
            model_path: Path to SadTalker model files
            device: Device to run the model on
        """
        super().__init__(device)
        self.model_path = model_path or config.get('models.sadtalker.model_path')
        self.model = None
        self.initialized = False
    
    def load_model(self, **kwargs) -> None:
        """Load the SadTalker model"""
        if not SADTALKER_AVAILABLE:
            raise ImportError("SadTalker is not available. Please install the required dependencies.")
        
        if not self.initialized:
            try:
                self.model = _SadTalker(
                    checkpoint_path=os.path.join(self.model_path, 'checkpoints'),
                    config_path=os.path.join(self.model_path, 'configs'),
                    lazy_load=True,
                    device=self.device
                )
                self.initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to load SadTalker: {e}")
    
    def generate(
        self,
        source_image: Union[str, np.ndarray],
        audio_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate talking head video
        
        Args:
            source_image: Path to source image or numpy array
            audio_path: Path to input audio file
            output_path: Path to save output video
            **kwargs: Additional parameters for generation
            
        Returns:
            Path to generated video file
        """
        if not self.initialized:
            self.load_model()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if output_path else config['paths.output']
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(output_dir, 'output.mp4')
        
        try:
            # Convert numpy array to temporary file if needed
            if isinstance(source_image, np.ndarray):
                temp_path = os.path.join(config['paths.temp'], 'temp_image.png')
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                cv2.imwrite(temp_path, cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
                source_image = temp_path
            
            # Call SadTalker's test method
            video_path = self.model.test(
                source_image=source_image,
                driven_audio=audio_path,
                result_dir=output_dir,
                **{
                    'preprocess': 'full',
                    'still': False,
                    'enhancer': 'gfpgan',
                    'expression_scale': 1.0,
                    'input_yaw': None,
                    'input_pitch': None,
                    'input_roll': None,
                    **kwargs
                }
            )
            
            # Move to final output path if needed
            if video_path and video_path != output_path:
                os.replace(video_path, output_path)
                return output_path
            
            return video_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate video: {e}")
    
    def is_loaded(self) -> bool:
        """Check if SadTalker is loaded"""
        return self.initialized and self.model is not None
    
    def unload(self) -> None:
        """Unload SadTalker and free resources"""
        if hasattr(self, 'model') and self.model is not None:
            # Cleanup any resources if needed
            pass
        super().unload()
    
    def __del__(self):
        """Cleanup on object deletion"""
        self.unload()
