'''Wav2Lip model implementation for PaksaTalker.'''
import os
import torch
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path
import cv2

from .base import BaseModel
from config import config

class Wav2LipModel(BaseModel):
    """Wav2Lip model for lip-syncing videos."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the Wav2Lip model.
        
        Args:
            device: Device to run the model on ('cuda', 'mps', 'cpu').
        """
        super().__init__(device)
        self.model = None
        self.initialized = False
    
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """Load the Wav2Lip model.
        
        Args:
            model_path: Path to the model directory.
            **kwargs: Additional arguments for model loading.
        """
        if self.initialized:
            return
            
        try:
            # Import Wav2Lip components
            from src.wav2lip.face_detection import FaceDetector
            from src.wav2lip.models import Wav2Lip as Wav2LipModel
            
            # Initialize paths
            model_path = model_path or config.get('models.wav2lip.model_path', 'models/wav2lip')
            checkpoint_path = os.path.join(model_path, 'wav2lip.pth')
            
            # Set up paths in the config
            config['wav2lip'] = {
                'checkpoint_path': checkpoint_path,
                'face_detection_weights': os.path.join(model_path, 's3fd-619a316812.pth'),
                'face_detection_arch': 'resnet50',
                'pads': [0, 10, 0, 0],
                'img_size': 96,
                'fps': 25,
                'static': False,
                'result_dir': config['paths.output'],
                'temp_dir': config['paths.temp'],
            }
            
            # Initialize face detector
            face_detector = FaceDetector(
                weights_path=config['wav2lip']['face_detection_weights'],
                arch=config['wav2lip']['face_detection_arch'],
                device=self.device
            )
            
            # Initialize Wav2Lip model
            model = Wav2LipModel()
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.model = {
                'model': model,
                'face_detector': face_detector,
            }
            
            self.initialized = True
            
        except Exception as e:
            self.initialized = False
            raise RuntimeError(f"Failed to load Wav2Lip model: {e}")
    
    def enhance(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Enhance lip-sync in a video.
        
        Args:
            video_path: Path to the input video file.
            audio_path: Path to the input audio file.
            output_path: Path to save the output video.
            **kwargs: Additional enhancement parameters.
            
        Returns:
            Path to the enhanced video file.
        """
        if not self.initialized:
            self.load_model()
        
        try:
            # Set up output path
            if output_path is None:
                output_dir = config.get('paths.output', 'output')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    f"wav2lip_{os.path.basename(video_path)}"
                )
            
            # Use the actual Wav2Lip implementation
            from src.wav2lip.inference import Wav2Lip as _Wav2Lip
            
            wav2lip = _Wav2Lip(
                checkpoint_path=config.get('wav2lip.checkpoint_path', 'models/wav2lip/wav2lip.pth'),
                device=self.device
            )
            
            result_path = wav2lip.inference(
                face=video_path,
                audio=audio_path,
                outfile=output_path,
                **kwargs
            )
            
            return result_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to enhance video: {e}")
    

    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.initialized and self.model is not None
    
    def unload(self) -> None:
        """Unload the model and free up memory."""
        if self.model is not None:
            # Unload the model and face detector
            if 'model' in self.model and hasattr(self.model['model'], 'to'):
                self.model['model'].to('cpu')
            
            if 'face_detector' in self.model and hasattr(self.model['face_detector'], 'to'):
                self.model['face_detector'].to('cpu')
            
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.initialized = False
