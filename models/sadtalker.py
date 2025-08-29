'''SadTalker model implementation for PaksaTalker.'''
import os
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import cv2
from dataclasses import dataclass, field
from enum import Enum, auto

from .base import BaseModel
from config import config
from werkzeug.utils import secure_filename

class SadTalkerModel(BaseModel):
    """SadTalker model for generating talking head videos with emotion control."""
    
    # Supported emotion types and their corresponding expression codes
    EMOTION_MAP = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'surprised': 4,
        'disgusted': 5,
        'fearful': 6
    }
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the SadTalker model with emotion control.
        
        Args:
            device: Device to run the model on ('cuda', 'mps', 'cpu').
        """
        super().__init__(device)
        self.model = None
        self.initialized = False
        
        # Emotion control state
        self.current_emotion = 'neutral'
        self.target_emotion = 'neutral'
        self.emotion_intensity = 1.0
        self.emotion_blend = {}
        self.transition_speed = 0.1  # Default transition speed (0-1)
        self._init_emotion_weights()
        
        # For smooth transitions
        self._transition_start_time = None
        self._transition_duration = 0.5  # seconds
        self._start_blend = None
        self._target_blend = None
        
    def _init_emotion_weights(self) -> None:
        """Initialize emotion blending weights."""
        self.emotion_blend = {emotion: 0.0 for emotion in self.EMOTION_MAP}
        self.emotion_blend['neutral'] = 1.0  # Start with neutral expression
    
    def set_emotion(self, emotion: str, intensity: float = 1.0) -> None:
        """Set the primary emotion and its intensity.
        
        Args:
            emotion: Name of the emotion (must be in EMOTION_MAP).
            intensity: Intensity of the emotion (0.0 to 1.0).
        """
        if emotion not in self.EMOTION_MAP:
            raise ValueError(f"Unsupported emotion: {emotion}. Must be one of {list(self.EMOTION_MAP.keys())}")
            
        intensity = max(0.0, min(1.0, intensity))  # Clamp between 0 and 1
        self.current_emotion = emotion
        self.emotion_intensity = intensity
        self._update_emotion_blend()
    
    def blend_emotions(self, emotion_weights: Dict[str, float]) -> None:
        """Blend multiple emotions with custom weights.
        
        Args:
            emotion_weights: Dictionary mapping emotions to their weights (0.0 to 1.0).
        """
        # Normalize weights to sum to 1.0
        total = sum(emotion_weights.values())
        if total <= 0:
            raise ValueError("Sum of emotion weights must be greater than 0")
            
        for emotion in self.EMOTION_MAP:
            self.emotion_blend[emotion] = emotion_weights.get(emotion, 0.0) / total
    
    def _update_emotion_blend(self) -> None:
        """Update emotion blend based on current emotion and intensity."""
        # Reset all weights
        for emotion in self.EMOTION_MAP:
            self.emotion_blend[emotion] = 0.0
        
        # Set the current emotion with intensity
        self.emotion_blend[self.current_emotion] = self.emotion_intensity
        
        # If not full intensity, blend with neutral
        if self.emotion_intensity < 1.0:
            self.emotion_blend['neutral'] = 1.0 - self.emotion_intensity
    
    def start_emotion_transition(self, target_emotion: str, duration: float = 0.5) -> None:
        """Start a smooth transition to a new emotion.
        
        Args:
            target_emotion: The emotion to transition to.
            duration: Duration of the transition in seconds.
        """
        if target_emotion not in self.EMOTION_MAP:
            raise ValueError(f"Unsupported emotion: {target_emotion}")
            
        self.target_emotion = target_emotion
        self._transition_duration = max(0.1, duration)
        self._transition_start_time = time.time()
        self._start_blend = self.emotion_blend.copy()
        
        # Initialize target blend
        self._target_blend = {emotion: 0.0 for emotion in self.EMOTION_MAP}
        self._target_blend[target_emotion] = self.emotion_intensity
        if self.emotion_intensity < 1.0:
            self._target_blend['neutral'] = 1.0 - self.emotion_intensity
    
    def update_emotion_transition(self) -> bool:
        """Update the current emotion blend based on transition progress.
        
        Returns:
            bool: True if transition is still in progress, False if complete.
        """
        if self._transition_start_time is None or self._start_blend is None or self._target_blend is None:
            return False
            
        elapsed = time.time() - self._transition_start_time
        t = min(1.0, elapsed / self._transition_duration)
        
        # Apply smooth step for more natural transitions
        t = t * t * (3 - 2 * t)
        
        # Interpolate between start and target blends
        for emotion in self.EMOTION_MAP:
            self.emotion_blend[emotion] = (
                self._start_blend[emotion] * (1 - t) + 
                self._target_blend[emotion] * t
            )
        
        # Update current emotion if transition is complete
        if t >= 1.0:
            self.current_emotion = self.target_emotion
            self._transition_start_time = None
            self._start_blend = None
            self._target_blend = None
            return False
            
        return True
    
    def get_expression_coefficients(self, audio_features: torch.Tensor, delta_time: float = 0.0) -> torch.Tensor:
        """Generate expression coefficients with emotion control and transitions.
        
        Args:
            audio_features: Raw audio features from the audio processing pipeline.
            delta_time: Time since last update (for smooth transitions).
            
        Returns:
            torch.Tensor: Expression coefficients with emotion modulation.
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        # Update emotion transition if active
        if self._transition_start_time is not None:
            self.update_emotion_transition()
            
        # Get base expression from audio
        base_exp = self.model['audio2exp'](audio_features)
        
        # Get current emotion blend
        current_blend = self._get_current_emotion_blend()
        
        # Apply emotion modulation
        emotion_weights = torch.tensor([
            current_blend[emotion] 
            for emotion in sorted(self.EMOTION_MAP.keys())
        ], device=self.device)
        
        # Blend base expression with emotion weights
        # Using a more sophisticated blending function that preserves speech dynamics
        emotion_scale = 0.8  # How much emotion affects the expression (0-1)
        modulated_exp = base_exp * (1.0 + (emotion_weights.unsqueeze(0) - 0.5) * emotion_scale * 2)
        
        return modulated_exp
    
    def _get_current_emotion_blend(self) -> Dict[str, float]:
        """Get the current emotion blend, ensuring it's normalized."""
        total = sum(self.emotion_blend.values())
        if total <= 0:
            return {emotion: 1.0 / len(self.EMOTION_MAP) for emotion in self.EMOTION_MAP}
        return {emotion: weight / total for emotion, weight in self.emotion_blend.items()}
    
    def set_emotion_intensity(self, intensity: float) -> None:
        """Set the intensity of the current emotion.
        
        Args:
            intensity: New intensity value (0.0 to 1.0).
        """
        self.emotion_intensity = max(0.0, min(1.0, intensity))
        self._update_emotion_blend()
    
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """Load the SadTalker model.
        
        Args:
            model_path: Path to the model directory.
            **kwargs: Additional arguments for model loading.
        """
        if self.initialized:
            return
            
        try:
            # Import SadTalker components
            from src.facerender.animate import AnimateFromCoeff
            from src.free_view import get_image_crop, get_semantic_radius
            from src.generate_batch import get_data
            from src.generate_facerender_batch import get_facerender_data
            from src.utils.init_path import init_path
            from src.utils.preprocessing import get_pose_params, get_img_coordinates
            from src.utils.videoio import save_video_with_watermark
            
            # Initialize paths
            model_path = model_path or config.get('models.sadtalker.model_path', 'models/sadtalker')
            checkpoint_dir = os.path.join(model_path, 'checkpoints')
            
            # Set up paths in the config
            config['sadtalker'] = {
                'checkpoint_dir': checkpoint_dir,
                'mapping_checkpoint': os.path.join(checkpoint_dir, 'mapping_00109-model.pth.tar'),
                'mapping_emo_checkpoint': os.path.join(checkpoint_dir, 'mapping_00229-model.pth.tar'),
                'facerender_checkpoint': os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar'),
                'audio2pose_checkpoint': os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth'),
                'audio2exp_checkpoint': os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth'),
                'still': False,
                'preprocess': 'full',
                'expression_scale': 1.0,
                'input_yaw': None,
                'input_pitch': None,
                'input_roll': None,
                'enhancer': 'gfpgan',
                'background_enhancer': None,
                'pitch_shift': 0,
                'still_threshold': 0.5,
                'result_dir': config['paths.output'],
                'temp_dir': config['paths.temp'],
            }
            
            # Initialize the model
            self.model = {
                'animate_from_coeff': AnimateFromCoeff(),
                'get_image_crop': get_image_crop,
                'get_semantic_radius': get_semantic_radius,
                'get_data': get_data,
                'get_facerender_data': get_facerender_data,
                'init_path': init_path,
                'get_pose_params': get_pose_params,
                'get_img_coordinates': get_img_coordinates,
                'save_video_with_watermark': save_video_with_watermark,
            }
            
            # Load checkpoints
            self._load_checkpoints()
            
            # Move to device
            self.to(self.device)
            self.eval()
            
            self.initialized = True
            
        except Exception as e:
            self.initialized = False
            raise RuntimeError(f"Failed to load SadTalker model: {e}")
    
    def _load_checkpoints(self) -> None:
        """Load all required checkpoints."""
        checkpoint_dir = config['sadtalker']['checkpoint_dir']
        
        # Load mapping network
        mapping_ckpt = torch.load(
            config['sadtalker']['mapping_checkpoint'],
            map_location=torch.device(self.device)
        )
        self.model['mapping_net'] = mapping_ckpt['net_g_ema']
        
        # Load audio2pose model
        audio2pose_ckpt = torch.load(
            config['sadtalker']['audio2pose_checkpoint'],
            map_location=torch.device(self.device)
        )
        self.model['audio2pose'] = audio2pose_ckpt['net_g']
        
        # Load audio2exp model
        audio2exp_ckpt = torch.load(
            config['sadtalker']['audio2exp_checkpoint'],
            map_location=torch.device(self.device)
        )
        self.model['audio2exp'] = audio2exp_ckpt['net_g']
        
        # Load face renderer
        facerender_ckpt = torch.load(
            config['sadtalker']['facerender_checkpoint'],
            map_location=torch.device(self.device)
        )
        self.model['facerender'] = facerender_ckpt['net_g_ema']
    
    def generate(
        self,
        image_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        style: str = "default",
        resolution: str = "480p",
        **kwargs
    ) -> str:
        """Generate a talking head video.
        
        Args:
            image_path: Path to the source image.
            audio_path: Path to the input audio file.
            output_path: Path to save the output video.
            style: Animation style (e.g., 'default', 'sad', 'happy').
            **kwargs: Additional generation parameters.
            
        Returns:
            Path to the generated video file.
        """
        if not self.initialized:
            self.load_model()
        
        try:
            # Set up output path
            if output_path is None:
                output_dir = config.get('paths.output', 'output')
                os.makedirs(output_dir, exist_ok=True)
                secure_image_name = secure_filename(os.path.basename(image_path))
                output_path = os.path.join(
                    output_dir,
                    f"sadtalker_{secure_image_name.split('.')[0]}.mp4"
                )
            
            # Use the actual SadTalker implementation
            from src.gradio_demo import SadTalker as _SadTalker
            
            sadtalker = _SadTalker(
                checkpoint_path=config.get('sadtalker.checkpoint_dir', 'models/sadtalker/checkpoints'),
                config_path=config.get('sadtalker.config_dir', 'models/sadtalker/configs'),
                device=self.device
            )
            
            # Map resolution to dimensions
            resolution_map = {
                "240p": (320, 240),
                "360p": (480, 360),
                "480p": (640, 480),
                "720p": (1280, 720),
                "1080p": (1920, 1080),
                "1440p": (2560, 1440),
                "4k": (3840, 2160)
            }
            
            width, height = resolution_map.get(resolution, (640, 480))
            
            result_path = sadtalker.test(
                source_image=image_path,
                driven_audio=audio_path,
                result_dir=os.path.dirname(output_path),
                size=height,  # SadTalker uses height for size parameter
                **kwargs
            )
            
            # Move result to final output path if different
            if result_path != output_path and os.path.exists(result_path):
                os.rename(result_path, output_path)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate video: {e}")
    

    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.initialized and self.model is not None
    
    def unload(self) -> None:
        """Unload the model and free up memory."""
        if self.model is not None:
            # Unload all sub-models
            for key in list(self.model.keys()):
                if hasattr(self.model[key], 'to'):
                    self.model[key].to('cpu')
                del self.model[key]
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.initialized = False
