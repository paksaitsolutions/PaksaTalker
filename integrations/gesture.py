"""Gesture Generator integration for PaksaTalker"""
import os
import torch
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path
import cv2
import time
from enum import Enum
import random

from integrations.base import BaseIntegration
from config import config
from models.emotion_gestures import EmotionGestureMapper, EmotionType, GestureType

# Import Gesture Generator with error handling
try:
    from awesome_gesture_generation import GestureGenerator as _GestureGenerator
    GESTURE_GEN_AVAILABLE = True
except ImportError:
    GESTURE_GEN_AVAILABLE = False
    _GestureGenerator = object

# Emotion mapping from string to EmotionType
EMOTION_MAP = {
    'neutral': EmotionType.NEUTRAL,
    'happy': EmotionType.HAPPY,
    'sad': EmotionType.SAD,
    'angry': EmotionType.ANGRY,
    'surprised': EmotionType.SURPRISED,
    'disgusted': EmotionType.DISGUSTED,
    'fearful': EmotionType.FEARFUL
}

class GestureGenerator(BaseIntegration):
    """Integration with Gesture Generation for body movements with emotion support"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """Initialize Gesture Generator integration with emotion support
        
        Args:
            model_path: Path to gesture generator model files
            device: Device to run the model on
        """
        super().__init__(device)
        
        # Initialize emotion gesture mapper
        self.emotion_mapper = EmotionGestureMapper()
        self.current_emotion = EmotionType.NEUTRAL
        self.emotion_intensity = 0.5
        self.active_gestures = []
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5  # Minimum time between gestures in seconds
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
    
    def set_emotion(self, emotion: Union[str, EmotionType], intensity: float = 0.5):
        """Set the current emotion for gesture generation
        
        Args:
            emotion: Emotion as string or EmotionType enum
            intensity: Intensity of the emotion (0.0 to 1.0)
        """
        if isinstance(emotion, str):
            emotion = emotion.lower()
            if emotion in EMOTION_MAP:
                self.current_emotion = EMOTION_MAP[emotion]
            else:
                self.current_emotion = EmotionType.NEUTRAL
        else:
            self.current_emotion = emotion
            
        self.emotion_intensity = max(0.0, min(1.0, intensity))
    
    def generate_gestures(self, audio: Optional[np.ndarray] = None, 
                         text: Optional[str] = None,
                         duration: float = 5.0,
                         emotion: Optional[Union[str, EmotionType]] = None,
                         intensity: Optional[float] = None) -> np.ndarray:
        """Generate gesture sequence with emotion support
        
        Args:
            audio: Audio signal (n_samples,) for audio-driven gestures
            text: Text for text-driven gestures
            duration: Duration of gesture sequence in seconds
            emotion: Optional emotion to override current emotion
            intensity: Optional intensity to override current intensity
            
        Returns:
            np.ndarray: Sequence of gesture parameters
        """
        if not GESTURE_GEN_AVAILABLE:
            return self._generate_emotion_gestures(duration, emotion, intensity)
        
        # Update emotion if provided
        if emotion is not None:
            self.set_emotion(emotion, intensity or self.emotion_intensity)
        
        # Get emotion-based gestures
        gestures = self.emotion_mapper.get_gesture_sequence(
            self.current_emotion,
            duration=duration,
            intensity=self.emotion_intensity
        )
        
        # Convert gestures to motion parameters
        motion_params = []
        for gesture in gestures:
            # Map gesture type to motion parameters
            params = self._gesture_to_motion_params(gesture)
            motion_params.append(params)
        
        # Combine motion parameters into a single array
        if motion_params:
            return np.concatenate(motion_params, axis=0)
        return np.zeros((int(duration * 30), 64))  # Default to 30fps, 64-dim params
    
    def _gesture_to_motion_params(self, gesture):
        """Convert a gesture to motion parameters"""
        # This is a simplified example - in practice, you'd map gesture types
        # to actual motion parameters for your animation system
        duration_frames = int(gesture.duration * 30)  # 30 FPS
        params = np.zeros((duration_frames, 64))  # Example: 64-dim motion parameters
        
        # Apply intensity and speed to parameters
        for i in range(duration_frames):
            # Example: Modify parameters based on gesture type and intensity
            if gesture.gesture_type == GestureType.NOD:
                # Nodding motion
                t = i / duration_frames
                params[i, 0] = np.sin(t * np.pi) * 0.5 * gesture.intensity
            elif gesture.gesture_type == GestureType.SHAKE_HEAD:
                # Head shake motion
                t = i / duration_frames
                params[i, 1] = np.sin(t * 4 * np.pi) * 0.3 * gesture.intensity
            # Add more gesture mappings as needed
            
        return params
    
    def _generate_emotion_gestures(self, duration: float, 
                                 emotion: Optional[Union[str, EmotionType]] = None,
                                 intensity: Optional[float] = None) -> np.ndarray:
        """Generate gestures based on emotion when gesture generation is not available"""
        # Update emotion if provided
        if emotion is not None:
            self.set_emotion(emotion, intensity or self.emotion_intensity)
        
        # Get emotion-based gestures
        gestures = self.emotion_mapper.get_gesture_sequence(
            self.current_emotion,
            duration=duration,
            intensity=self.emotion_intensity
        )
        
        # Generate simple motion parameters
        fps = 30
        total_frames = int(duration * fps)
        params = np.zeros((total_frames, 64))  # 64-dim motion parameters
        
        # Apply basic motion based on gestures
        current_frame = 0
        for gesture in gestures:
            frames = int(gesture.duration * fps)
            end_frame = min(current_frame + frames, total_frames)
            
            # Apply gesture effect to parameters
            for i in range(current_frame, end_frame):
                t = (i - current_frame) / frames
                # Simple example: map gesture type to parameter changes
                if gesture.gesture_type in [GestureType.NOD, GestureType.SHAKE_HEAD]:
                    # Head motion
                    params[i, 0] = np.sin(t * np.pi * 2) * 0.3 * gesture.intensity
                elif gesture.gesture_type in [GestureType.WAVE, GestureType.POINT]:
                    # Arm motion
                    params[i, 5] = np.sin(t * np.pi) * 0.4 * gesture.intensity
                # Add more mappings as needed
            
            current_frame = end_frame
            if current_frame >= total_frames:
                break
                
        return params
    
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
