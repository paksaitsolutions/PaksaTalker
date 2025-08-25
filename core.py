""Core PaksaTalker implementation"""
import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np
import cv2

from .config import config
from .integrations import (
    SadTalkerIntegration,
    Wav2LipIntegration,
    GestureGenerator,
    QwenIntegration
)

logger = logging.getLogger(__name__)

class PaksaTalker:
    """Main class for PaksaTalker - integrates all components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize PaksaTalker with optional config"""
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        if config_path:
            config._load_config(config_path)
        
        # Setup device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' 
            if torch.backends.mps.is_available() else 'cpu'
        )
        
        # Initialize components
        self.components = {}
        self._initialize_components()
        
        logger.info(f"PaksaTalker initialized on device: {self.device}")
    
    def _setup_logging(self):
        """Configure logging"""
        log_level = config.get('logging.level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.get('logging.file', 'paksatalker.log')),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize all enabled components"""
        # Initialize Qwen first as it might be needed by other components
        if config.get('models.qwen.enabled', False):
            try:
                self.components['qwen'] = QwenIntegration(
                    model_name=config.get('models.qwen.model_name'),
                    cache_dir=config.get('models.qwen.cache_dir'),
                    device=self.device
                )
                logger.info("Qwen integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Qwen: {e}")
        
        # Initialize SadTalker
        if config.get('models.sadtalker.enabled', False):
            try:
                self.components['sadtalker'] = SadTalkerIntegration(
                    model_path=config.get('models.sadtalker.model_path'),
                    device=self.device
                )
                logger.info("SadTalker integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SadTalker: {e}")
        
        # Initialize Wav2Lip
        if config.get('models.wav2lip.enabled', False):
            try:
                self.components['wav2lip'] = Wav2LipIntegration(
                    model_path=config.get('models.wav2lip.model_path'),
                    device=self.device
                )
                logger.info("Wav2Lip integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Wav2Lip: {e}")
        
        # Initialize Gesture Generator
        if config.get('models.gesture.enabled', False):
            try:
                self.components['gesture'] = GestureGenerator(
                    model_path=config.get('models.gesture.model_path'),
                    device=self.device
                )
                logger.info("Gesture Generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gesture Generator: {e}")
    
    def generate(
        self,
        source_image: Union[str, np.ndarray],
        audio_path: str,
        text: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate talking head video from source image and audio
        
        Args:
            source_image: Path to source image or numpy array
            audio_path: Path to input audio file
            text: Optional text for gesture/speech generation
            output_path: Path to save output video
            **kwargs: Additional parameters for generation
            
        Returns:
            Path to generated video file
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if output_path else config['paths.output']
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with Qwen if available
        if 'qwen' in self.components and text:
            try:
                enhanced_prompt = self.components['qwen'].enhance_prompt(text)
                logger.info(f"Enhanced prompt: {enhanced_prompt}")
                # TODO: Use enhanced prompt for better generation
            except Exception as e:
                logger.warning(f"Failed to enhance prompt: {e}")
        
        # Generate talking head animation
        if 'sadtalker' in self.components:
            try:
                # First generate basic talking head
                video_path = self.components['sadtalker'].generate(
                    source_image=source_image,
                    audio_path=audio_path,
                    output_path=output_path,
                    **kwargs
                )
                
                # Enhance with Wav2Lip if available
                if 'wav2lip' in self.components:
                    video_path = self.components['wav2lip'].enhance(
                        video_path=video_path,
                        audio_path=audio_path,
                        output_path=output_path
                    )
                
                # Add gestures if enabled
                if 'gesture' in self.components and text:
                    video_path = self.components['gesture'].add_gestures(
                        video_path=video_path,
                        text=text,
                        output_path=output_path
                    )
                
                return video_path
                
            except Exception as e:
                logger.error(f"Failed to generate video: {e}")
                raise
        else:
            raise RuntimeError("SadTalker component is not available")
    
    def __getattr__(self, name):
        """Delegate attribute access to components"""
        if name in self.components:
            return self.components[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
