"""Core PaksaTalker implementation"""
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
from .models.speaker import SpeakerManager
from .models.speaker_adaptation import adapt_speaker_model, SpeakerAdapter
from .models.animation_styles import AnimationStyleManager, AnimationStyle
from .models.voice_cloning import VoiceCloningManager

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
        
        # Setup adaptation settings
        self.adaptation_config = {
            'output_dir': config.get('adaptation.output_dir', 'output/adapted_models'),
            'epochs': config.get('adaptation.epochs', 10),
            'batch_size': config.get('adaptation.batch_size', 4),
            'learning_rate': config.get('adaptation.learning_rate', 1e-4),
            'num_workers': config.get('adaptation.num_workers', 4)
        }
        
        # Initialize components
        self.components = {}
        self._initialize_components()
        
        # Initialize speaker manager
        self.speaker_manager = SpeakerManager(
            storage_dir=str(Path(config.get('storage.speakers_dir', 'data/speakers')))
        )
        
        # Initialize animation style manager
        self.style_manager = AnimationStyleManager(
            storage_dir=str(Path(config.get('storage.animation_styles_dir', 'data/animation_styles')))
        )
        
        # Initialize voice cloning manager
        self.voice_manager = VoiceCloningManager(
            storage_dir=str(Path(config.get('storage.voices_dir', 'data/voices'))),
            device=self.device
        )
        
        # Initialize default styles if none exist
        self._initialize_default_styles()
        
        logger.info(f"PaksaTalker initialized on device: {self.device}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    def _initialize_default_styles(self):
        """Initialize default animation styles if none exist."""
        # Only create default styles if no styles exist
        if not self.style_manager.get_global_styles():
            logger.info("Initializing default animation styles...")
            
            # Default style
            self.style_manager.create_style(
                name="Default",
                description="Balanced animation style with natural movements",
                parameters={
                    "intensity": 1.0,
                    "smoothness": 0.8,
                    "expressiveness": 0.7,
                    "motion_scale": 1.0,
                    "head_movement": 0.5,
                    "eye_blink_rate": 0.5,
                    "lip_sync_strength": 0.9
                },
                is_global=True
            )
            
            # Expressive style
            self.style_manager.create_style(
                name="Expressive",
                description="More exaggerated facial expressions and movements",
                parameters={
                    "intensity": 1.3,
                    "smoothness": 0.7,
                    "expressiveness": 1.0,
                    "motion_scale": 1.2,
                    "head_movement": 0.8,
                    "eye_blink_rate": 0.7,
                    "lip_sync_strength": 1.1
                },
                is_global=True
            )
            
            # Subtle style
            self.style_manager.create_style(
                name="Subtle",
                description="Minimal, more natural movements",
                parameters={
                    "intensity": 0.7,
                    "smoothness": 0.9,
                    "expressiveness": 0.5,
                    "motion_scale": 0.8,
                    "head_movement": 0.3,
                    "eye_blink_rate": 0.3,
                    "lip_sync_strength": 0.8
                },
                is_global=True
            )
            
            logger.info("Default animation styles initialized")
    
    def adapt_to_speaker(self,
                        audio_dir: str,
                        speaker_id: str,
                        model_type: str = 'sadtalker',
                        **kwargs) -> str:
        """
        Fine-tune a model on a new speaker's data.
        
        Args:
            audio_dir: Directory containing the speaker's audio files
            speaker_id: Unique identifier for the speaker
            model_type: Type of model to adapt ('sadtalker', 'wav2lip', etc.)
            **kwargs: Additional arguments for adaptation
            
        Returns:
            Path to the adapted model
        """
        if model_type not in self.components:
            raise ValueError(f"Model type {model_type} not found in components")
            
        model = self.components[model_type].model
        
        # Update adaptation config with any provided kwargs
        adaptation_params = self.adaptation_config.copy()
        adaptation_params.update(kwargs)
        
        # Create output directory for this speaker
        speaker_output_dir = os.path.join(adaptation_params['output_dir'], speaker_id)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        # Run adaptation
        model_path = adapt_speaker_model(
            model=model,
            audio_dir=audio_dir,
            speaker_id=speaker_id,
            output_dir=speaker_output_dir,
            **{k: v for k, v in adaptation_params.items() 
               if k in ['epochs', 'batch_size', 'learning_rate', 'num_workers']}
        )
        
        # Update the model with the adapted weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Successfully adapted {model_type} model to speaker {speaker_id}")
        return model_path
        
    def register_speaker(self, 
                        audio_path: str, 
                        speaker_id: str, 
                        metadata: Optional[Dict] = None,
                        adapt_models: bool = False,
                        audio_dir: Optional[str] = None) -> bool:
        """Register a new speaker.
        
        Args:
            audio_path: Path to audio file for the speaker
            speaker_id: Unique identifier for the speaker
            metadata: Optional metadata about the speaker
            adapt_models: Whether to adapt models to this speaker
            audio_dir: Directory containing additional audio files for adaptation
            
        Returns:
            bool: True if registration was successful
        """
        # Register the speaker
        success = self.speaker_manager.register_speaker(audio_path, speaker_id, metadata)
        
        # Adapt models if requested
        if success and adapt_models:
            if not audio_dir:
                audio_dir = os.path.dirname(audio_path)
                
            try:
                logger.info(f"Starting model adaptation for speaker {speaker_id}")
                if 'sadtalker' in self.components:
                    self.adapt_to_speaker(
                        audio_dir=audio_dir,
                        speaker_id=speaker_id,
                        model_type='sadtalker'
                    )
                # Add other model types as needed
                
            except Exception as e:
                logger.error(f"Error during model adaptation: {e}")
                # Don't fail registration if adaptation fails
        
        return success
    
    def identify_speaker(self, audio_path: str, threshold: float = 0.7) -> Optional[str]:
        """Identify speaker from audio.
        
        Args:
            audio_path: Path to audio file
            threshold: Similarity threshold for positive identification
            
        Returns:
            Optional[str]: Speaker ID if identified, None otherwise
        """
        return self.speaker_manager.identify_speaker(audio_path, threshold)
    
    def get_speaker_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        """Get embedding for a registered speaker."""
        return self.speaker_manager.get_speaker_embedding(speaker_id)
    
    def generate_video(
        self,
        source_image: str,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
        output_dir: str = "output",
        speaker_id: Optional[str] = None,
        style_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate talking head video from source image and audio
        
        Args:
            source_image: Path to source image or numpy array
            audio_path: Path to input audio file
            text: Optional text for gesture/speech generation
            output_dir: Path to save output video
            speaker_id: Optional speaker ID for speaker-specific TTS
            style_id: Optional animation style ID
            **kwargs: Additional parameters for generation
            
        Returns:
            Path to generated video file
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with Qwen if available
        if 'qwen' in self.components and text:
            try:
                enhanced_prompt = self.components['qwen'].enhance_prompt(text)
                logger.info(f"Enhanced prompt: {enhanced_prompt}")
                # TODO: Use enhanced prompt for better generation
            except Exception as e:
                logger.warning(f"Failed to enhance prompt: {e}")
        
        # Get animation style parameters
        style = None
        if style_id:
            style = self.style_manager.get_style(style_id)
        elif speaker_id:
            # Get speaker's default style if no style_id provided
            style = self.style_manager.get_default_style(speaker_id)
        
        # Merge style parameters with any explicitly provided parameters
        style_params = {}
        if style:
            style_params.update(style.parameters)
            logger.info(f"Using animation style: {style.name}")
        
        # Override with any explicitly provided parameters
        style_params.update(kwargs)
        
        # Generate video using the appropriate model
        if 'sadtalker' in self.components:
            output_path = self.components['sadtalker'].generate(
                source_image=source_image,
                audio_path=audio_path,
                output_dir=output_dir,
                **style_params
            )
            
            # Enhance with Wav2Lip if available
            if 'wav2lip' in self.components:
                output_path = self.components['wav2lip'].enhance(
                    video_path=output_path,
                    audio_path=audio_path,
                    output_path=output_path
                )
            
            # If text is provided but no audio, generate speech first
            if text and not audio_path:
                # Use voice cloning if voice_id is provided
                if voice_id:
                    try:
                        audio_path = self.voice_manager.generate_speech(
                            text=text,
                            voice_id=voice_id,
                            output_path=str(Path(output_dir) / f"generated_speech_{int(datetime.now().timestamp())}.wav")
                        )
                    except Exception as e:
                        logger.warning(f"Voice cloning failed: {e}. Falling back to default TTS.")
                        audio_path = None
                
                # Fall back to default TTS if voice cloning wasn't used or failed
                if not audio_path:
                    tts_params = kwargs.get('tts_params', {})
                    if speaker_id and 'voice' not in tts_params:
                        tts_params['voice'] = speaker_id
                    
                    audio_path = self.components['tts'].generate_speech(
                        text=text,
                        output_dir=output_dir,
                        **tts_params
                    )
            
            # Add gestures if enabled and we have text
            if 'gesture' in self.components and text:
                output_path = self.components['gesture'].add_gestures(
                    video_path=output_path,
                    text=text,
                    output_path=output_path
                )
            
            return output_path
                
    def __getattr__(self, name):
        """Delegate attribute access to components"""
        if name in self.components:
            return self.components[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
