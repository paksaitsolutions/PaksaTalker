"""
Enhanced Generation Settings for PaksaTalker
Real working settings with validation and optimization
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import torch

class ResolutionPreset(Enum):
    """Standard resolution presets with real dimensions"""
    P240 = (426, 240)
    P360 = (640, 360) 
    P480 = (854, 480)
    P720 = (1280, 720)
    P1080 = (1920, 1080)
    P1440 = (2560, 1440)
    P4K = (3840, 2160)
    P8K = (7680, 4320)

class RenderQuality(Enum):
    """Render quality presets affecting processing time and output quality"""
    DRAFT = "draft"
    STANDARD = "standard" 
    HIGH = "high"
    ULTRA = "ultra"
    PRODUCTION = "production"

class EmotionType(Enum):
    """Supported emotion types for realistic expression"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    EXCITED = "excited"
    CONFIDENT = "confident"
    THOUGHTFUL = "thoughtful"

class CulturalStyle(Enum):
    """Cultural animation styles affecting gesture patterns"""
    GLOBAL = "global"
    WESTERN = "western"
    EAST_ASIAN = "east_asian"
    SOUTH_ASIAN = "south_asian"
    MIDDLE_EASTERN = "middle_eastern"
    LATIN_AMERICAN = "latin_american"
    AFRICAN = "african"

@dataclass
class GenerationSettings:
    """Comprehensive generation settings with real working parameters"""
    
    # Core AI Models
    use_emage: bool = True
    use_wav2lip2: bool = True
    use_sadtalker_full: bool = True
    
    # Quality & Performance
    resolution: str = "1080p"
    fps: int = 30
    render_quality: str = "high"
    lip_sync_quality: str = "ultra"
    
    # Animation Controls
    emotion: str = "neutral"
    emotion_intensity: float = 0.8
    body_style: str = "natural"
    avatar_type: str = "realistic"
    gesture_amplitude: float = 1.0
    
    # Visual Enhancement
    enhance_face: bool = True
    stabilization: bool = True
    background_type: str = "blur"
    lighting_style: str = "natural"
    post_processing: str = "enhanced"
    head_movement: str = "natural"
    
    # Advanced Features
    eye_tracking: bool = True
    breathing_effect: bool = True
    micro_expressions: bool = True
    
    # Cultural & Style
    cultural_style: str = "global"
    voice_sync: str = "precise"
    
    # Technical Optimization
    memory_optimization: bool = True
    gpu_acceleration: bool = True
    batch_processing: bool = False
    fp16_precision: bool = True
    tile_processing: bool = False
    
    # Audio Processing
    audio_enhancement: bool = True
    noise_reduction: bool = True
    voice_clarity: bool = True
    
    # Video Processing
    motion_blur: bool = False
    depth_of_field: bool = False
    color_grading: str = "natural"
    
    def __post_init__(self):
        """Validate and optimize settings after initialization"""
        self._validate_settings()
        self._optimize_for_hardware()
    
    def _validate_settings(self):
        """Validate all settings and apply constraints"""
        # Clamp numeric values
        self.emotion_intensity = max(0.0, min(1.0, self.emotion_intensity))
        self.gesture_amplitude = max(0.0, min(2.0, self.gesture_amplitude))
        self.fps = max(15, min(120, self.fps))
        
        # Validate resolution
        valid_resolutions = ["240p", "360p", "480p", "720p", "1080p", "1440p", "4k", "8k"]
        if self.resolution not in valid_resolutions:
            self.resolution = "1080p"
        
        # Validate enums
        if self.emotion not in [e.value for e in EmotionType]:
            self.emotion = "neutral"
        
        if self.cultural_style not in [c.value for c in CulturalStyle]:
            self.cultural_style = "global"
    
    def _optimize_for_hardware(self):
        """Optimize settings based on available hardware"""
        if not torch.cuda.is_available():
            # CPU fallback optimizations
            self.gpu_acceleration = False
            self.fp16_precision = False
            if self.resolution in ["4k", "8k"]:
                self.resolution = "1080p"
                self.tile_processing = True
        else:
            # GPU optimizations
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory < 6:  # Less than 6GB VRAM
                if self.resolution in ["4k", "8k"]:
                    self.resolution = "1080p"
                self.tile_processing = True
                self.batch_processing = False
            elif gpu_memory < 12:  # 6-12GB VRAM
                if self.resolution == "8k":
                    self.resolution = "4k"
                self.tile_processing = self.resolution in ["4k"]
            # 12GB+ can handle all settings
    
    def get_resolution_dimensions(self) -> tuple:
        """Get actual pixel dimensions for the selected resolution"""
        resolution_map = {
            "240p": (426, 240),
            "360p": (640, 360),
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "4k": (3840, 2160),
            "8k": (7680, 4320)
        }
        return resolution_map.get(self.resolution, (1920, 1080))
    
    def get_quality_multiplier(self) -> float:
        """Get quality multiplier based on render quality setting"""
        quality_map = {
            "draft": 0.5,
            "standard": 0.7,
            "high": 1.0,
            "ultra": 1.5,
            "production": 2.0
        }
        return quality_map.get(self.render_quality, 1.0)
    
    def estimate_processing_time(self, duration_seconds: float) -> float:
        """Estimate processing time in seconds based on settings"""
        base_time = duration_seconds * 2  # Base 2x real-time
        
        # Resolution multiplier
        width, height = self.get_resolution_dimensions()
        resolution_multiplier = (width * height) / (1920 * 1080)  # Relative to 1080p
        
        # Quality multiplier
        quality_multiplier = self.get_quality_multiplier()
        
        # Model complexity multiplier
        model_multiplier = 1.0
        if self.use_emage:
            model_multiplier *= 1.8
        if self.use_wav2lip2:
            model_multiplier *= 1.3
        if self.use_sadtalker_full:
            model_multiplier *= 1.2
        
        # Enhancement multiplier
        enhancement_multiplier = 1.0
        if self.enhance_face:
            enhancement_multiplier *= 1.2
        if self.stabilization:
            enhancement_multiplier *= 1.1
        if self.post_processing != "none":
            enhancement_multiplier *= 1.15
        
        # GPU acceleration speedup
        if self.gpu_acceleration and torch.cuda.is_available():
            gpu_speedup = 0.3  # 3x faster with GPU
        else:
            gpu_speedup = 1.0
        
        total_time = (base_time * resolution_multiplier * quality_multiplier * 
                     model_multiplier * enhancement_multiplier * gpu_speedup)
        
        return max(duration_seconds * 0.5, total_time)  # Minimum 0.5x real-time
    
    def estimate_memory_usage(self) -> float:
        """Estimate VRAM usage in GB"""
        width, height = self.get_resolution_dimensions()
        
        # Base memory for frame processing
        base_memory = (width * height * 3 * 4) / (1024**3)  # 4 bytes per pixel (float32)
        
        # Model memory requirements
        model_memory = 0
        if self.use_emage:
            model_memory += 4.0  # GB
        if self.use_wav2lip2:
            model_memory += 2.5  # GB
        if self.use_sadtalker_full:
            model_memory += 3.0  # GB
        
        # Processing overhead
        processing_overhead = base_memory * 8  # Multiple intermediate tensors
        
        # Batch processing multiplier
        batch_multiplier = 4 if self.batch_processing else 1
        
        total_memory = (base_memory + model_memory + processing_overhead) * batch_multiplier
        
        # FP16 optimization
        if self.fp16_precision:
            total_memory *= 0.6  # 40% memory reduction
        
        return total_memory
    
    def get_codec_settings(self) -> Dict[str, Any]:
        """Get optimized codec settings for the current quality"""
        quality_settings = {
            "draft": {"crf": 28, "preset": "ultrafast", "profile": "baseline"},
            "standard": {"crf": 23, "preset": "fast", "profile": "main"},
            "high": {"crf": 18, "preset": "medium", "profile": "high"},
            "ultra": {"crf": 15, "preset": "slow", "profile": "high"},
            "production": {"crf": 12, "preset": "veryslow", "profile": "high"}
        }
        
        base_settings = quality_settings.get(self.render_quality, quality_settings["high"])
        
        # Add resolution-specific optimizations
        width, height = self.get_resolution_dimensions()
        if width >= 3840:  # 4K+
            base_settings["level"] = "5.1"
            base_settings["refs"] = 3
        elif width >= 1920:  # 1080p+
            base_settings["level"] = "4.1"
            base_settings["refs"] = 2
        else:
            base_settings["level"] = "3.1"
            base_settings["refs"] = 1
        
        return base_settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for API/serialization"""
        return {
            # Core AI Models
            "use_emage": self.use_emage,
            "use_wav2lip2": self.use_wav2lip2,
            "use_sadtalker_full": self.use_sadtalker_full,
            
            # Quality & Performance
            "resolution": self.resolution,
            "fps": self.fps,
            "render_quality": self.render_quality,
            "lip_sync_quality": self.lip_sync_quality,
            
            # Animation Controls
            "emotion": self.emotion,
            "emotion_intensity": self.emotion_intensity,
            "body_style": self.body_style,
            "avatar_type": self.avatar_type,
            "gesture_amplitude": self.gesture_amplitude,
            
            # Visual Enhancement
            "enhance_face": self.enhance_face,
            "stabilization": self.stabilization,
            "background_type": self.background_type,
            "lighting_style": self.lighting_style,
            "post_processing": self.post_processing,
            "head_movement": self.head_movement,
            
            # Advanced Features
            "eye_tracking": self.eye_tracking,
            "breathing_effect": self.breathing_effect,
            "micro_expressions": self.micro_expressions,
            
            # Cultural & Style
            "cultural_style": self.cultural_style,
            "voice_sync": self.voice_sync,
            
            # Technical
            "memory_optimization": self.memory_optimization,
            "gpu_acceleration": self.gpu_acceleration,
            "batch_processing": self.batch_processing,
            "fp16_precision": self.fp16_precision,
            "tile_processing": self.tile_processing
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationSettings':
        """Create settings from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_preset(self, preset_name: str) -> 'GenerationSettings':
        """Get predefined quality presets"""
        presets = {
            "fast": GenerationSettings(
                resolution="720p",
                fps=25,
                render_quality="draft",
                use_emage=False,
                use_wav2lip2=False,
                enhance_face=False,
                stabilization=False,
                post_processing="none"
            ),
            "balanced": GenerationSettings(
                resolution="1080p",
                fps=30,
                render_quality="standard",
                use_emage=True,
                use_wav2lip2=True,
                use_sadtalker_full=False,
                enhance_face=True,
                stabilization=True,
                post_processing="basic"
            ),
            "quality": GenerationSettings(
                resolution="1080p",
                fps=30,
                render_quality="high",
                use_emage=True,
                use_wav2lip2=True,
                use_sadtalker_full=True,
                enhance_face=True,
                stabilization=True,
                post_processing="enhanced"
            ),
            "ultra": GenerationSettings(
                resolution="4k",
                fps=30,
                render_quality="ultra",
                use_emage=True,
                use_wav2lip2=True,
                use_sadtalker_full=True,
                enhance_face=True,
                stabilization=True,
                post_processing="professional",
                micro_expressions=True,
                eye_tracking=True,
                breathing_effect=True
            )
        }
        
        return presets.get(preset_name, self)

# Validation functions
def validate_generation_settings(settings: Dict[str, Any]) -> GenerationSettings:
    """Validate and create GenerationSettings from user input"""
    try:
        return GenerationSettings.from_dict(settings)
    except Exception as e:
        # Return safe defaults if validation fails
        return GenerationSettings()

def get_recommended_settings(
    target_quality: str = "balanced",
    available_vram_gb: float = 8.0,
    target_duration: float = 30.0
) -> GenerationSettings:
    """Get recommended settings based on hardware and requirements"""
    
    settings = GenerationSettings()
    
    # Adjust based on available VRAM
    if available_vram_gb < 4:
        settings.resolution = "720p"
        settings.use_emage = False
        settings.render_quality = "standard"
        settings.batch_processing = False
    elif available_vram_gb < 8:
        settings.resolution = "1080p"
        settings.render_quality = "high"
        settings.batch_processing = False
    elif available_vram_gb < 16:
        settings.resolution = "1080p"
        settings.render_quality = "ultra"
        settings.batch_processing = True
    else:
        settings.resolution = "4k"
        settings.render_quality = "production"
        settings.batch_processing = True
    
    # Adjust based on target quality
    if target_quality == "fast":
        settings = settings.get_preset("fast")
    elif target_quality == "quality":
        settings = settings.get_preset("quality")
    elif target_quality == "ultra":
        settings = settings.get_preset("ultra")
    
    return settings