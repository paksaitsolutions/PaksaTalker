"""
Style Presets Management Module

Handles saving, loading, and interpolating custom animation style presets.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from .emotion_gestures import CulturalContext, EmotionType


@dataclass
class StylePreset:
    """Represents a complete animation style preset."""
    preset_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    
    # Core animation parameters
    intensity: float = 0.7
    smoothness: float = 0.8
    expressiveness: float = 0.7
    motion_scale: float = 1.0
    head_movement: float = 0.5
    eye_blink_rate: float = 0.5
    lip_sync_strength: float = 0.9
    
    # Cultural and contextual parameters
    cultural_context: str = "GLOBAL"
    formality: float = 0.5
    engagement: float = 0.7
    dominance: float = 0.5
    
    # Advanced mannerism controls
    gesture_frequency: float = 0.7
    gesture_amplitude: float = 1.0
    micro_expression_rate: float = 0.5
    breathing_intensity: float = 0.3
    posture_variation: float = 0.4
    
    # Emotion-specific intensities
    emotion_intensities: Dict[str, float] = None
    
    # Custom parameters for extensions
    custom_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.emotion_intensities is None:
            self.emotion_intensities = {
                "NEUTRAL": 0.3,
                "HAPPY": 0.8,
                "SAD": 0.4,
                "ANGRY": 0.9,
                "SURPRISED": 0.7,
                "DISGUSTED": 0.6,
                "FEARFUL": 0.7,
                "EXCITED": 0.9,
                "CONFUSED": 0.5,
                "THINKING": 0.4,
                "CONFIDENT": 0.8,
                "DISAGREEING": 0.7,
                "AGREEING": 0.6
            }
        
        if self.custom_parameters is None:
            self.custom_parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StylePreset':
        """Create preset from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class StyleInterpolator:
    """Handles interpolation between style presets."""
    
    @staticmethod
    def interpolate_presets(preset1: StylePreset, preset2: StylePreset, 
                          ratio: float) -> StylePreset:
        """
        Interpolate between two presets.
        
        Args:
            preset1: First preset
            preset2: Second preset  
            ratio: Interpolation ratio (0.0 = all preset1, 1.0 = all preset2)
            
        Returns:
            New interpolated preset
        """
        # Clamp ratio
        ratio = max(0.0, min(1.0, ratio))
        
        # Create new preset with interpolated values
        interpolated = StylePreset(
            preset_id=str(uuid.uuid4()),
            name=f"Blend of {preset1.name} & {preset2.name}",
            description=f"Interpolated style ({ratio:.1%} {preset2.name})",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Interpolate numeric parameters
        numeric_fields = [
            'intensity', 'smoothness', 'expressiveness', 'motion_scale',
            'head_movement', 'eye_blink_rate', 'lip_sync_strength',
            'formality', 'engagement', 'dominance', 'gesture_frequency',
            'gesture_amplitude', 'micro_expression_rate', 'breathing_intensity',
            'posture_variation'
        ]
        
        for field in numeric_fields:
            val1 = getattr(preset1, field)
            val2 = getattr(preset2, field)
            interpolated_val = val1 * (1 - ratio) + val2 * ratio
            setattr(interpolated, field, interpolated_val)
        
        # Interpolate emotion intensities
        interpolated.emotion_intensities = {}
        for emotion in preset1.emotion_intensities:
            val1 = preset1.emotion_intensities.get(emotion, 0.5)
            val2 = preset2.emotion_intensities.get(emotion, 0.5)
            interpolated.emotion_intensities[emotion] = val1 * (1 - ratio) + val2 * ratio
        
        # Handle categorical fields (use threshold-based selection)
        interpolated.cultural_context = preset2.cultural_context if ratio > 0.5 else preset1.cultural_context
        
        # Merge custom parameters
        interpolated.custom_parameters = {**preset1.custom_parameters}
        for key, val2 in preset2.custom_parameters.items():
            if key in preset1.custom_parameters:
                val1 = preset1.custom_parameters[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    interpolated.custom_parameters[key] = val1 * (1 - ratio) + val2 * ratio
                else:
                    interpolated.custom_parameters[key] = val2 if ratio > 0.5 else val1
            else:
                interpolated.custom_parameters[key] = val2
        
        return interpolated
    
    @staticmethod
    def create_transition_sequence(preset1: StylePreset, preset2: StylePreset,
                                 steps: int = 10) -> List[StylePreset]:
        """
        Create a sequence of presets for smooth transition.
        
        Args:
            preset1: Starting preset
            preset2: Ending preset
            steps: Number of intermediate steps
            
        Returns:
            List of interpolated presets
        """
        sequence = []
        for i in range(steps + 1):
            ratio = i / steps
            interpolated = StyleInterpolator.interpolate_presets(preset1, preset2, ratio)
            interpolated.name = f"Transition Step {i+1}"
            sequence.append(interpolated)
        
        return sequence


class StylePresetManager:
    """Manages style presets with save/load functionality."""
    
    def __init__(self, storage_dir: str = "data/style_presets"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._presets: Dict[str, StylePreset] = {}
        self._load_presets()
        self._ensure_default_presets()
    
    def _load_presets(self):
        """Load all presets from storage."""
        self._presets = {}
        
        for preset_file in self.storage_dir.glob("*.json"):
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    preset = StylePreset.from_dict(data)
                    self._presets[preset.preset_id] = preset
            except Exception as e:
                print(f"Error loading preset {preset_file}: {e}")
    
    def _save_preset(self, preset: StylePreset):
        """Save a preset to disk."""
        preset_file = self.storage_dir / f"{preset.preset_id}.json"
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(preset.to_dict(), f, indent=2)
    
    def _ensure_default_presets(self):
        """Create default presets if they don't exist."""
        default_presets = [
            {
                "name": "Professional",
                "description": "Formal, controlled gestures for business presentations",
                "intensity": 0.6,
                "smoothness": 0.9,
                "expressiveness": 0.5,
                "formality": 0.9,
                "gesture_frequency": 0.4,
                "gesture_amplitude": 0.7
            },
            {
                "name": "Casual",
                "description": "Relaxed, natural gestures for informal conversations",
                "intensity": 0.7,
                "smoothness": 0.7,
                "expressiveness": 0.8,
                "formality": 0.2,
                "gesture_frequency": 0.8,
                "gesture_amplitude": 1.1
            },
            {
                "name": "Enthusiastic",
                "description": "Energetic, expressive gestures for engaging presentations",
                "intensity": 0.9,
                "smoothness": 0.6,
                "expressiveness": 1.0,
                "formality": 0.4,
                "gesture_frequency": 1.0,
                "gesture_amplitude": 1.3
            },
            {
                "name": "Academic",
                "description": "Thoughtful, measured gestures for educational content",
                "intensity": 0.5,
                "smoothness": 0.8,
                "expressiveness": 0.6,
                "formality": 0.7,
                "gesture_frequency": 0.5,
                "gesture_amplitude": 0.8
            },
            {
                "name": "Storyteller",
                "description": "Dramatic, narrative-focused gestures",
                "intensity": 0.8,
                "smoothness": 0.7,
                "expressiveness": 0.9,
                "formality": 0.3,
                "gesture_frequency": 0.9,
                "gesture_amplitude": 1.2,
                "emotion_intensities": {
                    "HAPPY": 0.9,
                    "SAD": 0.8,
                    "SURPRISED": 0.9,
                    "EXCITED": 1.0
                }
            }
        ]
        
        for preset_data in default_presets:
            # Check if preset already exists
            existing = next((p for p in self._presets.values() 
                           if p.name == preset_data["name"]), None)
            if not existing:
                preset = StylePreset(
                    preset_id=str(uuid.uuid4()),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    **preset_data
                )
                self._presets[preset.preset_id] = preset
                self._save_preset(preset)
    
    def create_preset(self, name: str, description: str = "", 
                     **parameters) -> StylePreset:
        """Create a new custom preset."""
        preset = StylePreset(
            preset_id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            **parameters
        )
        
        self._presets[preset.preset_id] = preset
        self._save_preset(preset)
        return preset
    
    def get_preset(self, preset_id: str) -> Optional[StylePreset]:
        """Get a preset by ID."""
        return self._presets.get(preset_id)
    
    def get_preset_by_name(self, name: str) -> Optional[StylePreset]:
        """Get a preset by name."""
        return next((p for p in self._presets.values() if p.name == name), None)
    
    def list_presets(self) -> List[StylePreset]:
        """List all available presets."""
        return list(self._presets.values())
    
    def update_preset(self, preset_id: str, **updates) -> Optional[StylePreset]:
        """Update an existing preset."""
        if preset_id not in self._presets:
            return None
        
        preset = self._presets[preset_id]
        preset.updated_at = datetime.now()
        
        for key, value in updates.items():
            if hasattr(preset, key):
                setattr(preset, key, value)
        
        self._save_preset(preset)
        return preset
    
    def delete_preset(self, preset_id: str) -> bool:
        """Delete a preset."""
        if preset_id not in self._presets:
            return False
        
        preset_file = self.storage_dir / f"{preset_id}.json"
        if preset_file.exists():
            preset_file.unlink()
        
        del self._presets[preset_id]
        return True
    
    def interpolate_presets(self, preset1_id: str, preset2_id: str, 
                          ratio: float) -> Optional[StylePreset]:
        """Interpolate between two presets."""
        preset1 = self.get_preset(preset1_id)
        preset2 = self.get_preset(preset2_id)
        
        if not preset1 or not preset2:
            return None
        
        return StyleInterpolator.interpolate_presets(preset1, preset2, ratio)
    
    def create_cultural_variants(self, base_preset_id: str) -> List[StylePreset]:
        """Create cultural variants of a base preset."""
        base_preset = self.get_preset(base_preset_id)
        if not base_preset:
            return []
        
        cultural_adjustments = {
            "WESTERN": {"formality": 0.1, "gesture_amplitude": 0.0},
            "EAST_ASIAN": {"formality": 0.2, "gesture_amplitude": -0.3, "expressiveness": -0.2},
            "MIDDLE_EASTERN": {"gesture_amplitude": 0.2, "expressiveness": 0.1},
            "SOUTH_ASIAN": {"gesture_amplitude": 0.1, "expressiveness": 0.1},
            "LATIN_AMERICAN": {"gesture_amplitude": 0.3, "expressiveness": 0.2},
            "AFRICAN": {"gesture_amplitude": 0.1, "expressiveness": 0.1}
        }
        
        variants = []
        for culture, adjustments in cultural_adjustments.items():
            if culture == base_preset.cultural_context:
                continue
            
            variant_data = base_preset.to_dict()
            variant_data['preset_id'] = str(uuid.uuid4())
            variant_data['name'] = f"{base_preset.name} ({culture.title()})"
            variant_data['description'] = f"{base_preset.description} - {culture.title()} cultural adaptation"
            variant_data['cultural_context'] = culture
            variant_data['created_at'] = datetime.now().isoformat()
            variant_data['updated_at'] = datetime.now().isoformat()
            
            # Apply cultural adjustments
            for param, adjustment in adjustments.items():
                if param in variant_data:
                    variant_data[param] = max(0.0, min(1.0, variant_data[param] + adjustment))
            
            variant = StylePreset.from_dict(variant_data)
            variants.append(variant)
            
            # Save variant
            self._presets[variant.preset_id] = variant
            self._save_preset(variant)
        
        return variants