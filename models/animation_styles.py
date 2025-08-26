"""Module for managing speaker-specific animation styles."""
import os
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AnimationStyle:
    """Class representing an animation style configuration."""
    
    def __init__(
        self,
        style_id: str,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        speaker_id: Optional[str] = None,
        is_global: bool = False
    ):
        """
        Initialize an animation style.
        
        Args:
            style_id: Unique identifier for the style
            name: Human-readable name of the style
            description: Description of the style
            parameters: Dictionary of animation parameters
            speaker_id: Optional speaker ID this style is associated with
            is_global: Whether this is a global style (available to all speakers)
        """
        self.style_id = style_id
        self.name = name
        self.description = description
        self.parameters = parameters
        self.speaker_id = speaker_id
        self.is_global = is_global
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert style to dictionary."""
        return {
            'style_id': self.style_id,
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'speaker_id': self.speaker_id,
            'is_global': self.is_global
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnimationStyle':
        """Create style from dictionary."""
        return cls(
            style_id=data['style_id'],
            name=data['name'],
            description=data['description'],
            parameters=data['parameters'],
            speaker_id=data.get('speaker_id'),
            is_global=data.get('is_global', False)
        )


class AnimationStyleManager:
    """Manages animation styles for speakers and global presets."""
    
    def __init__(self, storage_dir: str = "data/animation_styles"):
        """
        Initialize the animation style manager.
        
        Args:
            storage_dir: Directory to store style configurations
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded styles
        self._styles: Dict[str, AnimationStyle] = {}
        self._speaker_styles: Dict[str, List[str]] = {}  # speaker_id -> [style_ids]
        self._global_styles: List[str] = []
        
        # Load existing styles
        self._load_styles()
    
    def _load_styles(self) -> None:
        """Load all styles from storage."""
        self._styles = {}
        self._speaker_styles = {}
        self._global_styles = []
        
        if not self.storage_dir.exists():
            return
            
        for style_file in self.storage_dir.glob("*.json"):
            try:
                with open(style_file, 'r', encoding='utf-8') as f:
                    style_data = json.load(f)
                    style = AnimationStyle.from_dict(style_data)
                    self._styles[style.style_id] = style
                    
                    if style.is_global:
                        self._global_styles.append(style.style_id)
                    elif style.speaker_id:
                        if style.speaker_id not in self._speaker_styles:
                            self._speaker_styles[style.speaker_id] = []
                        self._speaker_styles[style.speaker_id].append(style.style_id)
                        
            except Exception as e:
                logger.error(f"Error loading style from {style_file}: {e}")
    
    def _save_style(self, style: AnimationStyle) -> None:
        """Save a style to disk."""
        style_file = self.storage_dir / f"{style.style_id}.json"
        with open(style_file, 'w', encoding='utf-8') as f:
            json.dump(style.to_dict(), f, indent=2)
    
    def create_style(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        speaker_id: Optional[str] = None,
        is_global: bool = False,
        style_id: Optional[str] = None
    ) -> AnimationStyle:
        """
        Create a new animation style.
        
        Args:
            name: Name of the style
            description: Description of the style
            parameters: Dictionary of animation parameters
            speaker_id: Optional speaker ID this style is for
            is_global: Whether this is a global style
            style_id: Optional custom style ID (will generate if not provided)
            
        Returns:
            The created AnimationStyle instance
        """
        if not style_id:
            style_id = f"style_{len(self._styles) + 1}"
            
        if style_id in self._styles:
            raise ValueError(f"Style with ID {style_id} already exists")
            
        if is_global and speaker_id:
            raise ValueError("A style cannot be both global and speaker-specific")
            
        style = AnimationStyle(
            style_id=style_id,
            name=name,
            description=description,
            parameters=parameters,
            speaker_id=speaker_id,
            is_global=is_global
        )
        
        self._styles[style_id] = style
        if is_global:
            self._global_styles.append(style_id)
        elif speaker_id:
            if speaker_id not in self._speaker_styles:
                self._speaker_styles[speaker_id] = []
            self._speaker_styles[speaker_id].append(style_id)
        
        self._save_style(style)
        return style
    
    def get_style(self, style_id: str) -> Optional[AnimationStyle]:
        """Get a style by ID."""
        return self._styles.get(style_id)
    
    def get_speaker_styles(self, speaker_id: str) -> List[AnimationStyle]:
        """Get all styles available for a speaker."""
        speaker_style_ids = self._speaker_styles.get(speaker_id, [])
        return [self._styles[style_id] for style_id in speaker_style_ids + self._global_styles 
                if style_id in self._styles]
    
    def get_global_styles(self) -> List[AnimationStyle]:
        """Get all global styles."""
        return [self._styles[style_id] for style_id in self._global_styles 
                if style_id in self._styles]
    
    def update_style(
        self,
        style_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[AnimationStyle]:
        """
        Update an existing style.
        
        Args:
            style_id: ID of the style to update
            name: New name (optional)
            description: New description (optional)
            parameters: New parameters (optional)
            
        Returns:
            Updated AnimationStyle or None if not found
        """
        if style_id not in self._styles:
            return None
            
        style = self._styles[style_id]
        
        if name is not None:
            style.name = name
        if description is not None:
            style.description = description
        if parameters is not None:
            style.parameters = parameters
        
        self._save_style(style)
        return style
    
    def delete_style(self, style_id: str) -> bool:
        """
        Delete a style.
        
        Args:
            style_id: ID of the style to delete
            
        Returns:
            True if deleted, False if not found
        """
        if style_id not in self._styles:
            return False
            
        style = self._styles[style_id]
        style_file = self.storage_dir / f"{style_id}.json"
        
        # Remove from caches
        if style.is_global and style_id in self._global_styles:
            self._global_styles.remove(style_id)
        elif style.speaker_id and style_id in self._speaker_styles.get(style.speaker_id, []):
            self._speaker_styles[style.speaker_id].remove(style_id)
        
        # Delete file and remove from styles
        if style_file.exists():
            try:
                style_file.unlink()
            except Exception as e:
                logger.error(f"Error deleting style file {style_file}: {e}")
        
        del self._styles[style_id]
        return True
    
    def get_default_style(self, speaker_id: Optional[str] = None) -> AnimationStyle:
        """
        Get the default animation style for a speaker.
        
        Args:
            speaker_id: Optional speaker ID to get speaker-specific default
            
        Returns:
            Default AnimationStyle
        """
        # Try to get speaker's default style first
        if speaker_id and speaker_id in self._speaker_styles:
            speaker_styles = self.get_speaker_styles(speaker_id)
            if speaker_styles:
                return speaker_styles[0]
        
        # Fall back to global default
        if self._global_styles:
            return self._styles[self._global_styles[0]]
        
        # Create a default style if none exists
        default_style = self.create_style(
            name="Default",
            description="Default animation style",
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
        
        return default_style
