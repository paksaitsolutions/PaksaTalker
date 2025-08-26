import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class VoiceModel:
    """Represents a cloned voice model."""
    voice_id: str
    speaker_name: str
    model_path: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class VoiceCloningManager:
    """Manages voice cloning operations and voice models."""
    
    def __init__(self, storage_dir: str, device: str = None):
        """Initialize the voice cloning manager.
        
        Args:
            storage_dir: Base directory to store voice models and metadata
            device: Device to run models on ('cuda', 'mps', 'cpu')
        """
        self.storage_dir = Path(storage_dir)
        self.voices_dir = self.storage_dir / 'voices'
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load available voices
        self.voices: Dict[str, VoiceModel] = {}
        self._load_voices()
        
        logger.info(f"VoiceCloningManager initialized with {len(self.voices)} voices on {self.device}")
    
    def _load_voices(self):
        """Load all voice models from storage."""
        self.voices = {}
        
        for voice_dir in self.voices_dir.iterdir():
            if not voice_dir.is_dir():
                continue
                
            metadata_path = voice_dir / 'metadata.json'
            model_path = voice_dir / 'model.pt'
            
            if not metadata_path.exists() or not model_path.exists():
                logger.warning(f"Incomplete voice model in {voice_dir}, skipping")
                continue
                
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                voice = VoiceModel(
                    voice_id=metadata['voice_id'],
                    speaker_name=metadata['speaker_name'],
                    model_path=str(model_path),
                    metadata=metadata.get('metadata', {}),
                    created_at=metadata.get('created_at'),
                    updated_at=metadata.get('updated_at')
                )
                
                # Load embedding if available
                embedding_path = voice_dir / 'embedding.pt'
                if embedding_path.exists():
                    voice.embedding = torch.load(embedding_path, map_location='cpu').numpy()
                
                self.voices[voice.voice_id] = voice
                
            except Exception as e:
                logger.error(f"Error loading voice from {voice_dir}: {e}")
    
    def create_voice(
        self,
        audio_path: str,
        speaker_name: str,
        voice_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reference_text: Optional[str] = None
    ) -> VoiceModel:
        """Create a new voice model from audio samples.
        
        Args:
            audio_path: Path to audio file or directory of audio files
            speaker_name: Name of the speaker
            voice_id: Optional custom voice ID
            metadata: Optional metadata to store with the voice
            reference_text: Optional reference text for better quality
            
        Returns:
            VoiceModel: The created voice model
        """
        from .speaker_adaptation import extract_speaker_embedding  # Lazy import to avoid circular imports
        
        # Generate voice ID if not provided
        voice_id = voice_id or f'voice_{len(self.voices) + 1}'
        
        # Create voice directory
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(exist_ok=True)
        
        # Process audio files
        audio_path = Path(audio_path)
        audio_files = []
        
        if audio_path.is_file():
            audio_files.append(audio_path)
        elif audio_path.is_dir():
            audio_files = list(audio_path.glob('*.wav')) + list(audio_path.glob('*.mp3'))
        
        if not audio_files:
            raise ValueError(f"No valid audio files found at {audio_path}")
        
        # Extract speaker embedding from audio
        # Note: This is a placeholder - in a real implementation, you would use a pre-trained model
        # like Resemblyzer, ECAPA-TDNN, or similar to extract speaker embeddings
        speaker_embeddings = []
        for audio_file in audio_files:
            try:
                embedding = extract_speaker_embedding(str(audio_file))
                if embedding is not None:
                    speaker_embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
        
        if not speaker_embeddings:
            raise ValueError("Failed to extract speaker embeddings from any audio files")
        
        # Average embeddings if multiple files
        avg_embedding = np.mean(speaker_embeddings, axis=0)
        
        # Create voice model (in a real implementation, this would train a voice model)
        # For now, we'll just save the embedding and metadata
        voice = VoiceModel(
            voice_id=voice_id,
            speaker_name=speaker_name,
            model_path=str(voice_dir / 'model.pt'),
            embedding=avg_embedding,
            metadata=metadata or {}
        )
        
        # Save the voice model
        self._save_voice(voice)
        
        # Update in-memory cache
        self.voices[voice_id] = voice
        
        return voice
    
    def _save_voice(self, voice: VoiceModel):
        """Save a voice model to disk."""
        voice_dir = self.voices_dir / voice.voice_id
        voice_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'voice_id': voice.voice_id,
            'speaker_name': voice.speaker_name,
            'created_at': voice.created_at,
            'updated_at': voice.updated_at,
            'metadata': voice.metadata
        }
        
        with open(voice_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embedding
        if voice.embedding is not None:
            torch.save(torch.from_numpy(voice.embedding), voice_dir / 'embedding.pt')
        
        # In a real implementation, save the actual voice model here
        # For now, we'll just save a placeholder
        torch.save({'voice_id': voice.voice_id}, voice_dir / 'model.pt')
    
    def get_voice(self, voice_id: str) -> Optional[VoiceModel]:
        """Get a voice model by ID."""
        return self.voices.get(voice_id)
    
    def list_voices(self) -> List[VoiceModel]:
        """List all available voice models."""
        return list(self.voices.values())
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice model."""
        if voice_id not in self.voices:
            return False
            
        voice_dir = self.voices_dir / voice_id
        if voice_dir.exists():
            import shutil
            shutil.rmtree(voice_dir)
            
        del self.voices[voice_id]
        return True
    
    def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate speech from text using a cloned voice.
        
        Args:
            text: Text to convert to speech
            voice_id: ID of the voice to use
            output_path: Optional path to save the generated audio
            **kwargs: Additional generation parameters
            
        Returns:
            Path to the generated audio file
        """
        if voice_id not in self.voices:
            raise ValueError(f"Voice {voice_id} not found")
            
        voice = self.voices[voice_id]
        
        # In a real implementation, this would use a TTS model conditioned on the voice embedding
        # For now, we'll just use a placeholder that saves a silent audio file
        
        # Create a silent audio file as a placeholder
        sample_rate = 22050
        duration = max(1.0, len(text) / 10.0)  # 0.1s per character as a rough estimate
        samples = torch.zeros(int(sample_rate * duration))
        
        if output_path is None:
            output_path = f"output_{voice_id}_{int(datetime.now().timestamp())}.wav"
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the audio
        torchaudio.save(output_path, samples.unsqueeze(0), sample_rate)
        
        logger.info(f"Generated speech for voice {voice_id} at {output_path}")
        return str(output_path)
