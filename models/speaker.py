"""Speaker embedding extraction and management."""
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings from audio using a pre-trained model."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """Initialize the speaker embedding extractor.
        
        Args:
            model_path: Path to pre-trained speaker embedding model
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' 
                               if torch.backends.mps.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.sample_rate = 16000  # Expected sample rate for the model
        
    def _load_model(self, model_path: Optional[str] = None):
        """Load the speaker embedding model.
        
        For now, we'll use a simple placeholder model. In production, replace this with
        a pre-trained speaker verification model like ECAPA-TDNN or x-vector.
        """
        class PlaceholderSpeakerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(32, 256)  # 256-dim speaker embedding
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.pool(x).squeeze(-1)
                return F.normalize(self.fc(x), p=2, dim=1)
        
        model = PlaceholderSpeakerModel().to(self.device)
        if model_path and Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            numpy.ndarray: Speaker embedding vector
        """
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.dim() > 1 and waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Normalize
            waveform = waveform / (waveform.abs().max() + 1e-8)
            
            # Extract features (this is a placeholder - replace with actual feature extraction)
            with torch.no_grad():
                self.model.eval()
                embedding = self.model(waveform.unsqueeze(0).to(self.device))
                
            return embedding.cpu().numpy().squeeze()
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {str(e)}")
            raise
    
    def compare_speakers(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two speaker embeddings using cosine similarity.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            float: Similarity score between 0 and 1
        """
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8))


class SpeakerManager:
    """Manage speaker profiles and their embeddings."""
    
    def __init__(self, storage_dir: str = "data/speakers"):
        """Initialize speaker manager.
        
        Args:
            storage_dir: Directory to store speaker profiles
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_extractor = SpeakerEmbeddingExtractor()
        self.speaker_profiles = self._load_speaker_profiles()
    
    def _load_speaker_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load existing speaker profiles from disk."""
        profiles = {}
        for profile_file in self.storage_dir.glob("*.npz"):
            try:
                data = np.load(profile_file, allow_pickle=True)
                speaker_id = profile_file.stem
                profiles[speaker_id] = {
                    'embedding': data['embedding'],
                    'metadata': dict(data['metadata'].item())
                }
            except Exception as e:
                logger.warning(f"Error loading speaker profile {profile_file}: {e}")
        return profiles
    
    def register_speaker(self, audio_path: str, speaker_id: str, metadata: Optional[Dict] = None) -> bool:
        """Register a new speaker.
        
        Args:
            audio_path: Path to audio file for the speaker
            speaker_id: Unique identifier for the speaker
            metadata: Optional metadata about the speaker
            
        Returns:
            bool: True if registration was successful
        """
        try:
            # Extract speaker embedding
            embedding = self.embedding_extractor.extract_embedding(audio_path)
            
            # Save speaker profile
            profile_path = self.storage_dir / f"{speaker_id}.npz"
            np.savez_compressed(
                profile_path,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            # Update in-memory cache
            self.speaker_profiles[speaker_id] = {
                'embedding': embedding,
                'metadata': metadata or {}
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering speaker {speaker_id}: {e}")
            return False
    
    def get_speaker_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        """Get embedding for a registered speaker."""
        if speaker_id in self.speaker_profiles:
            return self.speaker_profiles[speaker_id]['embedding']
        return None
    
    def identify_speaker(self, audio_path: str, threshold: float = 0.7) -> Optional[str]:
        """Identify speaker from audio.
        
        Args:
            audio_path: Path to audio file
            threshold: Similarity threshold for positive identification
            
        Returns:
            Optional[str]: Speaker ID if identified, None otherwise
        """
        try:
            # Extract embedding from input audio
            query_embedding = self.embedding_extractor.extract_embedding(audio_path)
            
            # Compare with registered speakers
            best_match = None
            best_score = -1
            
            for speaker_id, profile in self.speaker_profiles.items():
                score = self.embedding_extractor.compare_speakers(
                    query_embedding, profile['embedding'])
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = speaker_id
            
            return best_match if best_match else None
            
        except Exception as e:
            logger.error(f"Error identifying speaker: {e}")
            return None
