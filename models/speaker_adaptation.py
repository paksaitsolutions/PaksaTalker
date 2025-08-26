"""Speaker adaptation module for fine-tuning models to new speakers."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SpeakerAdaptationDataset(Dataset):
    """Dataset for speaker adaptation."""
    
    def __init__(self, audio_dir: str, transcript_file: Optional[str] = None, sample_rate: int = 16000):
        """
        Initialize the dataset.
        
        Args:
            audio_dir: Directory containing audio files
            transcript_file: Optional file containing transcriptions
            sample_rate: Target sample rate for audio
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.audio_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.mp3"))
        
        # Load transcripts if available
        self.transcripts = {}
        if transcript_file and os.path.exists(transcript_file):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    self.transcripts = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load transcripts: {e}")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get an audio file and its metadata."""
        audio_path = self.audio_files[idx]
        audio_id = audio_path.stem
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                
            # Convert to mono if needed
            if waveform.dim() > 1 and waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Normalize
            waveform = waveform / (waveform.abs().max() + 1e-8)
            
            # Get transcript if available
            transcript = self.transcripts.get(audio_id, "")
            
            return {
                'waveform': waveform,
                'sample_rate': self.sample_rate,
                'transcript': transcript,
                'audio_id': audio_id,
                'audio_path': str(audio_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise


class SpeakerAdapter:
    """Class for adapting models to new speakers."""
    
    def __init__(self, 
                model: nn.Module,
                device: Optional[torch.device] = None,
                output_dir: str = "output/adapted_models"):
        """
        Initialize the speaker adapter.
        
        Args:
            model: The model to adapt
            device: Device to run the model on
            output_dir: Directory to save adapted models
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def adapt_to_speaker(self,
                        audio_dir: str,
                        speaker_id: str,
                        epochs: int = 10,
                        batch_size: int = 4,
                        learning_rate: float = 1e-4,
                        num_workers: int = 4) -> str:
        """
        Fine-tune the model on a new speaker's data.
        
        Args:
            audio_dir: Directory containing the speaker's audio files
            speaker_id: Unique identifier for the speaker
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for the optimizer
            num_workers: Number of worker processes for data loading
            
        Returns:
            Path to the adapted model
        """
        # Create dataset and dataloader
        dataset = SpeakerAdaptationDataset(audio_dir)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                waveforms = batch['waveform'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(waveforms)
                
                # Compute loss (this is a placeholder - adjust based on your model's output)
                # In a real implementation, you'd compare the output to the target
                # For now, we'll use a dummy target for demonstration
                targets = torch.zeros_like(outputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update progress
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / len(dataloader)})
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        # Save the adapted model
        model_path = self.output_dir / f"{speaker_id}_adapted.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'speaker_id': speaker_id,
            'epochs': epochs,
            'learning_rate': learning_rate
        }, model_path)
        
        logger.info(f"Adapted model saved to {model_path}")
        return str(model_path)


def adapt_speaker_model(model: nn.Module,
                       audio_dir: str,
                       speaker_id: str,
                       output_dir: str = "output/adapted_models",
                       **kwargs) -> str:
    """
    Convenience function to adapt a model to a new speaker.
    
    Args:
        model: The model to adapt
        audio_dir: Directory containing the speaker's audio files
        speaker_id: Unique identifier for the speaker
        output_dir: Directory to save the adapted model
        **kwargs: Additional arguments for the adapter
        
    Returns:
        Path to the adapted model
    """
    adapter = SpeakerAdapter(model, output_dir=output_dir)
    return adapter.adapt_to_speaker(audio_dir=audio_dir, speaker_id=speaker_id, **kwargs)
