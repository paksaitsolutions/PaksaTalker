"""Integration test for multi-speaker support features."""
import os
import tempfile
import torch
import torchaudio
import numpy as np
from pathlib import Path
import pytest
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.speaker import SpeakerEmbeddingExtractor, SpeakerManager
from models.speaker_adaptation import SpeakerAdapter
from models.voice_cloning import VoiceCloningManager
from models.animation_styles import AnimationStyleManager

class TestMultiSpeakerIntegration:
    """Test multi-speaker support integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_audio(self, temp_dir):
        """Create sample audio file for testing."""
        # Generate a simple sine wave as test audio
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        frequency = 440  # A4 note
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        
        audio_path = Path(temp_dir) / "test_audio.wav"
        torchaudio.save(audio_path, waveform, sample_rate)
        return str(audio_path)
    
    def test_speaker_embedding_extraction(self, sample_audio):
        """Test speaker embedding extraction."""
        extractor = SpeakerEmbeddingExtractor()
        
        # Extract embedding
        embedding = extractor.extract_embedding(sample_audio)
        
        # Verify embedding properties
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (256,)  # Expected embedding dimension
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    def test_speaker_registration_and_identification(self, temp_dir, sample_audio):
        """Test speaker registration and identification."""
        manager = SpeakerManager(storage_dir=temp_dir)
        
        # Register a speaker
        success = manager.register_speaker(
            audio_path=sample_audio,
            speaker_id="test_speaker_1",
            metadata={"name": "Test Speaker", "gender": "neutral"}
        )
        assert success
        
        # Verify speaker is registered
        embedding = manager.get_speaker_embedding("test_speaker_1")
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        
        # Test speaker identification
        identified_speaker = manager.identify_speaker(sample_audio, threshold=0.5)
        assert identified_speaker == "test_speaker_1"
    
    def test_voice_cloning_workflow(self, temp_dir, sample_audio):
        """Test voice cloning creation and management."""
        voice_manager = VoiceCloningManager(storage_dir=temp_dir)
        
        # Create a voice model
        voice = voice_manager.create_voice(
            audio_path=sample_audio,
            speaker_name="Test Speaker",
            voice_id="test_voice_1",
            metadata={"language": "en", "accent": "neutral"}
        )
        
        # Verify voice creation
        assert voice.voice_id == "test_voice_1"
        assert voice.speaker_name == "Test Speaker"
        assert voice.embedding is not None
        
        # Test voice retrieval
        retrieved_voice = voice_manager.get_voice("test_voice_1")
        assert retrieved_voice is not None
        assert retrieved_voice.voice_id == voice.voice_id
        
        # Test speech generation (placeholder)
        output_path = voice_manager.generate_speech(
            text="Hello, this is a test.",
            voice_id="test_voice_1"
        )
        assert Path(output_path).exists()
    
    def test_animation_style_management(self, temp_dir):
        """Test animation style creation and management."""
        style_manager = AnimationStyleManager(storage_dir=temp_dir)
        
        # Create a speaker-specific style
        speaker_style = style_manager.create_style(
            name="Energetic Speaker",
            description="High-energy animation style",
            parameters={
                "intensity": 1.2,
                "expressiveness": 0.9,
                "head_movement": 0.8,
                "gesture_frequency": 1.1
            },
            speaker_id="test_speaker_1"
        )
        
        # Create a global style
        global_style = style_manager.create_style(
            name="Professional",
            description="Professional presentation style",
            parameters={
                "intensity": 0.8,
                "expressiveness": 0.6,
                "head_movement": 0.4,
                "gesture_frequency": 0.7
            },
            is_global=True
        )
        
        # Test style retrieval
        retrieved_speaker_style = style_manager.get_style(speaker_style.style_id)
        assert retrieved_speaker_style is not None
        assert retrieved_speaker_style.speaker_id == "test_speaker_1"
        
        retrieved_global_style = style_manager.get_style(global_style.style_id)
        assert retrieved_global_style is not None
        assert retrieved_global_style.is_global
        
        # Test getting styles for a speaker
        speaker_styles = style_manager.get_speaker_styles("test_speaker_1")
        assert len(speaker_styles) >= 2  # Speaker-specific + global styles
        
        # Verify both styles are available to the speaker
        style_ids = [s.style_id for s in speaker_styles]
        assert speaker_style.style_id in style_ids
        assert global_style.style_id in style_ids
    
    def test_speaker_adaptation_workflow(self, temp_dir, sample_audio):
        """Test speaker adaptation pipeline."""
        # Create a simple dummy model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1000, 256)
            
            def forward(self, x):
                # Reshape input to expected size
                if x.dim() == 2:
                    x = x.view(x.size(0), -1)
                if x.size(1) != 1000:
                    # Pad or truncate to expected size
                    if x.size(1) < 1000:
                        padding = torch.zeros(x.size(0), 1000 - x.size(1))
                        x = torch.cat([x, padding], dim=1)
                    else:
                        x = x[:, :1000]
                return self.linear(x)
        
        model = DummyModel()
        adapter = SpeakerAdapter(model, output_dir=temp_dir)
        
        # Create a directory with the sample audio for adaptation
        audio_dir = Path(temp_dir) / "speaker_audio"
        audio_dir.mkdir()
        
        # Copy sample audio to the directory
        import shutil
        shutil.copy(sample_audio, audio_dir / "sample1.wav")
        
        # Test adaptation (this will run a simplified training loop)
        adapted_model_path = adapter.adapt_to_speaker(
            audio_dir=str(audio_dir),
            speaker_id="adapted_speaker_1",
            epochs=1,  # Minimal epochs for testing
            batch_size=1
        )
        
        # Verify adapted model was saved
        assert Path(adapted_model_path).exists()
    
    def test_complete_multi_speaker_pipeline(self, temp_dir, sample_audio):
        """Test the complete multi-speaker pipeline integration."""
        # Initialize all managers
        speaker_manager = SpeakerManager(storage_dir=temp_dir + "/speakers")
        voice_manager = VoiceCloningManager(storage_dir=temp_dir + "/voices")
        style_manager = AnimationStyleManager(storage_dir=temp_dir + "/styles")
        
        # Step 1: Register speaker
        speaker_id = "pipeline_test_speaker"
        success = speaker_manager.register_speaker(
            audio_path=sample_audio,
            speaker_id=speaker_id,
            metadata={"name": "Pipeline Test Speaker"}
        )
        assert success
        
        # Step 2: Create voice model
        voice = voice_manager.create_voice(
            audio_path=sample_audio,
            speaker_name="Pipeline Test Speaker",
            voice_id=f"voice_{speaker_id}"
        )
        assert voice is not None
        
        # Step 3: Create speaker-specific animation style
        style = style_manager.create_style(
            name="Speaker Custom Style",
            description="Custom style for pipeline test speaker",
            parameters={
                "intensity": 0.9,
                "expressiveness": 0.8,
                "head_movement": 0.6
            },
            speaker_id=speaker_id
        )
        assert style is not None
        
        # Step 4: Verify everything works together
        # Get speaker embedding
        embedding = speaker_manager.get_speaker_embedding(speaker_id)
        assert embedding is not None
        
        # Get voice model
        retrieved_voice = voice_manager.get_voice(f"voice_{speaker_id}")
        assert retrieved_voice is not None
        
        # Get speaker styles (should include both speaker-specific and global)
        speaker_styles = style_manager.get_speaker_styles(speaker_id)
        assert len(speaker_styles) >= 1
        
        # Verify speaker-specific style is available
        style_names = [s.name for s in speaker_styles]
        assert "Speaker Custom Style" in style_names
        
        print(f"✅ Multi-speaker pipeline test completed successfully!")
        print(f"   - Speaker registered: {speaker_id}")
        print(f"   - Voice model created: {voice.voice_id}")
        print(f"   - Animation style created: {style.style_id}")
        print(f"   - Available styles: {len(speaker_styles)}")

if __name__ == "__main__":
    # Run a simple test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test = TestMultiSpeakerIntegration()
        
        # Create sample audio
        sample_rate = 16000
        duration = 2.0
        frequency = 440
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        
        audio_path = Path(tmpdir) / "test_audio.wav"
        torchaudio.save(audio_path, waveform, sample_rate)
        
        # Run the complete pipeline test
        test.test_complete_multi_speaker_pipeline(tmpdir, str(audio_path))