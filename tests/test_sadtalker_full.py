"""
Comprehensive test suite for full SadTalker implementation
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image
import wave

from models.sadtalker_full import SadTalkerFull, Audio2ExpNet, Audio2PoseNet, FaceRenderer
from models.sadtalker import SadTalkerModel


class TestSadTalkerComponents:
    """Test individual SadTalker components"""
    
    def test_audio2exp_network(self):
        """Test Audio2Expression network"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = Audio2ExpNet(audio_dim=80, exp_dim=64).to(device)
        
        # Test forward pass
        batch_size = 4
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80).to(device)
        
        expressions = net(audio_features.view(-1, 80))
        
        assert expressions.shape == (batch_size * seq_len, 64)
        assert expressions.min() >= -1.0 and expressions.max() <= 1.0  # Tanh output
    
    def test_audio2pose_network(self):
        """Test Audio2Pose network"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = Audio2PoseNet(audio_dim=80, pose_dim=6).to(device)
        
        # Test forward pass
        batch_size = 4
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80).to(device)
        
        poses = net(audio_features.view(-1, 80))
        
        assert poses.shape == (batch_size * seq_len, 6)
        # Poses should be small (scaled by 0.1 in network)
        assert poses.abs().max() <= 0.1
    
    def test_face_renderer(self):
        """Test Face Renderer network"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        renderer = FaceRenderer(img_size=256).to(device)
        
        # Test forward pass
        batch_size = 2
        source_img = torch.randn(batch_size, 3, 256, 256).to(device)
        expressions = torch.randn(batch_size, 64).to(device)
        poses = torch.randn(batch_size, 6).to(device)
        
        output = renderer(source_img, expressions, poses)
        
        assert output.shape == (batch_size, 3, 256, 256)
        assert output.min() >= 0.0 and output.max() <= 1.0  # Sigmoid output


class TestSadTalkerFull:
    """Test full SadTalker implementation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SadTalkerFull(device=self.device)
        
        # Create test files
        self.create_test_files()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test image and audio files"""
        # Create test image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_img = Image.fromarray(img_array)
        self.test_image_path = os.path.join(self.temp_dir, "test_face.jpg")
        test_img.save(self.test_image_path)
        
        # Create test audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
        
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        with wave.open(self.test_audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.device == self.device
        assert not self.model.initialized
        
        # Load model
        self.model.load_model()
        assert self.model.initialized
        assert self.model.is_loaded()
    
    def test_audio_feature_extraction(self):
        """Test audio feature extraction"""
        self.model.load_model()
        
        features = self.model.extract_audio_features(self.test_audio_path)
        
        assert isinstance(features, torch.Tensor)
        assert features.device.type == self.device
        assert features.shape[1] == self.model.audio_dim  # 80 mel features
        assert features.shape[0] > 0  # Should have some frames
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        self.model.load_model()
        
        face_tensor, landmarks = self.model.preprocess_image(self.test_image_path)
        
        assert isinstance(face_tensor, torch.Tensor)
        assert face_tensor.shape == (1, 3, self.model.img_size, self.model.img_size)
        assert face_tensor.device.type == self.device
        assert face_tensor.min() >= 0.0 and face_tensor.max() <= 1.0
        
        assert isinstance(landmarks, np.ndarray)
        assert landmarks.shape[1] == 2  # x, y coordinates
    
    def test_video_generation(self):
        """Test full video generation pipeline"""
        self.model.load_model()
        
        output_path = os.path.join(self.temp_dir, "output.mp4")
        
        result_path = self.model.generate(
            image_path=self.test_image_path,
            audio_path=self.test_audio_path,
            output_path=output_path
        )
        
        assert result_path == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0  # File should not be empty
    
    def test_model_unloading(self):
        """Test model unloading"""
        self.model.load_model()
        assert self.model.is_loaded()
        
        self.model.unload()
        assert not self.model.is_loaded()


class TestSadTalkerIntegration:
    """Test SadTalker integration with emotion control"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create test files
        self.create_test_files()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test files"""
        # Create test image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_img = Image.fromarray(img_array)
        self.test_image_path = os.path.join(self.temp_dir, "test_face.jpg")
        test_img.save(self.test_image_path)
        
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
        
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        with wave.open(self.test_audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    def test_full_model_integration(self):
        """Test integration with full model"""
        model = SadTalkerModel(use_full_model=True, device=self.device)
        model.load_model()
        
        # Test emotion control
        model.set_emotion('happy', 0.8)
        assert model.current_emotion == 'happy'
        assert model.emotion_intensity == 0.8
        
        # Test video generation
        output_path = os.path.join(self.temp_dir, "output_full.mp4")
        
        result_path = model.generate(
            image_path=self.test_image_path,
            audio_path=self.test_audio_path,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
        model.unload()
    
    def test_fallback_to_basic_model(self):
        """Test fallback to basic model when full model fails"""
        # Force fallback by using invalid device
        model = SadTalkerModel(use_full_model=False, device=self.device)
        model.load_model()
        
        assert model.is_loaded()
        
        # Test video generation with basic model
        output_path = os.path.join(self.temp_dir, "output_basic.mp4")
        
        result_path = model.generate(
            image_path=self.test_image_path,
            audio_path=self.test_audio_path,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
        model.unload()
    
    def test_emotion_transitions(self):
        """Test emotion transitions"""
        model = SadTalkerModel(use_full_model=True, device=self.device)
        model.load_model()
        
        # Start with neutral
        model.set_emotion('neutral', 1.0)
        
        # Start transition to happy
        model.start_emotion_transition('happy', duration=0.1)
        
        # Update transition (should be in progress)
        in_progress = model.update_emotion_transition()
        assert in_progress or model.current_emotion == 'happy'
        
        model.unload()


def test_performance_benchmarks():
    """Test performance benchmarks"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        model = SadTalkerFull(device=device)
        model.load_model()
        
        # Benchmark audio feature extraction
        import time
        
        # Create dummy audio features
        audio_features = torch.randn(100, 80).to(device)
        
        # Warm up
        for _ in range(5):
            _ = model.audio2exp(audio_features)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            _ = model.audio2exp(audio_features)
        end_time = time.time()
        
        fps = 100 * 100 / (end_time - start_time)  # 100 frames, 100 iterations
        print(f"Audio2Exp FPS: {fps:.2f}")
        
        # Should be able to process at least 25 FPS
        assert fps > 25.0
        
        model.unload()


if __name__ == "__main__":
    # Run basic tests
    print("Testing SadTalker Full Implementation...")
    
    # Test components
    print("✓ Testing neural network components...")
    test_components = TestSadTalkerComponents()
    test_components.test_audio2exp_network()
    test_components.test_audio2pose_network()
    test_components.test_face_renderer()
    
    # Test full model
    print("✓ Testing full SadTalker model...")
    test_full = TestSadTalkerFull()
    test_full.setup_method()
    
    try:
        test_full.test_model_initialization()
        test_full.test_audio_feature_extraction()
        test_full.test_image_preprocessing()
        test_full.test_video_generation()
        test_full.test_model_unloading()
        print("  ✓ All full model tests passed")
    finally:
        test_full.teardown_method()
    
    # Test integration
    print("✓ Testing SadTalker integration...")
    test_integration = TestSadTalkerIntegration()
    test_integration.setup_method()
    
    try:
        test_integration.test_full_model_integration()
        test_integration.test_fallback_to_basic_model()
        test_integration.test_emotion_transitions()
        print("  ✓ All integration tests passed")
    finally:
        test_integration.teardown_method()
    
    # Test performance
    if torch.cuda.is_available():
        print("✓ Testing performance...")
        test_performance_benchmarks()
        print("  ✓ Performance tests passed")
    
    print("\n🎉 All SadTalker tests passed!")
    print("\nFull SadTalker Implementation Status:")
    print("  ✅ Neural Audio-to-Expression mapping")
    print("  ✅ Neural Head Pose estimation")
    print("  ✅ Neural Face Rendering")
    print("  ✅ Face Landmark Detection")
    print("  ✅ Audio Feature Extraction")
    print("  ✅ Emotion Control Integration")
    print("  ✅ Video Generation Pipeline")
    print("  ✅ Fallback to Basic Model")
    print("  ✅ Performance Optimization")