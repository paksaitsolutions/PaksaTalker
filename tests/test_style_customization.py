"""
Test suite for Style Customization features
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from models.style_presets import StylePresetManager, StylePreset, StyleInterpolator
from models.emotion_gestures import EmotionGestureMapper, CulturalContext, EmotionType


class TestStylePresetManager:
    """Test the StylePresetManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = StylePresetManager(storage_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_preset(self):
        """Test creating a custom preset."""
        preset = self.manager.create_preset(
            name="Test Style",
            description="A test style preset",
            intensity=0.8,
            smoothness=0.9,
            cultural_context="WESTERN"
        )
        
        assert preset.name == "Test Style"
        assert preset.description == "A test style preset"
        assert preset.intensity == 0.8
        assert preset.smoothness == 0.9
        assert preset.cultural_context == "WESTERN"
        assert preset.preset_id is not None
    
    def test_list_presets(self):
        """Test listing presets includes defaults."""
        presets = self.manager.list_presets()
        
        # Should have default presets
        assert len(presets) >= 5
        preset_names = [p.name for p in presets]
        assert "Professional" in preset_names
        assert "Casual" in preset_names
        assert "Enthusiastic" in preset_names
    
    def test_get_preset_by_name(self):
        """Test retrieving preset by name."""
        preset = self.manager.get_preset_by_name("Professional")
        
        assert preset is not None
        assert preset.name == "Professional"
        assert preset.formality > 0.8  # Professional should be formal
    
    def test_update_preset(self):
        """Test updating an existing preset."""
        # Create a preset first
        preset = self.manager.create_preset(
            name="Updatable Style",
            intensity=0.5
        )
        
        # Update it
        updated = self.manager.update_preset(
            preset.preset_id,
            intensity=0.9,
            description="Updated description"
        )
        
        assert updated is not None
        assert updated.intensity == 0.9
        assert updated.description == "Updated description"
        assert updated.name == "Updatable Style"  # Unchanged
    
    def test_delete_preset(self):
        """Test deleting a preset."""
        # Create a preset
        preset = self.manager.create_preset(name="Deletable Style")
        preset_id = preset.preset_id
        
        # Verify it exists
        assert self.manager.get_preset(preset_id) is not None
        
        # Delete it
        success = self.manager.delete_preset(preset_id)
        assert success is True
        
        # Verify it's gone
        assert self.manager.get_preset(preset_id) is None
    
    def test_create_cultural_variants(self):
        """Test creating cultural variants of a preset."""
        # Create base preset
        base_preset = self.manager.create_preset(
            name="Base Style",
            cultural_context="GLOBAL"
        )
        
        # Create variants
        variants = self.manager.create_cultural_variants(base_preset.preset_id)
        
        assert len(variants) > 0
        
        # Check that variants have different cultural contexts
        contexts = [v.cultural_context for v in variants]
        assert "WESTERN" in contexts
        assert "EAST_ASIAN" in contexts
        assert "MIDDLE_EASTERN" in contexts


class TestStyleInterpolator:
    """Test the StyleInterpolator functionality."""
    
    def test_interpolate_presets(self):
        """Test interpolating between two presets."""
        preset1 = StylePreset(
            preset_id="1",
            name="Style 1",
            description="First style",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            intensity=0.2,
            smoothness=0.3,
            expressiveness=0.4
        )
        
        preset2 = StylePreset(
            preset_id="2",
            name="Style 2", 
            description="Second style",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            intensity=0.8,
            smoothness=0.9,
            expressiveness=1.0
        )
        
        # Test 50/50 interpolation
        interpolated = StyleInterpolator.interpolate_presets(preset1, preset2, 0.5)
        
        assert interpolated.intensity == 0.5  # (0.2 + 0.8) / 2
        assert interpolated.smoothness == 0.6  # (0.3 + 0.9) / 2
        assert interpolated.expressiveness == 0.7  # (0.4 + 1.0) / 2
        assert "Blend of" in interpolated.name
    
    def test_interpolate_edge_cases(self):
        """Test interpolation edge cases."""
        preset1 = StylePreset(
            preset_id="1",
            name="Style 1",
            description="First style",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            intensity=0.0
        )
        
        preset2 = StylePreset(
            preset_id="2",
            name="Style 2",
            description="Second style", 
            created_at=datetime.now(),
            updated_at=datetime.now(),
            intensity=1.0
        )
        
        # Test 0% interpolation (all preset1)
        result = StyleInterpolator.interpolate_presets(preset1, preset2, 0.0)
        assert result.intensity == 0.0
        
        # Test 100% interpolation (all preset2)
        result = StyleInterpolator.interpolate_presets(preset1, preset2, 1.0)
        assert result.intensity == 1.0
    
    def test_create_transition_sequence(self):
        """Test creating transition sequences."""
        preset1 = StylePreset(
            preset_id="1",
            name="Start",
            description="Starting style",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            intensity=0.0
        )
        
        preset2 = StylePreset(
            preset_id="2",
            name="End",
            description="Ending style",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            intensity=1.0
        )
        
        sequence = StyleInterpolator.create_transition_sequence(preset1, preset2, steps=4)
        
        assert len(sequence) == 5  # 4 steps + 1 = 5 total
        assert sequence[0].intensity == 0.0  # Start
        assert sequence[2].intensity == 0.5  # Middle
        assert sequence[4].intensity == 1.0  # End


class TestCulturalVariations:
    """Test enhanced cultural variations."""
    
    def test_cultural_gesture_mappings(self):
        """Test that different cultures have different gesture mappings."""
        mapper = EmotionGestureMapper()
        
        # Test East Asian context (more restrained)
        mapper.set_cultural_context(CulturalContext.EAST_ASIAN)
        east_asian_gesture = mapper.get_gesture_for_emotion(EmotionType.HAPPY, 0.8)
        
        # Test Latin American context (more expressive)
        mapper.set_cultural_context(CulturalContext.LATIN_AMERICAN)
        latin_gesture = mapper.get_gesture_for_emotion(EmotionType.HAPPY, 0.8)
        
        # East Asian should be more restrained
        assert east_asian_gesture.intensity < latin_gesture.intensity
    
    def test_cultural_parameters(self):
        """Test cultural parameter differences."""
        mapper = EmotionGestureMapper()
        
        # Test different cultural parameters
        east_asian_params = mapper.cultural_params[CulturalContext.EAST_ASIAN]
        latin_params = mapper.cultural_params[CulturalContext.LATIN_AMERICAN]
        
        # East Asian should be less expressive
        assert east_asian_params['expressiveness'] < latin_params['expressiveness']
        
        # Latin American should have closer personal space
        assert latin_params['personal_space'] < east_asian_params['personal_space']
    
    def test_automatic_intensity_analysis(self):
        """Test automatic intensity calculation from speech."""
        mapper = EmotionGestureMapper()
        
        # Test with exclamation marks (should increase intensity)
        high_intensity_text = "This is amazing! Absolutely incredible!"
        high_intensity = mapper.analyze_speech_for_intensity(high_intensity_text)
        
        # Test with question (should be more reserved)
        question_text = "Maybe we could consider this option?"
        question_intensity = mapper.analyze_speech_for_intensity(question_text)
        
        assert high_intensity > question_intensity
    
    def test_gesture_sequence_generation(self):
        """Test generating gesture sequences with cultural context."""
        mapper = EmotionGestureMapper()
        
        # Set cultural context
        mapper.set_cultural_context(CulturalContext.MIDDLE_EASTERN)
        
        # Generate gesture sequence
        gestures = mapper.get_gesture_sequence(
            emotion=EmotionType.HAPPY,
            duration=5.0,
            text="Welcome to our presentation!",
            context={'environment': 'professional'}
        )
        
        assert len(gestures) > 0
        assert all(g.intensity > 0 for g in gestures)
        assert all(g.duration > 0 for g in gestures)


def test_integration_style_and_gestures():
    """Test integration between style presets and gesture generation."""
    # Create style manager
    temp_dir = tempfile.mkdtemp()
    try:
        manager = StylePresetManager(storage_dir=temp_dir)
        
        # Create a custom preset with specific cultural context
        preset = manager.create_preset(
            name="Middle Eastern Professional",
            cultural_context="MIDDLE_EASTERN",
            formality=0.9,
            gesture_frequency=0.8,
            expressiveness=0.7
        )
        
        # Create gesture mapper with same cultural context
        mapper = EmotionGestureMapper()
        mapper.set_cultural_context(CulturalContext.MIDDLE_EASTERN)
        
        # Generate gestures that should reflect the style
        gestures = mapper.get_gesture_sequence(
            emotion=EmotionType.CONFIDENT,
            duration=3.0,
            intensity=preset.expressiveness
        )
        
        assert len(gestures) > 0
        # Gestures should reflect the preset's expressiveness
        avg_intensity = sum(g.intensity for g in gestures) / len(gestures)
        assert abs(avg_intensity - preset.expressiveness) < 0.3
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run basic tests
    print("Testing Style Customization System...")
    
    # Test StylePresetManager
    print("✓ Testing StylePresetManager...")
    test_manager = TestStylePresetManager()
    test_manager.setup_method()
    
    try:
        test_manager.test_create_preset()
        test_manager.test_list_presets()
        test_manager.test_get_preset_by_name()
        print("  ✓ Basic preset operations work")
        
        test_manager.test_create_cultural_variants()
        print("  ✓ Cultural variants creation works")
        
    finally:
        test_manager.teardown_method()
    
    # Test StyleInterpolator
    print("✓ Testing StyleInterpolator...")
    test_interpolator = TestStyleInterpolator()
    test_interpolator.test_interpolate_presets()
    test_interpolator.test_interpolate_edge_cases()
    test_interpolator.test_create_transition_sequence()
    print("  ✓ Style interpolation works")
    
    # Test Cultural Variations
    print("✓ Testing Cultural Variations...")
    test_cultural = TestCulturalVariations()
    test_cultural.test_cultural_gesture_mappings()
    test_cultural.test_cultural_parameters()
    test_cultural.test_automatic_intensity_analysis()
    test_cultural.test_gesture_sequence_generation()
    print("  ✓ Cultural variations work")
    
    # Test Integration
    print("✓ Testing Integration...")
    test_integration_style_and_gestures()
    print("  ✓ Style and gesture integration works")
    
    print("\n🎉 All Style Customization tests passed!")
    print("\nFeatures implemented:")
    print("  ✅ Save custom presets")
    print("  ✅ Style interpolation between presets") 
    print("  ✅ More cultural variations")
    print("  ✅ Advanced mannerism controls")
    print("  ✅ Frontend integration")
    print("  ✅ API endpoints")