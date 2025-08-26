"""Test script for emotion-based gesture generation"""
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.emotion_gestures import EmotionType, EmotionGestureMapper, GestureType

def test_emotion_gestures():
    """Test emotion-based gesture generation"""
    print("=== Testing Emotion-Based Gesture Generation ===")
    
    # Initialize gesture mapper
    mapper = EmotionGestureMapper()
    
    # Test emotions
    emotions = [
        ("Neutral", EmotionType.NEUTRAL),
        ("Happy", EmotionType.HAPPY),
        ("Sad", EmotionType.SAD),
        ("Angry", EmotionType.ANGRY),
        ("Surprised", EmotionType.SURPRISED),
        ("Disgusted", EmotionType.DISGUSTED),
        ("Fearful", EmotionType.FEARFUL)
    ]
    
    # Test each emotion with different intensities
    for name, emotion in emotions:
        print(f"\n--- Testing {name} Emotion ---")
        
        for intensity in [0.3, 0.6, 0.9]:
            print(f"\nIntensity: {intensity}")
            print("-" * 30)
            
            # Get gesture sequence
            gestures = mapper.get_gesture_sequence(
                emotion=emotion,
                duration=3.0,  # 3 seconds per test
                intensity=intensity
            )
            
            # Print gesture info
            for i, gesture in enumerate(gestures, 1):
                print(f"Gesture {i}: {gesture.gesture_type.name}")
                print(f"  - Duration: {gesture.duration:.2f}s")
                print(f"  - Intensity: {gesture.intensity:.2f}")
                print(f"  - Speed: {gesture.speed:.2f}")
            
            # Small delay between tests
            time.sleep(0.5)

def test_gesture_integration():
    """Test integration with the main gesture system"""
    print("\n=== Testing Gesture Integration ===")
    
    from integrations.gesture import GestureGenerator
    
    # Initialize gesture generator
    gesture_gen = GestureGenerator()
    
    # Test emotion setting
    print("\nTesting emotion setting:")
    gesture_gen.set_emotion("happy", intensity=0.8)
    print(f"Current emotion: {gesture_gen.current_emotion}")
    print(f"Current intensity: {gesture_gen.emotion_intensity}")
    
    # Test gesture generation
    print("\nGenerating gesture data...")
    gesture_data = gesture_gen.generate_gestures(duration=5.0)
    print(f"Generated gesture data shape: {gesture_data.shape}")
    print(f"Duration in seconds: {len(gesture_data) / 30:.2f}s")
    
    # Test with text input
    print("\nGenerating gestures from text...")
    text = "Hello, this is a test of the emotion-based gesture system."
    gesture_data = gesture_gen.generate_gestures(
        text=text,
        emotion="excited",  # Will fall back to happy
        intensity=0.9
    )
    print(f"Generated {len(gesture_data)} frames of gesture data")

if __name__ == "__main__":
    test_emotion_gestures()
    test_gesture_integration()
    print("\n=== Test Complete ===")
