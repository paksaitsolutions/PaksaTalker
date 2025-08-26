"""Simple test for emotion gestures"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    print("Testing Emotion Gesture System")
    print("=============================")
    
    try:
        from models.emotion_gestures import EmotionGestureMapper, EmotionType
        
        # Initialize the gesture mapper
        mapper = EmotionGestureMapper()
        
        # Test different emotions
        emotions = [
            ("Neutral", EmotionType.NEUTRAL),
            ("Happy", EmotionType.HAPPY),
            ("Sad", EmotionType.SAD),
            ("Angry", EmotionType.ANGRY)
        ]
        
        for name, emotion in emotions:
            print(f"\n--- {name} Emotion ---")
            
            # Get a gesture sequence
            gestures = mapper.get_gesture_sequence(
                emotion=emotion,
                duration=3.0,
                intensity=0.8
            )
            
            # Print the gestures
            for i, gesture in enumerate(gestures, 1):
                print(f"  {i}. {gesture.gesture_type.name} "
                      f"(Duration: {gesture.duration:.1f}s, "
                      f"Intensity: {gesture.intensity:.2f})")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
