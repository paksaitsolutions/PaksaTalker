"""Simple test for emotion gestures"""
from models.emotion_gestures import EmotionGestureMapper, EmotionType

def main():
    print("Testing Emotion Gesture System")
    print("=============================")
    
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

if __name__ == "__main__":
    main()
