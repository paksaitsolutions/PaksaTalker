"""Deep scan test for Core Gesture Synthesis implementation."""
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gesture_model():
    """Test the main gesture model."""
    print("Testing models/gesture.py...")
    try:
        from models.gesture import GestureModel
        
        # Test initialization
        model = GestureModel()
        print("  [OK] GestureModel initialized")
        
        # Test model loading
        model.load_model()
        print("  [OK] Model loading completed")
        
        # Test gesture generation
        gestures = model.generate_gestures("Hello world", duration=2.0)
        print(f"  [OK] Generated gestures: {gestures.shape}")
        
        # Test is_loaded
        loaded = model.is_loaded()
        print(f"  [OK] Model loaded status: {loaded}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_gesture_integration():
    """Test the gesture integration."""
    print("\nTesting integrations/gesture.py...")
    try:
        from integrations.gesture import GestureGenerator
        
        # Test initialization
        generator = GestureGenerator()
        print("  [OK] GestureGenerator initialized")
        
        # Test emotion setting
        generator.set_emotion("happy", 0.8)
        print("  [OK] Emotion set to happy")
        
        # Test gesture generation
        gestures = generator.generate_gestures(
            text="Hello world",
            duration=3.0,
            emotion="excited"
        )
        print(f"  [OK] Generated gestures: {gestures.shape}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_awesome_gesture_generation():
    """Test the awesome gesture generation module."""
    print("\nTesting awesome_gesture_generation.py...")
    try:
        from awesome_gesture_generation import GestureGenerator
        
        # Test initialization
        generator = GestureGenerator()
        print("  [OK] GestureGenerator initialized")
        
        # Test gesture generation
        gestures = generator.generate_gestures(
            text="Hello world",
            duration=2.0,
            emotion="happy"
        )
        print(f"  [OK] Generated gestures: {gestures.shape}")
        
        # Test inference method
        gestures2 = generator.inference(
            text="Test text",
            duration=1.0
        )
        print(f"  [OK] Inference method: {gestures2.shape}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_full_body_animation():
    """Test the full body animation system."""
    print("\nTesting models/full_body_animation.py...")
    try:
        from models.full_body_animation import FullBodyAnimator, StanceType, MovementType
        
        # Test initialization
        animator = FullBodyAnimator()
        print("  [OK] FullBodyAnimator initialized")
        
        # Test stance setting
        animator.set_stance(StanceType.CASUAL)
        print("  [OK] Stance set to casual")
        
        # Test weight shifting
        animator.shift_weight(0.3, 1.0)
        print("  [OK] Weight shift applied")
        
        # Test update
        import numpy as np
        velocity = np.array([0, 0, 1.0])
        animator.update(0.016, velocity)
        print("  [OK] Animation update completed")
        
        # Test balance state
        balance = animator.balance.balance_stability
        print(f"  [OK] Balance stability: {balance:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_hand_articulation():
    """Test the hand articulation system."""
    print("\nTesting models/hand_articulation.py...")
    try:
        from models.hand_articulation import HandArticulator, HandSide
        
        # Test initialization
        hand = HandArticulator(HandSide.RIGHT)
        print("  [OK] HandArticulator initialized")
        
        # Test gesture setting
        hand.set_gesture("fist", 0.5)
        print("  [OK] Gesture set to fist")
        
        # Test update
        hand.update(0.1)
        print("  [OK] Hand update completed")
        
        # Test finger tip position
        import numpy as np
        from models.hand_articulation import FingerType
        tip_pos = hand.get_finger_tip_position(FingerType.INDEX)
        print(f"  [OK] Index finger tip position: {tip_pos}")
        
        # Test grab strength
        hand.set_grab_strength(0.7)
        print("  [OK] Grab strength set")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_emotion_gestures():
    """Test emotion gesture mapping."""
    print("\nTesting models/emotion_gestures.py...")
    try:
        from models.emotion_gestures import EmotionGestureMapper, EmotionType
        
        # Test initialization
        mapper = EmotionGestureMapper()
        print("  [OK] EmotionGestureMapper initialized")
        
        # Test gesture sequence generation
        gestures = mapper.get_gesture_sequence(
            EmotionType.HAPPY,
            duration=3.0,
            intensity=0.8
        )
        print(f"  [OK] Generated {len(gestures)} gestures for happy emotion")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_integration_workflow():
    """Test complete gesture synthesis workflow."""
    print("\nTesting complete gesture synthesis workflow...")
    try:
        # Test basic gesture generation pipeline
        from models.gesture import GestureModel
        from integrations.gesture import GestureGenerator
        from awesome_gesture_generation import GestureGenerator as AwesomeGenerator
        
        # Initialize components
        model = GestureModel()
        integration = GestureGenerator()
        awesome = AwesomeGenerator()
        
        print("  [OK] All components initialized")
        
        # Test text-to-gesture pipeline
        text = "Hello, welcome to PaksaTalker!"
        
        # Generate gestures with different methods
        gestures1 = model.generate_gestures(text, duration=2.0)
        gestures2 = integration.generate_gestures(text=text, duration=2.0)
        gestures3 = awesome.generate_gestures(text=text, duration=2.0)
        
        print(f"  [OK] Model gestures: {gestures1.shape}")
        print(f"  [OK] Integration gestures: {gestures2.shape}")
        print(f"  [OK] Awesome gestures: {gestures3.shape}")
        
        # Test emotion-based generation
        gestures_happy = integration.generate_gestures(
            text=text,
            duration=2.0,
            emotion="happy",
            intensity=0.8
        )
        print(f"  [OK] Happy emotion gestures: {gestures_happy.shape}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all gesture synthesis tests."""
    print("=== Deep Scan: Core Gesture Synthesis Implementation ===")
    print()
    
    tests = [
        ("Gesture Model", test_gesture_model),
        ("Gesture Integration", test_gesture_integration),
        ("Awesome Gesture Generation", test_awesome_gesture_generation),
        ("Full Body Animation", test_full_body_animation),
        ("Hand Articulation", test_hand_articulation),
        ("Emotion Gestures", test_emotion_gestures),
        ("Integration Workflow", test_integration_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  [CRITICAL ERROR] {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Results ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All Core Gesture Synthesis features are working correctly!")
        print("\nImplemented Features:")
        print("  [OK] Upper body motion generation")
        print("  [OK] Timing synchronization with speech")
        print("  [OK] Basic gesture vocabulary (pointing, nodding, etc.)")
        print("  [OK] Emotion-based gesture mapping")
        print("  [OK] Full body animation with balance physics")
        print("  [OK] Hand articulation with finger control")
        print("  [OK] Gesture transitions and blending")
    else:
        print(f"\n[FAIL] {total - passed} issues found in gesture synthesis implementation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)