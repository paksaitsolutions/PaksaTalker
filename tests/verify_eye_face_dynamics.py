"""Verify natural eye and face dynamics implementation."""
import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_eye_face_dynamics_files():
    """Check if all eye and face dynamics files exist."""
    print("Checking Natural Eye and Face Dynamics Files...")
    
    required_files = [
        "models/eye_dynamics.py",
        "models/micro_expressions.py", 
        "models/asymmetrical_expressions.py",
        "models/breathing.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            all_exist = False
    
    return all_exist

def test_eye_dynamics():
    """Test eye dynamics module."""
    print("\nTesting Eye Dynamics...")
    
    try:
        from models.eye_dynamics import EyeDynamics, BlinkParameters, SaccadeParameters
        
        # Initialize eye dynamics
        eye_dynamics = EyeDynamics()
        
        print(f"  [OK] EyeDynamics initialized")
        
        # Test blink parameters
        blink_params = BlinkParameters(
            min_blink_interval=1.0,
            max_blink_interval=3.0,
            blink_duration=0.2
        )
        print(f"  [OK] BlinkParameters: {blink_params.blink_duration}s duration")
        
        # Test saccade parameters
        saccade_params = SaccadeParameters(
            min_saccade_interval=0.5,
            max_saccade_interval=2.0,
            saccade_duration=0.05
        )
        print(f"  [OK] SaccadeParameters: {saccade_params.saccade_duration}s duration")
        
        # Test eye state
        eye_dynamics.update()
        state = eye_dynamics.get_eye_states()
        
        print(f"  [OK] Eye state: left={state['left_eye_open']:.2f}, right={state['right_eye_open']:.2f}")
        print(f"  [OK] Gaze: x={state['look_x']:.2f}, y={state['look_y']:.2f}")
        
        # Test forced blink
        eye_dynamics.force_blink(0.3)
        print("  [OK] Forced blink triggered")
        
        # Test look at target
        eye_dynamics.look_at((0.5, 0.2), immediate=True)
        print("  [OK] Look at target set")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Eye dynamics test failed: {e}")
        return False

def test_micro_expressions():
    """Test micro-expressions module."""
    print("\nTesting Micro-Expressions...")
    
    try:
        from models.micro_expressions import MicroExpressionSystem, MicroExpressionParameters, MICRO_EXPRESSION_TYPES
        
        # Initialize micro-expression system
        micro_expr = MicroExpressionSystem()
        
        print(f"  [OK] MicroExpressionSystem initialized")
        print(f"  [OK] Available expressions: {len(MICRO_EXPRESSION_TYPES)}")
        
        # Test expression types
        for expr_name in list(MICRO_EXPRESSION_TYPES.keys())[:3]:  # Test first 3
            print(f"  [OK] Expression type: {expr_name}")
        
        # Test system update
        micro_expr.update()
        state = micro_expr.get_state()
        
        print(f"  [OK] System state: active={state['active']}")
        
        # Test manual trigger
        micro_expr.trigger_expression("brow_raise", duration=0.5, intensity=0.8)
        weights = micro_expr.get_current_weights()
        
        print(f"  [OK] Manual trigger: {len(weights)} weights")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Micro-expressions test failed: {e}")
        return False

def test_asymmetrical_expressions():
    """Test asymmetrical expressions module."""
    print("\nTesting Asymmetrical Expressions...")
    
    try:
        from models.asymmetrical_expressions import AsymmetricalExpressionSystem, AsymmetryParameters, FACIAL_REGIONS
        
        # Initialize asymmetrical expression system
        asym_expr = AsymmetricalExpressionSystem()
        
        print(f"  [OK] AsymmetricalExpressionSystem initialized")
        print(f"  [OK] Facial regions: {len(FACIAL_REGIONS)}")
        
        # Test facial regions
        for region_name in list(FACIAL_REGIONS.keys())[:3]:  # Test first 3
            print(f"  [OK] Facial region: {region_name}")
        
        # Test system update
        asym_expr.update()
        asymmetries = asym_expr.get_asymmetries()
        
        print(f"  [OK] Active asymmetries: {len(asymmetries)}")
        
        # Test forced asymmetry
        asym_expr.force_asymmetry("smile_l", 0.3, duration=1.0)
        print("  [OK] Forced asymmetry set")
        
        # Test enable/disable
        asym_expr.set_enabled(False)
        asym_expr.set_enabled(True)
        print("  [OK] Enable/disable functionality")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Asymmetrical expressions test failed: {e}")
        return False

def test_breathing_simulation():
    """Test breathing simulation module."""
    print("\nTesting Breathing Simulation...")
    
    try:
        from models.breathing import BreathingSimulation, BreathingParameters, EMOTIONAL_PRESETS
        
        # Initialize breathing simulation
        breathing = BreathingSimulation()
        
        print(f"  [OK] BreathingSimulation initialized")
        print(f"  [OK] Emotional presets: {len(EMOTIONAL_PRESETS)}")
        
        # Test emotional presets
        for emotion in list(EMOTIONAL_PRESETS.keys())[:3]:  # Test first 3
            print(f"  [OK] Emotion preset: {emotion}")
        
        # Test system update
        breathing.update(0.016)  # 60 FPS
        state = breathing.get_breathing_state()
        
        print(f"  [OK] Breathing value: {state['breath_value']:.3f}")
        print(f"  [OK] Chest movement: {state['chest_movement']:.3f}")
        print(f"  [OK] Rate: {state['rate']:.3f} cycles/sec")
        
        # Test emotional state change
        breathing.set_emotional_state("excited", intensity=0.8)
        print("  [OK] Emotional state set to 'excited'")
        
        # Test parameter changes
        breathing.set_breathing_rate(0.3)
        breathing.set_breathing_amplitude(0.7)
        print("  [OK] Breathing parameters updated")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Breathing simulation test failed: {e}")
        return False

def test_integration():
    """Test integration of all eye and face dynamics."""
    print("\nTesting Integration...")
    
    try:
        from models.eye_dynamics import EyeDynamics
        from models.micro_expressions import MicroExpressionSystem
        from models.asymmetrical_expressions import AsymmetricalExpressionSystem
        from models.breathing import BreathingSimulation
        
        # Initialize all systems
        eye_dynamics = EyeDynamics()
        micro_expr = MicroExpressionSystem()
        asym_expr = AsymmetricalExpressionSystem()
        breathing = BreathingSimulation()
        
        print("  [OK] All systems initialized")
        
        # Simulate a few frames
        for frame in range(5):
            # Update all systems
            eye_dynamics.update()
            micro_expr.update()
            asym_expr.update()
            breathing.update(0.016)  # 60 FPS
            
            # Get states
            eye_state = eye_dynamics.get_eye_states()
            micro_weights = micro_expr.get_current_weights()
            asymmetries = asym_expr.get_asymmetries()
            breath_state = breathing.get_breathing_state()
            
            print(f"  [OK] Frame {frame+1}: Eyes={eye_state['is_blinking']}, "
                  f"Micro={len(micro_weights)}, Asym={len(asymmetries)}, "
                  f"Breath={breath_state['breath_value']:.2f}")
        
        print("  [OK] Integration test completed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Integration test failed: {e}")
        return False

def main():
    """Main verification function."""
    print("=== Natural Eye and Face Dynamics Verification ===")
    
    # Check files exist
    files_ok = check_eye_face_dynamics_files()
    
    # Test components
    eye_dynamics_ok = test_eye_dynamics()
    micro_expr_ok = test_micro_expressions()
    asym_expr_ok = test_asymmetrical_expressions()
    breathing_ok = test_breathing_simulation()
    integration_ok = test_integration()
    
    print("\n=== Verification Results ===")
    
    if files_ok:
        print("[PASS] All required files present")
    else:
        print("[FAIL] Some files missing")
    
    if eye_dynamics_ok:
        print("[PASS] Eye dynamics working")
    else:
        print("[FAIL] Eye dynamics failed")
        
    if micro_expr_ok:
        print("[PASS] Micro-expressions working")
    else:
        print("[FAIL] Micro-expressions failed")
        
    if asym_expr_ok:
        print("[PASS] Asymmetrical expressions working")
    else:
        print("[FAIL] Asymmetrical expressions failed")
        
    if breathing_ok:
        print("[PASS] Breathing simulation working")
    else:
        print("[FAIL] Breathing simulation failed")
        
    if integration_ok:
        print("[PASS] Integration working")
    else:
        print("[FAIL] Integration failed")
    
    # Overall status
    all_ok = all([files_ok, eye_dynamics_ok, micro_expr_ok, asym_expr_ok, breathing_ok, integration_ok])
    
    print(f"\nOverall Status: {'PASS' if all_ok else 'FAIL'}")
    
    if all_ok:
        print("\nNatural Eye and Face Dynamics Features:")
        print("  [OK] Blink rate modeling")
        print("  [OK] Micro-expressions")
        print("  [OK] Eye saccades")
        print("  [OK] Asymmetrical expressions")
        print("  [OK] Breathing simulation")
        print("\nKey Capabilities:")
        print("  - Realistic blinking with natural timing")
        print("  - Subtle micro-expressions for realism")
        print("  - Natural eye movements and saccades")
        print("  - Asymmetrical facial expressions")
        print("  - Breathing patterns with emotional states")
        print("  - Integrated animation system")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)