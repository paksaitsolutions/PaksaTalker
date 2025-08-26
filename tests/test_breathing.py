"""Test script for breathing simulation."""
import cv2
import numpy as np
import time
from pathlib import Path
import sys
import math

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from models.breathing import BreathingSimulation, BreathingParameters

def create_breathing_visualization(breathing_value: float, params: dict, size=(400, 400)):
    """Create a visualization of the breathing simulation."""
    # Create a blank image with a light background
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240
    
    # Draw a simple character to visualize breathing
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Head (slightly bobs up and down with breathing)
    head_radius = 50
    head_y = center_y - 50 + int(20 * params['head_movement'])
    cv2.circle(img, (center_x, head_y), head_radius, (200, 180, 150), -1)  # Skin tone
    
    # Shoulders (rise and fall with breathing)
    shoulder_width = 120
    shoulder_y = center_y + int(30 * params['shoulder_movement'])
    cv2.line(img, 
             (center_x - shoulder_width // 2, shoulder_y),
             (center_x + shoulder_width // 2, shoulder_y),
             (100, 100, 200), 5)
    
    # Chest (expands and contracts with breathing)
    chest_width = int(80 * (1.0 + 0.3 * params['chest_movement']))
    chest_height = int(100 * (1.0 + 0.4 * params['chest_movement']))
    chest_top = shoulder_y - chest_height // 4
    
    # Draw chest with subtle expansion
    cv2.ellipse(img, 
                (center_x, chest_top + chest_height // 2),
                (chest_width // 2, chest_height // 2),
                0, 0, 360, (200, 200, 200), -1)
    
    # Breathing indicator (visualizes the breathing cycle)
    indicator_radius = 10
    indicator_x = center_x - 80
    indicator_y = 30
    
    # Breathing meter (fills up and down)
    meter_height = 100
    meter_width = 20
    meter_fill = int(meter_height * breathing_value)
    
    # Draw meter background
    cv2.rectangle(img, 
                 (indicator_x - meter_width // 2, indicator_y),
                 (indicator_x + meter_width // 2, indicator_y + meter_height),
                 (200, 200, 200), -1)
    
    # Draw meter fill (blue for inhale, green for exhale)
    if breathing_value > 0.5:  # Inhaling
        color = (255, int(255 * (1.0 - (breathing_value - 0.5) * 2)), 0)  # Yellow to green
    else:  # Exhaling
        color = (0, int(255 * breathing_value * 2), 255)  # Blue to cyan
    
    cv2.rectangle(img, 
                 (indicator_x - meter_width // 2, indicator_y + meter_height - meter_fill),
                 (indicator_x + meter_width // 2, indicator_y + meter_height),
                 color, -1)
    
    # Draw meter outline
    cv2.rectangle(img, 
                 (indicator_x - meter_width // 2, indicator_y),
                 (indicator_x + meter_width // 2, indicator_y + meter_height),
                 (100, 100, 100), 1)
    
    # Add status text
    status_y = 20
    cv2.putText(img, f"Breathing: {breathing_value:.2f}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    status_y += 20
    cv2.putText(img, f"Rate: {params['rate']:.2f} Hz", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    status_y += 20
    cv2.putText(img, f"Amplitude: {params['amplitude']:.2f}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    status_y += 20
    cv2.putText(img, f"State: {params['emotional_state'].title()}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add instructions
    instructions = [
        "Controls:",
        "1-9: Set breathing rate (1=slow, 9=fast)",
        "Q/A: Decrease/Increase amplitude",
        "W/S: Decrease/Increase chest movement",
        "E/D: Decrease/Increase shoulder movement",
        "R/F: Decrease/Increase head movement",
        "N: Toggle natural variation",
        "C: Calm", "X: Excited", "V: Nervous",
        "B: Tired", "M: Angry", "K: Sad",
        "Space: Reset to neutral",
        "ESC: Quit"
    ]
    
    for i, line in enumerate(instructions):
        y = size[1] - 20 * (len(instructions) - i) - 10
        cv2.putText(img, line, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize breathing simulation with custom parameters
    params = BreathingParameters(
        base_rate=0.2,
        base_amplitude=0.5,
        chest_rise=0.5,
        shoulder_rise=0.3,
        head_bob=0.2,
        enable_variation=True,
        smoothness=0.8
    )
    
    breathing = BreathingSimulation(params)
    
    # Create a window for display
    window_name = "Breathing Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Breathing Simulation Test")
    print("Use the on-screen controls to adjust the breathing pattern.")
    
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    # Main loop
    while True:
        # Update breathing simulation
        breathing.update()
        
        # Get current breathing state
        state = breathing.get_breathing_state()
        
        # Create visualization
        img = create_breathing_visualization(
            state['breath_value'],
            {
                'chest_movement': state['chest_movement'],
                'shoulder_movement': state['shoulder_movement'],
                'head_movement': state['head_movement'],
                'rate': state['rate'],
                'amplitude': state['amplitude'],
                'emotional_state': state['emotional_state']
            }
        )
        
        # Show the frame
        cv2.imshow(window_name, img)
        
        # Save frame as image (for debugging)
        if frame_count % 5 == 0:  # Save every 5th frame to avoid too many files
            cv2.imwrite(str(output_dir / f"breathing_{frame_count:04d}.png"), img)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC key
            break
        elif key == ord(' '):  # Space to reset
            breathing.set_emotional_state("neutral")
        elif key == ord('n'):  # Toggle natural variation
            breathing.params.enable_variation = not breathing.params.enable_variation
        # Emotional states
        elif key == ord('c'):  # Calm
            breathing.set_emotional_state("calm", 1.0)
        elif key == ord('x'):  # Excited
            breathing.set_emotional_state("excited", 1.0)
        elif key == ord('v'):  # Nervous
            breathing.set_emotional_state("nervous", 1.0)
        elif key == ord('b'):  # Tired
            breathing.set_emotional_state("tired", 1.0)
        elif key == ord('m'):  # Angry
            breathing.set_emotional_state("angry", 1.0)
        elif key == ord('k'):  # Sad
            breathing.set_emotional_state("sad", 1.0)
        # Breathing rate control (1-9)
        elif ord('1') <= key <= ord('9'):
            rate = (key - ord('0')) * 0.1  # 0.1 to 0.9
            breathing.set_breathing_rate(rate)
        # Amplitude control
        elif key == ord('a'):
            new_amp = max(0.0, breathing.params.base_amplitude - 0.1)
            breathing.set_breathing_amplitude(new_amp)
        elif key == ord('q'):
            new_amp = min(1.0, breathing.params.base_amplitude + 0.1)
            breathing.set_breathing_amplitude(new_amp)
        # Chest movement control
        elif key == ord('s'):
            breathing.params.chest_rise = max(0.0, breathing.params.chest_rise - 0.1)
        elif key == ord('w'):
            breathing.params.chest_rise = min(1.0, breathing.params.chest_rise + 0.1)
        # Shoulder movement control
        elif key == ord('d'):
            breathing.params.shoulder_rise = max(-1.0, breathing.params.shoulder_rise - 0.1)
        elif key == ord('e'):
            breathing.params.shoulder_rise = min(1.0, breathing.params.shoulder_rise + 0.1)
        # Head movement control
        elif key == ord('f'):
            breathing.params.head_bob = max(0.0, breathing.params.head_bob - 0.1)
        elif key == ord('r'):
            breathing.params.head_bob = min(1.0, breathing.params.head_bob + 0.1)
        
        frame_count += 1
        
        # Print FPS every second
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            fps = frame_count / (current_time - last_print_time)
            print(f"FPS: {fps:.1f} - Breath: {state['breath_value']:.2f} - Rate: {state['rate']:.2f} Hz")
            frame_count = 0
            last_print_time = current_time
    
    cv2.destroyAllWindows()
    print("Test complete. Check the 'output' directory for saved frames.")

if __name__ == "__main__":
    main()
