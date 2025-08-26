"""Test script for eye saccades functionality."""
import cv2
import numpy as np
import time
from pathlib import Path
import random
from models.eye_dynamics import EyeDynamics, BlinkParameters, SaccadeParameters

def create_eye_image(eye_state, size=(400, 200)):
    """Create a visualization of the eyes with gaze direction."""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 200  # Light gray background
    
    # Eye parameters
    eye_radius = 30
    eye_y = size[1] // 2
    left_eye_center = (size[0] // 3, eye_y)
    right_eye_center = (2 * size[0] // 3, eye_y)
    
    # Draw eye sockets
    cv2.circle(img, left_eye_center, eye_radius + 5, (150, 150, 150), -1)
    cv2.circle(img, right_eye_center, eye_radius + 5, (150, 150, 150), -1)
    
    # Calculate pupil positions based on look target
    look_x, look_y = eye_state['look_x'], eye_state['look_y']
    pupil_radius = 10
    
    # Left pupil
    left_pupil_x = int(left_eye_center[0] + look_x * eye_radius * 0.7)
    left_pupil_y = int(left_eye_center[1] + look_y * eye_radius * 0.7)
    
    # Right pupil
    right_pupil_x = int(right_eye_center[0] + look_x * eye_radius * 0.7)
    right_pupil_y = int(right_eye_center[1] + look_y * eye_radius * 0.7)
    
    # Draw eyes (white part)
    cv2.circle(img, left_eye_center, eye_radius, (255, 255, 255), -1)
    cv2.circle(img, right_eye_center, eye_radius, (255, 255, 255), -1)
    
    # Draw pupils
    cv2.circle(img, (left_pupil_x, left_pupil_y), pupil_radius, (0, 0, 0), -1)
    cv2.circle(img, (right_pupil_x, right_pupil_y), pupil_radius, (0, 0, 0), -1)
    
    # Draw eye outlines
    cv2.circle(img, left_eye_center, eye_radius, (0, 0, 0), 1)
    cv2.circle(img, right_eye_center, eye_radius, (0, 0, 0), 1)
    
    # Add status text
    status = f"Blinking: {'Yes' if eye_state['is_blinking'] else 'No'}"
    cv2.putText(img, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    saccade_status = f"Saccading: {'Yes' if eye_state['is_saccading'] else 'No'}"
    cv2.putText(img, saccade_status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    look_status = f"Gaze: ({eye_state['look_x']:.2f}, {eye_state['look_y']:.2f})"
    cv2.putText(img, look_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize eye dynamics with custom parameters
    blink_params = BlinkParameters(
        min_blink_interval=2.0,
        max_blink_interval=5.0,
        blink_duration=0.3,
        blink_speed=4.0
    )
    
    saccade_params = SaccadeParameters(
        min_saccade_interval=1.0,
        max_saccade_interval=3.0,
        saccade_duration=0.1,
        max_saccade_angle=30.0,
        saccade_speed=15.0
    )
    
    eye_dynamics = EyeDynamics(blink_params=blink_params, saccade_params=saccade_params)
    
    # Create a window for display
    window_name = "Eye Saccades Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Eye Saccades Test")
    print("Press 'b' to force a blink")
    print("Press 's' to force a saccade")
    print("Press 'l' to look at a random point")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Update eye dynamics
        eye_dynamics.update()
        eye_state = eye_dynamics.get_eye_states()
        
        # Create visualization
        frame = create_eye_image(eye_state)
        
        # Show the frame
        cv2.imshow(window_name, frame)
        
        # Save frame as image (for debugging)
        if frame_count % 5 == 0:  # Save every 5th frame to avoid too many files
            cv2.imwrite(str(output_dir / f"frame_{frame_count:04d}.png"), frame)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('b'):  # Force blink
            eye_dynamics.force_blink()
        elif key == ord('s'):  # Force random saccade
            eye_dynamics.force_saccade()
        elif key == ord('l'):  # Look at random point
            x = random.uniform(-0.8, 0.8)
            y = random.uniform(-0.5, 0.5)
            eye_dynamics.look_at((x, y))
        
        frame_count += 1
        
        # Print FPS every second
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
            frame_count = 0
            start_time = time.time()
    
    cv2.destroyAllWindows()
    print("Test complete. Check the 'output' directory for saved frames.")

if __name__ == "__main__":
    main()
