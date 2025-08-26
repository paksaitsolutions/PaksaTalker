"""Test script for eye dynamics functionality."""
import cv2
import numpy as np
import time
from pathlib import Path
from models.eye_dynamics import EyeDynamics, BlinkParameters

def create_eye_image(eye_state, size=(100, 100)):
    """Create a simple visualization of the eyes based on their state."""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw left eye
    left_eye_center = (size[0] // 3, size[1] // 2)
    right_eye_center = (2 * size[0] // 3, size[1] // 2)
    eye_radius = 20
    
    # Draw eye sockets
    cv2.circle(img, left_eye_center, eye_radius + 5, (150, 150, 150), -1)
    cv2.circle(img, right_eye_center, eye_radius + 5, (150, 150, 150), -1)
    
    # Draw eyes (open/closed based on state)
    left_height = max(2, int(eye_radius * 2 * eye_state['left_eye_open']))
    right_height = max(2, int(eye_radius * 2 * eye_state['right_eye_open']))
    
    # Left eye
    cv2.ellipse(img, left_eye_center, (eye_radius, eye_radius), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, left_eye_center, (eye_radius, eye_radius), 0, 0, 360, (0, 0, 0), 1)
    
    # Right eye
    cv2.ellipse(img, right_eye_center, (eye_radius, eye_radius), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, right_eye_center, (eye_radius, eye_radius), 0, 0, 360, (0, 0, 0), 1)
    
    # Add status text
    status = f"Blinking: {'Yes' if eye_state['is_blinking'] else 'No'}"
    cv2.putText(img, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    left_status = f"Left: {eye_state['left_eye_open']:.2f}"
    right_status = f"Right: {eye_state['right_eye_open']:.2f}"
    cv2.putText(img, left_status, (10, size[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, right_status, (10, size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize eye dynamics with custom parameters
    params = BlinkParameters(
        min_blink_interval=1.0,  # More frequent blinks for testing
        max_blink_interval=3.0,
        blink_duration=0.3,
        blink_speed=3.0,
        blink_intensity=1.0
    )
    eye_dynamics = EyeDynamics(params)
    
    # Create a window for display
    window_name = "Eye Dynamics Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Eye Dynamics Test")
    print("Press 'b' to force a blink")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Update eye dynamics
        eye_dynamics.update()
        eye_state = eye_dynamics.get_eye_states()
        
        # Create visualization
        frame = create_eye_image(eye_state, (400, 200))
        
        # Show the frame
        cv2.imshow(window_name, frame)
        
        # Save frame as image (for debugging)
        if frame_count % 10 == 0:  # Save every 10th frame to avoid too many files
            cv2.imwrite(str(output_dir / f"frame_{frame_count:04d}.png"), frame)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('b'):  # Force blink
            eye_dynamics.force_blink()
        
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
