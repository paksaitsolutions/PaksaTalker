"""Test script for micro-expressions functionality."""
import cv2
import numpy as np
import time
from pathlib import Path
import random
from models.micro_expressions import MicroExpressionSystem, MicroExpressionParameters

# Define face landmarks for visualization
FACE_LANDMARKS = {
    # Eyebrows (left and right, outer to inner)
    'left_eyebrow': [(100, 80), (120, 70), (140, 70), (160, 75)],
    'right_eyebrow': [(240, 75), (260, 70), (280, 70), (300, 80)],
    
    # Eyes (left and right, clockwise from top-left)
    'left_eye': [(110, 100), (130, 90), (150, 90), (170, 100), (150, 110), (130, 110)],
    'right_eye': [(230, 100), (250, 90), (270, 90), (290, 100), (270, 110), (250, 110)],
    
    # Nose (top to bottom)
    'nose_bridge': [(200, 100), (200, 130), (200, 160)],
    'nose_tip': [(190, 160), (200, 165), (210, 160)],
    
    # Mouth (outer to inner, top then bottom)
    'mouth_outer': [(150, 180), (175, 170), (200, 175), (225, 170), (250, 180), (225, 200), (200, 205), (175, 200)],
    'mouth_inner': [(175, 185), (200, 180), (225, 185), (200, 195)],
    
    # Jawline (left to right)
    'jaw': [(100, 100), (80, 150), (90, 200), (150, 240), (200, 250), (250, 240), (310, 200), (320, 150), (300, 100)]
}

def create_face_image(weights: dict, size=(400, 400)):
    """Create a visualization of the face with micro-expressions."""
    # Create a blank image with a light skin tone
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 220
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,0] = 15  # Hue for skin tone
    img[:,:,1] = 30  # Saturation
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    # Apply weights to landmarks
    landmarks = {k: [list(point) for point in points] for k, points in FACE_LANDMARKS.items()}
    
    # Apply micro-expression weights to landmarks
    for weight_name, weight in weights.items():
        if 'brow_raiser' in weight_name or 'brow_lowerer' in weight_name:
            # Affect eyebrows
            for i, point in enumerate(landmarks['left_eyebrow'] + landmarks['right_eyebrow']):
                point[1] -= int(10 * weight * (1 if 'l' in weight_name.lower() else -1))
        
        if 'nose_wrinkler' in weight_name:
            # Affect nose
            for i, point in enumerate(landmarks['nose_bridge'] + landmarks['nose_tip']):
                point[1] -= int(5 * weight)
        
        if 'lip_corner_puller' in weight_name:
            # Affect mouth corners
            landmarks['mouth_outer'][0][0] -= int(5 * weight)  # Left corner
            landmarks['mouth_outer'][4][0] += int(5 * weight)  # Right corner
            landmarks['mouth_outer'][0][1] -= int(3 * weight)
            landmarks['mouth_outer'][4][1] -= int(3 * weight)
        
        if 'lip_presser' in weight_name:
            # Press lips together
            for i in [1, 2, 6, 7]:  # Top lip points
                landmarks['mouth_outer'][i][1] += int(3 * weight)
            for i in [0, 3, 4, 5]:  # Bottom lip points
                landmarks['mouth_outer'][i][1] -= int(3 * weight)
        
        if 'chin_raiser' in weight_name:
            # Raise chin/lower lip
            for i in [5, 6, 7]:  # Lower lip points
                landmarks['mouth_outer'][i][1] -= int(5 * weight)
            landmarks['jaw'][3] = (landmarks['jaw'][3][0], landmarks['jaw'][3][1] - int(10 * weight))
            landmarks['jaw'][4] = (landmarks['jaw'][4][0], landmarks['jaw'][4][1] - int(15 * weight))
            landmarks['jaw'][5] = (landmarks['jaw'][5][0], landmarks['jaw'][5][1] - int(10 * weight))
    
    # Draw face features
    def draw_polygon(points, color, thickness=1, is_closed=True):
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], is_closed, color, thickness)
    
    # Draw face outline
    draw_polygon(landmarks['jaw'] + [landmarks['jaw'][0]], (0, 0, 0), 2)
    
    # Draw eyebrows
    draw_polygon(landmarks['left_eyebrow'], (50, 50, 50), 2)
    draw_polygon(landmarks['right_eyebrow'], (50, 50, 50), 2)
    
    # Draw eyes
    for eye in ['left_eye', 'right_eye']:
        draw_polygon(landmarks[eye], (0, 0, 0), 1)
        # Draw iris based on weights (simplified)
        center_x = sum(p[0] for p in landmarks[eye]) // len(landmarks[eye])
        center_y = sum(p[1] for p in landmarks[eye]) // len(landmarks[eye])
        cv2.circle(img, (center_x, center_y), 8, (0, 0, 0), -1)
    
    # Draw nose
    draw_polygon(landmarks['nose_bridge'] + landmarks['nose_tip'], (0, 0, 0), 1, False)
    
    # Draw mouth
    draw_polygon(landmarks['mouth_outer'], (0, 0, 0), 1)
    
    # Add status text
    status_y = 30
    cv2.putText(img, f"Active Weights: {', '.join(weights.keys()) if weights else 'None'}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize micro-expression system with custom parameters
    params = MicroExpressionParameters(
        min_interval=2.0,
        max_interval=5.0,
        min_duration=0.3,
        max_duration=1.0,
        intensity=0.5
    )
    
    micro_expr = MicroExpressionSystem(params)
    
    # Create a window for display
    window_name = "Micro-Expressions Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Micro-Expressions Test")
    print("Press 'r' to trigger a random micro-expression")
    print("Press 'b' for brow raise")
    print("Press 'f' for brow furrow")
    print("Press 'n' for nose scrunch")
    print("Press 's' for lip suck")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Update micro-expressions
        micro_expr.update()
        weights = micro_expr.get_current_weights()
        
        # Create visualization
        frame = create_face_image(weights)
        
        # Show the frame
        cv2.imshow(window_name, frame)
        
        # Save frame as image (for debugging)
        if frame_count % 5 == 0:  # Save every 5th frame to avoid too many files
            cv2.imwrite(str(output_dir / f"frame_{frame_count:04d}.png"), frame)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Random micro-expression
            micro_expr.trigger_expression()
        elif key == ord('b'):  # Brow raise
            micro_expr.trigger_expression("brow_raise", duration=0.8, intensity=0.7)
        elif key == ord('f'):  # Brow furrow
            micro_expr.trigger_expression("brow_furrow", duration=0.6, intensity=0.6)
        elif key == ord('n'):  # Nose scrunch
            micro_expr.trigger_expression("nose_scrunch", duration=0.5, intensity=0.8)
        elif key == ord('s'):  # Lip suck
            micro_expr.trigger_expression("lip_suck", duration=0.7, intensity=0.5)
        
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
