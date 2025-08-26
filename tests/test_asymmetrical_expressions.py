"""Test script for asymmetrical expressions functionality."""
import cv2
import numpy as np
import time
from pathlib import Path
import random
import sys
import os

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from models.asymmetrical_expressions import AsymmetricalExpressionSystem, AsymmetryParameters

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

def apply_asymmetry(landmarks, asymmetries):
    """Apply asymmetries to facial landmarks."""n    # Make a deep copy of the landmarks
    modified = {k: [list(p) for p in points] for k, points in landmarks.items()}
    
    # Apply each asymmetry
    for region, value in asymmetries.items():
        if 'brow_outer_l' in region:
            for i, point in enumerate(modified['left_eyebrow']):
                point[1] += int(10 * value)  # Move up/down
        elif 'brow_outer_r' in region:
            for i, point in enumerate(modified['right_eyebrow']):
                point[1] += int(10 * value)  # Move up/down
        elif 'brow_inner_l' in region:
            for i, point in enumerate(modified['left_eyebrow'][-2:], start=len(modified['left_eyebrow'])-2):
                point[1] += int(8 * value)  # Inner part of left eyebrow
        elif 'brow_inner_r' in region:
            for i, point in enumerate(modified['right_eyebrow'][:2]):
                point[1] += int(8 * value)  # Inner part of right eyebrow
        elif 'eye_open_l' in region:
            # Modify eye openness (vertical scale)
            for i in [1, 2, 4, 5]:  # Top and bottom points
                modified['left_eye'][i][1] += int(5 * value)
        elif 'eye_open_r' in region:
            for i in [1, 2, 4, 5]:  # Top and bottom points
                modified['right_eye'][i][1] += int(5 * value)
        elif 'eye_wide_l' in region:
            # Modify eye width
            modified['left_eye'][0][0] += int(3 * value)  # Left corner
            modified['left_eye'][3][0] -= int(3 * value)  # Right corner
        elif 'eye_wide_r' in region:
            modified['right_eye'][0][0] += int(3 * value)  # Left corner
            modified['right_eye'][3][0] -= int(3 * value)  # Right corner
        elif 'smile_l' in region:
            # Left side of mouth
            modified['mouth_outer'][0][0] += int(5 * value)
            modified['mouth_outer'][0][1] -= int(3 * value)
            modified['mouth_outer'][-1][0] += int(3 * value)
            modified['mouth_outer'][-1][1] -= int(2 * value)
        elif 'smile_r' in region:
            # Right side of mouth
            modified['mouth_outer'][4][0] -= int(5 * value)
            modified['mouth_outer'][4][1] -= int(3 * value)
            modified['mouth_outer'][5][0] -= int(3 * value)
            modified['mouth_outer'][5][1] -= int(2 * value)
        elif 'mouth_open' in region:
            # Vertical mouth opening
            for i in [0, 1, 2, 3]:  # Top lip points
                modified['mouth_outer'][i][1] += int(5 * value)
            for i in [4, 5, 6, 7]:  # Bottom lip points
                modified['mouth_outer'][i][1] -= int(5 * value)
        elif 'mouth_wide' in region:
            # Horizontal mouth stretch
            modified['mouth_outer'][0][0] -= int(5 * value)  # Left corner
            modified['mouth_outer'][4][0] += int(5 * value)  # Right corner
        elif 'nose_wrinkle_l' in region:
            # Left nose wrinkle
            modified['nose_bridge'][1][1] += int(3 * value)
            modified['nose_tip'][0][1] += int(2 * value)
        elif 'nose_wrinkle_r' in region:
            # Right nose wrinkle
            modified['nose_bridge'][1][1] += int(3 * value)
            modified['nose_tip'][2][1] += int(2 * value)
        elif 'cheek_raise_l' in region:
            # Left cheek raise
            for i in [2, 3]:  # Points under left eye
                modified['left_eye'][i][1] -= int(3 * value)
            modified['mouth_outer'][0][1] -= int(2 * value)  # Left mouth corner
        elif 'cheek_raise_r' in region:
            # Right cheek raise
            for i in [1, 2]:  # Points under right eye
                modified['right_eye'][i][1] -= int(3 * value)
            modified['mouth_outer'][4][1] -= int(2 * value)  # Right mouth corner
    
    return modified

def draw_face(image, landmarks, asymmetries):
    """Draw a face with the given landmarks and asymmetries."""
    # Apply asymmetries to landmarks
    modified_landmarks = apply_asymmetry(landmarks, asymmetries)
    
    # Draw face features
    def draw_polygon(points, color, thickness=1, is_closed=True):
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], is_closed, color, thickness)
    
    # Draw face outline
    draw_polygon(modified_landmarks['jaw'] + [modified_landmarks['jaw'][0]], (0, 0, 0), 2)
    
    # Draw eyebrows
    draw_polygon(modified_landmarks['left_eyebrow'], (50, 50, 50), 2)
    draw_polygon(modified_landmarks['right_eyebrow'], (50, 50, 50), 2)
    
    # Draw eyes
    for eye in ['left_eye', 'right_eye']:
        draw_polygon(modified_landmarks[eye], (0, 0, 0), 1)
        # Draw iris (simplified)
        center_x = sum(p[0] for p in modified_landmarks[eye]) // len(modified_landmarks[eye])
        center_y = sum(p[1] for p in modified_landmarks[eye]) // len(modified_landmarks[eye])
        cv2.circle(image, (center_x, center_y), 8, (0, 0, 0), -1)
    
    # Draw nose
    draw_polygon(modified_landmarks['nose_bridge'] + modified_landmarks['nose_tip'], (0, 0, 0), 1, False)
    
    # Draw mouth
    draw_polygon(modified_landmarks['mouth_outer'], (0, 0, 0), 1)
    
    # Add status text
    status_y = 30
    cv2.putText(image, "Active Asymmetries:", (10, status_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    for i, (region, value) in enumerate(asymmetries.items(), 1):
        text = f"  {region}: {value:.2f}"
        cv2.putText(image, text, (10, status_y + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize asymmetrical expression system
    params = AsymmetryParameters(
        intensity=0.5,
        min_delay=1.0,
        max_delay=3.0,
        min_duration=1.0,
        max_duration=2.5
    )
    
    asymm_system = AsymmetricalExpressionSystem(params)
    
    # Create a window for display
    window_name = "Asymmetrical Expressions Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("Asymmetrical Expressions Test")
    print("Press 'r' to force a random asymmetry")
    print("Press 'b' to toggle brow asymmetry")
    print("Press 'e' to toggle eye asymmetry")
    print("Press 'm' to toggle mouth asymmetry")
    print("Press 'n' to toggle nose asymmetry")
    print("Press 'c' to clear all asymmetries")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    # Test specific regions
    test_regions = {
        'brow': ['brow_outer_l', 'brow_outer_r', 'brow_inner_l', 'brow_inner_r'],
        'eye': ['eye_open_l', 'eye_open_r', 'eye_wide_l', 'eye_wide_r'],
        'mouth': ['smile_l', 'smile_r', 'mouth_open', 'mouth_wide'],
        'nose': ['nose_wrinkle_l', 'nose_wrinkle_r', 'cheek_raise_l', 'cheek_raise_r']
    }
    
    while True:
        # Create a blank image with a light skin tone
        img = np.ones((400, 400, 3), dtype=np.uint8) * 220
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,0] = 15  # Hue for skin tone
        img[:,:,1] = 30  # Saturation
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        # Update asymmetries
        asymm_system.update()
        asymmetries = asymm_system.get_asymmetries()
        
        # Draw the face with current asymmetries
        draw_face(img, FACE_LANDMARKS, asymmetries)
        
        # Show the frame
        cv2.imshow(window_name, img)
        
        # Save frame as image (for debugging)
        if frame_count % 5 == 0:  # Save every 5th frame to avoid too many files
            cv2.imwrite(str(output_dir / f"frame_{frame_count:04d}.png"), img)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Random asymmetry
            region = random.choice(list(FACIAL_REGIONS.keys()))
            value = random.uniform(-0.8, 0.8)
            duration = random.uniform(1.0, 2.0)
            asymm_system.force_asymmetry(region, value, duration)
        elif key == ord('b'):  # Toggle brow asymmetry
            region = random.choice(test_regions['brow'])
            value = random.uniform(-0.5, 0.5)
            asymm_system.force_asymmetry(region, value, 2.0)
        elif key == ord('e'):  # Toggle eye asymmetry
            region = random.choice(test_regions['eye'])
            value = random.uniform(-0.3, 0.3)
            asymm_system.force_asymmetry(region, value, 1.5)
        elif key == ord('m'):  # Toggle mouth asymmetry
            region = random.choice(test_regions['mouth'])
            value = random.uniform(-0.4, 0.4)
            asymm_system.force_asymmetry(region, value, 1.8)
        elif key == ord('n'):  # Toggle nose asymmetry
            region = random.choice(test_regions['nose'])
            value = random.uniform(0.0, 0.5)
            asymm_system.force_asymmetry(region, value, 1.2)
        elif key == ord('c'):  # Clear all asymmetries
            asymm_system.set_enabled(False)
            time.sleep(0.1)
            asymm_system.set_enabled(True)
        
        frame_count += 1
        
        # Print FPS every second
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f} - Active asymmetries: {len(asymmetries)}")
            frame_count = 0
            start_time = time.time()
    
    cv2.destroyAllWindows()
    print("Test complete. Check the 'output' directory for saved frames.")

if __name__ == "__main__":
    # Import FACIAL_REGIONS from the module if needed
    from models.asymmetrical_expressions import FACIAL_REGIONS
    main()
