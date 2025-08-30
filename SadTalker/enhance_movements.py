import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

def enhance_limb_movements(input_video_path, output_video_path, smoothness=0.7):
    """
    Enhance hand and arm movements in a video using MediaPipe.
    
    Args:
        input_video_path (str): Path to input video
        output_video_path (str): Path to save enhanced video
        smoothness (float): Smoothing factor (0.0 to 1.0)
    """
    # Initialize MediaPipe solutions
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize MediaPipe models
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        
        prev_pose_landmarks = None
        prev_hand_landmarks = None
        
        # Process each frame
        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            pose_results = pose.process(image)
            hand_results = hands.process(image)
            
            # Draw pose landmarks (skeleton)
            if pose_results.pose_landmarks:
                # Apply smoothing
                if prev_pose_landmarks is not None:
                    for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                        landmark.x = (landmark.x * smoothness) + (prev_pose_landmarks[i].x * (1 - smoothness))
                        landmark.y = (landmark.y * smoothness) + (prev_pose_landmarks[i].y * (1 - smoothness))
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Update previous landmarks
                prev_pose_landmarks = pose_results.pose_landmarks.landmark
            
            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )
            
            # Write the frame
            out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance hand and arm movements in a video')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='enhanced_output.mp4', help='Output video path')
    parser.add_argument('--smoothness', type=float, default=0.7, help='Smoothing factor (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    print(f"Enhancing movements in {args.input}...")
    enhance_limb_movements(args.input, args.output, args.smoothness)
    print(f"Enhanced video saved to {args.output}")
