import sys
import os
import cv2

# Add OpenSeeFace to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import OpenSeeFace modules
from OpenSeeFace.facetracker import FaceTracker
from OpenSeeFace.tracker import Tracker

def test_webcam():
    # Initialize the tracker
    tracker = Tracker()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
            
        # Convert to RGB (OpenSeeFace expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = tracker.predict(rgb)
        
        # Draw face landmarks
        for face in faces:
            for x, y in face.landmarks:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # Show the frame
        cv2.imshow('OpenSeeFace', frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Testing OpenSeeFace with webcam...")
    test_webcam()
