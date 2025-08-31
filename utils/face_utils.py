import cv2
import numpy as np

def cv_draw_landmark(img, pts, color=(0, 255, 0), size=2):
    """Draw facial landmarks on the image"""
    if pts is None or len(pts) == 0:
        return img
    
    img = img.copy()
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), size, color, -1)
    return img

def detect_faces(image, face_detector=None):
    """Detect faces in an image using OpenCV's Haar Cascade"""
    if face_detector is None:
        # Load a pre-trained face detector
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces
