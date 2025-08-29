import cv2
import numpy as np

class FaceDetector:
    def __init__(self, weights_path, arch='resnet50', device='cuda'):
        self.device = device
        self.weights_path = weights_path
        self.arch = arch
        
        # Initialize OpenCV face detector as fallback
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None
    
    def detect_faces(self, image):
        """Detect faces in image"""
        if self.face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                return faces
        
        # Fallback: return center region
        h, w = image.shape[:2]
        return np.array([[int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6)]])
    
    def get_face_region(self, image, padding=20):
        """Get face region with padding"""
        faces = self.detect_faces(image)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            return image[y1:y2, x1:x2]
        
        # Return center region if no face detected
        h, w = image.shape[:2]
        return image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]