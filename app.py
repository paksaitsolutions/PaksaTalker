import os
import sys
import logging
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import utility functions
try:
    from utils.face_utils import cv_draw_landmark, detect_faces
except ImportError:
    print("⚠️  Could not import face_utils. Some features may not work.")

# Initialize FastAPI app
app = FastAPI(title="PulsaTalker API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emotion Recognition imports
try:
    from models.emotion.fer_model import EmotionRecognizer
    from models.emotion.model_loader import get_model_path, ensure_model_downloaded
    
    # Initialize emotion recognizer
    EMOTION_MODEL = None
    if ensure_model_downloaded():
        model_path = get_model_path()
        if model_path and os.path.exists(model_path):
            EMOTION_MODEL = EmotionRecognizer(model_path)
            print("✅ Emotion recognition model loaded successfully")
        else:
            print("⚠️  Could not load emotion recognition model weights")
    else:
        print("⚠️  Could not download emotion recognition model")
except Exception as e:
    print(f"⚠️  Error initializing emotion recognition: {str(e)}")
    EMOTION_MODEL = None

# Health check endpoint
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "PulsaTalker API is running",
        "emotion_model_loaded": EMOTION_MODEL is not None
    }

# Emotion detection endpoint
@app.post("/api/v1/emotion/detect")
async def detect_emotion(image: UploadFile = File(...)):
    """
    Detect emotions from a face in the uploaded image.
    Returns the detected emotion with confidence scores.
    """
    if not EMOTION_MODEL:
        raise HTTPException(
            status_code=501,
            detail="Emotion recognition model is not available. Please check the installation."
        )
    
    try:
        # Read the uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        # Convert to RGB for emotion detection
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face using OpenCV's Haar Cascade
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {
                "status": "success",
                "message": "No faces detected",
                "faces": []
            }
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = rgb_img[y:y+h, x:x+w]
            
            # Predict emotion
            result = EMOTION_MODEL.predict_emotion(face_roi)
            
            # Add face coordinates to result
            result.update({
                "bbox": [int(x), int(y), int(w), int(h)],
                "face_image_size": f"{w}x{h}"
            })
            results.append(result)
        
        return {
            "status": "success",
            "num_faces": len(results),
            "faces": results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
