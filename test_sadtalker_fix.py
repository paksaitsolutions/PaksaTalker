#!/usr/bin/env python3

import requests
import json
from pathlib import Path

def test_fusion_generation():
    """Test fusion generation with SadTalker fix"""
    
    # Test image and audio
    test_image = Path("d:/PaksaTalker/test_assets/test_face.jpg")
    
    # Create test image if it doesn't exist
    if not test_image.exists():
        test_image.parent.mkdir(exist_ok=True)
        import cv2
        import numpy as np
        # Create a simple test face image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 150), (350, 350), (255, 255, 255), -1)  # Face
        cv2.circle(img, (200, 200), 20, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (300, 200), 20, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (250, 280), (50, 30), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        cv2.imwrite(str(test_image), img)
        print(f"Created test image: {test_image}")
    
    # Test fusion endpoint
    url = "http://localhost:8001/api/v1/generate/fusion-video"
    
    data = {
        'prompt': 'Hello, this is a test of the SadTalker integration.',
        'resolution': '480p',
        'fps': '25',
        'emotion': 'neutral',
        'style': 'natural',
        'preferWav2Lip2': 'false'
    }
    
    files = {
        'image': open(test_image, 'rb')
    }
    
    print("Testing fusion generation with SadTalker fix...")
    print(f"Image: {test_image}")
    print(f"Prompt: {data['prompt']}")
    
    try:
        response = requests.post(url, data=data, files=files, timeout=120)
        files['image'].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Video generated: {result.get('video_path')}")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_fusion_generation()