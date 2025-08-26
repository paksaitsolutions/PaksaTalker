"""Create a test image with artifacts for testing artifact reduction."""
import cv2
import numpy as np
from pathlib import Path
import os

def create_test_image():
    # Create output directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Create a 512x512 test image with a gradient background and text
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add a background gradient
    for i in range(512):
        for j in range(512):
            img[i, j] = [i//2, j//2, (i+j)//4]
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Test Image', (50, 256), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
    
    # Add some shapes
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 3)
    cv2.circle(img, (400, 150), 50, (255, 0, 0), -1)
    
    # Save the original image
    original_path = 'test_images/original_image.png'
    cv2.imwrite(original_path, img)
    print(f"Original test image saved to: {original_path}")
    
    # Add JPEG compression artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]  # Low quality
    _, buffer = cv2.imencode('.jpg', img, encode_param)
    artifact_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    # Save the image with artifacts
    artifact_path = 'test_images/artifact_image.jpg'
    cv2.imwrite(artifact_path, artifact_image, [cv2.IMWRITE_JPEG_QUALITY, 10])
    print(f"Image with artifacts saved to: {artifact_path}")
    
    return artifact_path

if __name__ == "__main__":
    create_test_image()
