import requests
import cv2
import numpy as np
import os

def test_emotion_detection(image_path):
    """Test the emotion detection API with an image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    url = "http://localhost:8000/api/v1/emotion/detect"
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(image_path), img_file, 'image/jpeg')}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("\n🎭 Emotion Detection Results:")
            print(f"Detected emotion: {result.get('emotion', 'unknown').upper()}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
            print("\nAll Probabilities:")
            for emotion, prob in result.get('probabilities', {}).items():
                print(f"- {emotion}: {prob:.4f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error making request: {str(e)}")

def capture_and_test():
    """Capture image from webcam and test emotion detection"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n📷 Press 's' to take a photo, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
            
        # Display the frame
        cv2.imshow('Press "s" to capture, "q" to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save the captured image
            temp_path = "temp_capture.jpg"
            cv2.imwrite(temp_path, frame)
            print("\nProcessing image...")
            
            # Test emotion detection
            test_emotion_detection(temp_path)
            
            # Remove temp file
            try:
                os.remove(temp_path)
            except:
                pass
                
            break
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Emotion Detection Tester")
    print("1. Test with webcam")
    print("2. Test with image file")
    
    choice = input("\nChoose an option (1/2): ")
    
    if choice == '1':
        capture_and_test()
    elif choice == '2':
        image_path = input("Enter image path: ").strip('"')
        test_emotion_detection(image_path)
    else:
        print("Invalid choice")
