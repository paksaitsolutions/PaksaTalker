#!/usr/bin/env python3
"""Test the API endpoints to verify video generation works."""

import requests
import time
import os
from pathlib import Path
from PIL import Image
import numpy as np
import wave

def create_test_files():
    """Create test image and audio files."""
    # Create test image
    img_array = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
    img = Image.fromarray(img_array)
    img.save("test_image.jpg")
    
    # Create test audio
    duration = 3.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open("test_audio.wav", 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print("Test files created: test_image.jpg, test_audio.wav")

def test_video_generation():
    """Test the video generation API."""
    base_url = "http://localhost:8000"
    
    # Create test files
    create_test_files()
    
    # Upload files and start generation
    print("Starting video generation...")
    
    with open("test_image.jpg", "rb") as img_file, open("test_audio.wav", "rb") as audio_file:
        files = {
            "image": ("test_image.jpg", img_file, "image/jpeg"),
            "audio": ("test_audio.wav", audio_file, "audio/wav")
        }
        
        data = {
            "resolution": "480p",
            "fps": 30,
            "expressionIntensity": 0.8,
            "gestureLevel": "medium",
            "voiceModel": "en-US-JennyNeural",
            "background": "blur",
            "enhanceFace": True,
            "stabilization": True
        }
        
        response = requests.post(f"{base_url}/api/generate/video", files=files, data=data)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return False
        
        result = response.json()
        if not result.get("success"):
            print(f"Generation failed: {result}")
            return False
        
        task_id = result["task_id"]
        print(f"Generation started with task ID: {task_id}")
        
        # Poll for completion
        print("Polling for completion...")
        max_attempts = 60  # 5 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(5)
            attempt += 1
            
            status_response = requests.get(f"{base_url}/api/status/{task_id}")
            if status_response.status_code != 200:
                print(f"Status check failed: {status_response.status_code}")
                continue
            
            status_data = status_response.json()
            if not status_data.get("success"):
                print(f"Status check error: {status_data}")
                continue
            
            task_info = status_data["data"]
            status = task_info["status"]
            progress = task_info.get("progress", 0)
            stage = task_info.get("stage", "Unknown")
            
            print(f"Attempt {attempt}: {status} - {progress}% - {stage}")
            
            if status == "completed":
                video_url = task_info.get("video_url")
                print(f"✅ Video generation completed!")
                print(f"Video URL: {video_url}")
                
                # Download the video
                if video_url:
                    video_response = requests.get(f"{base_url}{video_url}")
                    if video_response.status_code == 200:
                        with open("downloaded_video.mp4", "wb") as f:
                            f.write(video_response.content)
                        
                        file_size = len(video_response.content)
                        print(f"✅ Video downloaded successfully: {file_size:,} bytes")
                        
                        # Cleanup
                        os.remove("test_image.jpg")
                        os.remove("test_audio.wav")
                        
                        return True
                    else:
                        print(f"❌ Failed to download video: {video_response.status_code}")
                        return False
                else:
                    print("❌ No video URL provided")
                    return False
            
            elif status == "failed":
                error = task_info.get("error", "Unknown error")
                print(f"❌ Video generation failed: {error}")
                return False
        
        print("❌ Timeout waiting for video generation")
        return False

if __name__ == "__main__":
    print("Testing PaksaTalker API...")
    
    try:
        success = test_video_generation()
        if success:
            print("\n🎉 API test completed successfully!")
            print("The video generation system is working correctly.")
        else:
            print("\n💥 API test failed!")
            print("There are issues with the video generation system.")
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()