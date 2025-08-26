"""
PaksaTalker Advanced Usage
=========================

This script demonstrates advanced usage of the PaksaTalker API, including:
- Authentication
- Video generation with different emotions
- Batch processing
- Status checking
- Downloading results
"""

import os
import time
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class EmotionType(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"

@dataclass
class VideoGenerationTask:
    task_id: str
    status: str
    created_at: str
    updated_at: str
    result_url: Optional[str] = None
    error: Optional[str] = None

class PaksaTalkerClient:
    """Client for interacting with the PaksaTalker API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or API_KEY
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an API request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('detail', error_msg)
                except:
                    error_msg = e.response.text or error_msg
            raise Exception(f"API request failed: {error_msg}")
    
    def get_health(self) -> dict:
        """Get API health status."""
        return self._request("GET", "/health")
    
    def generate_video(
        self,
        image_path: str,
        text: str,
        emotion: EmotionType = EmotionType.HAPPY,
        format: str = "mp4"
    ) -> VideoGenerationTask:
        """Generate a talking head video."""
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"text": text, "emotion": emotion.value, "format": format}
            result = self._request("POST", "/generate/video", files=files, data=data)
            return VideoGenerationTask(**result)
    
    def get_task_status(self, task_id: str) -> VideoGenerationTask:
        """Get the status of a video generation task."""
        result = self._request("GET", f"/tasks/{task_id}")
        return VideoGenerationTask(**result)
    
    def wait_for_task_completion(
        self,
        task_id: str,
        poll_interval: int = 2,
        timeout: int = 300
    ) -> VideoGenerationTask:
        """Wait for a task to complete."""
        start_time = time.time()
        
        while True:
            task = self.get_task_status(task_id)
            
            if task.status in ["completed", "failed"]:
                return task
                
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
                
            time.sleep(poll_interval)
    
    def download_file(self, url: str, output_path: str) -> str:
        """Download a file from a URL."""
        with self.session.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return output_path

def main():
    print("=== PaksaTalker Advanced Usage ===\n")
    
    # Initialize client
    client = PaksaTalkerClient()
    
    # Test connection
    try:
        health = client.get_health()
        print(f"✅ Connected to PaksaTalker API (v{health.get('version', 'unknown')})")
        print(f"Status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to connect to PaksaTalker API: {e}")
        return
    
    # Example: Generate video with different emotions
    image_path = input("\nEnter path to image file: ").strip('"')
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    text = input("Enter text to speak: ")
    
    print("\nAvailable emotions:")
    for i, emotion in enumerate(EmotionType, 1):
        print(f"{i}. {emotion.value}")
    
    try:
        emotion_idx = int(input("\nSelect emotion (1-5): ")) - 1
        emotion = list(EmotionType)[emotion_idx]
    except (ValueError, IndexError):
        print("Invalid selection, using 'happy'")
        emotion = EmotionType.HAPPY
    
    # Generate video
    print(f"\nGenerating video with {emotion.value} emotion...")
    try:
        task = client.generate_video(image_path, text, emotion=emotion)
        print(f"Started task: {task.task_id}")
        
        # Wait for completion
        print("Waiting for task to complete...")
        task = client.wait_for_task_completion(task.task_id)
        
        if task.status == "completed":
            print("✅ Video generation completed!")
            
            # Download the result
            if task.result_url:
                output_path = OUTPUT_DIR / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                client.download_file(task.result_url, str(output_path))
                print(f"📥 Video saved to: {output_path}")
        else:
            print(f"❌ Task failed: {task.error}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
