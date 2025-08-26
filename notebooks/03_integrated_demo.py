"""
PaksaTalker Integrated Demo
==========================

This script demonstrates a complete workflow using all PaksaTalker features:
1. Generate speech from text using Qwen
2. Create a talking head video with SadTalker
3. Enhance lip sync with Wav2Lip
4. Add gestures based on speech content
"""

import os
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class ModelType(str, Enum):
    QWEN = "qwen"
    SADTALKER = "sadtalker"
    WAV2LIP = "wav2lip"
    GESTURE = "gesture"

@dataclass
class TaskStatus:
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

class PaksaTalkerDemo:
    """Demonstrates PaksaTalker's capabilities with a complete workflow."""
    
    def __init__(self, api_key: str = None, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or API_KEY
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an API request with error handling."""
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
    
    def check_models(self) -> Dict[str, bool]:
        """Check which models are available and ready."""
        try:
            health = self._request("GET", "/health")
            return health.get("models", {})
        except:
            return {}
    
    def generate_speech(self, text: str, voice: str = "default") -> Tuple[Optional[str], Optional[str]]:
        """Generate speech from text using Qwen."""
        try:
            data = {"text": text, "voice": voice}
            result = self._request("POST", "/qwen/tts", json=data)
            return result.get("audio_path"), result.get("duration")
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None, None
    
    def create_talking_head(
        self,
        image_path: str,
        audio_path: str,
        emotion: str = "happy",
        enhance: bool = True
    ) -> Optional[str]:
        """Generate a talking head video with optional enhancement."""
        try:
            with open(image_path, "rb") as img_file, open(audio_path, "rb") as audio_file:
                files = {
                    "image": (os.path.basename(image_path), img_file, "image/jpeg"),
                    "audio": (os.path.basename(audio_path), audio_file, "audio/wav")
                }
                data = {
                    "emotion": emotion,
                    "enhance": str(enhance).lower(),
                    "format": "mp4"
                }
                
                result = self._request("POST", "/generate/video", files=files, data=data)
                return result.get("output_path")
        except Exception as e:
            print(f"Error creating talking head: {e}")
            return None
    
    def enhance_lip_sync(self, video_path: str, audio_path: str) -> Optional[str]:
        """Enhance lip sync using Wav2Lip."""
        try:
            with open(video_path, "rb") as vid_file, open(audio_path, "rb") as audio_file:
                files = {
                    "video": (os.path.basename(video_path), vid_file, "video/mp4"),
                    "audio": (os.path.basename(audio_path), audio_file, "audio/wav")
                }
                result = self._request("POST", "/wav2lip/enhance", files=files)
                return result.get("output_path")
        except Exception as e:
            print(f"Error enhancing lip sync: {e}")
            return None
    
    def add_gestures(
        self,
        video_path: str,
        gesture_type: str = "natural",
        intensity: float = 0.5
    ) -> Optional[str]:
        """Add gestures to a talking head video."""
        try:
            with open(video_path, "rb") as vid_file:
                files = {"video": (os.path.basename(video_path), vid_file, "video/mp4")}
                data = {"gesture_type": gesture_type, "intensity": str(intensity)}
                result = self._request("POST", "/gesture/add", files=files, data=data)
                return result.get("output_path")
        except Exception as e:
            print(f"Error adding gestures: {e}")
            return None
    
    def run_demo(
        self,
        image_path: str,
        script: str,
        output_dir: str = "output",
        emotion: str = "happy",
        voice: str = "default"
    ) -> Dict[str, str]:
        """Run a complete demo with all processing steps."""
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Generate speech
        print("\n🔊 Generating speech from text...")
        audio_path, duration = self.generate_speech(script, voice=voice)
        if not audio_path:
            raise Exception("Failed to generate speech")
        results["audio"] = audio_path
        print(f"✅ Speech generated: {audio_path}")
        
        # Step 2: Create talking head
        print("\n🎭 Generating talking head video...")
        video_path = self.create_talking_head(image_path, audio_path, emotion=emotion)
        if not video_path:
            raise Exception("Failed to generate talking head")
        results["talking_head"] = video_path
        print(f"✅ Talking head video created: {video_path}")
        
        # Step 3: Enhance lip sync
        print("\n👄 Enhancing lip sync...")
        enhanced_path = self.enhance_lip_sync(video_path, audio_path)
        if enhanced_path:
            results["enhanced"] = enhanced_path
            print(f"✅ Lip sync enhanced: {enhanced_path}")
            video_path = enhanced_path
        
        # Step 4: Add gestures
        print("\n👋 Adding natural gestures...")
        gesture_path = self.add_gestures(video_path, gesture_type="natural")
        if gesture_path:
            results["with_gestures"] = gesture_path
            print(f"✅ Gestures added: {gesture_path}")
        
        return results

def main():
    print("=== PaksaTalker Integrated Demo ===\n")
    
    # Initialize client
    client = PaksaTalkerDemo()
    
    # Check available models
    print("Checking available models...")
    models = client.check_models()
    if not models:
        print("❌ Failed to check model status. Is the API server running?")
        return
    
    print("\nAvailable models:")
    for model, status in models.items():
        print(f"- {model}: {'✅ Ready' if status else '❌ Not available'}")
    
    # Get input files
    image_path = input("\n📷 Enter path to image file: ").strip('\"')
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    script = """Hello! I'm your AI assistant, created by PaksaTalker. 
    I can generate realistic talking head videos from just a photo and some text. 
    This demo shows how I can bring your images to life with natural speech and gestures."""
    
    use_custom_script = input("\n📝 Use custom script? (y/n): ").lower() == 'y'
    if use_custom_script:
        print("Enter your script (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        script = "\n".join(lines)
    
    # Select emotion
    emotions = ["happy", "sad", "angry", "surprised", "neutral"]
    print("\n🎭 Select emotion:")
    for i, e in enumerate(emotions, 1):
        print(f"{i}. {e.capitalize()}")
    
    try:
        emotion_idx = int(input("Enter number (1-5): ").strip()) - 1
        emotion = emotions[emotion_idx]
    except (ValueError, IndexError):
        print("Invalid selection, using 'happy'")
        emotion = "happy"
    
    # Run the demo
    print("\n🚀 Starting PaksaTalker demo...")
    try:
        timestamp = int(time.time())
        output_dir = OUTPUT_DIR / f"demo_{timestamp}"
        
        results = client.run_demo(
            image_path=image_path,
            script=script,
            output_dir=str(output_dir),
            emotion=emotion
        )
        
        print("\n🎉 Demo completed successfully!")
        print("\nGenerated files:")
        for name, path in results.items():
            print(f"- {name}: {path}")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")

if __name__ == "__main__":
    main()
