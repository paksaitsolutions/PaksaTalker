"""
PaksaTalker Quickstart
=====================

This script demonstrates how to use the PaksaTalker API to generate talking head videos.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY")

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_auth_headers():
    return {
        "Authorization": f"Bearer {API_KEY}" if API_KEY else "",
        "Content-Type": "application/json"
    }

def test_connection():
    """Test the API connection."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            headers=get_auth_headers()
        )
        response.raise_for_status()
        print("✅ Successfully connected to PaksaTalker API")
        print(f"📊 Status: {response.json()['status']}")
        print(f"📦 Version: {response.json()['version']}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to PaksaTalker API: {e}")
        return False

def generate_video(image_path, text, emotion="happy"):
    """Generate a talking head video from an image and text."""
    url = f"{API_BASE_URL}/generate/video"
    
    try:
        with open(image_path, "rb") as img_file:
            files = {"image": img_file}
            data = {
                "text": text,
                "emotion": emotion,
                "format": "mp4"
            }
            
            response = requests.post(
                url,
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error generating video: {e}")
        if hasattr(e, 'response') and e.response:
            print("Response content:", e.response.text)
        raise

def main():
    print("=== PaksaTalker Quickstart ===\n")
    
    # Test connection
    if not test_connection():
        return
    
    # Example usage
    image_path = input("Enter path to image file: ")
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    text = input("Enter text to speak: ")
    
    print("\nGenerating video...")
    try:
        result = generate_video(image_path, text)
        print(f"\n✅ Video generated successfully!")
        print(f"📁 Output file: {result.get('output_path')}")
        print(f"🔗 Video URL: {result.get('url', 'N/A')}")
    except Exception as e:
        print(f"\n❌ Failed to generate video: {e}")

if __name__ == "__main__":
    main()
