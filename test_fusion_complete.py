#!/usr/bin/env python3
"""
Complete Fusion Settings Test - Tests all frontend parameters
"""
import requests
import json
import os
from pathlib import Path

def test_fusion_complete():
    """Test fusion endpoint with all frontend parameters"""
    
    url = "http://localhost:8001/api/v1/generate/fusion-video"
    
    # Create test image
    test_image = Path("test_avatar.jpg")
    if not test_image.exists():
        import cv2, numpy as np
        img = np.ones((512, 512, 3), dtype=np.uint8) * 240
        cv2.circle(img, (256, 256), 200, (200, 200, 200), -1)
        cv2.circle(img, (200, 200), 30, (50, 50, 50), -1)
        cv2.circle(img, (312, 200), 30, (50, 50, 50), -1)
        cv2.ellipse(img, (256, 320), (80, 40), 0, 0, 180, (50, 50, 50), 5)
        cv2.imwrite(str(test_image), img)
    
    # Test all frontend parameters
    data = {
        # Core parameters
        'prompt': 'Hello! This is a complete test of PaksaTalker fusion mode with all settings.',
        'resolution': '720p',
        'fps': '30',
        
        # Fusion-specific
        'emotion': 'happy',
        'style': 'enthusiastic',
        'preferWav2Lip2': 'false',
        'useEmage': 'false',  # Set to false to avoid 503 if not fully configured
        'requireEmage': 'false',
        
        # Background processing
        'backgroundMode': 'blur',
        'backgroundColor': '#0066cc',
        
        # Expression engine
        'expressionEngine': 'auto'
    }
    
    files = {
        'image': ('test_avatar.jpg', open(test_image, 'rb'), 'image/jpeg')
    }
    
    try:
        print("🧪 Testing Fusion Endpoint with Complete Settings...")
        print(f"📡 URL: {url}")
        print(f"📋 Parameters: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, data=data, files=files, timeout=30)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        try:
            result = response.json()
            print(f"📝 Response Body: {json.dumps(result, indent=2)}")
        except:
            print(f"📝 Response Text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Fusion endpoint working with all parameters!")
                task_id = result.get('task_id')
                if task_id:
                    print(f"🎯 Task ID: {task_id}")
                    print("🔄 You can check status at: /api/v1/status/" + task_id)
                return True
            else:
                print("❌ Fusion endpoint returned success=false")
                return False
        else:
            print(f"❌ Fusion endpoint failed with status {response.status_code}")
            return False
        
    except requests.exceptions.ConnectionError:
        print("❌ Server not running on port 8001")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        files['image'][1].close()

def test_capabilities():
    """Test capabilities endpoint"""
    try:
        response = requests.get("http://localhost:8001/api/v1/capabilities")
        if response.status_code == 200:
            caps = response.json()
            print("🔧 Backend Capabilities:")
            if caps.get('success') and caps.get('data', {}).get('models'):
                models = caps['data']['models']
                for model, status in models.items():
                    status_icon = "✅" if status else "❌"
                    print(f"   {status_icon} {model}: {status}")
            return True
    except Exception as e:
        print(f"❌ Capabilities check failed: {e}")
    return False

if __name__ == "__main__":
    print("🚀 PaksaTalker Fusion Complete Integration Test")
    print("=" * 50)
    
    # Test capabilities first
    test_capabilities()
    print()
    
    # Test fusion endpoint
    test_fusion_complete()