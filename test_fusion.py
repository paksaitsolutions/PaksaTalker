#!/usr/bin/env python3
"""
Test fusion endpoint
"""
import requests
import json

def test_fusion_endpoint():
    url = "http://localhost:8001/api/v1/generate/fusion-video"
    
    # Test with text prompt (no files)
    data = {
        'prompt': 'Hello, this is a test message',
        'resolution': '480p',
        'fps': '25'
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✓ Fusion endpoint working!")
                return True
        
        print("✗ Fusion endpoint failed")
        return False
        
    except requests.exceptions.ConnectionError:
        print("✗ Server not running on port 8001")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    test_fusion_endpoint()