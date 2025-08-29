#!/usr/bin/env python3
"""
Quick test script to verify API endpoints work correctly
"""
import requests
import json

def test_style_presets():
    """Test the style presets endpoint"""
    try:
        response = requests.get("http://localhost:8080/api/v1/style-presets")
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response (JSON):")
            print(json.dumps(data, indent=2))
            return True
        else:
            print("❌ API Response (Text):")
            print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Server not running on port 8080")
        return False
    except json.JSONDecodeError:
        print("❌ Response is not valid JSON")
        print("Response content:", response.text[:200])
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing PaksaTalker API endpoints...")
    print("=" * 50)
    
    success = test_style_presets()
    
    if success:
        print("\n✅ API is working correctly!")
        print("The 'Failed to load presets' error should be resolved.")
    else:
        print("\n❌ API is not working. Server needs to be restarted.")
        print("Please stop the server (Ctrl+C) and run 'python app.py' again.")