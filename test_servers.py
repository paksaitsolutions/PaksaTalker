#!/usr/bin/env python3
"""
Simple test script to verify servers are accessible
"""
import requests
import time
import sys

def test_backend():
    """Test if backend server is running"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running at http://localhost:8000")
            return True
        else:
            print(f"❌ Backend server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend server is not running at http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def test_frontend():
    """Test if frontend server is running"""
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend server is running at http://localhost:5173")
            return True
        else:
            print(f"❌ Frontend server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Frontend server is not running at http://localhost:5173")
        return False
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")
        return False

def main():
    print("🔍 Testing PaksaTalker servers...")
    print()
    
    backend_ok = test_backend()
    frontend_ok = test_frontend()
    
    print()
    if backend_ok and frontend_ok:
        print("🎉 All servers are running correctly!")
        print("📱 Open http://localhost:5173 in your browser")
        print("🔧 API available at http://localhost:8000")
        print("📚 API docs at http://localhost:8000/api/docs")
    else:
        print("⚠️  Some servers are not running. Please start them first:")
        if not backend_ok:
            print("   - Start backend: python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
        if not frontend_ok:
            print("   - Start frontend: cd frontend && npm run dev")
        print("   - Or use: python run_dev.py")
    
    return 0 if (backend_ok and frontend_ok) else 1

if __name__ == "__main__":
    sys.exit(main())