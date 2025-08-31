import requests

def test_servers():
    # Test backend (port 5000 for this project)
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        print(f"Backend (5000): {response.status_code}")
    except Exception as e:
        print(f"Backend not running at http://localhost:5000")
    
    # Test frontend
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        print(f"Frontend (5173): {response.status_code}")
    except Exception as e:
        print(f"Frontend not running at http://localhost:5173")

if __name__ == "__main__":
    test_servers()