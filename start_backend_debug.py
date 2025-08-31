import sys
import os
from pathlib import Path

# Print debug information
print("=== Starting Backend Server ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# Try to import the required modules
try:
    import uvicorn
    print("Successfully imported uvicorn")
    
    # Try to import the FastAPI app
    try:
        from backend.app.main import app
        print("Successfully imported FastAPI app")
        
        # Start the server
        print("\n=== Starting Uvicorn Server ===")
        print(f"Host: 0.0.0.0")
        print(f"Port: 8000")
        print("===========================\n")
        
        uvicorn.run(
            "backend.app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )
        
    except ImportError as e:
        print(f"\n=== ERROR: Failed to import FastAPI app ===")
        print(f"Error: {e}")
        print("\nChecking directory structure...")
        
        # Check if backend directory exists
        backend_dir = os.path.join(project_root, 'backend')
        if not os.path.exists(backend_dir):
            print(f"ERROR: 'backend' directory not found in {project_root}")
        else:
            print(f"Found backend directory: {backend_dir}")
            
            # List contents of backend directory
            print("\nContents of backend directory:")
            for item in os.listdir(backend_dir):
                item_path = os.path.join(backend_dir, item)
                print(f"- {item} ({'dir' if os.path.isdir(item_path) else 'file'})")
        
        # Try to find the main.py file
        print("\nSearching for main.py...")
        for root, dirs, files in os.walk(project_root):
            if 'main.py' in files:
                print(f"Found main.py at: {os.path.join(root, 'main.py')}")
        
        print("\nPlease check the directory structure and ensure all required files exist.")
        
except ImportError as e:
    print(f"\n=== ERROR: Failed to import required modules ===")
    print(f"Error: {e}")
    print("\nPlease make sure you have installed all required dependencies.")
    print("You can install them using: pip install -r requirements.txt")
    
    # Check if requirements.txt exists
    req_file = os.path.join(project_root, 'requirements.txt')
    if os.path.exists(req_file):
        print(f"\nContents of {req_file}:")
        with open(req_file, 'r') as f:
            print(f.read())
    else:
        print(f"\n{req_file} not found.")
