import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

# Set environment variables
os.environ["UVICORN_RELOAD"] = "true"
os.environ["UVICORN_RELOAD_DIR"] = project_root

# Import uvicorn after setting up the path
import uvicorn

if __name__ == "__main__":
    print("Starting backend server...")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        import backend.app.main
        print("Successfully imported backend.app.main")
    except Exception as e:
        print(f"Error importing backend.app.main: {e}")
        raise
    
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
        log_level="debug"
    )
