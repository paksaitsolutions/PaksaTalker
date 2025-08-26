"""Development server script to run both frontend and backend."""
import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time

def run_development():
    """Run both frontend and backend in development mode."""
    root_dir = Path(__file__).parent
    frontend_dir = root_dir / "frontend"
    
    # Start the backend server
    print("Starting backend server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--reload", "--port", "8000"],
        cwd=str(root_dir)
    )
    
    # Start the frontend development server
    print("Starting frontend development server...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", "5173"],
        cwd=str(frontend_dir),
        shell=True
    )
    
    # Open the browser after a short delay
    time.sleep(3)
    webbrowser.open("http://localhost:5173")
    
    try:
        # Keep the script running until interrupted
        print("Development servers are running. Press Ctrl+C to stop.")
        print("Frontend: http://localhost:5173")
        print("Backend API: http://localhost:8000/api")
        print("API Docs: http://localhost:8000/api/docs")
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Servers stopped.")

if __name__ == "__main__":
    run_development()
