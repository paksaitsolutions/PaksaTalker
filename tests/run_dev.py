#!/usr/bin/env python3
"""
Development server runner for PaksaTalker
Starts both frontend and backend servers
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def run_backend():
    """Start the FastAPI backend server"""
    print("Starting backend server...")
    try:
        return subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], cwd=Path(__file__).parent)
    except Exception as e:
        print(f"Failed to start backend: {e}")
        return None

def run_frontend():
    """Start the Vite frontend server"""
    print("Starting frontend server...")
    frontend_dir = Path(__file__).parent / "frontend"
    try:
        return subprocess.Popen([
            "npm", "run", "dev"
        ], cwd=frontend_dir)
    except Exception as e:
        print(f"Failed to start frontend: {e}")
        return None

def main():
    """Main function to start both servers"""
    print("🚀 Starting PaksaTalker Development Servers...")
    
    # Start backend
    backend_process = run_backend()
    if not backend_process:
        print("❌ Failed to start backend server")
        return 1
    
    # Wait a moment for backend to start
    time.sleep(2)
    
    # Start frontend
    frontend_process = run_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend server")
        backend_process.terminate()
        return 1
    
    print("✅ Both servers started successfully!")
    print("📱 Frontend: http://localhost:5173")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/api/docs")
    print("\nPress Ctrl+C to stop both servers...")
    
    try:
        # Wait for processes
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("Backend process ended")
                break
            if frontend_process.poll() is not None:
                print("Frontend process ended")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for graceful shutdown
        backend_process.wait(timeout=5)
        frontend_process.wait(timeout=5)
        
        print("✅ Servers stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())