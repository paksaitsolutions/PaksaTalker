"""Build script for the frontend application."""
import shutil
import subprocess
from pathlib import Path

def build_frontend():
    """Build the frontend and copy files to the correct location."""
    print("Building frontend...")
    
    # Paths
    root_dir = Path(__file__).parent
    frontend_dir = root_dir / "frontend"
    dist_dir = frontend_dir / "dist"
    backend_static_dir = root_dir / "static"
    
    try:
        # Install frontend dependencies
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True)
        
        # Build the frontend
        print("Running frontend build...")
        subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True)
        
        # Ensure backend static directory exists
        backend_static_dir.mkdir(exist_ok=True)
        
        # Copy built files to backend static directory
        print(f"Copying files to {backend_static_dir}...")
        if backend_static_dir.exists():
            shutil.rmtree(backend_static_dir)
        shutil.copytree(dist_dir, backend_static_dir)
        
        print("Frontend build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building frontend: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    build_frontend()
