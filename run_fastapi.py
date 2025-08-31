import uvicorn
import os
from pathlib import Path

# Set up paths
current_dir = Path(__file__).parent.absolute()
backend_dir = current_dir / 'backend'
app_path = 'app.main:app'

print(f"Current directory: {current_dir}")
print(f"Backend directory: {backend_dir}")
print(f"App path: {app_path}")

# Check if backend directory exists
if not backend_dir.exists():
    print(f"Error: Backend directory not found at {backend_dir}")
    exit(1)

# Add backend directory to Python path
import sys
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Try to import the app
try:
    from app.main import app
    print("Successfully imported FastAPI app")
except ImportError as e:
    print(f"Error importing FastAPI app: {e}")
    print("\nPython path:")
    for p in sys.path:
        print(f"- {p}")
    exit(1)

# Start the server
if __name__ == "__main__":
    print("\nStarting FastAPI server...")
    uvicorn.run(
        app_path,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir / 'app')],
        log_level="debug"
    )
