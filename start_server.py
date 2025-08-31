import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Now import and run uvicorn
import uvicorn

if __name__ == "__main__":
    print(f"Starting server from: {current_dir}")
    print(f"Python path: {sys.path}")
    
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(current_dir / "backend")]
    )
