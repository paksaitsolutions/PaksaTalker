"""Script to install frontend dependencies and build the frontend."""
import os
import subprocess
import sys
from pathlib import Path

def install_frontend():
    """Install frontend dependencies and build the frontend."""
    print("Setting up frontend...")
    
    # Paths
    root_dir = Path(__file__).parent
    frontend_dir = root_dir / "frontend"
    
    # Check if Node.js is installed
    try:
        # On Windows, we need to use shell=True for proper command resolution
        node_check = subprocess.run(
            "node --version",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        npm_check = subprocess.run(
            "npm --version",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Node.js {node_check.stdout.strip()} is installed")
        print(f"✓ npm {npm_check.stdout.strip()} is installed")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Error: Node.js and npm must be installed to set up the frontend.")
        print("Please install Node.js from https://nodejs.org/ and try again.")
        print("If you just installed Node.js, please restart your terminal and try again.")
        return False
    
    try:
        # Install frontend dependencies
        print("Installing frontend dependencies...")
        subprocess.run(
            "npm install",
            cwd=str(frontend_dir),
            shell=True,
            check=True
        )
        
        print("\nFrontend setup completed successfully!")
        print("You can now start the development servers with:")
        print("  python run_dev.py")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError setting up frontend: {e}")
        if e.stderr:
            print(e.stderr.decode())
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    if install_frontend():
        sys.exit(0)
    else:
        sys.exit(1)
