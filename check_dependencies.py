"""Script to check if all required dependencies are installed."""
import sys
import subprocess


def check_node_installed():
    """Check if Node.js and npm are installed."""
    try:
        node_version = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True
        )
        npm_version = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True
        )
        
        print(f"✓ Node.js {node_version.stdout.strip()} is installed")
        print(f"✓ npm {npm_version.stdout.strip()} is installed")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Node.js and/or npm is not installed")
        print("  Please install Node.js from https://nodejs.org/")
        return False

def check_python_dependencies():
    """Check if required Python packages are installed."""
    required = ["fastapi", "uvicorn", "python-dotenv", "python-multipart"]
    missing = []
    
    for package in required:
        try:
            __import__(package.split('[')[0] if '[' in package else package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            missing.append(package)
    
    if missing:
        print("\nTo install missing Python packages, run:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def main():
    """Main function to check all dependencies."""
    print("Checking system dependencies...\n")
    
    node_ok = check_node_installed()
    print()
    python_ok = check_python_dependencies()
    
    if node_ok and python_ok:
        print("\n✓ All dependencies are installed!")
        print("You can now start the development servers with:")
        print("  python run_dev.py")
        return True
    else:
        print("\n✗ Some dependencies are missing. Please install them and try again.")
        return False

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
