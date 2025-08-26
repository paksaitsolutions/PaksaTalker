"""
Cleanup script for PaksaTalker.
This script helps clean up temporary files and verify the installation.
"""

import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Directories to clean
TEMP_DIRS = [
    os.path.join(BASE_DIR, 'temp'),
    os.path.join(BASE_DIR, 'output'),
    os.path.join(BASE_DIR, '__pycache__'),
    os.path.join(BASE_DIR, 'models', '__pycache__'),
    os.path.join(BASE_DIR, 'utils', '__pycache__'),
    os.path.join(BASE_DIR, 'integrations', '__pycache__'),
    os.path.join(BASE_DIR, 'api', '__pycache__'),
    os.path.join(BASE_DIR, 'config', '__pycache__'),
]

# Files to clean
TEMP_FILES = [
    os.path.join(BASE_DIR, '*.pyc'),
    os.path.join(BASE_DIR, '*.pyo'),
    os.path.join(BASE_DIR, '*.pyd'),
    os.path.join(BASE_DIR, '*.py,cover'),
    os.path.join(BASE_DIR, '*.so'),
    os.path.join(BASE_DIR, '*.c'),
    os.path.join(BASE_DIR, '*.html'),
    os.path.join(BASE_DIR, '*.log'),
]

def clean_directories():
    """Clean up temporary directories."""
    print("Cleaning directories...")
    for dir_path in TEMP_DIRS:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")

def clean_files():
    """Clean up temporary files."""
    import glob
    print("\nCleaning files...")
    for pattern in TEMP_FILES:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def verify_installation():
    """Verify the installation and directory structure."""
    print("\nVerifying installation...")
    
    # Check required directories
    required_dirs = [
        'models',
        'integrations',
        'utils',
        'api',
        'config'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n[WARNING] Missing directories: {', '.join(missing_dirs)}")
    else:
        print("✓ All required directories exist")
    
    # Check required files
    required_files = [
        'app.py',
        'config/config.py',
        'models/__init__.py',
        'models/base.py',
        'integrations/__init__.py',
        'integrations/base.py',
        'utils/__init__.py',
        'api/__init__.py',
        'api/routes.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(BASE_DIR, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n[WARNING] Missing files: {', '.join(missing_files)}")
    else:
        print("✓ All required files exist")
    
    # Check for duplicate config.py
    if os.path.exists(os.path.join(BASE_DIR, 'config.py')) and \
       os.path.exists(os.path.join(BASE_DIR, 'config', 'config.py')):
        print("\n[WARNING] Duplicate config.py found. Please keep only one in the config/ directory.")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    print("PaksaTalker Cleanup Utility\n" + "=" * 50)
    
    clean_directories()
    clean_files()
    verify_installation()
    
    print("\nCleanup complete!")
    print("You can now safely delete other model files and directories not in the PaksaTalker folder.")
