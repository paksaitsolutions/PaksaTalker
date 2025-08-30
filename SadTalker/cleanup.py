"""
Cleanup script for SadTalker folder.
This script helps identify and remove unnecessary files while keeping essential ones.
"""
import os
import shutil
from pathlib import Path

# Define essential files and directories to keep
ESSENTIAL_FILES = [
    # Core files
    'PaksaTalker/',
    'checkpoints/',
    'src/',
    'gfpgan/',
    'inference.py',
    'requirements.txt',
    'launcher.py',
    'webui.bat',
    'webui.sh',
    'README.md',
    'LICENSE',
    
    # Config files
    'cog.yaml',
    'configs/',
    
    # Documentation
    'docs/',
]

# Define patterns of files/directories to remove
REMOVE_PATTERNS = [
    # Cache and temporary files
    '__pycache__',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '.DS_Store',
    '*.swp',
    '*.swo',
    '*.egg-info',
    '.ipynb_checkpoints',
    
    # Unnecessary directories
    'examples/',
    'results/',
    'sadtalker_env/',
    'sadtalker_fresh/',
    'scripts/',
    'test/',
    'tests/',
    'temp/',
    'tmp/',
    'venv/',
    '.vscode/',
    '.idea/',
    
    # Large files that can be redownloaded
    '*.pth',
    '*.pth.tar',
    '*.safetensors',
    '*.dat',
    '*.zip',
    '*.tar.gz',
    '*.tar',
    '*.tgz',
]

def is_essential(file_path):
    """Check if a file or directory is essential"""
    rel_path = str(Path(file_path).relative_to(BASE_DIR))
    return any(rel_path.startswith(essential) for essential in ESSENTIAL_FILES)

def should_remove(file_path):
    """Check if a file or directory should be removed"""
    if is_essential(file_path):
        return False
        
    rel_path = str(Path(file_path).relative_to(BASE_DIR))
    return any(pattern in rel_path for pattern in REMOVE_PATTERNS)

def get_directory_size(path):
    """Get the total size of a directory in bytes"""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

def format_size(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def clean_directory(directory):
    """Clean up the specified directory"""
    total_saved = 0
    removed_items = []
    
    print("\n=== Cleaning Directory ===")
    print(f"Directory: {directory}\n")
    
    # First, identify all files and directories to remove
    for root, dirs, files in os.walk(directory, topdown=False):
        # Check directories
        for name in dirs:
            path = os.path.join(root, name)
            if should_remove(path):
                size = get_directory_size(path) if os.path.isdir(path) else os.path.getsize(path)
                total_saved += size
                removed_items.append((path, size, 'dir'))
        
        # Check files
        for name in files:
            path = os.path.join(root, name)
            if should_remove(path):
                size = os.path.getsize(path)
                total_saved += size
                removed_items.append((path, size, 'file'))
    
    # Sort by size (largest first)
    removed_items.sort(key=lambda x: x[1], reverse=True)
    
    # Print what will be removed
    print("The following items will be removed:")
    print("-" * 80)
    print(f"{'Path':<80} {'Type':<6} {'Size':>10}")
    print("-" * 80)
    
    for path, size, item_type in removed_items:
        rel_path = os.path.relpath(path, directory)
        print(f"{rel_path:<80} {item_type:<6} {format_size(size):>10}")
    
    print("-" * 80)
    print(f"Total space to be saved: {format_size(total_saved)}\n")
    
    # Ask for confirmation
    confirm = input("Do you want to proceed with the cleanup? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cleanup cancelled.")
        return
    
    # Perform the actual cleanup
    print("\nRemoving files and directories...")
    for path, _, item_type in removed_items:
        try:
            if item_type == 'dir':
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"Removed: {os.path.relpath(path, directory)}")
        except Exception as e:
            print(f"Error removing {path}: {e}")
    
    print("\nCleanup completed successfully!")
    print(f"Total space saved: {format_size(total_saved)}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    clean_directory(BASE_DIR)
