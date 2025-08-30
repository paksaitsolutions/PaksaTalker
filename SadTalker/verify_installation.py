"""
Verification script for SadTalker installation.
Checks for required files, directories, and dependencies.
"""
import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

# Required Python packages
REQUIRED_PACKAGES = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'numpy>=1.19.0',
    'opencv-python>=4.5.0',
    'tqdm>=4.64.0',
    'scipy>=1.7.0',
    'facexlib>=0.3.0',
    'gfpgan>=1.3.8',
    'gradio>=3.16.0',
    'ffmpeg-python>=0.2.0',
    'av>=9.2.0',
    'imageio>=2.9.0',
    'imageio-ffmpeg>=0.4.2',
    'librosa>=0.8.1',
    'yacs>=0.1.8',
    'scikit-image>=0.18.0',
    'kornia>=0.6.0',
    'transformers>=4.26.0',
    'safetensors>=0.2.7',
    'basicsr>=1.4.2',
    'facexlib>=0.3.0',
    'gfpgan>=1.3.8',
]

# Required directories and files
REQUIRED_DIRS = [
    'PaksaTalker',
    'checkpoints',
    'configs',
    'gfpgan',
    'src',
]

REQUIRED_FILES = [
    'inference.py',
    'launcher.py',
    'webui.bat',
    'webui.sh',
    'requirements.txt',
    'checkpoints/SadTalker_V0.0.2_256.safetensors',
    'checkpoints/SadTalker_V0.0.2_512.safetensors',
    'checkpoints/auido2exp_00300-model.pth',
    'checkpoints/auido2pose_00140-model.pth',
    'checkpoints/epoch_20.pth',
    'checkpoints/facevid2vid_00189-model.pth.tar',
    'checkpoints/mapping_00109-model.pth.tar',
    'checkpoints/mapping_00229-model.pth.tar',
    'checkpoints/shape_predictor_68_face_landmarks.dat',
    'checkpoints/wav2lip.pth',
]

def check_python_version():
    """Check Python version"""
    print("\n=== Python Version ===")
    print(f"Python {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_packages():
    """Check if required Python packages are installed"""
    print("\n=== Checking Python Packages ===")
    
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            pkg_name = package.split('>=')[0].split('<=')[0].split('==')[0].strip()
            pkg_resources.require(package)
            print(f"✅ {pkg_name} is installed")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
            print(f"❌ {package} is not installed or version is incompatible")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nMissing or incompatible packages. Install them with:")
        print(f"pip install {' '.join(missing_packages)}\n")
        return False
    
    print("\n✅ All required packages are installed")
    return True

def check_files_and_dirs():
    """Check if required files and directories exist"""
    print("\n=== Checking Files and Directories ===")
    
    all_ok = True
    
    # Check directories
    for dir_path in REQUIRED_DIRS:
        if os.path.isdir(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            all_ok = False
    
    # Check files
    for file_path in REQUIRED_FILES:
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File exists: {file_path} ({file_size:.2f} MB)")
        else:
            print(f"❌ File missing: {file_path}")
            all_ok = False
    
    return all_ok

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible"""
    print("\n=== Checking FFmpeg ===")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg is installed: {version_line}")
            return True
        else:
            print("❌ FFmpeg is not properly installed")
            return False
            
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH")
        print("Download FFmpeg from: https://ffmpeg.org/download.html")
        return False

def main():
    """Main function to run all checks"""
    print("\n" + "="*50)
    print("SadTalker Installation Verification")
    print("="*50)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Python Packages", check_packages()),
        ("Files and Directories", check_files_and_dirs()),
        ("FFmpeg Installation", check_ffmpeg()),
    ]
    
    print("\n" + "="*50)
    print("Verification Summary:")
    print("="*50)
    
    all_passed = all(passed for _, passed in checks)
    
    for check_name, passed in checks:
        status = "PASSED" if passed else "FAILED"
        print(f"{check_name}: {'✅' if passed else '❌'} {status}")
    
    if all_passed:
        print("\n✅ All checks passed! Your SadTalker installation is complete.")
    else:
        print("\n❌ Some checks failed. Please address the issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
