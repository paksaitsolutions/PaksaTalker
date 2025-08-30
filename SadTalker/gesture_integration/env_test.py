import sys
import platform
import importlib.metadata

def print_section(title):
    print("\n" + "="*50)
    print(f"{title:^50}")
    print("="*50)

def check_python():
    print_section("Python Environment")
    print(f"Python Version: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")

def check_imports():
    print_section("Required Packages")
    packages = [
        'torch', 'torchvision', 'torchaudio',
        'numpy', 'opencv-python', 'scipy',
        'librosa', 'soundfile', 'pyyaml'
    ]
    
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            print(f"✅ {pkg}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ {pkg}: Not installed")

def check_gpu():
    print_section("GPU Availability")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  No GPU available. Using CPU.")
    except Exception as e:
        print(f"Error checking GPU: {str(e)}")

def main():
    print("Environment Test for Gesture Integration")
    print("="*50)
    
    check_python()
    check_imports()
    check_gpu()
    
    print("\nTest completed. Check for any ❌ or ⚠️  warnings above.")

if __name__ == "__main__":
    main()
