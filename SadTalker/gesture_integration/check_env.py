import sys
import platform
import os

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
    print_section("Checking Imports")
    
    def try_import(name):
        try:
            __import__(name)
            print(f"✅ {name}")
            return True
        except ImportError:
            print(f"❌ {name} (Not installed)")
            return False
    
    # Core dependencies
    try_import("numpy")
    try_import("torch")
    try_import("cv2")
    try_import("yaml")
    
    # SadTalker specific
    try:
        sys.path.append("D:/SadTalker")
        __import__("src.facerender.animate")
        print("✅ SadTalker (facerender.animate)")
    except ImportError as e:
        print(f"❌ SadTalker: {str(e)}")

def main():
    print("Environment Check")
    print("="*50)
    check_python()
    check_imports()
    print("\nCheck complete. Look for ❌ to identify missing dependencies.")

if __name__ == "__main__":
    main()
