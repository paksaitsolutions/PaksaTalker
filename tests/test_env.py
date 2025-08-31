import sys
import os

def main():
    with open('test_output.txt', 'w') as f:
        f.write(f"Python executable: {sys.executable}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Current working directory: {os.getcwd()}\n")
        f.write("Python path:\n")
        for p in sys.path:
            f.write(f"  {p}\n")
        
        f.write("\nEnvironment variables:\n")
        for key, value in os.environ.items():
            if 'PYTHON' in key.upper() or 'PATH' in key.upper():
                f.write(f"  {key}: {value}\n")

if __name__ == "__main__":
    main()
