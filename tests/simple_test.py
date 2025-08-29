import sys
import os

def main():
    # Print to console
    print("This is a test script from PaksaTalker")
    print("Python executable:", sys.executable)
    
    # Create a test file
    with open('test_output.txt', 'w') as f:
        f.write("Test successful!\n")
        f.write(f"Python executable: {sys.executable}\n")
        f.write(f"Current directory: {os.getcwd()}\n")
    
    print("Test file created successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
