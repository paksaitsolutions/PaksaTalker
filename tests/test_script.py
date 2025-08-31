# This is a simple test script to verify Python execution
import sys
import os

def main():
    # Create a test file in the current directory
    test_file = 'test_output.txt'
    
    try:
        # Write to a file
        with open(test_file, 'w') as f:
            f.write('Test successful!\n')
            f.write(f'Python executable: {sys.executable}\n')
            f.write(f'Python version: {sys.version}\n')
            f.write(f'Current working directory: {os.getcwd()}\n')
            f.write('Environment variables:\n')
            for key in os.environ:
                f.write(f'  {key}={os.environ[key]}\n')
        
        # Read the file back
        with open(test_file, 'r') as f:
            print(f.read())
            
        # Clean up
        os.remove(test_file)
        return 0
        
    except Exception as e:
        print(f'Error: {str(e)}', file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
