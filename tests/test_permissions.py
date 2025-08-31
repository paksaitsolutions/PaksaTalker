import os
import sys

def test_file_permissions():
    test_file = 'test_permission_file.txt'
    
    # Test file creation
    try:
        with open(test_file, 'w') as f:
            f.write('Test file content')
        print(f"Successfully created file: {os.path.abspath(test_file)}")
        
        # Test file reading
        with open(test_file, 'r') as f:
            content = f.read()
            print(f"Successfully read file content: {content}")
            
        # Test file deletion
        os.remove(test_file)
        print("Successfully deleted test file")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing file system permissions...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    if test_file_permissions():
        print("File system permissions test: PASSED")
    else:
        print("File system permissions test: FAILED")
