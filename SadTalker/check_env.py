import os
import sys

def main():
    print("Environment Check")
    print("=" * 50)
    
    # Print Python info
    print(f"Python {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check checkpoints directory
    checkpoints_path = os.path.join("D:\\", "SadTalker", "checkpoints")
    print(f"\nChecking directory: {checkpoints_path}")
    
    if not os.path.exists(checkpoints_path):
        print("Directory does not exist!")
        try:
            os.makedirs(checkpoints_path, exist_ok=True)
            print("Created checkpoints directory.")
        except Exception as e:
            print(f"Failed to create directory: {e}")
            return
    else:
        print("Directory exists!")
    
    # List files
    try:
        files = os.listdir(checkpoints_path)
        print(f"\nFound {len(files)} files in checkpoints directory:")
        for f in files:
            print(f"- {f}")
    except Exception as e:
        print(f"Error listing directory: {e}")

if __name__ == "__main__":
    main()
