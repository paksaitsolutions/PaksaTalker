import os

def main():
    path = r"D:\\SadTalker\\checkpoints"
    print(f"Contents of {path}:")
    if not os.path.exists(path):
        print("Directory does not exist!")
        return
        
    for item in os.listdir(path):
        print(f"- {item}")

if __name__ == "__main__":
    main()
