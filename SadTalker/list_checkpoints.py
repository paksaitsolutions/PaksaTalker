import os

def main():
    path = r"D:\\SadTalker\\checkpoints"
    print(f"Contents of {path}:")
    for item in os.listdir(path):
        print(f"- {item}")

if __name__ == "__main__":
    main()
