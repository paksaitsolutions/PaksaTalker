import sys
print("Python version:", sys.version)
print("\nPython path:", sys.path)

try:
    import numpy
    print("\n✅ NumPy version:", numpy.__version__)
except ImportError:
    print("\n❌ NumPy not installed")

try:
    import torch
    print("✅ PyTorch version:", torch.__version__)
    print("   CUDA available:", torch.cuda.is_available())
except ImportError:
    print("❌ PyTorch not installed")
