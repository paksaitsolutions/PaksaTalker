"""
Simple test script for model optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model_size_mb(model):
    """Get the size of the model in MB."""
    temp_path = "temp_model.pt"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb

def main():
    print("Testing model optimization...")
    
    # Create a test model
    model = TestModel()
    model.eval()
    
    # Get original size
    original_size = get_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Apply dynamic quantization
    print("\nApplying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # Get quantized size
    quantized_size = get_model_size_mb(quantized_model)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(original_size - quantized_size) / original_size * 100:.1f}%")
    
    # Test inference
    print("\nTesting inference...")
    input_tensor = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = quantized_model(input_tensor)
    print("Output shape:", output.shape)
    print("Optimization test completed successfully!")

if __name__ == "__main__":
    main()
