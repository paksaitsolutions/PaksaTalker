"""
Optimize the SadTalker model for better performance.
"""

import torch
from models.sadtalker import SadTalkerModel
from models.optimization import optimize_model_for_device

def main():
    print("Loading SadTalker model...")
    model = SadTalkerModel()
    
    # Set to evaluation mode
    model.eval()
    
    print("Optimizing model for CUDA...")
    optimized_model, report = optimize_model_for_device(
        model,
        device_type='cuda',
        precision='float16'  # Use mixed precision for better performance
    )
    
    # Print optimization report
    print("\nOptimization Report:")
    print(f"Original size: {report['original_size']:.2f} MB")
    print(f"Optimized size: {report['optimized_size']:.2f} MB")
    print(f"Size reduction: {report['size_reduction']:.1f}%")
    print("Optimizations applied:", ", ".join(report['optimizations_applied']))
    
    # Save the optimized model
    output_path = "optimized_models/sadtalker_optimized.pt"
    torch.save(optimized_model.state_dict(), output_path)
    print(f"\nOptimized model saved to {output_path}")

if __name__ == "__main__":
    main()
