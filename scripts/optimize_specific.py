"""
Optimize specific models with advanced techniques.
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sadtalker import SadTalkerModel
from models.wav2lip import Wav2LipModel
from models.gesture import GestureModel
from models.qwen import QwenModel

class QuantizedSadTalker(nn.Module):
    """Quantized version of SadTalker model."""
    
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)

class ModelOptimizer:
    """Advanced model optimizer with hardware-specific optimizations."""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.supported_models = {
            'sadtalker': {
                'class': SadTalkerModel,
                'optimizations': ['quantization', 'pruning', 'fusion'],
                'input_shape': (1, 3, 256, 256)
            },
            'wav2lip': {
                'class': Wav2LipModel,
                'optimizations': ['quantization', 'pruning'],
                'input_shape': (1, 6, 96, 96)
            },
            'gesture': {
                'class': GestureModel,
                'optimizations': ['quantization', 'pruning'],
                'input_shape': (1, 3, 256, 256)
            },
            'qwen': {
                'class': QwenModel,
                'optimizations': ['quantization'],
                'input_shape': (1, 512)
            }
        }
    
    def optimize(self, model_name: str, output_dir: str = 'optimized_models') -> Dict[str, Any]:
        """Optimize a specific model with all available optimizations."""
        if model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model_config = self.supported_models[model_name]
        model = model_config['class']().to(self.device).eval()
        
        # Create output directory
        output_path = Path(output_dir) / model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Apply optimizations
        optimized_model = model
        optimizations_applied = []
        
        # 1. Apply model fusion (combine conv+bn, etc.)
        if 'fusion' in model_config['optimizations']:
            optimized_model = self._fuse_model(optimized_model)
            optimizations_applied.append('fusion')
        
        # 2. Apply quantization
        if 'quantization' in model_config['optimizations']:
            optimized_model = self._quantize_model(optimized_model, model_config['input_shape'])
            optimizations_applied.append('quantization')
        
        # 3. Apply pruning (if supported)
        if 'pruning' in model_config['optimizations']:
            optimized_model = self._prune_model(optimized_model, amount=0.2)  # Prune 20% of weights
            optimizations_applied.append('pruning')
        
        # Save the optimized model
        model_path = output_path / f"{model_name}_optimized.pt"
        torch.save(optimized_model.state_dict(), model_path)
        
        # Generate optimization report
        original_size = self._get_model_size_mb(model)
        optimized_size = self._get_model_size_mb(optimized_model)
        
        report = {
            'model': model_name,
            'device': self.device,
            'optimizations_applied': optimizations_applied,
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'size_reduction_pct': (1 - (optimized_size / original_size)) * 100,
            'model_path': str(model_path)
        }
        
        # Save report
        report_path = output_path / f"{model_name}_optimization_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _fuse_model(self, model: nn.Module) -> nn.Module:
        """Fuse Conv+BN layers for better performance."""
        # This is a simplified example - real implementation would need to handle model-specific architectures
        if hasattr(model, 'fuse_model'):
            model.fuse_model()
        return model
    
    def _quantize_model(
        self, 
        model: nn.Module, 
        input_shape: tuple,
        quant_dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """Apply dynamic quantization to the model."""
        # Skip quantization if already quantized
        if hasattr(model, 'is_quantized') and model.is_quantized:
            return model
        
        # For demonstration, we'll use dynamic quantization
        # In a real scenario, you might want to use static quantization with calibration
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=quant_dtype
        )
        
        return quantized_model
    
    def _prune_model(
        self, 
        model: nn.Module, 
        amount: float = 0.2
    ) -> nn.Module:
        """Apply pruning to the model."""
        # Skip if already pruned
        if hasattr(model, 'is_pruned') and model.is_pruned:
            return model
        
        # Simple global pruning example
        # In practice, you might want to use a more sophisticated pruning strategy
        parameters_to_prune = [
            (module, 'weight') 
            for module in model.modules() 
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        
        if parameters_to_prune:
            torch.nn.utils.prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=amount
            )
        
        return model
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get the size of the model in MB."""
        temp_path = "temp_model.pt"
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb

def parse_args():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize PaksaTalker models.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['sadtalker', 'wav2lip', 'gesture', 'qwen'],
                        help='Model to optimize')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='optimized_models',
                        help='Directory to save optimized models')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    print(f"Optimizing {args.model} on {args.device or 'default device'}...")
    
    optimizer = ModelOptimizer(device=args.device)
    report = optimizer.optimize(args.model, output_dir=args.output_dir)
    
    print("\nOptimization complete!")
    print(f"Model: {report['model']}")
    print(f"Device: {report['device']}")
    print(f"Original size: {report['original_size_mb']:.2f} MB")
    print(f"Optimized size: {report['optimized_size_mb']:.2f} MB")
    print(f"Size reduction: {report['size_reduction_pct']:.1f}%")
    print(f"Optimizations applied: {', '.join(report['optimizations_applied'])}")
    print(f"\nOptimized model saved to: {report['model_path']}")

if __name__ == "__main__":
    main()
