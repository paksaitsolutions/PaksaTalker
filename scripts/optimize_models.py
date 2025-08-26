"""
PaksaTalker Model Optimization Script
====================================

This script demonstrates how to optimize PaksaTalker's AI models for better performance.
It applies various optimization techniques and benchmarks the results.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models
from models.sadtalker import SadTalkerModel
from models.wav2lip import Wav2LipModel
from models.gesture import GestureModel
from models.qwen import QwenModel
from models.optimization import (
    ModelOptimizer, 
    optimize_model_for_device,
    benchmark_model
)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizerCLI:
    """Command-line interface for model optimization."""
    
    def __init__(self, output_dir: str = "optimized_models"):
        """Initialize the optimizer.
        
        Args:
            output_dir: Directory to save optimized models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = ModelOptimizer(device=self.device)
        
        # Default optimization configs
        self.configs = {
            'cpu': {
                'quantize': True,
                'use_mixed_precision': False,
                'precision': 'float32',
                'fuse_conv_bn': True,
                'optimize_for_inference': True,
                'quantization_config': {
                    'quantization_type': 'dynamic',
                    'dtype': 'qint8'
                }
            },
            'cuda': {
                'quantize': False,  # Quantization may hurt performance on GPU
                'use_mixed_precision': True,
                'precision': 'float16',
                'fuse_conv_bn': True,
                'optimize_for_inference': True
            },
            'tensorrt': {
                'quantize': True,
                'use_mixed_precision': True,
                'precision': 'float16',
                'fuse_conv_bn': True,
                'optimize_for_inference': True,
                'quantization_config': {
                    'quantization_type': 'static',
                    'dtype': 'qint8'
                }
            }
        }
    
    def load_model(self, model_name: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load a model by name."""
        model_map = {
            'sadtalker': (SadTalkerModel, {
                'input_shape': (1, 3, 256, 256),  # Example input shape
                'description': 'SadTalker talking head model'
            }),
            'wav2lip': (Wav2LipModel, {
                'input_shape': (1, 6, 96, 96),  # Example input shape
                'description': 'Wav2Lip lip-sync model'
            }),
            'gesture': (GestureModel, {
                'input_shape': (1, 3, 256, 256),  # Example input shape
                'description': 'Gesture generation model'
            }),
            'qwen': (QwenModel, {
                'input_shape': (1, 512),  # Example input shape
                'description': 'Qwen language model'
            })
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
        
        model_class, model_info = model_map[model_name]
        logger.info(f"Loading {model_info['description']}...")
        
        # Initialize model with default parameters
        model = model_class()
        model.eval()
        
        return model, model_info
    
    def optimize_model(
        self, 
        model: nn.Module, 
        config_name: str = 'cuda'
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Optimize a model using the specified configuration."""
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}. Available configs: {list(self.configs.keys())}")
        
        config = self.configs[config_name]
        logger.info(f"Optimizing model with {config_name} configuration...")
        
        return self.optimizer.optimize_model(model, config)
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        num_warmup: int = 10,
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """Benchmark model performance."""
        logger.info(f"Benchmarking model with input shape {input_shape}...")
        
        # Create a random input tensor
        input_tensor = torch.randn(input_shape, device=self.device)
        if self.device == 'cuda':
            model = model.cuda()
        
        # Run benchmark
        results = benchmark_model(
            model=model,
            input_shape=input_shape,
            device=self.device,
            num_warmup=num_warmup,
            num_runs=num_runs
        )
        
        # Print results
        print("\n" + "="*50)
        print(f"Benchmark Results ({self.device.upper()}):")
        print("-"*50)
        print(f"Average inference time: {results['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {results['fps']:.2f} FPS")
        if self.device == 'cuda':
            print(f"GPU Memory: {results['memory_allocated_mb']:.2f} MB allocated")
            print(f"GPU Memory: {results['memory_reserved_mb']:.2f} MB reserved")
        print("="*50 + "\n")
        
        return results
    
    def save_model(
        self, 
        model: nn.Module, 
        model_name: str, 
        config_name: str,
        optimization_report: Dict[str, Any]
    ) -> str:
        """Save the optimized model and its report."""
        # Create model directory
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{model_name}_{config_name}.pt"
        report_path = model_dir / f"{model_name}_{config_name}_report.json"
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Save optimization report
        with open(report_path, 'w') as f:
            json.dump(optimization_report, f, indent=2)
        
        logger.info(f"Saved optimized model to {model_path}")
        logger.info(f"Saved optimization report to {report_path}")
        
        return str(model_path)
    
    def run_optimization_pipeline(
        self, 
        model_name: str, 
        config_name: str = 'cuda',
        benchmark: bool = True
    ) -> Dict[str, Any]:
        """Run the complete optimization pipeline."""
        # Load model
        model, model_info = self.load_model(model_name)
        
        # Benchmark original model
        original_results = {}
        if benchmark:
            original_results = self.benchmark_model(
                model, 
                model_info['input_shape']
            )
        
        # Optimize model
        optimized_model, optimization_report = self.optimize_model(model, config_name)
        
        # Benchmark optimized model
        optimized_results = {}
        if benchmark:
            optimized_results = self.benchmark_model(
                optimized_model,
                model_info['input_shape']
            )
        
        # Save optimized model
        model_path = self.save_model(
            optimized_model,
            model_name,
            config_name,
            {
                'optimization_report': optimization_report,
                'original_benchmark': original_results,
                'optimized_benchmark': optimized_results,
                'device': self.device,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        
        return {
            'model_name': model_name,
            'config_name': config_name,
            'model_path': model_path,
            'optimization_report': optimization_report,
            'original_benchmark': original_results,
            'optimized_benchmark': optimized_results
        }

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Optimize PaksaTalker models for better performance.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['sadtalker', 'wav2lip', 'gesture', 'qwen', 'all'],
                        help='Model to optimize (or "all" for all models)')
    parser.add_argument('--config', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'tensorrt'],
                        help='Optimization configuration')
    parser.add_argument('--output-dir', type=str, default='optimized_models',
                        help='Directory to save optimized models')
    parser.add_argument('--no-benchmark', action='store_true',
                        help='Skip benchmarking')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda", "cpu")')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizerCLI(output_dir=args.output_dir)
    
    # Set device if specified
    if args.device:
        optimizer.device = args.device
    
    # Determine which models to optimize
    models_to_optimize = []
    if args.model == 'all':
        models_to_optimize = ['sadtalker', 'wav2lip', 'gesture', 'qwen']
    else:
        models_to_optimize = [args.model]
    
    # Optimize models
    results = {}
    for model_name in models_to_optimize:
        try:
            print(f"\n{'='*80}")
            print(f"Optimizing {model_name.upper()} with {args.config.upper()} configuration")
            print(f"{'='*80}\n")
            
            result = optimizer.run_optimization_pipeline(
                model_name=model_name,
                config_name=args.config,
                benchmark=not args.no_benchmark
            )
            results[model_name] = result
            
            # Print summary
            if 'optimization_report' in result and 'optimized_benchmark' in result:
                report = result['optimization_report']
                benchmark = result['optimized_benchmark']
                print(f"\n{'='*80}")
                print(f"Optimization Summary for {model_name.upper()}:")
                print(f"- Size reduction: {report.get('size_reduction', 0):.1f}%")
                print(f"- Optimizations applied: {', '.join(report.get('optimizations_applied', []))}")
                print(f"- Average inference time: {benchmark.get('avg_inference_time_ms', 0):.2f} ms")
                print(f"- Throughput: {benchmark.get('fps', 0):.2f} FPS")
                if 'memory_allocated_mb' in benchmark:
                    print(f"- GPU Memory: {benchmark['memory_allocated_mb']:.2f} MB")
                print(f"{'='*80}\n")
                
        except Exception as e:
            print(f"\n❌ Error optimizing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

if __name__ == "__main__":
    main()
