"""
Benchmark script for PaksaTalker models.
"""

import time
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sadtalker import SadTalkerModel
from models.wav2lip import Wav2LipModel
from models.gesture import GestureModel
from models.qwen import QwenModel

# Model configurations
MODEL_CONFIGS = {
    'sadtalker': {
        'class': SadTalkerModel,
        'input_shape': (1, 3, 256, 256),  # batch, channels, height, width
        'dtype': torch.float32,
        'warmup_iters': 3,
        'benchmark_iters': 10
    },
    'wav2lip': {
        'class': Wav2LipModel,
        'input_shape': (1, 6, 96, 96),  # batch, channels, height, width
        'dtype': torch.float32,
        'warmup_iters': 3,
        'benchmark_iters': 10
    },
    'gesture': {
        'class': GestureModel,
        'input_shape': (1, 3, 256, 256),  # batch, channels, height, width
        'dtype': torch.float32,
        'warmup_iters': 3,
        'benchmark_iters': 10
    },
    'qwen': {
        'class': QwenModel,
        'input_shape': (1, 512),  # batch, sequence_length
        'dtype': torch.long,
        'warmup_iters': 1,
        'benchmark_iters': 5
    }
}

class ModelBenchmark:
    """Benchmark class for PaksaTalker models."""
    
    def __init__(self, device=None):
        """Initialize the benchmark.
        
        Args:
            device: Device to run benchmark on (default: 'cuda' if available, else 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def benchmark_model(self, model_name, config):
        """Benchmark a single model."""
        print(f"\n{'='*80}")
        print(f"Benchmarking {model_name} on {self.device.upper()}")
        print(f"{'='*80}")
        
        # Initialize model
        model_class = config['class']
        model = model_class().to(self.device).eval()
        
        # Create random input
        if config['dtype'] == torch.long:
            # For language models with integer input
            input_tensor = torch.randint(
                0, 10000, 
                config['input_shape'], 
                dtype=config['dtype'], 
                device=self.device
            )
        else:
            # For vision models with float input
            input_tensor = torch.randn(
                *config['input_shape'], 
                dtype=config['dtype'], 
                device=self.device
            )
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in tqdm(range(config['warmup_iters'])):
                _ = model(input_tensor)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        with torch.no_grad():
            for _ in tqdm(range(config['benchmark_iters'])):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(input_tensor)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            'model': model_name,
            'device': self.device,
            'batch_size': config['input_shape'][0],
            'iterations': config['benchmark_iters'],
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps': 1000 / np.mean(times) * config['input_shape'][0],
            'memory_allocated_mb': torch.cuda.memory_allocated(self.device) / (1024 ** 2) if self.device == 'cuda' else 0,
            'memory_reserved_mb': torch.cuda.memory_reserved(self.device) / (1024 ** 2) if self.device == 'cuda' else 0
        }
        
        # Print results
        self._print_results(stats)
        
        # Save results
        self.results[model_name] = stats
        return stats
    
    def _print_results(self, stats):
        """Print benchmark results."""
        print("\nBenchmark Results:")
        print("-" * 50)
        print(f"Model:           {stats['model']}")
        print(f"Device:          {stats['device'].upper()}")
        print(f"Batch size:      {stats['batch_size']}")
        print(f"Iterations:      {stats['iterations']}")
        print(f"Average time:    {stats['avg_time_ms']:.2f} ms")
        print(f"Std dev:         {stats['std_time_ms']:.2f} ms")
        print(f"Min time:        {stats['min_time_ms']:.2f} ms")
        print(f"Max time:        {stats['max_time_ms']:.2f} ms")
        print(f"Throughput:      {stats['fps']:.2f} FPS")
        
        if self.device == 'cuda':
            print(f"GPU Memory:      {stats['memory_allocated_mb']:.2f} MB allocated")
            print(f"                {stats['memory_reserved_mb']:.2f} MB reserved")
        
        print("-" * 50 + "\n")
    
    def save_results(self, filename='benchmark_results.json'):
        """Save benchmark results to a JSON file."""
        import json
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(x) for x in obj]
            return obj
        
        results = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark PaksaTalker models.')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'sadtalker', 'wav2lip', 'gesture', 'qwen'],
                        help='Model to benchmark (default: all)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run benchmark on (default: cuda if available, else cpu)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for benchmark results (default: benchmark_results.json)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize benchmark
    benchmark = ModelBenchmark(device=args.device)
    
    # Determine which models to benchmark
    if args.model == 'all':
        models_to_benchmark = MODEL_CONFIGS.keys()
    else:
        models_to_benchmark = [args.model]
    
    # Run benchmarks
    for model_name in models_to_benchmark:
        if model_name in MODEL_CONFIGS:
            benchmark.benchmark_model(model_name, MODEL_CONFIGS[model_name])
    
    # Save results
    benchmark.save_results(args.output)

if __name__ == "__main__":
    main()
