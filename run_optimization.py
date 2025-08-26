"""
Run optimization and benchmarking for PaksaTalker models.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import optimization and benchmarking scripts
from scripts.optimize_specific import ModelOptimizer
from scripts.benchmark_models import ModelBenchmark, MODEL_CONFIGS

def optimize_models(
    model_names: List[str],
    device: Optional[str] = None,
    output_dir: str = 'optimized_models'
) -> Dict[str, Any]:
    """Optimize the specified models."""
    optimizer = ModelOptimizer(device=device)
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Optimizing {model_name}...")
        print(f"{'='*80}")
        
        try:
            report = optimizer.optimize(model_name, output_dir=output_dir)
            results[model_name] = report
            
            print(f"\nOptimization complete for {model_name}:")
            print(f"- Original size: {report['original_size_mb']:.2f} MB")
            print(f"- Optimized size: {report['optimized_size_mb']:.2f} MB")
            print(f"- Size reduction: {report['size_reduction_pct']:.1f}%")
            print(f"- Optimizations applied: {', '.join(report['optimizations_applied'])}")
            
        except Exception as e:
            print(f"Error optimizing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def benchmark_models(
    model_names: List[str],
    device: Optional[str] = None,
    output_file: str = 'benchmark_results.json'
) -> Dict[str, Any]:
    """Benchmark the specified models."""
    benchmark = ModelBenchmark(device=device)
    
    for model_name in model_names:
        if model_name in MODEL_CONFIGS:
            print(f"\n{'='*80}")
            print(f"Benchmarking {model_name}...")
            print(f"{'='*80}")
            
            try:
                benchmark.benchmark_model(model_name, MODEL_CONFIGS[model_name])
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    benchmark.save_results(output_file)
    return benchmark.results

def generate_report(
    optimization_results: Dict[str, Any],
    benchmark_results: Dict[str, Any],
    output_file: str = 'optimization_report.md'
) -> None:
    """Generate a markdown report with optimization and benchmark results."""
    report = ["# PaksaTalker Optimization Report\n"]
    
    # Add summary section
    report.append("## Summary\n")
    report.append("| Model | Original Size (MB) | Optimized Size (MB) | Reduction | Avg Inference Time (ms) | FPS |")
    report.append("|-------|-------------------|---------------------|-----------|-------------------------|-----|")
    
    for model_name, opt_result in optimization_results.items():
        bm_result = benchmark_results.get(model_name, {})
        report.append(
            f"| {model_name} | "
            f"{opt_result['original_size_mb']:.2f} | "
            f"{opt_result['optimized_size_mb']:.2f} | "
            f"{opt_result['size_reduction_pct']:.1f}% | "
            f"{bm_result.get('avg_time_ms', 'N/A'):.2f} | "
            f"{bm_result.get('fps', 'N/A'):.2f} |"
        )
    
    # Add detailed sections for each model
    for model_name, opt_result in optimization_results.items():
        bm_result = benchmark_results.get(model_name, {})
        
        report.append(f"\n## {model_name.capitalize()}\n")
        
        # Optimization details
        report.append("### Optimization Details\n")
        report.append(f"- **Original size:** {opt_result['original_size_mb']:.2f} MB")
        report.append(f"- **Optimized size:** {opt_result['optimized_size_mb']:.2f} MB")
        report.append(f"- **Size reduction:** {opt_result['size_reduction_pct']:.1f}%")
        report.append(f"- **Optimizations applied:** {', '.join(opt_result['optimizations_applied'])}\n")
        
        # Benchmark results
        if bm_result:
            report.append("### Benchmark Results\n")
            report.append(f"- **Device:** {bm_result.get('device', 'N/A')}")
            report.append(f"- **Batch size:** {bm_result.get('batch_size', 'N/A')}")
            report.append(f"- **Average inference time:** {bm_result.get('avg_time_ms', 'N/A'):.2f} ms")
            report.append(f"- **Throughput:** {bm_result.get('fps', 'N/A'):.2f} FPS")
            
            if bm_result.get('memory_allocated_mb'):
                report.append(f"- **GPU Memory allocated:** {bm_result['memory_allocated_mb']:.2f} MB")
                report.append(f"- **GPU Memory reserved:** {bm_result['memory_reserved_mb']:.2f} MB")
    
    # Write report to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport generated: {output_file}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Optimize and benchmark PaksaTalker models.')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=['sadtalker'],
                        choices=['sadtalker', 'wav2lip', 'gesture', 'qwen', 'all'],
                        help='Models to optimize and benchmark')
    
    # Device selection
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='optimized_models',
                        help='Directory to save optimized models')
    parser.add_argument('--benchmark-file', type=str, default='benchmark_results.json',
                        help='File to save benchmark results')
    parser.add_argument('--report-file', type=str, default='optimization_report.md',
                        help='File to save the optimization report')
    
    # Action selection
    parser.add_argument('--optimize', action='store_true',
                        help='Run optimization')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmarking')
    parser.add_argument('--all', action='store_true',
                        help='Run both optimization and benchmarking')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Handle 'all' models case
    if 'all' in args.models:
        model_names = ['sadtalker', 'wav2lip', 'gesture', 'qwen']
    else:
        model_names = args.models
    
    # Set default action if none specified
    if not (args.optimize or args.benchmark or args.all):
        args.optimize = True
        args.benchmark = True
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run optimization if requested
    optimization_results = {}
    if args.optimize or args.all:
        optimization_results = optimize_models(
            model_names=model_names,
            device=args.device,
            output_dir=args.output_dir
        )
    
    # Run benchmarking if requested
    benchmark_results = {}
    if args.benchmark or args.all:
        benchmark_results = benchmark_models(
            model_names=model_names,
            device=args.device,
            output_file=args.benchmark_file
        )
    
    # Generate report if we have both optimization and benchmark results
    if (args.optimize or args.all) and (args.benchmark or args.all):
        generate_report(
            optimization_results=optimization_results,
            benchmark_results=benchmark_results,
            output_file=args.report_file
        )

if __name__ == "__main__":
    main()
