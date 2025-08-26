"""
Model Optimization Module
========================

This module provides utilities for optimizing the performance of PaksaTalker's AI models.
It includes techniques for model quantization, mixed precision training, and inference optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import logging
import os
import time

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Handles optimization of AI models for better performance."""
    
    def __init__(self, device: str = None):
        """Initialize the model optimizer.
        
        Args:
            device: The device to use for optimization (e.g., 'cuda', 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.supported_precisions = ['float32', 'float16', 'bfloat16']
        
    def optimize_model(
        self, 
        model: nn.Module, 
        optimization_config: Dict[str, Any]
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply optimizations to a model based on the provided configuration.
        
        Args:
            model: The PyTorch model to optimize
            optimization_config: Dictionary containing optimization settings
            
        Returns:
            Tuple of (optimized_model, optimization_report)
        """
        optimization_report = {
            'original_size': self._get_model_size_mb(model),
            'optimizations_applied': [],
            'performance_metrics': {}
        }
        
        # Apply optimizations in a specific order
        if optimization_config.get('quantize', False):
            model = self._quantize_model(model, optimization_config.get('quantization_config', {}))
            optimization_report['optimizations_applied'].append('quantization')
        
        if optimization_config.get('use_mixed_precision', False):
            model = self._enable_mixed_precision(model, optimization_config.get('precision', 'float16'))
            optimization_report['optimizations_applied'].append(f'mixed_precision_{optimization_config.get("precision", "float16")}')
        
        if optimization_config.get('fuse_conv_bn', False):
            model = self._fuse_conv_bn(model)
            optimization_report['optimizations_applied'].append('fuse_conv_bn')
        
        if optimization_config.get('optimize_for_inference', False):
            model = self._optimize_for_inference(model)
            optimization_report['optimizations_applied'].append('inference_optimization')
        
        # Update report
        optimization_report['optimized_size'] = self._get_model_size_mb(model)
        optimization_report['size_reduction'] = (
            optimization_report['original_size'] - optimization_report['optimized_size']
        ) / optimization_report['original_size'] * 100
        
        return model, optimization_report
    
    def _quantize_model(
        self, 
        model: nn.Module, 
        config: Dict[str, Any]
    ) -> nn.Module:
        """Apply quantization to the model.
        
        Args:
            model: The model to quantize
            config: Quantization configuration
            
        Returns:
            Quantized model
        """
        quant_type = config.get('quantization_type', 'dynamic')
        dtype = getattr(torch, config.get('dtype', 'qint8'))
        
        try:
            if quant_type == 'dynamic':
                return torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=dtype
                )
            elif quant_type == 'static':
                # Static quantization requires calibration
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                model = torch.quantization.prepare(model, inplace=False)
                # TODO: Add calibration step with representative dataset
                model = torch.quantization.convert(model, inplace=False)
                return model
            else:
                logger.warning(f"Unsupported quantization type: {quant_type}")
                return model
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
            return model
    
    def _enable_mixed_precision(
        self, 
        model: nn.Module, 
        precision: str = 'float16'
    ) -> nn.Module:
        """Enable mixed precision training/inference.
        
        Args:
            model: The model to optimize
            precision: The precision to use ('float16' or 'bfloat16')
            
        Returns:
            Model with mixed precision enabled
        """
        if precision not in self.supported_precisions[1:]:  # Skip float32
            logger.warning(f"Unsupported precision: {precision}, using float16")
            precision = 'float16'
        
        if precision == 'float16':
            model = model.half()
        elif precision == 'bfloat16' and hasattr(torch, 'bfloat16'):
            model = model.bfloat16()
        
        return model
    
    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """Fuse Conv + BatchNorm layers for better performance."""
        def _fuse_conv_bn_recursive(module):
            for child_name, child in module.named_children():
                if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    # Check if previous layer is a conv layer
                    for name, m in module.named_children():
                        if m is child and hasattr(module, name + '_conv'):
                            # Fuse conv + bn
                            fused_conv = torch.quantization.fuse_conv_bn(
                                getattr(module, name + '_conv'),
                                child
                            )
                            setattr(module, name + '_conv', fused_conv)
                            delattr(module, name)
                            break
                else:
                    _fuse_conv_bn_recursive(child)
        
        _fuse_conv_bn_recursive(model)
        return model
    
    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply optimizations specifically for inference."""
        model.eval()
        
        # Enable inference mode (PyTorch 1.9+)
        if hasattr(torch, 'inference_mode'):
            model.forward = torch.inference_mode()(model.forward)
        
        # Enable cudnn benchmarking for fixed input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # JIT compile the model if possible
        try:
            with torch.no_grad():
                model = torch.jit.script(model)
                logger.info("Model successfully compiled with TorchScript")
        except Exception as e:
            logger.warning(f"Could not compile model with TorchScript: {e}")
        
        return model
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get the size of the model in megabytes."""
        # Save model to a temporary file to get size
        temp_path = "temp_model.pt"
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb


def optimize_model_for_device(
    model: nn.Module,
    device_type: str = None,
    precision: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Convenience function to optimize a model for a specific device.
    
    Args:
        model: The model to optimize
        device_type: Target device ('cuda', 'cpu', or None for auto-detect)
        precision: Target precision ('float32', 'float16', 'bfloat16')
        
    Returns:
        Tuple of (optimized_model, optimization_report)
    """
    device = device_type or ('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = ModelOptimizer(device)
    
    # Default optimization config
    config = {
        'quantize': device == 'cpu',  # Quantize for CPU
        'use_mixed_precision': device == 'cuda',  # Use mixed precision for GPU
        'precision': precision or ('float16' if device == 'cuda' else 'float32'),
        'fuse_conv_bn': True,
        'optimize_for_inference': True,
        'quantization_config': {
            'quantization_type': 'dynamic',
            'dtype': 'qint8'
        }
    }
    
    return optimizer.optimize_model(model, config)


def benchmark_model(
    model: nn.Module, 
    input_shape: Tuple[int, ...],
    device: str = None,
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, Any]:
    """Benchmark the model's inference performance.
    
    Args:
        model: The model to benchmark
        input_shape: Shape of input tensor (batch size should be 1)
        device: Device to run benchmark on
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        
    Returns:
        Dictionary with benchmark results
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    # Create random input
    input_tensor = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs * 1000  # Convert to milliseconds
    
    return {
        'device': device,
        'batch_size': input_shape[0],
        'total_time_ms': total_time * 1000,
        'avg_inference_time_ms': avg_time,
        'fps': num_runs / total_time if total_time > 0 else float('inf'),
        'memory_allocated_mb': torch.cuda.memory_allocated(device) / (1024 * 1024) if device == 'cuda' else 0,
        'memory_reserved_mb': torch.cuda.memory_reserved(device) / (1024 * 1024) if device == 'cuda' else 0
    }
