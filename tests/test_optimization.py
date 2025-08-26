"""
Test script for model optimization functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.optimization import ModelOptimizer, optimize_model_for_device, benchmark_model

# Test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 8 * 8, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Fixtures
@pytest.fixture
def test_model():
    return TestModel()

@pytest.fixture
def optimizer():
    return ModelOptimizer()

def test_model_optimizer_init(optimizer):
    """Test ModelOptimizer initialization."""
    assert optimizer is not None
    assert hasattr(optimizer, 'device')
    assert hasattr(optimizer, 'supported_precisions')

def test_quantization(optimizer, test_model):
    '''Test model quantization.'''
    config = {
        'quantize': True,
        'quantization_config': {
            'quantization_type': 'dynamic',
            'dtype': 'qint8'
        }
    }
    
    optimized_model, report = optimizer.optimize_model(test_model, config)
    
    assert optimized_model is not None
    assert 'quantization' in report['optimizations_applied']
    assert report['optimized_size'] < report['original_size']

def test_mixed_precision(optimizer, test_model):
    '''Test mixed precision optimization.'''
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for mixed precision testing")
    
    config = {
        'use_mixed_precision': True,
        'precision': 'float16'
    }
    
    optimized_model, report = optimizer.optimize_model(test_model, config)
    
    assert optimized_model is not None
    assert 'mixed_precision_float16' in report['optimizations_applied']
    
    # Check if model is using half precision
    for param in optimized_model.parameters():
        assert param.dtype == torch.float16

def test_conv_bn_fusion(optimizer, test_model):
    '''Test Conv+BN fusion.'''
    config = {
        'fuse_conv_bn': True
    }
    
    optimized_model, report = optimizer.optimize_model(test_model, config)
    
    assert optimized_model is not None
    assert 'fuse_conv_bn' in report['optimizations_applied']
    
    # Check if BN layers are fused
    for name, module in optimized_model.named_modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

def test_optimize_for_device_cpu():
    '''Test optimization for CPU.'''
    model = TestModel()
    optimized_model, report = optimize_model_for_device(model, device_type='cpu')
    
    assert optimized_model is not None
    assert report['optimizations_applied']  # Should have some optimizations applied
    assert report['optimized_size'] <= report['original_size']

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_optimize_for_device_cuda():
    '''Test optimization for CUDA.'''
    model = TestModel()
    optimized_model, report = optimize_model_for_device(model, device_type='cuda')
    
    assert optimized_model is not None
    assert 'mixed_precision_float16' in report['optimizations_applied']
    assert report['optimized_size'] <= report['original_size']

def test_benchmark_model():
    '''Test model benchmarking.'''
    model = TestModel()
    input_shape = (1, 3, 32, 32)  # batch_size, channels, height, width
    
    results = benchmark_model(
        model=model,
        input_shape=input_shape,
        device='cpu',
        num_warmup=2,
        num_runs=5
    )
    
    assert 'avg_inference_time_ms' in results
    assert 'fps' in results
    assert results['avg_inference_time_ms'] > 0
    assert results['fps'] > 0

if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
