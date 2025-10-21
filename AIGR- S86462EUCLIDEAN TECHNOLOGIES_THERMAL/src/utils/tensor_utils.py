#!/usr/bin/env python3
"""
Tensor utilities for GPU acceleration and mixed precision training.
"""

import torch
import numpy as np
from torch.cuda.amp import autocast
from typing import Union, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def to_device(data: Union[torch.Tensor, Dict, list], device: torch.device) -> Union[torch.Tensor, Dict, list]:
    """
    Move data to specified device recursively.
    
    Args:
        data: Data to move (tensor, dict, or list)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    else:
        return data


def mixed_precision_context(enabled: bool = True):
    """
    Context manager for mixed precision training.
    
    Args:
        enabled: Whether to enable mixed precision
        
    Returns:
        Context manager
    """
    if enabled and torch.cuda.is_available():
        return autocast()
    else:
        return torch.no_grad()


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'total': total,
        'free': free
    }


def optimize_model_for_inference(model: torch.nn.Module, 
                                use_jit: bool = True,
                                channels_last: bool = True) -> torch.nn.Module:
    """
    Optimize model for inference performance.
    
    Args:
        model: PyTorch model
        use_jit: Whether to use JIT compilation
        channels_last: Whether to use channels_last memory format
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Convert to channels_last for better performance
    if channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    
    # JIT compilation
    if use_jit:
        try:
            # Create example input
            example_input = torch.randn(1, 1, 512, 512)
            if torch.cuda.is_available():
                example_input = example_input.cuda()
            if channels_last:
                example_input = example_input.to(memory_format=torch.channels_last)
            
            model = torch.jit.trace(model, example_input)
            logger.info("Model optimized with JIT compilation")
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
    
    return model


def benchmark_model(model: torch.nn.Module, 
                   input_shape: tuple = (1, 1, 512, 512),
                   num_runs: int = 100,
                   warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Performance statistics
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Create input tensor
    input_tensor = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize()
    
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'fps': float(1000 / np.mean(times)),
        'throughput': float(input_shape[0] * 1000 / np.mean(times))  # samples per second
    }


class TensorStats:
    """Utility class for tensor statistics."""
    
    @staticmethod
    def summarize_tensor(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Get comprehensive tensor statistics."""
        tensor_np = tensor.detach().cpu().numpy()
        
        return {
            'name': name,
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'min': float(tensor_np.min()),
            'max': float(tensor_np.max()),
            'mean': float(tensor_np.mean()),
            'std': float(tensor_np.std()),
            'memory_mb': tensor.numel() * tensor.element_size() / 1024**2,
            'requires_grad': tensor.requires_grad
        }
    
    @staticmethod
    def check_for_nans(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains NaN values."""
        has_nan = torch.isnan(tensor).any().item()
        if has_nan:
            logger.warning(f"NaN values detected in {name}")
        return has_nan
    
    @staticmethod
    def check_for_infs(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains infinite values."""
        has_inf = torch.isinf(tensor).any().item()
        if has_inf:
            logger.warning(f"Infinite values detected in {name}")
        return has_inf


def setup_cuda_optimization():
    """Setup CUDA optimizations for better performance."""
    if torch.cuda.is_available():
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Enable optimized attention (if available)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass
        
        logger.info("CUDA optimizations enabled")
    else:
        logger.warning("CUDA not available - optimizations skipped")


if __name__ == "__main__":
    # Test utilities
    print("Testing tensor utilities...")
    
    # Test device movement
    if torch.cuda.is_available():
        device = torch.device('cuda')
        tensor = torch.randn(10, 10)
        tensor_gpu = to_device(tensor, device)
        print(f"Tensor moved to: {tensor_gpu.device}")
        
        # Test memory usage
        memory = get_memory_usage()
        print(f"GPU memory: {memory}")
        
        # Setup optimizations
        setup_cuda_optimization()
        
    else:
        print("CUDA not available - using CPU")
    
    # Test tensor stats
    test_tensor = torch.randn(100, 100)
    stats = TensorStats.summarize_tensor(test_tensor, "test_tensor")
    print(f"Tensor stats: {stats}")
    
    print("Tensor utilities test completed!")