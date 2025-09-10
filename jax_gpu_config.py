"""
JAX GPU Configuration for HPFRACC Library
This module automatically configures JAX to use GPU when available,
with graceful fallback to CPU when GPU is not supported.
"""

import os
import warnings
from typing import Optional

def configure_jax_gpu() -> bool:
    """
    Configure JAX to use GPU when available.
    
    Returns:
        bool: True if GPU is available and configured, False if using CPU fallback
    """
    try:
        import jax
        
        # Set environment variables to prefer GPU
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        
        # Check if GPU is available
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if gpu_devices:
            print(f"✅ JAX GPU detected: {gpu_devices}")
            print(f"✅ JAX backend: {jax.default_backend()}")
            return True
        else:
            print("⚠️  JAX GPU not available, using CPU fallback")
            print(f"⚠️  JAX devices: {devices}")
            print(f"⚠️  JAX backend: {jax.default_backend()}")
            return False
            
    except Exception as e:
        warnings.warn(f"Failed to configure JAX GPU: {e}")
        return False

def get_jax_device_info() -> dict:
    """
    Get information about available JAX devices.
    
    Returns:
        dict: Device information including count, types, and backend
    """
    try:
        import jax
        devices = jax.devices()
        
        return {
            'version': jax.__version__,
            'devices': devices,
            'device_count': len(devices),
            'backend': jax.default_backend(),
            'gpu_available': any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        }
    except Exception as e:
        return {'error': str(e)}

def test_jax_performance(size: int = 10000) -> dict:
    """
    Test JAX performance with a simple computation.
    
    Args:
        size: Size of the test array
        
    Returns:
        dict: Performance metrics
    """
    import time
    import jax.numpy as jnp
    from jax import random
    
    # Create test data
    key = random.PRNGKey(42)
    x = random.normal(key, shape=(size, size))
    
    # Time matrix multiplication
    start_time = time.time()
    result = jnp.dot(x, x)
    end_time = time.time()
    
    return {
        'computation_time': end_time - start_time,
        'array_size': size,
        'result_shape': result.shape,
        'device': str(result.device()) if hasattr(result, 'device') else 'unknown'
    }

if __name__ == "__main__":
    print("=== JAX GPU Configuration for HPFRACC ===")
    
    # Configure JAX
    gpu_available = configure_jax_gpu()
    
    # Get device info
    device_info = get_jax_device_info()
    print(f"\nDevice Info: {device_info}")
    
    # Test performance
    print("\n=== Performance Test ===")
    perf = test_jax_performance(1000)
    print(f"Performance: {perf}")
    
    if gpu_available:
        print("\n🎉 JAX is configured for GPU acceleration!")
    else:
        print("\n💡 JAX is using CPU fallback. GPU support will be enabled automatically when available.")
