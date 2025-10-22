"""
JAX GPU Setup for HPFRACC Library
Automatically configures JAX to use GPU when available.
"""

import os
import warnings
from typing import Optional


def setup_jax_gpu() -> bool:
    """
    Set up JAX to use GPU when available.

    This function should be called at the beginning of any HPFRACC script
    that uses JAX to ensure optimal performance.

    Returns:
        bool: True if GPU is available and configured, False if using CPU fallback
    """
    try:
        import jax

        # Set environment variables to prefer GPU
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'

        # Check if GPU is available
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(
            d).lower() or 'cuda' in str(d).lower()]

        if gpu_devices:
            print(f"✅ JAX GPU detected: {gpu_devices}")
            return True
        else:
            # Silent fallback to CPU - no warning needed
            return False

    except Exception as e:
        warnings.warn(f"Failed to configure JAX GPU: {e}")
        return False


def get_jax_info() -> dict:
    """
    Get JAX device information.

    Returns:
        dict: JAX device and backend information
    """
    try:
        import jax
        devices = jax.devices()

        return {
            'version': jax.__version__,
            'devices': [str(d) for d in devices],
            'device_count': len(devices),
            'backend': jax.default_backend(),
            'gpu_available': any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        }
    except Exception as e:
        return {'error': str(e)}


# Auto-configure JAX on import
_jax_gpu_available = setup_jax_gpu()

# Export the configuration status
JAX_GPU_AVAILABLE = _jax_gpu_available
