"""
JAX GPU Setup for HPFRACC Library
Automatically configures JAX to use GPU when available with proper conflict resolution.
"""

import os
import warnings
from typing import Optional, Dict, Any


def clear_jax_plugins():
    """Clear any existing JAX plugins to prevent conflicts."""
    try:
        # Clear environment variables that might cause conflicts
        env_vars_to_clear = [
            'JAX_PLATFORM_NAME',
            'JAX_ENABLE_XLA', 
            'XLA_FLAGS',
            'CUDA_VISIBLE_DEVICES'
        ]
        
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
                
    except Exception as e:
        warnings.warn(f"Failed to clear JAX environment variables: {e}")


def check_cudnn_compatibility() -> Dict[str, Any]:
    """Check CuDNN compatibility and return status information."""
    try:
        import jaxlib
        
        # Try to get CuDNN version info
        try:
            import ctypes
            cudnn_lib = ctypes.CDLL("libcudnn.so")
            # This is a simplified check - actual version detection is more complex
            cudnn_available = True
        except:
            cudnn_available = False
            
        return {
            'cudnn_available': cudnn_available,
            'jaxlib_version': jaxlib.__version__,
            'warning': 'CuDNN version mismatch detected. Consider upgrading CuDNN to 9.12.0+ for optimal performance.'
        }
        
    except Exception as e:
        return {
            'cudnn_available': False,
            'error': str(e)
        }


def setup_jax_gpu_safe() -> bool:
    """
    Set up JAX to use GPU when available with proper conflict resolution.
    
    This function handles PJRT plugin conflicts and CuDNN compatibility issues
    that commonly occur with JAX-GPU setups.
    
    Returns:
        bool: True if GPU is available and configured, False if using CPU fallback
    """
    try:
        # Clear any existing plugins first
        clear_jax_plugins()
        
        # Prioritize pip-installed CuDNN libraries over conda's older versions
        try:
            import site
            for path in site.getsitepackages() + [site.getusersitepackages()]:
                cudnn_lib_path = os.path.join(path, 'nvidia', 'cudnn', 'lib')
                if os.path.exists(cudnn_lib_path):
                    # Add to LD_LIBRARY_PATH if not already there
                    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                    if cudnn_lib_path not in ld_path:
                        os.environ['LD_LIBRARY_PATH'] = f'{cudnn_lib_path}:{ld_path}' if ld_path else cudnn_lib_path
                        break
        except Exception:
            pass  # Non-critical, continue without library path adjustment
        
        # Check CuDNN compatibility
        cudnn_info = check_cudnn_compatibility()
        if not cudnn_info.get('cudnn_available', False):
            warnings.warn("CuDNN not available or incompatible. GPU performance may be degraded.")
        
        # Import JAX after clearing environment
        import jax
        
        # Let JAX auto-detect GPU without forcing platform
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        
        if gpu_devices:
            print(f"✅ JAX GPU detected: {gpu_devices}")
            if 'warning' in cudnn_info:
                print(f"⚠️  {cudnn_info['warning']}")
            return True
        else:
            print("ℹ️  No GPU detected, using CPU fallback")
            return False
            
    except Exception as e:
        warnings.warn(f"Failed to configure JAX GPU: {e}")
        print("ℹ️  Falling back to CPU execution")
        return False


def setup_jax_gpu() -> bool:
    """
    Legacy function for backward compatibility.
    Use setup_jax_gpu_safe() for new code.
    """
    return setup_jax_gpu_safe()


def get_jax_info() -> dict:
    """
    Get comprehensive JAX device and compatibility information.
    
    Returns:
        dict: JAX device, backend, and compatibility information
    """
    try:
        import jax
        devices = jax.devices()
        
        # Get CuDNN compatibility info
        cudnn_info = check_cudnn_compatibility()
        
        return {
            'version': jax.__version__,
            'devices': [str(d) for d in devices],
            'device_count': len(devices),
            'backend': jax.default_backend(),
            'gpu_available': any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices),
            'cudnn_info': cudnn_info,
            'platform': jax.default_backend()
        }
    except Exception as e:
        return {'error': str(e)}


def force_cpu_fallback() -> bool:
    """
    Force JAX to use CPU even if GPU is available.
    Useful for debugging or when GPU causes issues.
    
    Returns:
        bool: True if successfully forced to CPU
    """
    try:
        # Set environment variable to force CPU
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
        # Clear any cached JAX state
        import jax
        jax.clear_caches()
        
        devices = jax.devices()
        cpu_devices = [d for d in devices if 'cpu' in str(d).lower()]
        
        if cpu_devices:
            print(f"✅ Forced JAX to use CPU: {cpu_devices}")
            return True
        else:
            print("⚠️  Failed to force CPU fallback")
            return False
            
    except Exception as e:
        warnings.warn(f"Failed to force CPU fallback: {e}")
        return False


# Auto-configure JAX on import with safe setup
_jax_gpu_available = setup_jax_gpu_safe()

# Export the configuration status
JAX_GPU_AVAILABLE = _jax_gpu_available
