"""
Centralized JAX Configuration for HPFRACC Library

This module provides a single point of JAX initialization that prevents
PJRT plugin conflicts, CuDNN warnings, and multiple initialization attempts.

Key features:
- Single initialization point (prevents PJRT conflicts)
- Graceful error handling for system-level issues
- Suppresses known warnings (CuDNN version mismatches)
- Thread-safe initialization
"""

import os
import sys
import warnings
import logging
from typing import Optional, Dict, Any
from threading import Lock

# Ensure os is available globally
assert os is not None

# Thread-safe initialization lock
_init_lock = Lock()
_jax_initialized = False
_jax_available = False
_jax_gpu_available = False
_jax_config = {}

# Suppress JAX/XLA warnings that are system-level issues
_original_warn = warnings.warn

def _suppress_jax_warnings(message, category, *args, **kwargs):
    """Suppress known JAX system-level warnings that users can't fix."""
    # Suppress PJRT plugin conflicts (system-level issue)
    if "PJRT_Api already exists" in str(message):
        return
    # Suppress CuDNN version mismatch warnings (system-level issue)
    if "CuDNN" in str(message) and "version" in str(message).lower():
        return
    # Suppress JAX plugin configuration errors
    if "Jax plugin configuration error" in str(message):
        return
    # Call original warning function for everything else
    _original_warn(message, category, *args, **kwargs)

# Redirect JAX internal logging to suppress noisy errors
class JAXErrorFilter(logging.Filter):
    """Filter out known JAX system-level errors."""
    
    def filter(self, record):
        msg = record.getMessage()
        # Suppress PJRT plugin conflicts
        if "PJRT_Api already exists" in msg:
            return False
        # Suppress CuDNN version warnings (all variations)
        if any(cudnn_term in msg for cudnn_term in [
            "CuDNN", "cuda_dnn", "Loaded runtime CuDNN",
            "but source was compiled with", "CuDNN library needs"
        ]):
            return False
        # Suppress plugin configuration errors
        if "Jax plugin configuration error" in msg:
            return False
        return True

# Apply filter to JAX loggers and set level to suppress warnings
try:
    _jax_logger = logging.getLogger('jax')
    _jax_logger.addFilter(JAXErrorFilter())
    _jax_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
    
    _xla_logger = logging.getLogger('jax._src.xla_bridge')
    _xla_logger.addFilter(JAXErrorFilter())
    _xla_logger.setLevel(logging.ERROR)
    
    # Also filter tensorflow logging (used by JAX internally)
    _tf_logger = logging.getLogger('absl')
    _tf_logger.setLevel(logging.ERROR)
except Exception:
    pass  # Logging setup may fail in some environments


def _suppress_jax_stderr():
    """Suppress JAX stderr output for known system-level issues."""
    import io
    import contextlib
    
    class JAXErrorSuppressor(io.TextIOWrapper):
        """Suppress specific JAX error messages."""
        
        def __init__(self, original):
            self._original = original
            super().__init__(original.buffer, encoding=original.encoding, 
                           errors=original.errors, line_buffering=original.line_buffering)
        
        def write(self, s):
            # Suppress known error messages
            if "PJRT_Api already exists" in s:
                return len(s)
            if "CuDNN" in s and ("version" in s.lower() or "library" in s.lower()):
                return len(s)
            if "Jax plugin configuration error" in s:
                return len(s)
            return self._original.write(s)
    
    return JAXErrorSuppressor


def initialize_jax_once(force_cpu: bool = False) -> Dict[str, Any]:
    """
    Initialize JAX once and only once, preventing PJRT conflicts.
    
    This function:
    1. Clears problematic environment variables
    2. Handles PJRT plugin conflicts gracefully
    3. Suppresses known system-level warnings
    4. Returns JAX availability status
    
    Args:
        force_cpu: If True, force CPU-only mode (avoids GPU issues)
        
    Returns:
        Dictionary with JAX initialization status
    """
    global _jax_initialized, _jax_available, _jax_gpu_available, _jax_config
    
    with _init_lock:
        if _jax_initialized:
            return _jax_config
        
        # Clear environment variables that cause conflicts
        env_vars_to_clear = [
            'JAX_PLATFORM_NAME',  # Don't set this - causes PJRT conflicts
            'JAX_ENABLE_XLA',
            'XLA_FLAGS',
        ]
        
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
        
        # Force CPU mode if requested (avoids GPU/CuDNN issues)
        if force_cpu:
            os.environ['JAX_PLATFORMS'] = 'cpu'
        
        # Suppress warnings during initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn = _suppress_jax_warnings
            
            try:
                # Suppress stderr for JAX initialization
                import contextlib
                import io
                
                # Try to import JAX with error suppression
                try:
                    # Redirect stderr temporarily to catch and filter errors
                    # Use a more aggressive approach that works with C-level stderr writes
                    old_stderr_fd = None
                    old_stderr = sys.stderr
                    
                    try:
                        # Create a pipe to filter stderr (os already imported at top of file)
                        read_fd, write_fd = os.pipe()
                        old_stderr_fd = os.dup(2)  # Save stderr file descriptor
                        os.dup2(write_fd, 2)  # Redirect stderr to pipe
                        
                        # Create a filter for stderr that suppresses JAX system-level errors
                        class FilteredStderr:
                            def __init__(self, original, read_fd):
                                self._original = original
                                self._read_fd = read_fd
                                self._buffer = b''
                            
                            def write(self, s):
                                if isinstance(s, bytes):
                                    self._buffer += s
                                else:
                                    self._buffer += s.encode('utf-8', errors='ignore')
                                
                                # Process buffer line by line
                                while b'\n' in self._buffer:
                                    line, self._buffer = self._buffer.split(b'\n', 1)
                                    line_str = line.decode('utf-8', errors='ignore')
                                    
                                    # Filter out known JAX system-level errors
                                    if not any(err in line_str for err in [
                                        "PJRT_Api already exists",
                                        "CuDNN",
                                        "cuda_dnn.cc",
                                        "Jax plugin configuration error",
                                        "Loaded runtime CuDNN library",
                                        "but source was compiled with",
                                        "CuDNN library needs to have matching",
                                        "DNN library initialization failed",
                                        "FAILED_PRECONDITION"
                                    ]):
                                        # Only write non-JAX-error messages
                                        self._original.write(line_str + '\n')
                                
                                return len(s) if isinstance(s, str) else len(s)
                            
                            def flush(self):
                                self._original.flush()
                            
                            def __getattr__(self, name):
                                return getattr(self._original, name)
                        
                        sys.stderr = FilteredStderr(old_stderr, read_fd)
                        
                    except Exception:
                        # Fallback to simpler filtering if pipe method fails
                        class FilteredStderr:
                            def __init__(self, original):
                                self._original = original
                            
                            def write(self, s):
                                # Filter out all known JAX system-level errors
                                if isinstance(s, str):
                                    if any(err in s for err in [
                                        "PJRT_Api already exists",
                                        "CuDNN",
                                        "cuda_dnn.cc",
                                        "Jax plugin configuration error",
                                        "Loaded runtime CuDNN library",
                                        "but source was compiled with",
                                        "CuDNN library needs to have matching",
                                        "DNN library initialization failed",
                                        "FAILED_PRECONDITION"
                                    ]):
                                        return len(s)
                                return self._original.write(s)
                            
                            def flush(self):
                                self._original.flush()
                            
                            def __getattr__(self, name):
                                return getattr(self._original, name)
                        
                        sys.stderr = FilteredStderr(old_stderr)
                    
                    try:
                        # Import JAX with error handling
                        try:
                            import jax
                            import jax.numpy as jnp
                        except Exception as import_error:
                            # If import fails due to DNN/CuDNN issues, force CPU mode and retry
                            error_str = str(import_error)
                            if any(err in error_str for err in [
                                "DNN library initialization failed",
                                "FAILED_PRECONDITION",
                                "CuDNN"
                            ]):
                                # Force CPU mode and retry
                                try:
                                    os.environ['JAX_PLATFORMS'] = 'cpu'
                                    import jax
                                    import jax.numpy as jnp
                                    import jax.config
                                    jax.config.update('jax_platform_name', 'cpu')
                                except:
                                    _jax_available = False
                                    _jax_gpu_available = False
                                    raise
                            else:
                                raise
                        
                        # Try to get devices (this is where PJRT conflicts occur)
                        try:
                            devices = jax.devices()
                            _jax_available = True
                            
                            # Check for GPU
                            gpu_devices = [d for d in devices 
                                         if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
                            _jax_gpu_available = len(gpu_devices) > 0
                            
                        except Exception as device_error:
                            error_str = str(device_error)
                            # PJRT conflict or other device error - continue with CPU
                            if "PJRT_Api already exists" in error_str:
                                # Plugin already registered - this is OK, JAX will use it
                                _jax_available = True
                                _jax_gpu_available = False
                                # Force CPU mode to avoid further conflicts
                                try:
                                    import jax.config
                                    jax.config.update('jax_platform_name', 'cpu')
                                except:
                                    pass
                            elif "DNN library initialization failed" in error_str:
                                # CuDNN/DNN initialization failed - use CPU mode
                                _jax_available = True
                                _jax_gpu_available = False
                                try:
                                    import jax.config
                                    jax.config.update('jax_platform_name', 'cpu')
                                except:
                                    pass
                            else:
                                # Other error - assume JAX is available but problematic
                                _jax_available = True
                                _jax_gpu_available = False
                        
                    finally:
                        # Restore stderr
                        sys.stderr = old_stderr
                        if old_stderr_fd is not None:
                            try:
                                os.dup2(old_stderr_fd, 2)  # Restore original stderr
                                os.close(old_stderr_fd)
                                os.close(write_fd)
                                os.close(read_fd)
                            except Exception:
                                pass
                        
                except ImportError:
                    _jax_available = False
                    _jax_gpu_available = False
                except Exception as e:
                    # Any other error - assume JAX is not properly available
                    _jax_available = False
                    _jax_gpu_available = False
                
                # Restore original warning function
                warnings.warn = _original_warn
                
            except Exception as e:
                # Final fallback - JAX not available
                _jax_available = False
                _jax_gpu_available = False
                warnings.warn = _original_warn
        
        _jax_initialized = True
        
        _jax_config = {
            'available': _jax_available,
            'gpu_available': _jax_gpu_available,
            'initialized': True
        }
        
        return _jax_config


def get_jax_safely():
    """
    Get JAX modules safely, ensuring initialization has occurred.
    
    Returns:
        Tuple of (jax, jnp) if available, or (None, None) otherwise
    """
    if not _jax_initialized:
        initialize_jax_once()
    
    if not _jax_available:
        return None, None
    
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except ImportError:
        return None, None


def is_jax_available() -> bool:
    """Check if JAX is available."""
    if not _jax_initialized:
        initialize_jax_once()
    return _jax_available


def is_jax_gpu_available() -> bool:
    """Check if JAX GPU is available."""
    if not _jax_initialized:
        initialize_jax_once()
    return _jax_gpu_available


# Auto-initialize on import (but only once)
initialize_jax_once()

