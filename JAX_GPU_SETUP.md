# JAX GPU Setup for HPFRACC Library

This document explains how JAX GPU support is configured in the HPFRACC library.

## Current Status

- **PyTorch GPU**: ✅ **Fully supported** - RTX 5070 detected and working
- **JAX GPU**: ⚠️ **CPU fallback** - RTX 5070 not yet supported by JAX
- **Automatic detection**: ✅ **Configured** - Will use GPU when available

## How It Works

The HPFRACC library automatically configures JAX to use GPU when available:

1. **Auto-detection**: On import, `hpfracc.jax_gpu_setup` automatically detects GPU availability
2. **Environment setup**: Sets `JAX_PLATFORM_NAME=gpu` to prefer GPU when available
3. **Graceful fallback**: Falls back to CPU when GPU is not supported
4. **No user intervention**: Works automatically without any configuration needed

## Usage

Simply import HPFRACC modules that use JAX - no additional setup required:

```python
from hpfracc.jax_gpu_setup import JAX_GPU_AVAILABLE
import hpfracc.ml.probabilistic_fractional_orders  # Uses JAX

print(f"JAX GPU available: {JAX_GPU_AVAILABLE}")
```

## GPU Support Status

### RTX 5070 (Current GPU)
- **PyTorch**: ✅ Fully supported
- **JAX**: ⚠️ Not yet supported (very new architecture)
- **Expected**: JAX support likely in future releases

### Future GPU Support
When JAX adds support for RTX 5070, the library will automatically detect and use it without any code changes.

## Performance Impact

- **PyTorch operations**: Full GPU acceleration (8GB VRAM)
- **JAX operations**: CPU fallback (still fast for most use cases)
- **Mixed workloads**: Optimal performance through PyTorch GPU + JAX CPU

## Troubleshooting

If you encounter issues:

1. **Check GPU detection**: Run `python jax_gpu_config.py`
2. **Verify PyTorch GPU**: Run `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check JAX status**: Run `python -c "from hpfracc.jax_gpu_setup import get_jax_info; print(get_jax_info())"`

## Technical Details

- **JAX version**: 0.4.21 (compatible with NumPy 1.26.4)
- **CUDA support**: Configured for CUDA 12.x
- **Environment variables**: `JAX_PLATFORM_NAME=gpu` set automatically
- **Fallback mechanism**: Silent CPU fallback when GPU unavailable
