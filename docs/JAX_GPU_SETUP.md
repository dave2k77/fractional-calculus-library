# JAX GPU Setup for HPFRACC Library

This document explains how JAX GPU support is configured in the HPFRACC library.

## Current Status

- **PyTorch GPU**: ✅ **Fully supported** - RTX 5070 detected and working with CUDA 12.8
- **JAX GPU**: ✅ **Fully supported** - RTX 5070 detected and working with CUDA 12
- **Automatic detection**: ✅ **Configured** - Will use GPU when available
- **CuDNN**: ✅ **Compatible** - CuDNN 9.12.0+ recommended for JAX 0.8.0

## Installation

### Recommended Installation for GPU Support

For optimal GPU performance with JAX and PyTorch compatibility:

```bash
# Install PyTorch with CUDA 12.8 first
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Then install JAX with CUDA 12 support
pip install --upgrade "jax[cuda12]"

# Install HPFRACC with GPU extras
pip install hpfracc[gpu]
```

**Important Notes:**
- JAX's CUDA 12 wheels are built with CUDA 12.3 but are compatible with CUDA ≥12.1 (including CUDA 12.8)
- CUDA libraries are backward compatible, so JAX will work with PyTorch's CUDA 12.8 installation
- Ensure CuDNN 9.12.0+ is installed for JAX 0.8.0 compatibility
- If you have conda-installed CuDNN that conflicts, use `scripts/setup_jax_gpu_env.sh` to configure library paths

### CuDNN Compatibility

If you encounter CuDNN version mismatch errors:

1. **Upgrade CuDNN** to 9.12.0+:
   ```bash
   pip install --upgrade "nvidia-cudnn-cu12>=9.12.0"
   ```

2. **Configure library paths** (if conda CuDNN conflicts):
   ```bash
   source scripts/setup_jax_gpu_env.sh
   ```

3. **Verify installation**:
   ```bash
   python -c "import jax; print(jax.devices()); print(jax.default_backend())"
   ```

## How It Works

The HPFRACC library automatically configures JAX to use GPU when available:

1. **Auto-detection**: On import, `hpfracc.jax_gpu_setup` automatically detects GPU availability
2. **Library path setup**: Automatically prioritizes pip-installed CuDNN over conda's older versions
3. **Environment setup**: Configures `LD_LIBRARY_PATH` to find correct CuDNN libraries
4. **Graceful fallback**: Falls back to CPU when GPU is not supported
5. **No user intervention**: Works automatically without any configuration needed

## Usage

Simply import HPFRACC modules that use JAX - no additional setup required:

```python
from hpfracc.jax_gpu_setup import JAX_GPU_AVAILABLE
import hpfracc.ml.probabilistic_fractional_orders  # Uses JAX

print(f"JAX GPU available: {JAX_GPU_AVAILABLE}")
```

## GPU Support Status

### RTX 5070 (Current GPU)
- **PyTorch**: ✅ Fully supported with CUDA 12.8
- **JAX**: ✅ Fully supported with CUDA 12
- **CUDA Compatibility**: ✅ JAX CUDA 12 wheels compatible with CUDA 12.8
- **CuDNN**: ✅ 9.12.0+ recommended for JAX 0.8.0
- **Status**: Complete GPU acceleration available

### CUDA Version Compatibility

| Component | CUDA Version | Status |
|-----------|--------------|--------|
| PyTorch | 12.8 | ✅ Fully supported |
| JAX | 12.3 (wheels) → 12.8 (runtime) | ✅ Compatible |
| CuDNN | 9.12.0+ | ✅ Recommended |

**Key Point**: JAX's CUDA 12 wheels are built with CUDA 12.3 but work with CUDA ≥12.1, including 12.8. This ensures compatibility between JAX and PyTorch installations.

### Future GPU Support
When JAX adds support for newer GPUs, the library will automatically detect and use them without any code changes.

## Performance Impact

- **PyTorch operations**: Full GPU acceleration (8GB VRAM)
- **JAX operations**: Full GPU acceleration (8GB VRAM)
- **Mixed workloads**: Optimal performance through PyTorch GPU + JAX GPU

## Troubleshooting

If you encounter issues:

1. **Check GPU detection**: Run `python jax_gpu_config.py`
2. **Verify PyTorch GPU**: Run `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check JAX status**: Run `python -c "from hpfracc.jax_gpu_setup import get_jax_info; print(get_jax_info())"`

## Technical Details

- **JAX version**: 0.8.0 (compatible with NumPy 2.3+)
- **JAXlib version**: 0.8.0
- **CUDA support**: CUDA 12 (wheels built with 12.3, compatible with ≥12.1 including 12.8)
- **CuDNN**: 9.12.0+ required for JAX 0.8.0
- **PyTorch CUDA**: 12.8 (compatible with JAX CUDA 12)
- **Environment variables**: Automatically configured for optimal library resolution
- **GPU acceleration**: Full RTX 5070 support with 8GB VRAM
- **Library path management**: Automatic prioritization of pip-installed CuDNN

## Troubleshooting CuDNN Issues

If you see CuDNN version mismatch errors:

1. **Upgrade CuDNN**:
   ```bash
   pip install --upgrade "nvidia-cudnn-cu12>=9.12.0"
   ```

2. **Use setup script** (if conda CuDNN conflicts):
   ```bash
   source scripts/setup_jax_gpu_env.sh
   ```

3. **Manual library path** (if needed):
   ```bash
   export LD_LIBRARY_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia/cudnn/lib:$LD_LIBRARY_PATH
   ```

4. **Verify installation**:
   ```bash
   python -c "from hpfracc.jax_gpu_setup import get_jax_info; import json; print(json.dumps(get_jax_info(), indent=2))"
   ```
