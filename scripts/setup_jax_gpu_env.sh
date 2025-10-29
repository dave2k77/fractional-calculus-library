#!/usr/bin/env bash
# Setup script to configure LD_LIBRARY_PATH for JAX GPU with upgraded CuDNN
# 
# Usage: source scripts/setup_jax_gpu_env.sh
# Or add to your ~/.bashrc or shell profile

# Find pip-installed CuDNN location
PYTHON_CMD="python3"
CUDNN_LIB=$(python3 -c "
import site, os
for path in site.getsitepackages() + [site.getusersitepackages()]:
    cudnn_path = os.path.join(path, 'nvidia', 'cudnn', 'lib')
    if os.path.exists(cudnn_path):
        print(cudnn_path)
        break
" 2>/dev/null)

if [ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ]; then
    # Add pip-installed CuDNN to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
    echo "✓ Configured LD_LIBRARY_PATH for CuDNN: $CUDNN_LIB"
else
    echo "⚠️  Could not find pip-installed CuDNN library path"
    echo "   JAX may use conda's older CuDNN version"
fi

# Verify JAX can detect GPU
python3 -c "
import warnings
warnings.filterwarnings('ignore')
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
if gpu_devices:
    print(f'✓ JAX GPU detected: {gpu_devices}')
else:
    print('⚠️  No JAX GPU devices found')
" 2>/dev/null || echo "⚠️  Could not verify JAX GPU status"

