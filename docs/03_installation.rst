Installation and Quick Start
=============================

This guide covers installation of HPFRACC with various configurations and provides quick start examples to get you up and running immediately.

Basic Installation
------------------

Install the core library:

.. code-block:: bash

   pip install hpfracc

This installs HPFRACC with all core dependencies including NumPy, SciPy, and Matplotlib.

Installation with GPU Support
-----------------------------

For GPU acceleration with PyTorch and JAX:

**Recommended Installation for CUDA 12.8 (PyTorch) and CUDA 12 (JAX):**

.. code-block:: bash

   # Install PyTorch with CUDA 12.8 first
   pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
   
   # Then install JAX with CUDA 12 support
   pip install --upgrade "jax[cuda12]"
   
   # Install HPFRACC with GPU extras
   pip install hpfracc[gpu]

**Note:** JAX's CUDA 12 wheels are built with CUDA 12.3 but are compatible with CUDA ≥12.1, which includes CUDA 12.8. CUDA libraries are backward compatible, so JAX will work with PyTorch's CUDA 12.8 installation.

**Alternative (simpler but may have version conflicts):**

.. code-block:: bash

   pip install hpfracc[gpu]

Installation with Machine Learning Dependencies
-----------------------------------------------

For full machine learning capabilities including PyTorch, JAX, and NUMBA with intelligent backend selection:

.. code-block:: bash

   pip install hpfracc[ml]

Development Installation
------------------------

For development and contribution:

.. code-block:: bash

   git clone https://github.com/dave2k77/fractional_calculus_library.git
   cd fractional_calculus_library
   pip install -e .[dev]
   pip install -e .[ml]

Requirements
------------

**Python**: 3.9+ (dropped 3.8 support)

**Required Dependencies**:
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

**Optional Dependencies**:
- PyTorch >= 1.12.0 (for GPU acceleration and ML)
- JAX >= 0.4.0 (for JAX backend and GPU)
- Numba >= 0.56.0 (for JIT compilation)
- NumPyro >= 0.13.0 (for probabilistic programming)

**GPU Support**:
- CUDA-compatible GPU (optional)
- CuDNN >= 9.12.0 (for JAX 0.8.0)

GPU Setup Details
-----------------

JAX GPU Configuration
~~~~~~~~~~~~~~~~~~~~~

HPFRACC automatically configures JAX to use GPU when available:

1. **Auto-detection**: On import, `hpfracc.jax_gpu_setup` automatically detects GPU availability
2. **Library path setup**: Automatically prioritizes pip-installed CuDNN over conda's older versions
3. **Environment setup**: Configures `LD_LIBRARY_PATH` to find correct CuDNN libraries
4. **Graceful fallback**: Falls back to CPU when GPU is not supported
5. **No user intervention**: Works automatically without any configuration needed

CuDNN Compatibility
~~~~~~~~~~~~~~~~~~~

If you encounter CuDNN version mismatch errors:

1. **Upgrade CuDNN** to 9.12.0+:
   .. code-block:: bash
      
      pip install --upgrade "nvidia-cudnn-cu12>=9.12.0"

2. **Configure library paths** (if conda CuDNN conflicts):
   .. code-block:: bash
      
      source scripts/setup_jax_gpu_env.sh

3. **Verify installation**:
   .. code-block:: bash
      
      python -c "import jax; print(jax.devices()); print(jax.default_backend())"

CUDA Version Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------+-----------------------------------+-------------------+
| Component | CUDA Version                     | Status            |
+===========+===================================+===================+
| PyTorch   | 12.8                              | ✅ Fully supported|
+-----------+-----------------------------------+-------------------+
| JAX       | 12.3 (wheels) → 12.8 (runtime)   | ✅ Compatible     |
+-----------+-----------------------------------+-------------------+
| CuDNN     | 9.12.0+                           | ✅ Recommended    |
+-----------+-----------------------------------+-------------------+

**Key Point**: JAX's CUDA 12 wheels are built with CUDA 12.3 but work with CUDA ≥12.1, including 12.8. This ensures compatibility between JAX and PyTorch installations.

Quick Start Examples
--------------------

Basic Fractional Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hpfracc
   import numpy as np
   from hpfracc import FractionalOrder, optimized_riemann_liouville

   # Create a test function
   def f(x):
       return np.sin(x)

   # Define fractional order
   alpha = FractionalOrder(0.5)

   # Compute fractional derivative
   x = np.linspace(0, 2*np.pi, 100)
   result = optimized_riemann_liouville(x, f(x), alpha)

   print(f"Fractional derivative computed for {len(x)} points")
   print(f"First 5 values: {result[:5]}")

Spectral Autograd with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

   # Create input with gradient support
   x = torch.randn(32, requires_grad=True)
   alpha = 0.5  # fractional order

   # Apply spectral fractional derivative
   result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
   
   # Gradients flow properly through fractional derivatives
   loss = torch.sum(result)
   loss.backward()
   
   print(f"Input gradient norm: {x.grad.norm().item():.6f}")

Learnable Fractional Orders
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

   # Create learnable alpha parameter
   alpha_param = BoundedAlphaParameter(alpha_init=1.0)
   
   x = torch.randn(32, requires_grad=True)
   
   # Use in computation
   alpha_val = alpha_param()
   result = SpectralFractionalDerivative.apply(x, alpha_val, -1, "fft")
   
   # Alpha gradients are computed automatically
   loss = torch.sum(result)
   loss.backward()
   
   print(f"Alpha value: {alpha_val.item():.4f}")
   print(f"Alpha gradient: {alpha_param.rho.grad.item():.6f}")

Intelligent Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hpfracc
   from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector
   from hpfracc.ml.intelligent_backend_selector import WorkloadCharacteristics

   # Create intelligent backend selector
   selector = IntelligentBackendSelector(enable_learning=True)

   # Define workload characteristics
   workload = WorkloadCharacteristics(
       operation_type="fractional_derivative",
       data_size=10000,
       data_shape=(100, 100),
       requires_gradient=True
   )

   # Select optimal backend automatically
   backend = selector.select_backend(workload)
   print(f"Selected backend: {backend}")

   # Use with fractional operations
   frac_deriv = hpfracc.create_fractional_derivative(alpha=0.5, definition="caputo")
   result = frac_deriv(f, x)  # Automatically uses optimal backend

Verification
------------

Check Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hpfracc
   print(f"HPFRACC version: {hpfracc.__version__}")

   # Test basic functionality
   from hpfracc.core.derivatives import CaputoDerivative
   caputo = CaputoDerivative(order=0.5)
   print("✅ Installation successful!")

Check GPU Availability
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check PyTorch GPU
   import torch
   print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"PyTorch CUDA version: {torch.version.cuda}")

   # Check JAX GPU
   try:
       import jax
       print(f"JAX devices: {jax.devices()}")
       print(f"JAX backend: {jax.default_backend()}")
   except ImportError:
       print("JAX not installed")

   # Check HPFRACC GPU setup
   from hpfracc.jax_gpu_setup import JAX_GPU_AVAILABLE
   print(f"HPFRACC JAX GPU available: {JAX_GPU_AVAILABLE}")

Troubleshooting Installation
-----------------------------

Common Issues
~~~~~~~~~~~~~

**Import Errors After Installation**:
- Ensure you're using Python 3.9+
- Try reinstalling: ``pip install --upgrade --force-reinstall hpfracc``
- Check that dependencies are installed: ``pip list | grep hpfracc``

**GPU Not Detected**:
- Verify CUDA is installed: ``nvidia-smi``
- Check PyTorch CUDA: ``python -c "import torch; print(torch.cuda.is_available())"``
- For JAX, ensure ``jax[cuda12]`` is installed correctly

**CuDNN Version Mismatch**:
- Upgrade CuDNN: ``pip install --upgrade "nvidia-cudnn-cu12>=9.12.0"``
- Use setup script: ``source scripts/setup_jax_gpu_env.sh``
- Check library paths match installed CuDNN version

**Package Conflicts**:
- Use virtual environment: ``python -m venv venv && source venv/bin/activate``
- Install in isolated environment to avoid conflicts
- Consider using conda if pip conflicts persist

Next Steps
----------

Once installation is complete:

1. **Start with**: :doc:`04_basic_examples` for basic usage examples
2. **Explore**: :doc:`02_advanced_features` for advanced capabilities
3. **Learn**: :doc:`06_derivatives_integrals` for comprehensive operator guide
4. **Build**: :doc:`07_fractional_neural_networks` for ML integration

For detailed GPU setup information, see :doc:`JAX_GPU_SETUP`.

