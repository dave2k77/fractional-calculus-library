Advanced Usage
==============

This chapter covers advanced configuration, troubleshooting, best practices, and backend optimization for HPFRACC.

Configuration and Settings
--------------------------

Precision Settings
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.core.utilities import (
       get_default_precision, set_default_precision,
       get_available_methods, get_method_properties
   )

   # Get current precision settings
   precision = get_default_precision()
   print(f"Current precision: {precision}")

   # Set precision
   set_default_precision(64)  # Use 64-bit precision

   # Get available methods
   methods = get_available_methods()
   print(f"Available methods: {methods}")

   # Get method properties
   properties = get_method_properties("riemann_liouville")
   print(f"Riemann-Liouville properties: {properties}")

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.core.utilities import setup_logging, get_logger

   # Setup logging
   logger = setup_logging(level="INFO", log_file="hpfracc.log")

   # Get logger for specific module
   logger = get_logger("hpfracc.core.derivatives")

   # Use logger
   logger.info("Starting fractional derivative computation")
   logger.debug("Computing with alpha=0.5")
   logger.warning("Large data size detected")
   logger.error("Computation failed")

Backend Configuration
---------------------

Manual Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType

   # Check available backends
   available = BackendManager.get_available_backends()
   print(f"Available backends: {available}")

   # Set preferred backend
   BackendManager.set_backend(BackendType.JAX)

   # Get current backend
   current = BackendManager.get_current_backend()
   print(f"Current backend: {current}")

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Control backend behavior with environment variables:

.. code-block:: bash

   export HPFRACC_FORCE_JAX=1        # Force JAX
   export HPFRACC_DISABLE_TORCH=1    # Disable PyTorch
   export JAX_PLATFORM_NAME=cpu      # Force CPU

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**:

.. code-block:: python

   # Check installation
   import hpfracc
   print(hpfracc.__version__)

   # Check available backends
   from hpfracc.ml.backends import BackendManager
   available = BackendManager.get_available_backends()
   print(f"Available backends: {available}")

**Memory Issues**:

.. code-block:: python

   from hpfracc.core.utilities import memory_usage_decorator
   import gc

   @memory_usage_decorator
   def process_large_data(data, chunk_size=1000):
       results = []
       for i in range(0, len(data), chunk_size):
           chunk = data[i:i+chunk_size]
           # Process chunk
           chunk_result = process_chunk(chunk)
           results.append(chunk_result)
           
           # Clear memory
           del chunk
           gc.collect()
       
       return np.concatenate(results)

**Performance Issues**:

.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType

   # Try different backends
   backends_to_try = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]
   
   for backend in backends_to_try:
       if BackendManager.is_backend_available(backend):
           BackendManager.set_backend(backend)
           print(f"Using backend: {backend}")
           break

**Validation Errors**:

.. code-block:: python

   from hpfracc.core.utilities import validate_fractional_order, validate_function

   # Validate inputs before computation
   alpha = 0.5
   if not validate_fractional_order(alpha):
       raise ValueError(f"Invalid fractional order: {alpha}")

   def f(x):
       return x**2
   
   if not validate_function(f):
       raise ValueError("Invalid function")

GPU Troubleshooting
-------------------

JAX GPU Setup
~~~~~~~~~~~~~

If JAX GPU is not detected:

.. code-block:: bash

   # Upgrade CuDNN
   pip install --upgrade "nvidia-cudnn-cu12>=9.12.0"

   # Use setup script
   source scripts/setup_jax_gpu_env.sh

   # Verify installation
   python -c "import jax; print(jax.devices()); print(jax.default_backend())"

PyTorch GPU Verification
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"PyTorch CUDA version: {torch.version.cuda}")
       print(f"GPU device: {torch.cuda.get_device_name(0)}")

See :doc:`JAX_GPU_SETUP` for comprehensive GPU setup documentation.

Best Practices
--------------

Code Organization
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Organize your code with proper imports
   import numpy as np
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.integrals import create_fractional_integral
   from hpfracc.special import gamma_function, mittag_leffler_function

   # Use consistent naming conventions
   alpha = FractionalOrder(0.5)
   x = np.linspace(0, 10, 100)
   
   # Create reusable functions
   def compute_fractional_derivative(f, alpha, method="RL"):
       deriv = create_fractional_derivative(alpha, method=method)
       return deriv(f, x)

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from hpfracc.core.utilities import validate_fractional_order

   def safe_fractional_derivative(f, alpha, method="RL"):
       """Safely compute fractional derivative with error handling."""
       try:
           # Validate inputs
           if not validate_fractional_order(alpha):
               raise ValueError(f"Invalid fractional order: {alpha}")
           
           # Create derivative
           from hpfracc.core.derivatives import create_fractional_derivative
           from hpfracc.core.definitions import FractionalOrder
           
           deriv = create_fractional_derivative(FractionalOrder(alpha), method=method)
           
           # Compute result
           x = np.linspace(0, 10, 100)
           result = deriv(f, x)
           
           return result
           
       except Exception as e:
           print(f"Error computing fractional derivative: {e}")
           return None

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.core.utilities import timing_decorator
   from hpfracc.ml.backends import BackendManager, BackendType

   @timing_decorator
   def optimized_computation(data, alpha, method="RL"):
       """Optimized computation with backend selection."""
       # Choose best available backend
       if BackendManager.is_backend_available(BackendType.TORCH):
           BackendManager.set_backend(BackendType.TORCH)
       elif BackendManager.is_backend_available(BackendType.JAX):
           BackendManager.set_backend(BackendType.JAX)
       else:
           BackendManager.set_backend(BackendType.NUMPY)
       
       # Perform computation
       from hpfracc.core.derivatives import create_fractional_derivative
       from hpfracc.core.definitions import FractionalOrder
       
       deriv = create_fractional_derivative(FractionalOrder(alpha), method=method)
       return deriv(lambda x: data, np.arange(len(data)))

Backend Optimization
---------------------

Intelligent Selection
~~~~~~~~~~~~~~~~~~~~~

The intelligent backend selector automatically optimizes performance:

.. code-block:: python

   from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector
   from hpfracc.ml.intelligent_backend_selector import WorkloadCharacteristics

   # Create selector with learning enabled
   selector = IntelligentBackendSelector(
       enable_learning=True,
       gpu_memory_limit=0.8,
       performance_threshold=0.1
   )

   # Define workload
   workload = WorkloadCharacteristics(
       operation_type="fractional_derivative",
       data_size=10000,
       data_shape=(100, 100),
       requires_gradient=True
   )

   # Select optimal backend
   backend = selector.select_backend(workload)
   print(f"Selected backend: {backend}")

See :doc:`02_advanced_features` for comprehensive backend optimization guide.

Known Limitations
------------------

This section documents intentional limitations and planned future enhancements.

Solver Limitations
~~~~~~~~~~~~~~~~~~

**SDE Solvers - Matrix Diffusion**:
- Full matrix diffusion (:math:`g_\theta: \mathbb{R}^{d} \to \mathbb{R}^{d \times d}`) is not yet implemented
- Currently supported: scalar diffusion (additive noise) and vector diffusion (diagonal multiplicative noise)
- **Workaround**: Use diagonal approximation or standard (non-fractional) SDE solvers first

**ODE Solvers - FFT Convolution**:
- FFT convolution is currently only implemented for `axis=0` (time axis)
- **Workaround**: Transpose your data so time is the first axis, or use direct convolution methods

**ODE Solvers - Predictor-Corrector**:
- Currently implemented for Caputo derivative only
- **Workaround**: Use fixed-step Euler method or convert problem to Caputo formulation

**PDE Solvers - Spectral Scheme**:
- Spectral scheme is implemented for :math:`0 < \alpha < 1` only
- **Workaround**: For :math:`\alpha \geq 1`, decompose into integer and fractional parts, or use finite difference schemes

**Adaptive ODE Solver**:
- Currently disabled due to known implementation flaw
- **Workaround**: Use fixed-step solver with `adaptive=False`

Backend Support
~~~~~~~~~~~~~~~

**Multi-Backend Fractional Derivatives**:
- ✅ PyTorch: Full support with autograd
- ✅ JAX: Full support with Caputo derivatives
- ✅ NumPy/NUMBA: Fractional scaling approximation
- All ML modules (training, losses, data) now support all backends

**GPU Optimization**:
- ✅ FFT-based methods: Fully implemented
- ✅ Mellin transform: GPU-optimized implementation available
- ✅ Fractional Laplacian: GPU-optimized

Summary
-------

Advanced Usage covers:

✅ **Configuration**: Precision settings, logging, backend management  
✅ **Troubleshooting**: Common issues and solutions  
✅ **Best Practices**: Code organization, error handling, optimization  
✅ **GPU Setup**: JAX and PyTorch GPU configuration  
✅ **Backend Optimization**: Intelligent selection and manual configuration  
✅ **Known Limitations**: Documented limitations with workarounds  

Next Steps
----------

- **Advanced Features**: See :doc:`02_advanced_features` for intelligent backend selection
- **Installation**: See :doc:`03_installation` for GPU setup details
- **Performance**: See :doc:`10_scientific_applications` for optimization strategies

