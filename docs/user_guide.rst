"""
User Guide
==========

Welcome to the HPFRACC User Guide! This guide will help you get started with the High-Performance Fractional Calculus Library and its machine learning integration.

Installation
-----------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install the core library:

.. code-block:: bash

   pip install hpfracc

Installation with Machine Learning Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full machine learning capabilities including PyTorch, JAX, and NUMBA:

.. code-block:: bash

   pip install hpfracc[ml]

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development and contribution:

.. code-block:: bash

   git clone https://github.com/dave2k77/fractional_calculus_library.git
   cd fractional_calculus_library
   pip install -e .[dev]
   pip install -e .[ml]

Quick Start
----------

Basic Fractional Calculus Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fractional Derivatives
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from hpfracc import FractionalOrder, optimized_riemann_liouville
   import numpy as np

   # Define fractional order
   alpha = FractionalOrder(0.5)

   # Create a test function
   def f(x):
       return np.sin(x)

   # Compute fractional derivative
   x = np.linspace(0, 2*np.pi, 100)
   result = optimized_riemann_liouville(x, f(x), alpha)

   print(f"Fractional derivative of sin(x) with order {alpha}:")
   print(result[:5])  # Show first 5 values

Fractional Integrals
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from hpfracc import FractionalOrder, riemann_liouville_integral
   import numpy as np

   # Define fractional order
   alpha = FractionalOrder(0.5)

   # Create a test function
   def f(x):
       return x**2

   # Compute fractional integral
   x = np.linspace(0, 5, 100)
   result = riemann_liouville_integral(x, f(x), alpha)

   print(f"Fractional integral of x² with order {alpha}:")
   print(result[:5])  # Show first 5 values

Advanced Fractional Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HPFRACC provides a comprehensive collection of fractional operators beyond the classical definitions:

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.fractional_implementations import create_riesz_fisher_operator
   import numpy as np

   # Novel derivatives
   cf_derivative = create_fractional_derivative('caputo_fabrizio', 0.5)
   ab_derivative = create_fractional_derivative('atangana_baleanu', 0.5)

   # Advanced methods
   weyl_derivative = create_fractional_derivative('weyl', 0.5)
   marchaud_derivative = create_fractional_derivative('marchaud', 0.5)

   # Special operators
   rf_operator = create_riesz_fisher_operator(0.5)  # Derivative behavior
   rf_integral = create_riesz_fisher_operator(-0.5)  # Integral behavior

   # Test function
   def f(x): return np.exp(-x**2)
   x = np.linspace(-2, 2, 100)

   # Compute results
   cf_result = cf_derivative.compute(f, x)
   rf_result = rf_operator.compute(f, x)

   print(f"Caputo-Fabrizio result shape: {cf_result.shape}")
   print(f"Riesz-Fisher result shape: {rf_result.shape}")

Special Functions
^^^^^^^^^^^^^^^^

.. code-block:: python

   from hpfracc.special import (
       gamma_function, beta_function, binomial_coefficient,
       mittag_leffler_function
   )
   import numpy as np

   # Gamma function
   x = 2.5
   gamma_val = gamma_function(x)
   print(f"Γ({x}) = {gamma_val}")

   # Beta function
   a, b = 2.0, 3.0
   beta_val = beta_function(a, b)
   print(f"B({a}, {b}) = {beta_val}")

   # Binomial coefficient
   n, k = 5, 2
   binomial_val = binomial_coefficient(n, k)
   print(f"({n} choose {k}) = {binomial_val}")

   # Mittag-Leffler function
   alpha, z = 0.5, 1.0
   ml_val = mittag_leffler_function(alpha, z)
   print(f"E_{alpha}({z}) = {ml_val}")

Backend Management
~~~~~~~~~~~~~~~~~

HPFRACC supports multiple computation backends:

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

Core Features
------------

Fractional Derivatives
~~~~~~~~~~~~~~~~~~~~~

HPFRACC provides multiple definitions of fractional derivatives:

**Riemann-Liouville Definition:**

.. code-block:: python

   from hpfracc import FractionalOrder, optimized_riemann_liouville

   # Create Riemann-Liouville fractional derivative
   alpha = FractionalOrder(0.5)

   # Apply to function
   def f(x):
       return np.sin(x)
   
   x = np.linspace(0, 2*np.pi, 100)
   result = optimized_riemann_liouville(x, f(x), alpha)

**Caputo Definition:**

.. code-block:: python

   from hpfracc import optimized_caputo

   # Create Caputo fractional derivative
   result = optimized_caputo(x, f(x), alpha)

**Grünwald-Letnikov Definition:**

.. code-block:: python

   from hpfracc import optimized_grunwald_letnikov

   # Create Grünwald-Letnikov fractional derivative
   result = optimized_grunwald_letnikov(x, f(x), alpha)

Fractional Integrals
~~~~~~~~~~~~~~~~~~~

HPFRACC supports various types of fractional integrals:

**Riemann-Liouville Integral:**

.. code-block:: python

   from hpfracc import riemann_liouville_integral

   # Create Riemann-Liouville fractional integral
   alpha = FractionalOrder(0.5)

   # Apply to function
   def f(x):
       return x**2
   
   x = np.linspace(0, 5, 100)
   result = riemann_liouville_integral(x, f(x), alpha)

**Caputo Integral:**

.. code-block:: python

   from hpfracc import caputo_integral

   # Create Caputo fractional integral
   result = caputo_integral(x, f(x), alpha)

**Note**: Weyl and Hadamard integrals are available but require specific implementations. For now, use Riemann-Liouville and Caputo integrals which are fully implemented.

Special Functions
~~~~~~~~~~~~~~~~

**Gamma and Beta Functions:**

.. code-block:: python

   from hpfracc.special import gamma_function, beta_function

   # Gamma function
   x = np.linspace(0.1, 5, 100)
   gamma_vals = [gamma_function(xi) for xi in x]

   # Beta function
   a, b = 2.0, 3.0
   beta_val = beta_function(a, b)

**Binomial Coefficients:**

.. code-block:: python

   from hpfracc.special import binomial_coefficient, generalized_binomial

   # Standard binomial coefficient
   n, k = 5, 2
   binomial_val = binomial_coefficient(n, k)

   # Fractional binomial coefficient
   alpha = 0.5
   frac_binomial_val = generalized_binomial(alpha, k)

**Mittag-Leffler Functions:**

.. code-block:: python

   from hpfracc.special import mittag_leffler_function

   # One-parameter Mittag-Leffler function
   alpha = 0.5
   z = np.linspace(-5, 5, 100)
   ml_vals = [mittag_leffler_function(alpha, zi) for zi in z]

# Green's functions have been removed from this release
# They will be re-implemented in future releases with improved stability







# HPM and VIM solvers have been removed from this release
# They will be re-implemented in future releases with improved stability

Mathematical Utilities
~~~~~~~~~~~~~~~~~~~~~

HPFRACC provides various mathematical utilities:

**Validation Functions:**

.. code-block:: python

   from hpfracc.core.utilities import (
       validate_fractional_order, validate_function,
       validate_tensor_input
   )

   # Validate fractional order
   is_valid = validate_fractional_order(0.5)  # True
   is_valid = validate_fractional_order(-1.0)  # False

   # Validate function
   def test_func(x):
       return x**2
   
   is_valid = validate_function(test_func)  # True
   is_valid = validate_function("not a function")  # False

   # Validate tensor input
   import numpy as np
   tensor = np.random.randn(10, 5)
   is_valid = validate_tensor_input(tensor)  # True

**Mathematical Functions:**

.. code-block:: python

   from hpfracc.core.utilities import (
       factorial_fractional, binomial_coefficient,
       pochhammer_symbol, hypergeometric_series
   )

   # Fractional factorial
   x = 2.5
   factorial_val = factorial_fractional(x)

   # Binomial coefficient
   n, k = 5, 2
   binomial_val = binomial_coefficient(n, k)

   # Pochhammer symbol
   a, n = 0.5, 3
   pochhammer_val = pochhammer_symbol(a, n)

   # Hypergeometric series
   a, b, c, z = 1, 1, 1, 0.5
   hypergeometric_val = hypergeometric_series(a, b, c, z)

**Performance Monitoring:**

.. code-block:: python

   from hpfracc.core.utilities import (
       timing_decorator, memory_usage_decorator,
       PerformanceMonitor
   )

   # Timing decorator
   @timing_decorator
   def expensive_function(n):
       return sum(i**2 for i in range(n))

   result = expensive_function(10000)

   # Memory usage decorator
   @memory_usage_decorator
   def memory_intensive_function(n):
       return np.random.randn(n, n)

   result = memory_intensive_function(1000)

   # Performance monitor
   monitor = PerformanceMonitor()
   
   with monitor.timer("computation"):
       result = expensive_function(10000)
   
   print(f"Computation time: {monitor.get_timing('computation')}")

Fractional Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~

Create and use fractional neural networks:

.. code-block:: python

   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType
   import numpy as np

   # Create a fractional neural network
   model = FractionalNeuralNetwork(
       input_dim=10,
       hidden_dims=[64, 32, 16],
       output_dim=1,
       fractional_order=FractionalOrder(0.5),
       activation='relu',
       dropout_rate=0.2
   )

   # Generate sample data
   X = np.random.randn(1000, 10)
   y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(1000)

   # Train the model
   history = model.fit(
       X, y,
       epochs=100,
       batch_size=32,
       learning_rate=0.001,
       verbose=True
   )

   # Make predictions
   predictions = model.predict(X)

Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~

Work with fractional graph neural networks:

.. code-block:: python

   from hpfracc.ml.gnn_layers import FractionalGraphConvolution
   from hpfracc.core.definitions import FractionalOrder
   import numpy as np
   import networkx as nx

   # Create a graph
   G = nx.erdos_renyi_graph(20, 0.3)
   adj_matrix = nx.adjacency_matrix(G).toarray()
   
   # Create node features
   node_features = np.random.randn(20, 5)
   
   # Create fractional graph convolution layer
   fractional_order = FractionalOrder(0.5)
   fgc_layer = FractionalGraphConvolution(
       input_dim=5,
       output_dim=3,
       fractional_order=fractional_order,
       activation='relu'
   )
   
   # Apply fractional graph convolution
   output_features = fgc_layer(adj_matrix, node_features)

Advanced Usage
-------------

Error Analysis and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Numerical Error Analysis:**

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder
   import numpy as np

   def analytical_solution(x, alpha):
       """Analytical solution for D^α sin(x)."""
       return np.sin(x + alpha * np.pi / 2)

   # Compare numerical and analytical solutions
   x = np.linspace(0, 2*np.pi, 100)
   alpha = 0.5
   
   # Numerical solution
   deriv = create_fractional_derivative(FractionalOrder(alpha), method="RL")
   numerical = deriv(lambda x: np.sin(x), x)
   
   # Analytical solution
   analytical = analytical_solution(x, alpha)
   
   # Compute error
   error = np.mean(np.abs((numerical - analytical) / analytical))
   print(f"Relative error: {error:.6f}")

**Convergence Analysis:**

.. code-block:: python

   from hpfracc.solvers.homotopy_perturbation import HomotopyPerturbationMethod

   # Analyze convergence of HPM
   def source_function(t):
       return t**2

   def initial_condition(t):
       return 0.0

   alpha = 0.5
   hpm_solver = HomotopyPerturbationMethod(alpha)
   t = np.linspace(0, 2, 100)

   convergence = hpm_solver.analyze_convergence(
       source_function=source_function,
       initial_condition=initial_condition,
       t_span=t,
       max_iterations=10
   )

   print(f"Final residual: {convergence['final_residual']:.6f}")
   print(f"Convergence rate: {convergence['convergence_rate']:.6f}")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**GPU Acceleration:**

.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder
   import time

   def benchmark_cpu_vs_gpu(data_size):
       # Generate data
       x = np.linspace(0, 10, data_size)
       signal = np.sin(2*np.pi*x) + 0.1*np.random.randn(data_size)
       
       # CPU computation
       BackendManager.set_backend(BackendType.NUMPY)
       deriv_cpu = create_fractional_derivative(FractionalOrder(0.5), method="RL")
       
       start_time = time.time()
       result_cpu = deriv_cpu(lambda x: signal, x)
       cpu_time = time.time() - start_time
       
       # GPU computation (if available)
       if BackendManager.is_backend_available(BackendType.TORCH):
           BackendManager.set_backend(BackendType.TORCH)
           deriv_gpu = create_fractional_derivative(FractionalOrder(0.5), method="RL")
           
           start_time = time.time()
           result_gpu = deriv_gpu(lambda x: signal, x)
           gpu_time = time.time() - start_time
           
           print(f"CPU time: {cpu_time:.4f}s")
           print(f"GPU time: {gpu_time:.4f}s")
           print(f"Speedup: {cpu_time/gpu_time:.2f}x")

**Memory Optimization:**

.. code-block:: python

   from hpfracc.core.utilities import memory_usage_decorator
   import numpy as np

   @memory_usage_decorator
   def memory_intensive_computation(data_size):
       # Generate large dataset
       x = np.linspace(0, 10, data_size)
       signal = np.sin(2*np.pi*x) + 0.1*np.random.randn(data_size)
       
       # Create multiple fractional derivatives
       derivatives = []
       for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
           from hpfracc.core.derivatives import create_fractional_derivative
           from hpfracc.core.definitions import FractionalOrder
           deriv = create_fractional_derivative(FractionalOrder(alpha), method="RL")
           result = deriv(lambda x: signal, x)
           derivatives.append(result)
       
       return derivatives

   # Test memory usage
   result = memory_intensive_computation(10000)

Signal Processing Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fractional Signal Processing:**

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder
   import numpy as np
   from scipy.fft import fft, fftfreq

   # Generate test signal
   t = np.linspace(0, 10, 1000)
   signal = np.sin(2*np.pi*t) + 0.5*np.sin(4*np.pi*t) + 0.1*np.random.randn(len(t))

   # Apply fractional derivatives
   alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
   derivatives = {}

   for alpha in alpha_values:
       deriv = create_fractional_derivative(FractionalOrder(alpha), method="RL")
       derivatives[alpha] = deriv(lambda x: signal, t)

   # Frequency domain analysis
   fft_original = np.abs(fft(signal))
   fft_derivatives = {}
   
   for alpha in alpha_values:
       fft_derivatives[alpha] = np.abs(fft(derivatives[alpha]))

Image Processing Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fractional Image Processing:**

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder
   import numpy as np
   from scipy import ndimage

   # Create a test image
   x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
   image = np.sin(x) * np.cos(y) + 0.1 * np.random.randn(100, 100)

   # Apply fractional derivatives in x and y directions
   alpha = 0.5
   deriv_x = create_fractional_derivative(FractionalOrder(alpha), method="RL")
   deriv_y = create_fractional_derivative(FractionalOrder(alpha), method="RL")

   # Compute fractional gradients
   gradient_x = np.zeros_like(image)
   gradient_y = np.zeros_like(image)
   
   for i in range(image.shape[0]):
       gradient_x[i, :] = deriv_x(lambda x: image[i, :], np.arange(image.shape[1]))
   
   for j in range(image.shape[1]):
       gradient_y[:, j] = deriv_y(lambda y: image[:, j], np.arange(image.shape[0]))

   # Compute gradient magnitude
   gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

Configuration and Settings
-------------------------

Precision Settings
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

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

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Import Errors:**

.. code-block:: python

   # If you get import errors, check your installation
   import hpfracc
   print(hpfracc.__version__)

   # Check available backends
   from hpfracc.ml.backends import BackendManager
   available = BackendManager.get_available_backends()
   print(f"Available backends: {available}")

**Memory Issues:**

.. code-block:: python

   # For large computations, use memory-efficient processing
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

**Performance Issues:**

.. code-block:: python

   # Use GPU acceleration when available
   from hpfracc.ml.backends import BackendManager, BackendType

   # Try different backends
   backends_to_try = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]
   
   for backend in backends_to_try:
       if BackendManager.is_backend_available(backend):
           BackendManager.set_backend(backend)
           print(f"Using backend: {backend}")
           break

**Validation Errors:**

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

Best Practices
-------------

**Code Organization:**

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

**Error Handling:**

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

**Performance Optimization:**

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

**Documentation and Testing:**

.. code-block:: python

   def well_documented_function(f, alpha, method="RL"):
       """
       Compute fractional derivative with comprehensive documentation.
       
       Parameters:
       -----------
       f : callable
           Function to differentiate
       alpha : float
           Fractional order (0 < alpha < 2)
       method : str, optional
           Method to use ("RL", "Caputo", "GL")
       
       Returns:
       --------
       numpy.ndarray
           Fractional derivative values
       
       Raises:
       -------
       ValueError
           If alpha is not in valid range
       TypeError
           If f is not callable
       
       Examples:
       --------
       >>> def f(x): return np.sin(x)
       >>> result = well_documented_function(f, 0.5)
       """
       # Input validation
       if not validate_fractional_order(alpha):
           raise ValueError(f"Invalid fractional order: {alpha}")
       
       if not validate_function(f):
           raise TypeError("f must be callable")
       
       # Computation
       from hpfracc.core.derivatives import create_fractional_derivative
       from hpfracc.core.definitions import FractionalOrder
       
       deriv = create_fractional_derivative(FractionalOrder(alpha), method=method)
       x = np.linspace(0, 10, 100)
       return deriv(f, x)

This comprehensive user guide covers all the major features of HPFRACC, from basic usage to advanced applications. For more detailed examples and tutorials, see the Examples & Tutorials section.
"""
