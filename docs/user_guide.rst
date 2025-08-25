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

.. code-block:: python

   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative
   import numpy as np

   # Define fractional order
   alpha = FractionalOrder(0.5)

   # Create a test function
   def f(x):
       return np.sin(x)

   # Create fractional derivative
   fractional_deriv = create_fractional_derivative(alpha, method="RL")

   # Compute fractional derivative
   x = np.linspace(0, 2*np.pi, 100)
   result = fractional_deriv(f, x)

   print(f"Fractional derivative of sin(x) with order {alpha}:")
   print(result[:5])  # Show first 5 values

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
       hidden_dims=[32, 16],
       output_dim=1,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.TORCH
   )

   # Generate sample data
   X = np.random.randn(100, 10)
   y = np.random.randn(100, 1)

   # Forward pass
   output = model.forward(X)
   print(f"Output shape: {output.shape}")

Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~

Use fractional Graph Neural Networks:

.. code-block:: python

   from hpfracc.ml import FractionalGNNFactory
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType
   import numpy as np

   # Create a fractional GCN
   gnn = FractionalGNNFactory.create_model(
       model_type='gcn',
       input_dim=16,
       hidden_dim=32,
       output_dim=4,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.JAX
   )

   # Generate graph data
   num_nodes = 50
   node_features = np.random.randn(num_nodes, 16)
   edge_index = np.random.randint(0, num_nodes, (2, 100))

   # Forward pass
   output = gnn.forward(node_features, edge_index)
   print(f"GNN output shape: {output.shape}")

Core Concepts
------------

Fractional Orders
~~~~~~~~~~~~~~~~

Fractional orders define the degree of differentiation:

.. code-block:: python

   from hpfracc.core.definitions import FractionalOrder

   # Integer order (classical derivative)
   order_1 = FractionalOrder(1.0)
   
   # Fractional order (fractional derivative)
   order_half = FractionalOrder(0.5)
   
   # Negative order (fractional integral)
   order_neg = FractionalOrder(-0.5)

   print(f"Order 1.0: {order_1}")
   print(f"Order 0.5: {order_half}")
   print(f"Order -0.5: {order_neg}")

Fractional Derivative Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HPFRACC supports multiple fractional derivative definitions:

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder
   import numpy as np

   alpha = FractionalOrder(0.5)
   x = np.linspace(0, 1, 50)

   # Riemann-Liouville definition
   rl_deriv = create_fractional_derivative(alpha, method="RL")
   
   # Caputo definition
   caputo_deriv = create_fractional_derivative(alpha, method="Caputo")
   
   # Grünwald-Letnikov definition
   gl_deriv = create_fractional_derivative(alpha, method="GL")

   def test_function(x):
       return np.exp(-x)

   # Compare different methods
   result_rl = rl_deriv(test_function, x)
   result_caputo = caputo_deriv(test_function, x)
   result_gl = gl_deriv(test_function, x)

Backend Types
~~~~~~~~~~~~

HPFRACC supports three main computation backends:

**PyTorch Backend**
- Full neural network support
- GPU acceleration
- Automatic differentiation
- Rich ecosystem

**JAX Backend**
- Functional programming
- GPU/TPU acceleration
- JIT compilation
- High performance

**NUMBA Backend**
- JIT compilation
- CPU optimization
- Lightweight
- Easy deployment

Advanced Usage
-------------

Custom Fractional Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom fractional derivative implementations:

.. code-block:: python

   from hpfracc.core.derivatives import BaseFractionalDerivative
   from hpfracc.core.definitions import FractionalOrder
   import numpy as np

   class CustomFractionalDerivative(BaseFractionalDerivative):
       def __init__(self, order: FractionalOrder):
           super().__init__(order)
           
       def compute(self, func, x):
           # Custom implementation
           # This is a simplified example
           h = x[1] - x[0]
           n = len(x)
           result = np.zeros_like(x)
           
           for i in range(n):
               # Custom computation logic
               result[i] = func(x[i]) * (h ** self.order.value)
               
           return result

   # Use custom derivative
   custom_deriv = CustomFractionalDerivative(FractionalOrder(0.5))
   x = np.linspace(0, 1, 100)
   result = custom_deriv.compute(lambda x: np.sin(x), x)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

Optimize performance for large-scale computations:

.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType
   import numpy as np
   import time

   # Benchmark different backends
   def benchmark_backend(backend_type):
       BackendManager.set_backend(backend_type)
       
       # Large dataset
       X = np.random.randn(10000, 100)
       
       start_time = time.time()
       # Perform computation
       result = np.sum(X, axis=1)
       end_time = time.time()
       
       return end_time - start_time

   # Compare performance
   torch_time = benchmark_backend(BackendType.TORCH)
   jax_time = benchmark_backend(BackendType.JAX)
   numba_time = benchmark_backend(BackendType.NUMBA)

   print(f"PyTorch: {torch_time:.4f}s")
   print(f"JAX: {jax_time:.4f}s")
   print(f"NUMBA: {numba_time:.4f}s")

Error Handling
-------------

Handle common errors and exceptions:

.. code-block:: python

   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendManager, BackendType

   # Invalid fractional order
   try:
       invalid_order = FractionalOrder(-2.0)  # Should raise error
   except ValueError as e:
       print(f"Error: {e}")

   # Backend not available
   try:
       BackendManager.set_backend(BackendType.TORCH)
       if not BackendManager.is_backend_available(BackendType.TORCH):
           print("PyTorch backend not available")
   except Exception as e:
       print(f"Backend error: {e}")

Best Practices
-------------

Code Organization
~~~~~~~~~~~~~~~~

Organize your HPFRACC code effectively:

.. code-block:: python

   # 1. Import organization
   import numpy as np
   import matplotlib.pyplot as plt
   
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.ml.backends import BackendManager, BackendType

   # 2. Configuration
   class Config:
       FRACTIONAL_ORDER = 0.5
       BACKEND = BackendType.JAX
       INPUT_DIM = 10
       HIDDEN_DIMS = [32, 16]
       OUTPUT_DIM = 1

   # 3. Setup
   BackendManager.set_backend(Config.BACKEND)
   alpha = FractionalOrder(Config.FRACTIONAL_ORDER)

   # 4. Model creation
   model = FractionalNeuralNetwork(
       input_dim=Config.INPUT_DIM,
       hidden_dims=Config.HIDDEN_DIMS,
       output_dim=Config.OUTPUT_DIM,
       fractional_order=alpha
   )

Memory Management
~~~~~~~~~~~~~~~~

Optimize memory usage for large computations:

.. code-block:: python

   import gc
   import numpy as np
   from hpfracc.ml.backends import BackendManager, BackendType

   # Use JAX for memory efficiency
   BackendManager.set_backend(BackendType.JAX)

   # Process data in batches
   def process_in_batches(data, batch_size=1000):
       results = []
       for i in range(0, len(data), batch_size):
           batch = data[i:i+batch_size]
           # Process batch
           result = np.sum(batch, axis=1)
           results.append(result)
           
           # Clear memory
           del batch
           gc.collect()
       
       return np.concatenate(results)

   # Large dataset
   large_data = np.random.randn(100000, 100)
   results = process_in_batches(large_data)

Debugging
---------

Debug HPFRACC applications effectively:

.. code-block:: python

   import logging
   from hpfracc.ml.backends import BackendManager

   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)

   # Debug backend information
   logger.debug(f"Available backends: {BackendManager.get_available_backends()}")
   logger.debug(f"Current backend: {BackendManager.get_current_backend()}")

   # Debug model parameters
   def debug_model(model):
       logger.debug(f"Model type: {type(model)}")
       logger.debug(f"Input dimension: {model.input_dim}")
       logger.debug(f"Output dimension: {model.output_dim}")
       logger.debug(f"Fractional order: {model.fractional_order}")

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Backend not available**
.. code-block:: python

   # Solution: Check and install dependencies
   from hpfracc.ml.backends import BackendManager
   
   available = BackendManager.get_available_backends()
   print(f"Available: {available}")
   
   # Install missing backend
   # pip install torch  # for PyTorch
   # pip install jax jaxlib  # for JAX
   # pip install numba  # for NUMBA

**Issue: Memory errors with large datasets**
.. code-block:: python

   # Solution: Use batch processing
   def process_large_dataset(data, batch_size=1000):
       results = []
       for i in range(0, len(data), batch_size):
           batch = data[i:i+batch_size]
           # Process batch
           results.append(process_batch(batch))
       return np.concatenate(results)

**Issue: Slow performance**
.. code-block:: python

   # Solution: Choose appropriate backend
   from hpfracc.ml.backends import BackendType
   
   # For GPU acceleration
   BackendManager.set_backend(BackendType.TORCH)  # or BackendType.JAX
   
   # For CPU optimization
   BackendManager.set_backend(BackendType.NUMBA)

Getting Help
-----------

If you encounter issues:

1. **Check the documentation**: Visit the full documentation at https://fractional-calculus-library.readthedocs.io
2. **Search existing issues**: Check GitHub issues for similar problems
3. **Create a new issue**: Report bugs or request features on GitHub
4. **Contact support**: Email d.r.chin@pgr.reading.ac.uk for academic inquiries

Next Steps
----------

Now that you're familiar with the basics:

1. **Explore Examples**: Check the examples directory for practical applications
2. **Read API Reference**: Understand all available functions and classes
3. **Study Model Theory**: Learn the mathematical foundations
4. **Contribute**: Help improve the library by contributing code or documentation

For advanced usage, see the :doc:`api_reference` and :doc:`examples` sections.
