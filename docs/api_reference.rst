API Reference
============

This section provides comprehensive documentation for all functions, classes, and methods in the HPFRACC library.

Core Module
----------

Fractional Order Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc
   :members:
   :undoc-members:
   :show-inheritance:

Core Algorithms
~~~~~~~~~~~~~~

.. automodule:: hpfracc.algorithms.optimized_methods
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.algorithms.advanced_methods
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.algorithms.special_methods
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.core.fractional_implementations
   :members:
   :undoc-members:
   :show-inheritance:

Core Derivatives
~~~~~~~~~~~~~~~

.. automodule:: hpfracc.core.derivatives
   :members:
   :undoc-members:
   :show-inheritance:

Core Integrals
~~~~~~~~~~~~~

.. automodule:: hpfracc.core.integrals
   :members:
   :undoc-members:
   :show-inheritance:

Machine Learning Module
----------------------

Backend Management
~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.backends
   :members:
   :undoc-members:
   :show-inheritance:

Tensor Operations
~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.tensor_ops
   :members:
   :undoc-members:
   :show-inheritance:

Core ML Components
~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.core
   :members:
   :undoc-members:
   :show-inheritance:

Neural Network Layers
~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.layers
   :members:
   :undoc-members:
   :show-inheritance:

Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.gnn_layers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.ml.gnn_models
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.losses
   :members:
   :undoc-members:
   :show-inheritance:

Optimizers
~~~~~~~~~

.. automodule:: hpfracc.ml.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

Detailed API Documentation
-------------------------

Core Definitions
~~~~~~~~~~~~~~~

FractionalOrder
^^^^^^^^^^^^^^

.. autoclass:: hpfracc.FractionalOrder
   :members:
   :undoc-members:
   :special-members: __init__, __str__, __repr__

   .. automethod:: __init__

Core Fractional Calculus Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptimizedRiemannLiouville
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.OptimizedRiemannLiouville
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

OptimizedCaputo
^^^^^^^^^^^^^^

.. autoclass:: hpfracc.OptimizedCaputo
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

OptimizedGrunwaldLetnikov
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.OptimizedGrunwaldLetnikov
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: compute

RiemannLiouvilleDerivative
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.core.derivatives.RiemannLiouvilleDerivative
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: compute

CaputoDerivative
^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.core.derivatives.CaputoDerivative
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: compute

GrunwaldLetnikovDerivative
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.core.derivatives.GrunwaldLetnikovDerivative
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: compute

Backend Management
~~~~~~~~~~~~~~~~~

BackendType
^^^^^^^^^^

.. autoclass:: hpfracc.ml.backends.BackendType
   :members:
   :undoc-members:

BackendManager
^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.backends.BackendManager
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: set_backend
   .. automethod:: get_current_backend
   .. automethod:: get_available_backends
   .. automethod:: is_backend_available

Tensor Operations
~~~~~~~~~~~~~~~~

TensorOps
^^^^^^^^^

.. autoclass:: hpfracc.ml.tensor_ops.TensorOps
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: zeros
   .. automethod:: ones
   .. automethod:: random_normal
   .. automethod:: matmul
   .. automethod:: transpose
   .. automethod:: sum
   .. automethod:: mean
   .. automethod:: sqrt
   .. automethod:: exp
   .. automethod:: log
   .. automethod:: sin
   .. automethod:: cos
   .. automethod:: tanh
   .. automethod:: relu
   .. automethod:: sigmoid
   .. automethod:: softmax
   .. automethod:: dropout
   .. automethod:: batch_norm

Neural Networks
~~~~~~~~~~~~~~

FractionalNeuralNetwork
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.neural_networks.FractionalNeuralNetwork
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_parameters
   .. automethod:: set_parameters

FractionalLayer
^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.neural_networks.FractionalLayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_weights
   .. automethod:: set_weights

Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~

FractionalGCN
^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGCN
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_parameters

FractionalGAT
^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGAT
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_parameters

FractionalGraphSAGE
^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGraphSAGE
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_parameters

FractionalGraphUNet
^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGraphUNet
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_parameters

GNN Factory
^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGNNFactory
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: create_model

GNN Layers
^^^^^^^^^^

FractionalGCNLayer
^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_layers.FractionalGCNLayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

FractionalGATLayer
^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_layers.FractionalGATLayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

FractionalGraphSAGELayer
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_layers.FractionalGraphSAGELayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

Attention Mechanisms
~~~~~~~~~~~~~~~~~~~

FractionalAttention
^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.attention.FractionalAttention
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_attention_weights

Utility Functions
----------------

Fractional Derivative Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hpfracc.core.derivatives.create_fractional_derivative

.. autofunction:: hpfracc.core.derivatives.riemann_liouville

.. autofunction:: hpfracc.core.derivatives.caputo

.. autofunction:: hpfracc.core.derivatives.grunwald_letnikov

Backend Utilities
~~~~~~~~~~~~~~~~

.. autofunction:: hpfracc.ml.backends.get_backend_ops

.. autofunction:: hpfracc.ml.backends.set_default_backend

.. autofunction:: hpfracc.ml.backends.check_backend_compatibility

Tensor Utilities
~~~~~~~~~~~~~~~

.. autofunction:: hpfracc.ml.tensor_ops.create_tensor_ops

.. autofunction:: hpfracc.ml.tensor_ops.convert_tensor

.. autofunction:: hpfracc.ml.tensor_ops.get_tensor_info

Model Utilities
~~~~~~~~~~~~~~

.. autofunction:: hpfracc.ml.neural_networks.create_fractional_model

.. autofunction:: hpfracc.ml.gnn_models.create_gnn_model

.. autofunction:: hpfracc.ml.attention.create_attention_model

Configuration
-------------

Default Parameters
~~~~~~~~~~~~~~~~~

.. data:: hpfracc.core.definitions.DEFAULT_FRACTIONAL_ORDER
   :annotation: = 0.5

.. data:: hpfracc.ml.backends.DEFAULT_BACKEND
   :annotation: = BackendType.JAX

.. data:: hpfracc.ml.tensor_ops.DEFAULT_DTYPE
   :annotation: = 'float32'

Supported Backends
~~~~~~~~~~~~~~~~~

.. data:: hpfracc.ml.backends.SUPPORTED_BACKENDS
   :annotation: = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]

Supported GNN Types
~~~~~~~~~~~~~~~~~~

.. data:: hpfracc.ml.gnn_models.SUPPORTED_GNN_TYPES
   :annotation: = ['gcn', 'gat', 'sage', 'unet']

Supported Derivative Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: hpfracc.core.derivatives.SUPPORTED_METHODS
   :annotation: = ['RL', 'Caputo', 'GL']

Error Classes
------------

.. autoclass:: hpfracc.core.exceptions.FractionalOrderError
   :members:
   :undoc-members:

.. autoclass:: hpfracc.core.exceptions.BackendError
   :members:
   :undoc-members:

.. autoclass:: hpfracc.core.exceptions.TensorError
   :members:
   :undoc-members:

.. autoclass:: hpfracc.core.exceptions.ModelError
   :members:
   :undoc-members:

Type Hints
----------

Core Types
~~~~~~~~~~

.. autodata:: hpfracc.core.types.FractionalOrderType
   :annotation: Union[float, FractionalOrder]

.. autodata:: hpfracc.core.types.BackendTypeType
   :annotation: Union[str, BackendType]

.. autodata:: hpfracc.core.types.TensorType
   :annotation: Union[np.ndarray, torch.Tensor, jax.numpy.ndarray]

ML Types
~~~~~~~~

.. autodata:: hpfracc.ml.types.ModelType
   :annotation: Union[FractionalNeuralNetwork, FractionalGCN, FractionalGAT, FractionalGraphSAGE, FractionalGraphUNet]

.. autodata:: hpfracc.ml.types.LayerType
   :annotation: Union[FractionalLayer, FractionalGCNLayer, FractionalGATLayer, FractionalGraphSAGELayer]

.. autodata:: hpfracc.ml.types.AttentionType
   :annotation: FractionalAttention

Usage Examples
-------------

Basic Fractional Calculus
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative
   import numpy as np

   # Create fractional derivative
   alpha = FractionalOrder(0.5)
   deriv = create_fractional_derivative(alpha, method="RL")

   # Test function
   def f(x):
       return np.exp(-x)

   # Compute derivative
   x = np.linspace(0, 1, 100)
   result = deriv(f, x)

Neural Network Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType
   import numpy as np

   # Create model
   model = FractionalNeuralNetwork(
       input_dim=10,
       hidden_dims=[32, 16],
       output_dim=1,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.JAX
   )

   # Forward pass
   X = np.random.randn(100, 10)
   output = model.forward(X)

GNN Usage
~~~~~~~~~

.. code-block:: python

   from hpfracc.ml import FractionalGNNFactory
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType
   import numpy as np

   # Create GNN
   gnn = FractionalGNNFactory.create_model(
       model_type='gcn',
       input_dim=16,
       hidden_dim=32,
       output_dim=4,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.TORCH
   )

   # Graph data
   node_features = np.random.randn(50, 16)
   edge_index = np.random.randint(0, 50, (2, 100))

   # Forward pass
   output = gnn.forward(node_features, edge_index)

Backend Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType

   # Check available backends
   available = BackendManager.get_available_backends()
   print(f"Available: {available}")

   # Set backend
   BackendManager.set_backend(BackendType.JAX)

   # Get current backend
   current = BackendManager.get_current_backend()
   print(f"Current: {current}")

Performance Considerations
-------------------------

Backend Selection
~~~~~~~~~~~~~~~~

- **PyTorch**: Best for GPU acceleration and complex neural networks
- **JAX**: Best for functional programming and TPU acceleration
- **NUMBA**: Best for CPU optimization and lightweight deployment

Memory Management
~~~~~~~~~~~~~~~~

- Use batch processing for large datasets
- Clear intermediate tensors when possible
- Monitor memory usage with large models

Computation Optimization
~~~~~~~~~~~~~~~~~~~~~~~

- Choose appropriate fractional derivative method for your use case
- Use JIT compilation when available
- Profile performance with different backends

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Backend not available**
.. code-block:: python

   # Check available backends
   from hpfracc.ml.backends import BackendManager
   available = BackendManager.get_available_backends()
   print(f"Available: {available}")

**Invalid fractional order**
.. code-block:: python

   # Valid orders: -1 < order < 2
   from hpfracc.core.definitions import FractionalOrder
   try:
       order = FractionalOrder(0.5)  # Valid
   except ValueError as e:
       print(f"Error: {e}")

**Tensor shape mismatch**
.. code-block:: python

   # Ensure input dimensions match model expectations
   model = FractionalNeuralNetwork(input_dim=10, ...)
   X = np.random.randn(100, 10)  # Correct shape
   # X = np.random.randn(100, 5)  # Wrong shape - will fail

Debugging Tips
~~~~~~~~~~~~~

1. **Enable debug logging**
2. **Check tensor shapes and types**
3. **Verify backend compatibility**
4. **Test with small datasets first**

For more detailed examples, see the :doc:`examples` section.
