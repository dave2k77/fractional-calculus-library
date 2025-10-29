Fractional Neural Networks
==========================

HPFRACC provides comprehensive support for fractional neural networks with spectral autograd, stochastic memory sampling, and probabilistic fractional orders. This chapter consolidates all neural network-related functionality.

For complete API documentation, see :doc:`api/fnn_api`.

Overview
--------

The Fractional Neural Network framework in HPFRACC extends standard neural networks by incorporating fractional calculus operations that preserve gradient flow through automatic differentiation.

Key Components:
- **Spectral Autograd Framework**: FFT, Mellin, and Laplacian transforms for efficient computation
- **Stochastic Memory Sampling**: Approximate fractional operators by sampling from memory history
- **Probabilistic Fractional Orders**: Treat fractional orders as random variables for uncertainty quantification
- **GPU Optimization**: Automatic GPU acceleration with memory management

Spectral Autograd Framework
----------------------------

The spectral autograd framework is the core innovation that enables gradient flow through fractional derivatives.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

   # Create input with gradient support
   x = torch.randn(32, 128, requires_grad=True)
   alpha = 0.5  # fractional order

   # Apply spectral fractional derivative
   result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
   
   # Gradients flow properly through fractional derivatives
   loss = torch.sum(result)
   loss.backward()
   
   print(f"Input gradient norm: {x.grad.norm().item():.6f}")

Engine Types
~~~~~~~~~~~~

**FFT Engine**: Best for large sequences, periodic functions
- O(N log N) complexity
- Frequency domain multiplication

**Mellin Engine**: Best for power-law functions, scale-invariant problems
- O(N log N) complexity
- Mellin transform domain

**Laplacian Engine**: Best for spatial problems, diffusion equations
- O(N log N) complexity
- Fractional Laplacian in frequency domain

See :doc:`fractional_autograd_guide` for detailed engine documentation.

Stochastic Memory Sampling
---------------------------

Approximate fractional operators by sampling from memory history for memory-efficient computation.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.stochastic_memory_sampling import StochasticFractionalLayer

   # Create stochastic fractional layer
   layer = StochasticFractionalLayer(
       alpha=0.5,
       k=32,  # Number of samples
       method="importance"  # or "stratified", "control_variate"
   )

   # Forward pass
   x = torch.randn(32, 128)
   output = layer(x)

Sampling Methods
~~~~~~~~~~~~~~~~

- **Importance Sampling**: General purpose, power-law distributions
- **Stratified Sampling**: When recent history is important
- **Control Variate Sampling**: When baseline estimates are available

Probabilistic Fractional Orders
-------------------------------

Treat fractional orders as random variables for uncertainty quantification.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer

   # Create probabilistic fractional layer
   layer = create_normal_alpha_layer(
       mean=0.5,
       std=0.1,
       learnable=True
   )

   # Forward pass
   x = torch.randn(32, 128)
   output = layer(x)

Distribution Types
~~~~~~~~~~~~~~~~~~

- **Normal Distribution**: Continuous fractional orders
- **Uniform Distribution**: Bounded fractional orders
- **Beta Distribution**: Fractional orders in [0, 1]

Neural Fractional ODE Framework
-------------------------------

Learning-based solution of fractional differential equations.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import hpfracc.ml.neural_ode as nfode
   import torch

   # Create a neural ODE model
   model = nfode.NeuralODE(
       input_dim=2,      # Input dimension
       hidden_dim=32,    # Hidden layer dimension
       output_dim=1,     # Output dimension
       num_layers=3,     # Number of hidden layers
       activation="tanh" # Activation function
   )

   # Create input data
   x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Batch of initial conditions
   t = torch.linspace(0, 1, 100)              # Time points

   # Forward pass
   solution = model(x, t)
   print(f"Solution shape: {solution.shape}")  # (batch_size, time_steps, output_dim)

Fractional Neural ODE
~~~~~~~~~~~~~~~~~~~~

Extend to fractional calculus with configurable fractional order:

.. code-block:: python

   # Fractional neural ODE
   fode_model = nfode.NeuralFODE(
       input_dim=2,
       hidden_dim=32,
       output_dim=1,
       fractional_order=0.5,  # Fractional order α
       num_layers=3,
       activation="tanh"
   )

   solution = fode_model(x, t)

See :doc:`neural_fode_guide` for comprehensive neural fODE documentation.

Complete Examples
-----------------

Training Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder
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

GPU Optimization
-----------------

Automatic GPU acceleration with intelligent memory management.

.. code-block:: python

   from hpfracc.ml.gpu_optimization import GPUOptimizedSpectralEngine
   import torch

   # Automatic chunking for large arrays
   x = torch.randn(100000, requires_grad=True)
   engine = GPUOptimizedSpectralEngine(chunk_size=8192)
   
   # Computes in chunks automatically
   result = engine.apply_spectral_transform(x, alpha=0.5)

Summary
-------

The Fractional Neural Networks framework provides:

✅ **Spectral Autograd**: FFT, Mellin, Laplacian engines for O(N log N) complexity  
✅ **Stochastic Memory**: Memory-efficient sampling methods  
✅ **Probabilistic Orders**: Uncertainty quantification through random fractional orders  
✅ **Neural fODEs**: Learning-based fractional ODE solving  
✅ **GPU Acceleration**: Automatic optimization with memory management  

Next Steps
----------

- **API Reference**: See :doc:`api/fnn_api` for complete API documentation
- **Examples**: Check :doc:`05_advanced_examples` for ML integration examples
- **Autograd Guide**: See :doc:`fractional_autograd_guide` for detailed autograd documentation
- **Neural fODE Guide**: See :doc:`neural_fode_guide` for neural ODE framework

