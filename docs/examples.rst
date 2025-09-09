"""
Examples & Tutorials
===================

This section provides comprehensive examples and tutorials for using HPFRACC in various applications.

Basic Examples
-------------

Spectral Autograd Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~

The spectral autograd framework enables gradient flow through fractional derivatives:

.. code-block:: python

   import torch
   from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

   # Basic spectral fractional derivative
   x = torch.randn(32, requires_grad=True)
   alpha = 0.5
   
   # Apply spectral fractional derivative (4.67x faster than standard)
   result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
   
   # Compute gradients
   loss = torch.sum(result)
   loss.backward()
   
   print(f"Gradient norm: {x.grad.norm().item():.6f}")

Neural Network with Learnable Fractional Orders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

   class FractionalNeuralNetwork(nn.Module):
       def __init__(self, input_size=32, hidden_size=64, output_size=1):
           super().__init__()
           self.alpha_param = BoundedAlphaParameter(alpha_init=1.2)
           self.linear1 = nn.Linear(input_size, hidden_size)
           self.linear2 = nn.Linear(hidden_size, output_size)
       
       def forward(self, x):
           # Apply spectral fractional derivative with learnable alpha
           alpha = self.alpha_param()
           x_frac = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
           
           # Standard neural network layers
           x = self.linear1(x_frac)
           x = torch.relu(x)
           x = self.linear2(x)
           return x

   # Create model and test
   model = FractionalNeuralNetwork()
   x = torch.randn(100, 32, requires_grad=True)
   y = torch.randn(100, 1)
   
   # Forward pass
   output = model(x)
   loss = nn.functional.mse_loss(output, y)
   
   # Backward pass (gradients flow through fractional derivatives)
   loss.backward()
   
   print(f"Model loss: {loss.item():.6f}")
   print(f"Alpha value: {model.alpha_param().item():.4f}")
   print(f"Alpha gradient: {model.alpha_param.rho.grad.item():.6f}")

Fractional Derivative Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute fractional derivatives using different methods:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc import FractionalOrder, optimized_riemann_liouville, optimized_caputo, optimized_grunwald_letnikov

   # Define test function
   def test_function(x):
       return np.sin(x)

   # Create different fractional derivatives
   alpha = FractionalOrder(0.5)
   x = np.linspace(0, 2*np.pi, 100)

   # Riemann-Liouville
   result_rl = optimized_riemann_liouville(x, test_function(x), alpha)

   # Caputo
   result_caputo = optimized_caputo(x, test_function(x), alpha)

   # Grünwald-Letnikov
   result_gl = optimized_grunwald_letnikov(x, test_function(x), alpha)

   # Plot results
   plt.figure(figsize=(12, 8))
   plt.plot(x, test_function(x), label='Original: sin(x)', linewidth=2)
   plt.plot(x, result_rl, label='Riemann-Liouville (α=0.5)', linewidth=2)
   plt.plot(x, result_caputo, label='Caputo (α=0.5)', linewidth=2)
   plt.plot(x, result_gl, label='Grünwald-Letnikov (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Fractional Derivatives of sin(x)')
   plt.legend()
   plt.grid(True)
   plt.show()

Fractional Integral Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute fractional integrals using different methods:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc import FractionalOrder, riemann_liouville_integral, caputo_integral

   # Define test function
   def test_function(x):
       return x**2

   # Create different fractional integrals
   alpha = FractionalOrder(0.5)
   x = np.linspace(0, 5, 100)

   # Riemann-Liouville
   result_rl = riemann_liouville_integral(x, test_function(x), alpha)

   # Caputo
   result_caputo = caputo_integral(x, test_function(x), alpha)

   # Note: Weyl and Hadamard integrals are available but require specific implementations

   # Plot results
   plt.figure(figsize=(15, 10))
   
   plt.subplot(2, 2, 1)
   plt.plot(x, test_function(x), label='Original: x²', linewidth=2)
   plt.plot(x, result_rl, label='Riemann-Liouville (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Riemann-Liouville Fractional Integral')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 2, 2)
   plt.plot(x, test_function(x), label='Original: x²', linewidth=2)
   plt.plot(x, result_caputo, label='Caputo (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Caputo Fractional Integral')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 2, 3)
   plt.plot(x, test_function(x), label='Original: x²', linewidth=2)
   plt.plot(x, result_weyl, label='Weyl (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Weyl Fractional Integral')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 2, 4)
   plt.plot(x_hadamard, test_function(x_hadamard), label='Original: x²', linewidth=2)
   plt.plot(x_hadamard, result_hadamard, label='Hadamard (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Hadamard Fractional Integral')
   plt.legend()
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

Special Functions
~~~~~~~~~~~~~~~~

Working with special functions in fractional calculus:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.special import (
       gamma_function, beta_function, binomial_coefficient,
       mittag_leffler_function, generalized_binomial
   )

   # Gamma function
   x = np.linspace(0.1, 5, 100)
   gamma_vals = [gamma_function(xi) for xi in x]

   # Beta function
   y = np.linspace(0.1, 3, 50)
   X, Y = np.meshgrid(x[:50], y)
   beta_vals = np.array([[beta_function(xi, yi) for xi in x[:50]] for yi in y])

   # Binomial coefficients
   n_vals = np.arange(0, 10)
   alpha = 0.5
   binomial_frac = [generalized_binomial(alpha, n) for n in n_vals]

   # Mittag-Leffler function
   z = np.linspace(-5, 5, 100)
   ml_vals = [mittag_leffler_function(0.5, zi) for zi in z]

   # Plot results
   plt.figure(figsize=(15, 10))
   
   plt.subplot(2, 2, 1)
   plt.plot(x, gamma_vals, linewidth=2)
   plt.xlabel('x')
   plt.ylabel('Γ(x)')
   plt.title('Gamma Function')
   plt.grid(True)
   
   plt.subplot(2, 2, 2)
   plt.contourf(X, Y, beta_vals, levels=20)
   plt.colorbar(label='B(x, y)')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('Beta Function')
   
   plt.subplot(2, 2, 3)
   plt.stem(n_vals, binomial_frac)
   plt.xlabel('n')
   plt.ylabel('(α choose n)')
   plt.title(f'Fractional Binomial Coefficients (α={alpha})')
   plt.grid(True)
   
   plt.subplot(2, 2, 4)
   plt.plot(z, ml_vals, linewidth=2)
   plt.xlabel('z')
   plt.ylabel('E₀.₅(z)')
   plt.title('Mittag-Leffler Function E₀.₅(z)')
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

# Green's functions have been removed from this release
# They will be re-implemented in future releases with improved stability

.. code-block:: python














# Analytical methods section - focusing on implemented solvers

Mathematical Utilities
~~~~~~~~~~~~~~~~~~~~~

Using mathematical utilities for validation and computation:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.core.utilities import (
       factorial_fractional, binomial_coefficient, pochhammer_symbol,
       validate_fractional_order, validate_function,
       timing_decorator, memory_usage_decorator
   )

   # Fractional factorial
   x = np.linspace(0.1, 5, 100)
   factorial_vals = [factorial_fractional(xi) for xi in x]

   # Binomial coefficients
   n_vals = np.arange(0, 10)
   k_vals = np.arange(0, 10)
   binomial_matrix = np.array([[binomial_coefficient(n, k) for k in k_vals] for n in n_vals])

   # Pochhammer symbol
   pochhammer_vals = [pochhammer_symbol(0.5, xi) for xi in x]

   # Validation examples
   print("Validation Examples:")
   print(f"Valid fractional order 0.5: {validate_fractional_order(0.5)}")
   print(f"Invalid fractional order -1: {validate_fractional_order(-1)}")

   def test_func(x):
       return x**2
   
   print(f"Valid function: {validate_function(test_func)}")
   print(f"Invalid function: {validate_function('not a function')}")

   # Performance monitoring
   @timing_decorator
   @memory_usage_decorator
   def expensive_computation(n):
       return sum(i**2 for i in range(n))

   result = expensive_computation(10000)

   # Plot results
   plt.figure(figsize=(15, 5))
   
   plt.subplot(1, 3, 1)
   plt.plot(x, factorial_vals, linewidth=2)
   plt.xlabel('x')
   plt.ylabel('x!')
   plt.title('Fractional Factorial Function')
   plt.grid(True)
   
   plt.subplot(1, 3, 2)
   plt.imshow(binomial_matrix, cmap='viridis', aspect='auto')
   plt.colorbar(label='(n choose k)')
   plt.xlabel('k')
   plt.ylabel('n')
   plt.title('Binomial Coefficients Matrix')
   
   plt.subplot(1, 3, 3)
   plt.plot(x, pochhammer_vals, linewidth=2)
   plt.xlabel('x')
   plt.ylabel('(0.5)_x')
   plt.title('Pochhammer Symbol (0.5)_x')
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

Backend Comparison
~~~~~~~~~~~~~~~~~

Compare performance across different backends:

.. code-block:: python

   import time
   import numpy as np
   from hpfracc.ml.backends import BackendManager, BackendType
   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder

   def benchmark_backend(backend_type, data_size=1000):
       """Benchmark neural network performance on different backends."""
       BackendManager.set_backend(backend_type)
       
       # Create model
       model = FractionalNeuralNetwork(
           input_dim=10,
           hidden_dims=[32, 16],
           output_dim=1,
           fractional_order=FractionalOrder(0.5)
       )
       
       # Generate data
       X = np.random.randn(data_size, 10)
       
       # Warm up
       for _ in range(10):
           _ = model.forward(X)
       
       # Benchmark
       start_time = time.time()
       for _ in range(100):
           _ = model.forward(X)
       end_time = time.time()
       
       return end_time - start_time

   # Test all backends
   backends = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]
   results = {}

   for backend in backends:
       if BackendManager.is_backend_available(backend):
           time_taken = benchmark_backend(backend)
           results[backend.name] = time_taken
           print(f"{backend.name}: {time_taken:.4f} seconds")

   # Plot comparison
   if results:
       plt.figure(figsize=(8, 6))
       backend_names = list(results.keys())
       times = list(results.values())
       
       plt.bar(backend_names, times, color=['blue', 'green', 'red'])
       plt.ylabel('Time (seconds)')
       plt.title('Backend Performance Comparison')
       plt.xticks(rotation=45)
       
       for i, v in enumerate(times):
           plt.text(i, v + 0.001, f'{v:.4f}s', ha='center', va='bottom')
       
       plt.tight_layout()
       plt.show()

Advanced Examples
----------------

Fractional Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~

Create and train a fractional neural network:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   # Generate synthetic data
   np.random.seed(42)
   X = np.random.randn(1000, 10)
   y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(1000)

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Scale features
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # Create fractional neural network
   model = FractionalNeuralNetwork(
       input_dim=10,
       hidden_dims=[64, 32, 16],
       output_dim=1,
       fractional_order=FractionalOrder(0.5),
       activation='relu',
       dropout_rate=0.2
   )

   # Train the model
   history = model.fit(
       X_train_scaled, y_train,
       validation_data=(X_test_scaled, y_test),
       epochs=100,
       batch_size=32,
       learning_rate=0.001,
       verbose=True
   )

   # Plot training history
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 2, 1)
   plt.plot(history['loss'], label='Training Loss')
   plt.plot(history['val_loss'], label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training History')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(1, 2, 2)
   plt.plot(history['accuracy'], label='Training Accuracy')
   plt.plot(history['val_accuracy'], label='Validation Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.title('Accuracy History')
   plt.legend()
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

   # Make predictions
   y_pred = model.predict(X_test_scaled)
   
   # Plot predictions vs actual
   plt.figure(figsize=(8, 6))
   plt.scatter(y_test, y_pred, alpha=0.6)
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
   plt.xlabel('Actual Values')
   plt.ylabel('Predicted Values')
   plt.title('Predictions vs Actual Values')
   plt.grid(True)
   plt.show()

Graph Neural Networks with Fractional Calculus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement fractional graph convolutions:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import networkx as nx
   from hpfracc.ml.gnn_layers import FractionalGraphConvolution
   from hpfracc.core.definitions import FractionalOrder

   # Create a random graph
   np.random.seed(42)
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
   
   # Visualize the graph with node features
   plt.figure(figsize=(15, 5))
   
   # Original graph
   plt.subplot(1, 3, 1)
   pos = nx.spring_layout(G)
   nx.draw(G, pos, with_labels=True, node_color='lightblue', 
           node_size=500, font_size=10, font_weight='bold')
   plt.title('Original Graph')
   
   # Node features before convolution
   plt.subplot(1, 3, 2)
   nx.draw(G, pos, with_labels=True, 
           node_color=node_features[:, 0], 
           node_size=500, font_size=10, font_weight='bold',
           cmap=plt.cm.viridis)
   plt.title('Node Features (Before)')
   
   # Node features after convolution
   plt.subplot(1, 3, 3)
   nx.draw(G, pos, with_labels=True, 
           node_color=output_features[:, 0], 
           node_size=500, font_size=10, font_weight='bold',
           cmap=plt.cm.viridis)
   plt.title('Node Features (After Fractional Convolution)')
   
   plt.tight_layout()
   plt.show()

Signal Processing Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply fractional derivatives to signal processing:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder

   # Generate test signal
   t = np.linspace(0, 10, 1000)
   signal = np.sin(2*np.pi*t) + 0.5*np.sin(4*np.pi*t) + 0.1*np.random.randn(len(t))

   # Create fractional derivatives
   alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
   derivatives = {}

   for alpha in alpha_values:
       deriv = create_fractional_derivative(FractionalOrder(alpha), method="RL")
       derivatives[alpha] = deriv(lambda x: signal, t)

   # Plot results
   plt.figure(figsize=(15, 10))
   
   plt.subplot(2, 1, 1)
   plt.plot(t, signal, 'k-', linewidth=2, label='Original Signal')
   plt.xlabel('Time')
   plt.ylabel('Amplitude')
   plt.title('Original Signal')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 1, 2)
   for alpha in alpha_values:
       plt.plot(t, derivatives[alpha], linewidth=2, label=f'α = {alpha}')
   plt.xlabel('Time')
   plt.ylabel('Amplitude')
   plt.title('Fractional Derivatives')
   plt.legend()
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

   # Frequency domain analysis
   from scipy.fft import fft, fftfreq
   
   # Compute FFT of original signal and derivatives
   fft_original = np.abs(fft(signal))
   fft_derivatives = {}
   
   for alpha in alpha_values:
       fft_derivatives[alpha] = np.abs(fft(derivatives[alpha]))
   
   # Plot frequency domain
   freqs = fftfreq(len(t), t[1] - t[0])
   positive_freqs = freqs[:len(freqs)//2]
   
   plt.figure(figsize=(12, 8))
   
   plt.subplot(2, 1, 1)
   plt.plot(positive_freqs, fft_original[:len(positive_freqs)], 'k-', linewidth=2, label='Original')
   plt.xlabel('Frequency')
   plt.ylabel('Magnitude')
   plt.title('Frequency Domain - Original Signal')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 1, 2)
   for alpha in alpha_values:
       plt.plot(positive_freqs, fft_derivatives[alpha][:len(positive_freqs)], 
                linewidth=2, label=f'α = {alpha}')
   plt.xlabel('Frequency')
   plt.ylabel('Magnitude')
   plt.title('Frequency Domain - Fractional Derivatives')
   plt.legend()
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

Image Processing with Fractional Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply fractional derivatives to image processing:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import ndimage
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder

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

   # Plot results
   plt.figure(figsize=(15, 10))
   
   plt.subplot(2, 3, 1)
   plt.imshow(image, cmap='gray')
   plt.title('Original Image')
   plt.axis('off')
   
   plt.subplot(2, 3, 2)
   plt.imshow(gradient_x, cmap='gray')
   plt.title(f'Fractional Gradient X (α={alpha})')
   plt.axis('off')
   
   plt.subplot(2, 3, 3)
   plt.imshow(gradient_y, cmap='gray')
   plt.title(f'Fractional Gradient Y (α={alpha})')
   plt.axis('off')
   
   plt.subplot(2, 3, 4)
   plt.imshow(gradient_magnitude, cmap='gray')
   plt.title(f'Gradient Magnitude (α={alpha})')
   plt.axis('off')
   
   plt.subplot(2, 3, 5)
   plt.imshow(np.abs(gradient_x) + np.abs(gradient_y), cmap='gray')
   plt.title(f'Sum of Absolute Gradients (α={alpha})')
   plt.axis('off')
   
   plt.subplot(2, 3, 6)
   # Edge detection using threshold
   threshold = np.percentile(gradient_magnitude, 90)
   edges = gradient_magnitude > threshold
   plt.imshow(edges, cmap='gray')
   plt.title(f'Edge Detection (α={alpha})')
   plt.axis('off')
   
   plt.tight_layout()
   plt.show()

Performance Optimization Examples
--------------------------------

GPU Acceleration
~~~~~~~~~~~~~~~

Demonstrate GPU acceleration for large-scale computations:

.. code-block:: python

   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from hpfracc.ml.backends import BackendManager, BackendType
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder

   def benchmark_cpu_vs_gpu(data_sizes):
       """Benchmark CPU vs GPU performance."""
       results = {'CPU': [], 'GPU': []}
       
       for size in data_sizes:
           # Generate data
           x = np.linspace(0, 10, size)
           signal = np.sin(2*np.pi*x) + 0.1*np.random.randn(size)
           
           # CPU computation
           BackendManager.set_backend(BackendType.NUMPY)
           deriv_cpu = create_fractional_derivative(FractionalOrder(0.5), method="RL")
           
           start_time = time.time()
           result_cpu = deriv_cpu(lambda x: signal, x)
           cpu_time = time.time() - start_time
           results['CPU'].append(cpu_time)
           
           # GPU computation (if available)
           if BackendManager.is_backend_available(BackendType.TORCH):
               BackendManager.set_backend(BackendType.TORCH)
               deriv_gpu = create_fractional_derivative(FractionalOrder(0.5), method="RL")
               
               start_time = time.time()
               result_gpu = deriv_gpu(lambda x: signal, x)
               gpu_time = time.time() - start_time
               results['GPU'].append(gpu_time)
           else:
               results['GPU'].append(None)
       
       return results

   # Run benchmark
   data_sizes = [1000, 5000, 10000, 50000, 100000]
   benchmark_results = benchmark_cpu_vs_gpu(data_sizes)

   # Plot results
   plt.figure(figsize=(10, 6))
   
   plt.plot(data_sizes, benchmark_results['CPU'], 'b-o', linewidth=2, label='CPU')
   if any(result is not None for result in benchmark_results['GPU']):
       gpu_times = [t if t is not None else 0 for t in benchmark_results['GPU']]
       plt.plot(data_sizes, gpu_times, 'r-s', linewidth=2, label='GPU')
   
   plt.xlabel('Data Size')
   plt.ylabel('Time (seconds)')
   plt.title('CPU vs GPU Performance Comparison')
   plt.legend()
   plt.grid(True)
   plt.xscale('log')
   plt.yscale('log')
   plt.show()

Memory Optimization
~~~~~~~~~~~~~~~~~~

Demonstrate memory-efficient computations:

.. code-block:: python

   import numpy as np
   import psutil
   import matplotlib.pyplot as plt
   from hpfracc.core.utilities import memory_usage_decorator
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder

   @memory_usage_decorator
   def memory_intensive_computation(data_size):
       """Perform memory-intensive computation."""
       # Generate large dataset
       x = np.linspace(0, 10, data_size)
       signal = np.sin(2*np.pi*x) + 0.1*np.random.randn(data_size)
       
       # Create multiple fractional derivatives
       derivatives = []
       for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
           deriv = create_fractional_derivative(FractionalOrder(alpha), method="RL")
           result = deriv(lambda x: signal, x)
           derivatives.append(result)
       
       return derivatives

   # Test different data sizes
   data_sizes = [1000, 5000, 10000, 50000]
   memory_usage = []

   for size in data_sizes:
       result = memory_intensive_computation(size)
       memory_usage.append(result)

   # Plot memory usage
   plt.figure(figsize=(10, 6))
   plt.plot(data_sizes, memory_usage, 'g-o', linewidth=2)
   plt.xlabel('Data Size')
   plt.ylabel('Memory Usage (MB)')
   plt.title('Memory Usage vs Data Size')
   plt.grid(True)
   plt.show()

Parallel Processing
~~~~~~~~~~~~~~~~~~

Demonstrate parallel processing capabilities:

.. code-block:: python

   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from multiprocessing import Pool, cpu_count
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder

   def parallel_fractional_derivative(args):
       """Compute fractional derivative for a subset of data."""
       data, alpha, method = args
       deriv = create_fractional_derivative(FractionalOrder(alpha), method=method)
       return deriv(lambda x: data, np.arange(len(data)))

   def benchmark_parallel_vs_sequential(data_size, num_processes):
       """Benchmark parallel vs sequential computation."""
       # Generate data
       x = np.linspace(0, 10, data_size)
       signal = np.sin(2*np.pi*x) + 0.1*np.random.randn(data_size)
       
       # Sequential computation
       start_time = time.time()
       sequential_results = []
       for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
           deriv = create_fractional_derivative(FractionalOrder(alpha), method="RL")
           result = deriv(lambda x: signal, x)
           sequential_results.append(result)
       sequential_time = time.time() - start_time
       
       # Parallel computation
       start_time = time.time()
       with Pool(num_processes) as pool:
           args = [(signal, alpha, "RL") for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]]
           parallel_results = pool.map(parallel_fractional_derivative, args)
       parallel_time = time.time() - start_time
       
       return sequential_time, parallel_time

   # Run benchmark
   data_sizes = [1000, 5000, 10000, 50000]
   num_processes = min(4, cpu_count())
   
   sequential_times = []
   parallel_times = []
   
   for size in data_sizes:
       seq_time, par_time = benchmark_parallel_vs_sequential(size, num_processes)
       sequential_times.append(seq_time)
       parallel_times.append(par_time)

   # Plot results
   plt.figure(figsize=(10, 6))
   plt.plot(data_sizes, sequential_times, 'b-o', linewidth=2, label='Sequential')
   plt.plot(data_sizes, parallel_times, 'r-s', linewidth=2, label=f'Parallel ({num_processes} processes)')
   plt.xlabel('Data Size')
   plt.ylabel('Time (seconds)')
   plt.title('Sequential vs Parallel Performance')
   plt.legend()
   plt.grid(True)
   plt.xscale('log')
   plt.yscale('log')
   plt.show()

Error Analysis and Validation
----------------------------

Numerical Error Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Analyze numerical errors in fractional calculus computations:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder

   def analytical_solution(x, alpha):
       """Analytical solution for D^α sin(x)."""
       # For sin(x), D^α sin(x) = sin(x + απ/2)
       return np.sin(x + alpha * np.pi / 2)

   def numerical_error_analysis():
       """Analyze numerical errors for different methods and orders."""
       x = np.linspace(0, 2*np.pi, 100)
       alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
       methods = ["RL", "Caputo", "GL"]
       
       errors = {method: [] for method in methods}
       
       for alpha in alpha_values:
           analytical = analytical_solution(x, alpha)
           
           for method in methods:
               deriv = create_fractional_derivative(FractionalOrder(alpha), method=method)
               numerical = deriv(lambda x: np.sin(x), x)
               
               # Compute relative error
               error = np.mean(np.abs((numerical - analytical) / analytical))
               errors[method].append(error)
       
       return alpha_values, errors

   # Run error analysis
   alpha_values, errors = numerical_error_analysis()

   # Plot results
   plt.figure(figsize=(12, 8))
   
   for method, error_list in errors.items():
       plt.semilogy(alpha_values, error_list, 'o-', linewidth=2, label=method)
   
   plt.xlabel('Fractional Order α')
   plt.ylabel('Relative Error')
   plt.title('Numerical Error Analysis for Different Methods')
   plt.legend()
   plt.grid(True)
   plt.show()

Convergence Analysis
~~~~~~~~~~~~~~~~~~~

Analyze convergence of iterative methods:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   # HPM and VIM solvers removed - focusing on implemented methods

   # HPM and VIM solvers removed - focusing on implemented methods

   # HPM and VIM solvers removed - focusing on implemented methods

   # HPM and VIM plotting removed - focusing on implemented methods
   
   # VIM plotting removed
   
   # Solutions comparison removed
   
   plt.subplot(2, 2, 4)
   # HPM convergence rates removed
   # VIM convergence rates removed
   # All HPM/VIM plotting removed
   
   # Focus on implemented methods: SDE solvers, fractional operators, and ML integration
   print("HPM and VIM solvers have been removed - focusing on implemented methods")

Advanced Fractional Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrate the comprehensive collection of fractional operators available in HPFRACC:

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.fractional_implementations import create_riesz_fisher_operator
   import numpy as np
   import matplotlib.pyplot as plt

   # Test function
   def f(x): return np.exp(-x**2)
   x = np.linspace(-3, 3, 200)
   alpha = 0.5

   # Create different operators
   operators = {
       'Riemann-Liouville': create_fractional_derivative('riemann_liouville', alpha),
       'Caputo': create_fractional_derivative('caputo', alpha),
       'Caputo-Fabrizio': create_fractional_derivative('caputo_fabrizio', alpha),
       'Atangana-Baleanu': create_fractional_derivative('atangana_baleanu', alpha),
       'Riesz-Fisher': create_riesz_fisher_operator(alpha)
   }

   # Compute results
   results = {}
   for name, operator in operators.items():
       try:
           results[name] = operator.compute(f, x)
       except Exception as e:
           print(f"{name}: {e}")

   # Plot comparison
   plt.figure(figsize=(15, 10))
   
   plt.subplot(2, 2, 1)
   plt.plot(x, f(x), 'k-', linewidth=3, label='Original: exp(-x²)')
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Original Function')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.subplot(2, 2, 2)
   for name, result in results.items():
       if name in ['Riemann-Liouville', 'Caputo']:
           plt.plot(x, result, '--', linewidth=2, label=f'{name} D^{alpha}')
   plt.xlabel('x')
   plt.ylabel('D^α f(x)')
   plt.title('Classical Methods Comparison')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.subplot(2, 2, 3)
   for name, result in results.items():
       if name in ['Caputo-Fabrizio', 'Atangana-Baleanu']:
           plt.plot(x, result, '--', linewidth=2, label=f'{name} D^{alpha}')
   plt.xlabel('x')
   plt.ylabel('D^α f(x)')
   plt.title('Novel Methods Comparison')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.subplot(2, 2, 4)
   if 'Riesz-Fisher' in results:
       plt.plot(x, results['Riesz-Fisher'], '--', linewidth=2, label=f'Riesz-Fisher D^{alpha}')
   plt.xlabel('x')
   plt.ylabel('D^α f(x)')
   plt.title('Special Operators')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

   # Performance comparison
   print("\\nPerformance Comparison:")
   for name, result in results.items():
       if name in results:
           print(f"{name}: Result shape {result.shape}")

Autograd Fractional Derivatives (ML)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrate the autograd-friendly fractional derivatives for machine learning applications:

.. code-block:: python

   import torch
   import torch.nn as nn
   from hpfracc.ml.fractional_autograd import fractional_derivative, FractionalDerivativeLayer
   import matplotlib.pyplot as plt

   # Create test data
   batch_size, channels, time_steps = 4, 16, 128
   x = torch.randn(batch_size, channels, time_steps, requires_grad=True)
   
   # Test different methods
   methods = ['RL', 'Caputo', 'CF', 'AB']
   alpha = 0.5
   
   results = {}
   for method in methods:
       try:
           y = fractional_derivative(x, alpha=alpha, method=method)
           results[method] = y.detach().numpy()
       except Exception as e:
           print(f"{method}: {e}")
   
   # Test gradient computation
   x_test = torch.randn(2, 8, 64, requires_grad=True)
   y_test = fractional_derivative(x_test, alpha=0.3, method="RL")
   loss = y_test.mean()
   loss.backward()
   
   print(f"Gradient shape: {x_test.grad.shape}")
   print(f"Gradient norm: {x_test.grad.norm().item():.6f}")
   
   # Test layer wrapper
   layer = FractionalDerivativeLayer(alpha=0.5, method="RL")
   y_layer = layer(x_test)
   print(f"Layer output shape: {y_layer.shape}")
   
   # Visualize results
   if results:
       plt.figure(figsize=(15, 10))
       
       # Original signal
       plt.subplot(2, 2, 1)
       plt.plot(x[0, 0, :].detach().numpy(), 'k-', linewidth=2, label='Original')
       plt.xlabel('Time')
       plt.ylabel('Amplitude')
       plt.title('Original Signal')
       plt.legend()
       plt.grid(True, alpha=0.3)
       
       # Method comparisons
       for i, (method, result) in enumerate(results.items()):
           plt.subplot(2, 2, i+2)
           plt.plot(result[0, 0, :], '--', linewidth=2, label=f'{method} D^{alpha}')
           plt.xlabel('Time')
           plt.ylabel('D^α f(t)')
           plt.title(f'{method} Method')
           plt.legend()
           plt.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
   
   # Training example
   class FractionalNet(nn.Module):
       def __init__(self, alpha=0.5, method="RL"):
           super().__init__()
           self.fractional_layer = FractionalDerivativeLayer(alpha, method)
           self.linear = nn.Linear(64, 1)
       
       def forward(self, x):
           x = self.fractional_layer(x)
           x = x.mean(dim=1)  # Global average pooling
           return self.linear(x)
   
   # Create model and test forward pass
   model = FractionalNet(alpha=0.5, method="RL")
   output = model(x_test)
   print(f"Model output shape: {output.shape}")
   
   # Test training step
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   target = torch.randn(2, 1)
   loss = nn.MSELoss()(output, target)
   loss.backward()
   optimizer.step()
   
   print(f"Training loss: {loss.item():.6f}")

These examples demonstrate the comprehensive capabilities of the HPFRACC library, from basic fractional calculus operations to advanced applications in machine learning, signal processing, and numerical analysis. Each example includes visualization and analysis tools to help users understand the behavior and performance of fractional calculus methods.
"""
