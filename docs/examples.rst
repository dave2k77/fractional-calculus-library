"""
Examples & Tutorials
===================

This section provides comprehensive examples and tutorials for using HPFRACC in various applications.

Basic Examples
-------------

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

Fractional Green's Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using fractional Green's functions for solving differential equations:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.special.greens_function import (
       FractionalDiffusionGreensFunction,
       FractionalWaveGreensFunction,
       FractionalAdvectionGreensFunction
   )

   # Parameters
   alpha = 0.5
   D = 1.0  # Diffusion coefficient
   c = 1.0  # Wave speed
   v = 1.0  # Advection velocity

   # Create Green's functions
   diffusion_gf = FractionalDiffusionGreensFunction(alpha, D)
   wave_gf = FractionalWaveGreensFunction(alpha, c)
   advection_gf = FractionalAdvectionGreensFunction(alpha, v)

   # Spatial and temporal grids
   x = np.linspace(-5, 5, 200)
   t = np.linspace(0.1, 2, 100)
   X, T = np.meshgrid(x, t)

   # Compute Green's functions
   diffusion_result = np.array([[diffusion_gf.compute(xi, ti) for xi in x] for ti in t])
   wave_result = np.array([[wave_gf.compute(xi, ti) for xi in x] for ti in t])
   advection_result = np.array([[advection_gf.compute(xi, ti) for xi in x] for ti in t])

   # Plot results
   plt.figure(figsize=(15, 5))
   
   plt.subplot(1, 3, 1)
   plt.contourf(X, T, diffusion_result, levels=20)
   plt.colorbar(label='G(x, t)')
   plt.xlabel('x')
   plt.ylabel('t')
   plt.title(f'Fractional Diffusion Green\'s Function (α={alpha})')
   
   plt.subplot(1, 3, 2)
   plt.contourf(X, T, wave_result, levels=20)
   plt.colorbar(label='G(x, t)')
   plt.xlabel('x')
   plt.ylabel('t')
   plt.title(f'Fractional Wave Green\'s Function (α={alpha})')
   
   plt.subplot(1, 3, 3)
   plt.contourf(X, T, advection_result, levels=20)
   plt.colorbar(label='G(x, t)')
   plt.xlabel('x')
   plt.ylabel('t')
   plt.title(f'Fractional Advection Green\'s Function (α={alpha})')
   
   plt.tight_layout()
   plt.show()

Analytical Methods: Homotopy Perturbation Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solving fractional differential equations using HPM:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.solvers.homotopy_perturbation import HomotopyPerturbationMethod

   # Define the fractional differential equation
   # D^α u + u = f(t), where f(t) = t^2
   def source_function(t):
       return t**2

   def initial_condition(t):
       return 0.0

   # Create HPM solver
   alpha = 0.5
   hpm_solver = HomotopyPerturbationMethod(alpha)

   # Solve the equation
   t = np.linspace(0, 2, 100)
   solution = hpm_solver.solve(
       source_function=source_function,
       initial_condition=initial_condition,
       t_span=t,
       max_iterations=5
   )

   # Plot solution
   plt.figure(figsize=(10, 6))
   plt.plot(t, solution, 'b-', linewidth=2, label=f'HPM Solution (α={alpha})')
   plt.plot(t, source_function(t), 'r--', linewidth=2, label='Source Function f(t) = t²')
   plt.xlabel('t')
   plt.ylabel('u(t)')
   plt.title('Solution of Fractional Differential Equation using HPM')
   plt.legend()
   plt.grid(True)
   plt.show()

   # Analyze convergence
   convergence = hpm_solver.analyze_convergence(
       source_function=source_function,
       initial_condition=initial_condition,
       t_span=t,
       max_iterations=10
   )
   
   print("Convergence Analysis:")
   print(f"Final residual: {convergence['final_residual']:.6f}")
   print(f"Convergence rate: {convergence['convergence_rate']:.6f}")

Analytical Methods: Variational Iteration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solving fractional differential equations using VIM:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.solvers.variational_iteration import VariationalIterationMethod

   # Define the fractional differential equation
   # D^α u + u^2 = f(t), where f(t) = 1
   def source_function(t):
       return np.ones_like(t)

   def initial_condition(t):
       return 0.0

   def nonlinear_term(u):
       return u**2

   # Create VIM solver
   alpha = 0.5
   vim_solver = VariationalIterationMethod(alpha)

   # Solve the equation
   t = np.linspace(0, 2, 100)
   solution = vim_solver.solve(
       source_function=source_function,
       initial_condition=initial_condition,
       nonlinear_term=nonlinear_term,
       t_span=t,
       max_iterations=5
   )

   # Plot solution
   plt.figure(figsize=(10, 6))
   plt.plot(t, solution, 'g-', linewidth=2, label=f'VIM Solution (α={alpha})')
   plt.plot(t, source_function(t), 'r--', linewidth=2, label='Source Function f(t) = 1')
   plt.xlabel('t')
   plt.ylabel('u(t)')
   plt.title('Solution of Nonlinear Fractional Differential Equation using VIM')
   plt.legend()
   plt.grid(True)
   plt.show()

   # Compare HPM and VIM
   hpm_solver = HomotopyPerturbationMethod(alpha)
   hpm_solution = hpm_solver.solve(
       source_function=source_function,
       initial_condition=initial_condition,
       t_span=t,
       max_iterations=5
   )

   plt.figure(figsize=(10, 6))
   plt.plot(t, hpm_solution, 'b-', linewidth=2, label='HPM Solution')
   plt.plot(t, solution, 'g-', linewidth=2, label='VIM Solution')
   plt.xlabel('t')
   plt.ylabel('u(t)')
   plt.title('Comparison of HPM and VIM Solutions')
   plt.legend()
   plt.grid(True)
   plt.show()

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
   from hpfracc.solvers.homotopy_perturbation import HomotopyPerturbationMethod
   from hpfracc.solvers.variational_iteration import VariationalIterationMethod

   def convergence_analysis():
       """Analyze convergence of HPM and VIM methods."""
       # Define test problem
       def source_function(t):
           return t**2
       
       def initial_condition(t):
           return 0.0
       
       t = np.linspace(0, 2, 100)
       alpha = 0.5
       
       # HPM convergence
       hpm_solver = HomotopyPerturbationMethod(alpha)
       hpm_convergence = hpm_solver.analyze_convergence(
           source_function=source_function,
           initial_condition=initial_condition,
           t_span=t,
           max_iterations=10
       )
       
       # VIM convergence
       vim_solver = VariationalIterationMethod(alpha)
       vim_convergence = vim_solver.analyze_convergence(
           source_function=source_function,
           initial_condition=initial_condition,
           t_span=t,
           max_iterations=10
       )
       
       return hpm_convergence, vim_convergence

   # Run convergence analysis
   hpm_conv, vim_conv = convergence_analysis()

   # Plot convergence
   plt.figure(figsize=(12, 8))
   
   plt.subplot(2, 2, 1)
   plt.semilogy(hpm_conv['residuals'], 'b-o', linewidth=2, label='HPM')
   plt.xlabel('Iteration')
   plt.ylabel('Residual')
   plt.title('HPM Convergence')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 2, 2)
   plt.semilogy(vim_conv['residuals'], 'r-s', linewidth=2, label='VIM')
   plt.xlabel('Iteration')
   plt.ylabel('Residual')
   plt.title('VIM Convergence')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 2, 3)
   plt.plot(hpm_conv['solutions'][-1], 'b-', linewidth=2, label='HPM Final Solution')
   plt.plot(vim_conv['solutions'][-1], 'r--', linewidth=2, label='VIM Final Solution')
   plt.xlabel('t')
   plt.ylabel('u(t)')
   plt.title('Final Solutions Comparison')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(2, 2, 4)
   plt.plot(hpm_conv['convergence_rates'], 'b-o', linewidth=2, label='HPM')
   plt.plot(vim_conv['convergence_rates'], 'r-s', linewidth=2, label='VIM')
   plt.xlabel('Iteration')
   plt.ylabel('Convergence Rate')
   plt.title('Convergence Rates')
   plt.legend()
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

These examples demonstrate the comprehensive capabilities of the HPFRACC library, from basic fractional calculus operations to advanced applications in machine learning, signal processing, and numerical analysis. Each example includes visualization and analysis tools to help users understand the behavior and performance of fractional calculus methods.
"""
