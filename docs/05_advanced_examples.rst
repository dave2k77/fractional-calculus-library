Advanced Examples
=================

This section demonstrates advanced applications of fractional calculus including signal processing, image processing, and neural network integration.

Signal Processing Applications
------------------------------

Fractional Signal Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply fractional derivatives to signal processing for frequency analysis:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder
   from scipy.fft import fft, fftfreq

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

Image Processing Applications
-----------------------------

Fractional Image Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply fractional derivatives for edge detection and image enhancement:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
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

Fractional Neural Networks
-------------------------

Basic Fractional Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and train a fractional neural network for regression:

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

Fractional Graph Neural Networks
--------------------------------

Graph Convolution with Fractional Calculus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement fractional graph convolutions for node feature learning:

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

Performance Benchmarking
------------------------

Comparing Different Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark performance across different computation backends:

.. code-block:: python

   import numpy as np
   import time
   from hpfracc.ml.backends import BackendManager, BackendType
   from hpfracc.core.derivatives import create_fractional_derivative
   from hpfracc.core.definitions import FractionalOrder

   def benchmark_backend(backend_type, data_size=10000):
       """Benchmark a specific backend"""
       BackendManager.set_backend(backend_type)
       
       # Generate test data
       x = np.linspace(0, 10, data_size)
       signal = np.sin(2*np.pi*x)
       
       # Create fractional derivative
       deriv = create_fractional_derivative(FractionalOrder(0.5), method="RL")
       
       # Time computation
       start_time = time.time()
       result = deriv(lambda x: signal, x)
       elapsed_time = time.time() - start_time
       
       return elapsed_time

   # Benchmark different backends
   backends = [BackendType.NUMPY, BackendType.NUMBA]
   if BackendManager.is_backend_available(BackendType.TORCH):
       backends.append(BackendType.TORCH)
   if BackendManager.is_backend_available(BackendType.JAX):
       backends.append(BackendType.JAX)

   print("Benchmarking fractional derivative computation...")
   print(f"{'Backend':<15} {'Time (s)':<15} {'Speedup':<15}")
   print("-" * 45)
   
   times = {}
   for backend in backends:
       if BackendManager.is_backend_available(backend):
           elapsed = benchmark_backend(backend, data_size=10000)
           times[backend] = elapsed
           print(f"{backend.value:<15} {elapsed:<15.4f}")

   # Calculate speedup relative to NumPy
   if BackendType.NUMPY in times:
       numpy_time = times[BackendType.NUMPY]
       print("\nSpeedup relative to NumPy:")
       for backend, elapsed in times.items():
           speedup = numpy_time / elapsed
           print(f"{backend.value}: {speedup:.2f}x")

Summary
-------

These advanced examples demonstrate:

✅ **Signal Processing**: Fractional derivatives for frequency analysis  
✅ **Image Processing**: Fractional gradients for edge detection  
✅ **Neural Networks**: Fractional layers for machine learning  
✅ **Graph Neural Networks**: Fractional convolutions for graph data  
✅ **Performance**: Benchmarking across different backends  

Next Steps
----------

- **Deep dive**: Explore :doc:`07_fractional_neural_networks` for comprehensive ML guide
- **Graph networks**: See :doc:`08_fractional_gnn` for GNN details
- **Scientific applications**: Check :doc:`10_scientific_applications` for research examples

