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
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative

   # Define test function
   def test_function(x):
       return np.sin(x)

   # Create different fractional derivatives
   alpha = FractionalOrder(0.5)
   x = np.linspace(0, 2*np.pi, 100)

   # Riemann-Liouville
   rl_deriv = create_fractional_derivative(alpha, method="RL")
   result_rl = rl_deriv(test_function, x)

   # Caputo
   caputo_deriv = create_fractional_derivative(alpha, method="Caputo")
   result_caputo = caputo_deriv(test_function, x)

   # Grünwald-Letnikov
   gl_deriv = create_fractional_derivative(alpha, method="GL")
   result_gl = gl_deriv(test_function, x)

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
           results[backend.value] = time_taken
           print(f"{backend.value}: {time_taken:.4f}s")
       else:
           print(f"{backend.value}: Not available")

   # Plot results
   if results:
       plt.figure(figsize=(8, 6))
       plt.bar(results.keys(), results.values())
       plt.ylabel('Time (seconds)')
       plt.title('Backend Performance Comparison')
       plt.show()

Neural Network Examples
----------------------

Fractional Neural Network Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a fractional neural network on a simple regression task:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType

   # Generate synthetic data
   np.random.seed(42)
   X = np.random.randn(1000, 5)
   y = np.sum(X, axis=1, keepdims=True) + 0.1 * np.random.randn(1000, 1)

   # Split data
   train_size = int(0.8 * len(X))
   X_train, X_test = X[:train_size], X[train_size:]
   y_train, y_test = y[:train_size], y[train_size:]

   # Create model
   model = FractionalNeuralNetwork(
       input_dim=5,
       hidden_dims=[32, 16],
       output_dim=1,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.JAX
   )

   # Simple training loop
   learning_rate = 0.01
   epochs = 100
   train_losses = []
   test_losses = []

   for epoch in range(epochs):
       # Forward pass
       train_pred = model.forward(X_train)
       test_pred = model.forward(X_test)
       
       # Compute losses (MSE)
       train_loss = np.mean((train_pred - y_train) ** 2)
       test_loss = np.mean((test_pred - y_test) ** 2)
       
       train_losses.append(train_loss)
       test_losses.append(test_loss)
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

   # Plot training progress
   plt.figure(figsize=(10, 6))
   plt.plot(train_losses, label='Training Loss')
   plt.plot(test_losses, label='Test Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Mean Squared Error')
   plt.title('Training Progress')
   plt.legend()
   plt.grid(True)
   plt.show()

   # Plot predictions
   plt.figure(figsize=(10, 6))
   plt.scatter(y_test, test_pred, alpha=0.6)
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
   plt.xlabel('True Values')
   plt.ylabel('Predictions')
   plt.title('Model Predictions vs True Values')
   plt.grid(True)
   plt.show()

Graph Neural Network Examples
---------------------------

Node Classification with Fractional GCN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classify nodes in a graph using fractional Graph Convolutional Networks:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.ml import FractionalGNNFactory
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType

   # Generate synthetic graph data
   np.random.seed(42)
   num_nodes = 100
   num_features = 16
   num_classes = 4

   # Node features
   node_features = np.random.randn(num_nodes, num_features)

   # Generate edges (random graph)
   num_edges = 200
   edge_index = np.random.randint(0, num_nodes, (2, num_edges))

   # Node labels (random for demonstration)
   node_labels = np.random.randint(0, num_classes, num_nodes)

   # Create fractional GCN
   gnn = FractionalGNNFactory.create_model(
       model_type='gcn',
       input_dim=num_features,
       hidden_dim=32,
       output_dim=num_classes,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.TORCH
   )

   # Forward pass
   output = gnn.forward(node_features, edge_index)
   predictions = np.argmax(output, axis=1)

   # Calculate accuracy
   accuracy = np.mean(predictions == node_labels)
   print(f"Node Classification Accuracy: {accuracy:.4f}")

   # Visualize results
   plt.figure(figsize=(12, 5))

   # Original labels
   plt.subplot(1, 2, 1)
   scatter = plt.scatter(node_features[:, 0], node_features[:, 1], 
                        c=node_labels, cmap='viridis', alpha=0.7)
   plt.colorbar(scatter)
   plt.title('Original Node Labels')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')

   # Predicted labels
   plt.subplot(1, 2, 2)
   scatter = plt.scatter(node_features[:, 0], node_features[:, 1], 
                        c=predictions, cmap='viridis', alpha=0.7)
   plt.colorbar(scatter)
   plt.title('Predicted Node Labels')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')

   plt.tight_layout()
   plt.show()

Graph Attention Network Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use fractional Graph Attention Networks for node classification:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.ml import FractionalGNNFactory
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType

   # Generate synthetic graph data
   np.random.seed(42)
   num_nodes = 50
   num_features = 8
   num_classes = 3

   # Node features
   node_features = np.random.randn(num_nodes, num_features)

   # Generate edges with some structure
   edge_index = []
   for i in range(num_nodes):
       # Connect to 3-5 random neighbors
       num_neighbors = np.random.randint(3, 6)
       neighbors = np.random.choice(num_nodes, num_neighbors, replace=False)
       for neighbor in neighbors:
           if i != neighbor:
               edge_index.append([i, neighbor])
   
   edge_index = np.array(edge_index).T

   # Node labels (based on feature clustering)
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=num_classes, random_state=42)
   node_labels = kmeans.fit_predict(node_features)

   # Create fractional GAT
   gat = FractionalGNNFactory.create_model(
       model_type='gat',
       input_dim=num_features,
       hidden_dim=16,
       output_dim=num_classes,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.JAX
   )

   # Forward pass
   output = gat.forward(node_features, edge_index)
   predictions = np.argmax(output, axis=1)

   # Calculate accuracy
   accuracy = np.mean(predictions == node_labels)
   print(f"GAT Node Classification Accuracy: {accuracy:.4f}")

   # Visualize attention patterns (simplified)
   plt.figure(figsize=(10, 6))
   plt.scatter(node_features[:, 0], node_features[:, 1], 
              c=node_labels, cmap='viridis', s=100, alpha=0.7)
   
   # Draw edges
   for i in range(edge_index.shape[1]):
       src, dst = edge_index[:, i]
       plt.plot([node_features[src, 0], node_features[dst, 0]], 
                [node_features[src, 1], node_features[dst, 1]], 
                'k-', alpha=0.1, linewidth=0.5)
   
   plt.title('Graph Structure with Node Labels')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   plt.colorbar()
   plt.show()

Advanced Examples
----------------

Fractional Attention Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement and use fractional attention mechanisms:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.ml.attention import FractionalAttention
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType

   # Generate sequence data
   seq_length = 20
   d_model = 64
   batch_size = 4

   # Input sequence
   input_sequence = np.random.randn(batch_size, seq_length, d_model)

   # Create fractional attention
   attention = FractionalAttention(
       d_model=d_model,
       n_heads=8,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.TORCH
   )

   # Apply attention
   output = attention.forward(input_sequence, method="RL")
   attention_weights = attention.get_attention_weights()

   print(f"Input shape: {input_sequence.shape}")
   print(f"Output shape: {output.shape}")
   print(f"Attention weights shape: {attention_weights.shape}")

   # Visualize attention weights for first batch
   plt.figure(figsize=(10, 8))
   plt.imshow(attention_weights[0, 0], cmap='viridis', aspect='auto')
   plt.colorbar()
   plt.title('Attention Weights (First Head, First Batch)')
   plt.xlabel('Key Position')
   plt.ylabel('Query Position')
   plt.show()

Multi-Backend Comparison
~~~~~~~~~~~~~~~~~~~~~~~~

Compare different backends for the same computation:

.. code-block:: python

   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from hpfracc.ml.backends import BackendManager, BackendType
   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder

   def benchmark_model(backend_type, data_sizes):
       """Benchmark model performance across different data sizes."""
       BackendManager.set_backend(backend_type)
       
       times = []
       for size in data_sizes:
           # Create model
           model = FractionalNeuralNetwork(
               input_dim=10,
               hidden_dims=[32, 16],
               output_dim=1,
               fractional_order=FractionalOrder(0.5)
           )
           
           # Generate data
           X = np.random.randn(size, 10)
           
           # Warm up
           for _ in range(5):
               _ = model.forward(X)
           
           # Benchmark
           start_time = time.time()
           for _ in range(50):
               _ = model.forward(X)
           end_time = time.time()
           
           times.append(end_time - start_time)
       
       return times

   # Test parameters
   data_sizes = [100, 500, 1000, 2000, 5000]
   backends = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]
   
   results = {}
   
   for backend in backends:
       if BackendManager.is_backend_available(backend):
           print(f"Testing {backend.value}...")
           times = benchmark_model(backend, data_sizes)
           results[backend.value] = times
       else:
           print(f"{backend.value} not available")

   # Plot results
   plt.figure(figsize=(12, 8))
   
   for backend, times in results.items():
       plt.plot(data_sizes, times, marker='o', label=backend, linewidth=2)
   
   plt.xlabel('Data Size')
   plt.ylabel('Time (seconds)')
   plt.title('Backend Performance Comparison')
   plt.legend()
   plt.grid(True)
   plt.xscale('log')
   plt.yscale('log')
   plt.show()

Real-World Applications
----------------------

Signal Processing with Fractional Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply fractional calculus to signal processing:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative
   from scipy import signal

   # Generate noisy signal
   t = np.linspace(0, 10, 1000)
   clean_signal = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
   noisy_signal = clean_signal + 0.1 * np.random.randn(len(t))

   # Create fractional derivatives
   orders = [0.25, 0.5, 0.75, 1.0]
   derivatives = {}

   for order in orders:
       alpha = FractionalOrder(order)
       deriv = create_fractional_derivative(alpha, method="RL")
       derivatives[order] = deriv(noisy_signal, t)

   # Plot results
   plt.figure(figsize=(15, 10))

   # Original signal
   plt.subplot(2, 2, 1)
   plt.plot(t, clean_signal, label='Clean Signal', linewidth=2)
   plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.7)
   plt.title('Original Signal')
   plt.legend()
   plt.grid(True)

   # Fractional derivatives
   for i, order in enumerate(orders):
       plt.subplot(2, 2, i + 2)
       plt.plot(t, derivatives[order], linewidth=2)
       plt.title(f'Fractional Derivative (α={order})')
       plt.grid(True)

   plt.tight_layout()
   plt.show()

   # Frequency domain analysis
   plt.figure(figsize=(12, 8))
   
   for order in orders:
       # Compute FFT
       fft_result = np.fft.fft(derivatives[order])
       freqs = np.fft.fftfreq(len(t), t[1] - t[0])
       
       # Plot magnitude spectrum
       plt.semilogy(freqs[:len(freqs)//2], 
                   np.abs(fft_result[:len(freqs)//2]), 
                   label=f'α={order}', linewidth=2)
   
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Magnitude')
   plt.title('Frequency Domain Analysis')
   plt.legend()
   plt.grid(True)
   plt.show()

Image Processing with Fractional Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply fractional calculus to image processing:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative
   from scipy import ndimage

   # Generate test image
   x = np.linspace(-5, 5, 100)
   y = np.linspace(-5, 5, 100)
   X, Y = np.meshgrid(x, y)
   
   # Create a test image (Gaussian + noise)
   image = np.exp(-(X**2 + Y**2) / 2) + 0.1 * np.random.randn(100, 100)

   # Apply fractional derivatives along rows
   orders = [0.25, 0.5, 0.75, 1.0]
   filtered_images = {}

   for order in orders:
       alpha = FractionalOrder(order)
       deriv = create_fractional_derivative(alpha, method="RL")
       
       # Apply to each row
       filtered = np.zeros_like(image)
       for i in range(image.shape[0]):
           filtered[i, :] = deriv(image[i, :], x)
       
       filtered_images[order] = filtered

   # Plot results
   plt.figure(figsize=(15, 10))

   # Original image
   plt.subplot(2, 3, 1)
   plt.imshow(image, cmap='gray')
   plt.title('Original Image')
   plt.colorbar()

   # Filtered images
   for i, order in enumerate(orders):
       plt.subplot(2, 3, i + 2)
       plt.imshow(filtered_images[order], cmap='gray')
       plt.title(f'Fractional Filter (α={order})')
       plt.colorbar()

   plt.tight_layout()
   plt.show()

   # Cross-section comparison
   plt.figure(figsize=(12, 8))
   center_row = image.shape[0] // 2
   
   plt.plot(x, image[center_row, :], label='Original', linewidth=2)
   for order in orders:
       plt.plot(x, filtered_images[order][center_row, :], 
                label=f'α={order}', linewidth=2)
   
   plt.xlabel('Position')
   plt.ylabel('Intensity')
   plt.title('Cross-section Comparison')
   plt.legend()
   plt.grid(True)
   plt.show()

Performance Optimization Examples
-------------------------------

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Process large datasets efficiently:

.. code-block:: python

   import numpy as np
   import gc
   import time
   from hpfracc.ml.backends import BackendManager, BackendType
   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder

   def process_large_dataset_efficiently(data, batch_size=1000):
       """Process large dataset in batches to manage memory."""
       results = []
       num_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
       
       for i in range(num_batches):
           start_idx = i * batch_size
           end_idx = min((i + 1) * batch_size, len(data))
           batch = data[start_idx:end_idx]
           
           # Process batch
           batch_result = np.sum(batch, axis=1)
           results.append(batch_result)
           
           # Clear memory
           del batch
           gc.collect()
           
           if i % 10 == 0:
               print(f"Processed batch {i+1}/{num_batches}")
       
       return np.concatenate(results)

   # Test with large dataset
   large_data = np.random.randn(50000, 100)
   
   print("Processing large dataset...")
   start_time = time.time()
   results = process_large_dataset_efficiently(large_data)
   end_time = time.time()
   
   print(f"Processing time: {end_time - start_time:.2f}s")
   print(f"Result shape: {results.shape}")

Parallel Processing
~~~~~~~~~~~~~~~~~~

Use multiple backends for parallel processing:

.. code-block:: python

   import numpy as np
   import multiprocessing as mp
   from hpfracc.ml.backends import BackendType
   from hpfracc.core.definitions import FractionalOrder

   def process_chunk(chunk_data, backend_type):
       """Process a chunk of data with specified backend."""
       from hpfracc.ml.backends import BackendManager
       from hpfracc.ml import FractionalNeuralNetwork
       
       BackendManager.set_backend(backend_type)
       
       model = FractionalNeuralNetwork(
           input_dim=chunk_data.shape[1],
           hidden_dims=[16],
           output_dim=1,
           fractional_order=FractionalOrder(0.5)
       )
       
       return model.forward(chunk_data)

   def parallel_processing(data, num_processes=4):
       """Process data in parallel using multiple backends."""
       chunk_size = len(data) // num_processes
       chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
       
       backends = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA, BackendType.TORCH]
       
       with mp.Pool(num_processes) as pool:
           results = pool.starmap(process_chunk, 
                                 zip(chunks, backends[:len(chunks)]))
       
       return np.concatenate(results)

   # Test parallel processing
   test_data = np.random.randn(1000, 10)
   print("Testing parallel processing...")
   
   result = parallel_processing(test_data)
   print(f"Result shape: {result.shape}")

Interactive Examples
-------------------

Jupyter Notebook Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For interactive examples, you can use Jupyter notebooks. Here's a template:

.. code-block:: python

   # %matplotlib inline
   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.ml.backends import BackendManager, BackendType

   # Interactive parameter adjustment
   def interactive_fractional_network(alpha=0.5, backend='jax'):
       """Create and test a fractional neural network with interactive parameters."""
       
       # Set backend
       backend_map = {'torch': BackendType.TORCH, 
                     'jax': BackendType.JAX, 
                     'numba': BackendType.NUMBA}
       BackendManager.set_backend(backend_map[backend])
       
       # Create model
       model = FractionalNeuralNetwork(
           input_dim=5,
           hidden_dims=[32, 16],
           output_dim=1,
           fractional_order=FractionalOrder(alpha)
       )
       
       # Test data
       X = np.random.randn(100, 5)
       output = model.forward(X)
       
       # Plot results
       plt.figure(figsize=(10, 6))
       plt.hist(output.flatten(), bins=30, alpha=0.7)
       plt.title(f'Output Distribution (α={alpha}, Backend={backend})')
       plt.xlabel('Output Value')
       plt.ylabel('Frequency')
       plt.show()
       
       return model, output

   # Example usage
   model, output = interactive_fractional_network(alpha=0.7, backend='jax')

For more interactive examples, see the Jupyter notebooks in the `examples/` directory.

Troubleshooting Examples
-----------------------

Common Error Solutions
~~~~~~~~~~~~~~~~~~~~~

**Handling Backend Errors**
.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType

   def safe_backend_usage():
       """Safely use backends with error handling."""
       backends_to_try = [BackendType.JAX, BackendType.TORCH, BackendType.NUMBA]
       
       for backend in backends_to_try:
           try:
               if BackendManager.is_backend_available(backend):
                   BackendManager.set_backend(backend)
                   print(f"Successfully set backend to {backend.value}")
                   return backend
           except Exception as e:
               print(f"Failed to set {backend.value}: {e}")
       
       print("No backends available!")
       return None

   # Usage
   backend = safe_backend_usage()

**Memory Management**
.. code-block:: python

   import gc
   import numpy as np

   def memory_efficient_processing(data, chunk_size=1000):
       """Process data in chunks to manage memory."""
       results = []
       
       for i in range(0, len(data), chunk_size):
           chunk = data[i:i+chunk_size]
           
           # Process chunk
           chunk_result = np.sum(chunk, axis=1)
           results.append(chunk_result)
           
           # Clear memory
           del chunk
           gc.collect()
       
       return np.concatenate(results)

   # Test with large data
   large_data = np.random.randn(10000, 100)
   result = memory_efficient_processing(large_data)

For more examples and detailed tutorials, visit the GitHub repository and check the `examples/` directory for additional code samples and Jupyter notebooks.
