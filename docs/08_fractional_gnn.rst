Fractional Graph Neural Networks
================================

HPFRACC provides comprehensive support for fractional graph neural networks, extending standard GNN architectures with fractional calculus operations.

For complete API documentation, see :doc:`api/fgnn_api`.

Overview
--------

Fractional Graph Neural Networks combine graph convolution operations with fractional calculus to capture long-range dependencies and memory effects in graph-structured data.

Key Features:
- **Multi-backend support** (PyTorch, JAX, NUMBA) - Fully implemented with backend-agnostic fractional derivatives
- Various GNN architectures (GCN, GAT, GraphSAGE, U-Net)
- **Proper fractional derivative integration** - Mathematically correct implementations (no placeholders)
- Performance optimization across backends

Basic Usage
-----------

Fractional Graph Convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import networkx as nx
   from hpfracc.ml.gnn_layers import FractionalGraphConvolution
   from hpfracc.core.definitions import FractionalOrder

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
   print(f"Output shape: {output_features.shape}")

Architecture Types
------------------

Graph Convolutional Network (GCN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.gnn_layers import FractionalGCNLayer

   layer = FractionalGCNLayer(
       input_dim=16,
       output_dim=32,
       fractional_order=0.5,
       activation='relu'
   )

Graph Attention Network (GAT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.gnn_layers import FractionalGATLayer

   layer = FractionalGATLayer(
       input_dim=16,
       output_dim=32,
       num_heads=4,
       fractional_order=0.5,
       activation='relu'
   )

GraphSAGE
~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.gnn_layers import FractionalGraphSAGELayer

   layer = FractionalGraphSAGELayer(
       input_dim=16,
       output_dim=32,
       fractional_order=0.5,
       aggregation='mean'
   )

Complete Example
----------------

Full GNN Model
~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.gnn_layers import FractionalGraphConvolution
   from hpfracc.core.definitions import FractionalOrder
   import numpy as np
   import networkx as nx
   import matplotlib.pyplot as plt

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

Backend Selection
-----------------

Multi-Backend Support
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml import BackendType, get_backend_manager, switch_backend

   # Switch backends for optimal performance
   switch_backend(BackendType.TORCH)  # PyTorch
   switch_backend(BackendType.JAX)    # JAX
   switch_backend(BackendType.NUMBA)  # NUMBA

Performance Benchmarking
------------------------

Compare performance across different backends:

.. code-block:: python

   from hpfracc.ml import BackendType, switch_backend
   import time

   def benchmark_backend(backend_type, adj_matrix, node_features):
       switch_backend(backend_type)
       layer = FractionalGraphConvolution(5, 3, FractionalOrder(0.5))
       
       start_time = time.time()
       output = layer(adj_matrix, node_features)
       elapsed_time = time.time() - start_time
       
       return elapsed_time

   # Benchmark different backends
   backends = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]
   for backend in backends:
       if get_backend_manager().is_backend_available(backend):
           elapsed = benchmark_backend(backend, adj_matrix, node_features)
           print(f"{backend.value}: {elapsed:.4f}s")

Summary
-------

Fractional Graph Neural Networks provide:

✅ **Multiple Architectures**: GCN, GAT, GraphSAGE, U-Net  
✅ **Fractional Integration**: Long-range dependencies through fractional calculus  
✅ **Multi-Backend**: PyTorch, JAX, NUMBA support  
✅ **Performance**: Automatic optimization across backends  

Next Steps
----------

- **API Reference**: See :doc:`api/fgnn_api` for complete API documentation
- **Examples**: Check :doc:`05_advanced_examples` for GNN examples
- **User Guide**: See :doc:`user_guide` for comprehensive usage guide

