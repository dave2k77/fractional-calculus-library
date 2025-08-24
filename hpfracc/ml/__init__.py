"""
Machine Learning Integration for Fractional Calculus

This module provides comprehensive ML components that integrate fractional calculus
with neural networks, including:

- **Core Neural Networks**: FractionalNeuralNetwork, FractionalAttention
- **Neural Network Layers**: FractionalConv1D, FractionalConv2D, FractionalLSTM, FractionalTransformer, FractionalPooling, FractionalBatchNorm1d
- **Loss Functions**: FractionalMSELoss, FractionalCrossEntropyLoss, FractionalHuberLoss, and more
- **Optimizers**: FractionalAdam, FractionalSGD, FractionalRMSprop, FractionalAdagrad, FractionalAdamW
- **Fractional Graph Neural Networks (GNNs) with multi-backend support**
- **Multi-backend support (PyTorch, JAX, NUMBA)**
- **Backend Management System**: BackendManager, BackendType, unified tensor operations
- **Unified Tensor Operations**: Cross-backend tensor manipulations
"""

# Backend Management System
from .backends import (
    BackendManager,
    BackendType,
    get_backend_manager,
    set_backend_manager,
    get_active_backend,
    switch_backend
)

# Unified Tensor Operations
from .tensor_ops import (
    TensorOps,
    get_tensor_ops,
    create_tensor
)

# Core ML Components
from .core import (
    MLConfig,
    FractionalNeuralNetwork,
    FractionalAttention,
    FractionalLossFunction,
    FractionalMSELoss,
    FractionalCrossEntropyLoss,
    FractionalAutoML
)

# Neural Network Layers
from .layers import (
    LayerConfig,
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d
)

# Loss Functions
from .losses import (
    FractionalMSELoss as FractionalMSELoss,
    FractionalCrossEntropyLoss as FractionalCrossEntropyLoss,
    FractionalHuberLoss,
    FractionalSmoothL1Loss,
    FractionalKLDivLoss,
    FractionalBCELoss,
    FractionalNLLLoss,
    FractionalPoissonNLLLoss,
    FractionalCosineEmbeddingLoss,
    FractionalMarginRankingLoss,
    FractionalMultiMarginLoss,
    FractionalTripletMarginLoss,
    FractionalCTCLoss,
    FractionalCustomLoss,
    FractionalCombinedLoss
)

# Optimizers
from .optimizers import (
    FractionalOptimizer,
    FractionalAdam,
    FractionalSGD,
    FractionalRMSprop,
    FractionalAdagrad,
    FractionalAdamW
)

# Fractional Graph Neural Network Components
from .gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling
)

from .gnn_models import (
    BaseFractionalGNN,
    FractionalGCN,
    FractionalGAT,
    FractionalGraphSAGE,
    FractionalGraphUNet,
    FractionalGNNFactory
)

# Export all components
__all__ = [
    # Backend Management
    'BackendManager',
    'BackendType',
    'get_backend_manager',
    'set_backend_manager',
    'get_active_backend',
    'switch_backend',
    
    # Tensor Operations
    'TensorOps',
    'get_tensor_ops',
    'create_tensor',
    
    # Core ML Components
    'MLConfig',
    'FractionalNeuralNetwork',
    'FractionalAttention',
    'FractionalLossFunction',
    'FractionalMSELoss',
    'FractionalCrossEntropyLoss',
    'FractionalAutoML',
    
    # Neural Network Layers
    'LayerConfig',
    'FractionalConv1D',
    'FractionalConv2D',
    'FractionalLSTM',
    'FractionalTransformer',
    'FractionalPooling',
    'FractionalBatchNorm1d',
    
    # Loss Functions
    'FractionalMSELoss',
    'FractionalCrossEntropyLoss',
    'FractionalHuberLoss',
    'FractionalSmoothL1Loss',
    'FractionalKLDivLoss',
    'FractionalBCELoss',
    'FractionalNLLLoss',
    'FractionalPoissonNLLLoss',
    'FractionalCosineEmbeddingLoss',
    'FractionalMarginRankingLoss',
    'FractionalMultiMarginLoss',
    'FractionalTripletMarginLoss',
    'FractionalCTCLoss',
    'FractionalCustomLoss',
    'FractionalCombinedLoss',
    
    # Optimizers
    'FractionalOptimizer',
    'FractionalAdam',
    'FractionalSGD',
    'FractionalRMSprop',
    'FractionalAdagrad',
    'FractionalAdamW',
    
    # Fractional GNN Components
    'BaseFractionalGNNLayer',
    'FractionalGraphConv',
    'FractionalGraphAttention',
    'FractionalGraphPooling',
    'BaseFractionalGNN',
    'FractionalGCN',
    'FractionalGAT',
    'FractionalGraphSAGE',
    'FractionalGraphUNet',
    'FractionalGNNFactory'
]

__version__ = "0.1.0"
__author__ = "Davian R. Chin"
__email__ = "d.r.chin@pgr.reading.ac.uk"
__institution__ = "Department of Biomedical Engineering, University of Reading"
