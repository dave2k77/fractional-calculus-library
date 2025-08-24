"""
Machine Learning Integration Module for hpfracc

This module provides comprehensive ML integration for fractional calculus,
including neural networks, attention mechanisms, loss functions, optimizers,
and a complete development-to-production workflow.

Features:
- Fractional Neural Networks with various architectures
- Fractional Graph Neural Networks (GNNs) with multi-backend support
- Fractional Attention Mechanisms
- Fractional Loss Functions
- Fractional Optimizers
- Multi-backend support (PyTorch, JAX, NUMBA)
- Model Registry and Versioning
- Development vs. Production Workflow
- Quality Gates and Validation
- Model Monitoring and Rollback
- Fractional Neural Network Layers
- PyTorch Autograd Integration for Fractional Derivatives
"""

from .core import (
    FractionalNeuralNetwork,
    FractionalAttention,
    FractionalLossFunction,
    FractionalAutoML,
    MLConfig,
)

from .backends import (
    BackendManager,
    BackendType,
    get_backend_manager,
    set_backend_manager,
    get_active_backend,
    switch_backend,
)

from .tensor_ops import (
    TensorOps,
    get_tensor_ops,
    create_tensor,
)

from .registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetadata,
    DeploymentStatus,
)

from .workflow import (
    DevelopmentWorkflow,
    ProductionWorkflow,
    ModelValidator,
    QualityGate,
    QualityMetric,
    QualityThreshold,
)

from .layers import (
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d,
    LayerConfig,
)

from .gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling,
)

from .gnn_models import (
    BaseFractionalGNN,
    FractionalGCN,
    FractionalGAT,
    FractionalGraphSAGE,
    FractionalGraphUNet,
    FractionalGNNFactory,
)

from .losses import (
    FractionalMSELoss,
    FractionalCrossEntropyLoss,
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
    FractionalCombinedLoss,
)

from .optimizers import (
    FractionalAdam,
    FractionalSGD,
    FractionalRMSprop,
    FractionalAdagrad,
    FractionalAdamW,
)

from .fractional_autograd import (
    FractionalDerivativeFunction,
    FractionalDerivativeLayer,
    fractional_derivative,
    rl_derivative,
    caputo_derivative,
    gl_derivative,
)

__all__ = [
    # Core ML components
    "FractionalNeuralNetwork",
    "FractionalAttention", 
    "FractionalLossFunction",
    "FractionalAutoML",
    "MLConfig",
    
    # Backend management
    "BackendManager",
    "BackendType",
    "get_backend_manager",
    "set_backend_manager",
    "get_active_backend",
    "switch_backend",
    
    # Tensor operations
    "TensorOps",
    "get_tensor_ops",
    "create_tensor",
    
    # Model management
    "ModelRegistry",
    "ModelVersion",
    "ModelMetadata",
    "DeploymentStatus",
    
    # Workflow management
    "DevelopmentWorkflow",
    "ProductionWorkflow",
    "ModelValidator",
    "QualityGate",
    "QualityMetric",
    "QualityThreshold",
    
    # Neural network layers
    "FractionalConv1D",
    "FractionalConv2D",
    "FractionalLSTM",
    "FractionalTransformer",
    "FractionalPooling",
    "FractionalBatchNorm1d",
    "LayerConfig",
    
    # Graph Neural Network layers
    "BaseFractionalGNNLayer",
    "FractionalGraphConv",
    "FractionalGraphAttention",
    "FractionalGraphPooling",
    
    # Graph Neural Network models
    "BaseFractionalGNN",
    "FractionalGCN",
    "FractionalGAT",
    "FractionalGraphSAGE",
    "FractionalGraphUNet",
    "FractionalGNNFactory",
    
    # Loss functions
    "FractionalMSELoss",
    "FractionalCrossEntropyLoss",
    "FractionalHuberLoss",
    "FractionalSmoothL1Loss",
    "FractionalKLDivLoss",
    "FractionalBCELoss",
    "FractionalNLLLoss",
    "FractionalPoissonNLLLoss",
    "FractionalCosineEmbeddingLoss",
    "FractionalMarginRankingLoss",
    "FractionalMultiMarginLoss",
    "FractionalTripletMarginLoss",
    "FractionalCTCLoss",
    "FractionalCustomLoss",
    "FractionalCombinedLoss",
    
    # Optimizers
    "FractionalAdam",
    "FractionalSGD",
    "FractionalRMSprop",
    "FractionalAdagrad",
    "FractionalAdamW",
    
    # Autograd integration
    "FractionalDerivativeFunction",
    "FractionalDerivativeLayer",
    "fractional_derivative",
    "rl_derivative",
    "caputo_derivative",
    "gl_derivative",
]

__version__ = "0.1.0"
__author__ = "Davian R. Chin"
__email__ = "d.r.chin@pgr.reading.ac.uk"
__institution__ = "Department of Biomedical Engineering, University of Reading"
