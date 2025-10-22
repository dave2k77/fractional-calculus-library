"""
Data Loaders and Datasets for Fractional Calculus ML

This module provides data loading utilities and dataset classes specifically
designed for fractional calculus machine learning applications.
Supports multiple backends: PyTorch, JAX, and NUMBA.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union, Tuple, Iterator, Callable
from collections import defaultdict
import warnings

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


class FractionalDataset(ABC):
    """
    Base class for datasets with fractional calculus integration

    This class provides a framework for datasets that can apply
    fractional derivatives to data during loading or preprocessing.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
            self,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None,
            apply_fractional: bool = True):
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        self.apply_fractional = apply_fractional

    def fractional_transform(self, data: Any) -> Any:
        """
        Apply fractional derivative to input data

        Args:
            data: Input data

        Returns:
            Data with fractional derivative applied
        """
        if not self.apply_fractional:
            return data

        # Only apply fractional derivative for PyTorch backend for now
        if self.backend == BackendType.TORCH:
            return fractional_derivative(
                data, self.fractional_order.alpha, self.method)
        else:
            # TODO: Implement backend-agnostic fractional derivatives
            # For now, return the input unchanged
            return data

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a sample from the dataset"""


class FractionalTensorDataset(FractionalDataset):
    """Dataset for tensor data with fractional calculus integration"""

    def __init__(
            self,
            tensors: List[Any],
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None,
            apply_fractional: bool = True):
        super().__init__(fractional_order, method, backend, apply_fractional)
        
        if not tensors:
            raise ValueError("Tensors list cannot be empty")
        
        # Ensure all tensors have the same first dimension
        first_dim = len(tensors[0])
        if not all(len(tensor) == first_dim for tensor in tensors):
            raise ValueError("All tensors must have the same first dimension")
        
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        # Get data at index
        data = [tensor[index] for tensor in self.tensors]
        
        # Apply fractional transform to input data (first tensor)
        if len(data) > 1:
            data[0] = self.fractional_transform(data[0])
            return data[0], data[1:]
        else:
            data[0] = self.fractional_transform(data[0])
            return data[0], None


class FractionalTimeSeriesDataset(FractionalDataset):
    """Dataset for time series data with fractional calculus integration"""

    def __init__(
            self,
            data: Any,
            targets: Any,
            sequence_length: int = 10,
            stride: int = 1,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None,
            apply_fractional: bool = True):
        super().__init__(fractional_order, method, backend, apply_fractional)
        
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Calculate number of sequences
        if len(data) < sequence_length:
            raise ValueError("Data length must be at least sequence_length")
        
        self.num_sequences = (len(data) - sequence_length) // stride + 1

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        # Calculate start and end indices
        start_idx = index * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Extract sequence
        sequence = self.data[start_idx:end_idx]
        target = self.targets[end_idx - 1] if end_idx <= len(self.targets) else self.targets[-1]
        
        # Apply fractional transform to sequence
        sequence = self.fractional_transform(sequence)
        
        return sequence, target


class FractionalGraphDataset(FractionalDataset):
    """Dataset for graph data with fractional calculus integration"""

    def __init__(
            self,
            node_features: List[Any],
            edge_indices: List[Any],
            edge_weights: Optional[List[Any]] = None,
            node_labels: Optional[List[Any]] = None,
            graph_labels: Optional[List[Any]] = None,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None,
            apply_fractional: bool = True):
        super().__init__(fractional_order, method, backend, apply_fractional)
        
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights or [None] * len(edge_indices)
        self.node_labels = node_labels or [None] * len(node_features)
        self.graph_labels = graph_labels or [None] * len(node_features)
        
        if not (len(node_features) == len(edge_indices) == len(self.edge_weights) == 
                len(self.node_labels) == len(self.graph_labels)):
            raise ValueError("All input lists must have the same length")

    def __len__(self) -> int:
        return len(self.node_features)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        # Get graph components
        node_feat = self.node_features[index]
        edge_idx = self.edge_indices[index]
        edge_weight = self.edge_weights[index]
        node_label = self.node_labels[index]
        graph_label = self.graph_labels[index]
        
        # Apply fractional transform to node features
        node_feat = self.fractional_transform(node_feat)
        
        return {
            'node_features': node_feat,
            'edge_index': edge_idx,
            'edge_weight': edge_weight,
            'node_labels': node_label,
            'graph_labels': graph_label
        }


class FractionalDataLoader:
    """Data loader with fractional calculus integration"""

    def __init__(
            self,
            dataset: FractionalDataset,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            drop_last: bool = False,
            collate_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.collate_fn = collate_fn or self._default_collate
        
        # Generate indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)

    def _default_collate(self, batch: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
        """Default collate function for batching"""
        if not batch:
            return [], []
        
        # Separate inputs and targets
        inputs, targets = zip(*batch)
        
        # Handle different target formats
        if targets[0] is None:
            # Single tensor case
            return self.dataset.tensor_ops.stack(inputs), None
        elif isinstance(targets[0], (list, tuple)):
            # Multiple target case
            target_lists = list(zip(*targets))
            stacked_targets = [self.dataset.tensor_ops.stack(t_list) for t_list in target_lists]
            return self.dataset.tensor_ops.stack(inputs), stacked_targets
        else:
            # Single target case
            return self.dataset.tensor_ops.stack(inputs), self.dataset.tensor_ops.stack(targets)

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        # Reset indices if shuffling
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Generate batches
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Get batch data
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Collate batch
            yield self.collate_fn(batch_data)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class FractionalDataProcessor:
    """Data processor with fractional calculus integration"""

    def __init__(
            self,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)

    def normalize_data(self, data: Any, method: str = "standard") -> Tuple[Any, Dict[str, Any]]:
        """
        Normalize data using specified method
        
        Args:
            data: Input data
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Normalized data and normalization parameters
        """
        if method == "standard":
            mean = self.tensor_ops.mean(data)
            std = self.tensor_ops.std(data)
            normalized = (data - mean) / (std + 1e-8)
            params = {'mean': mean, 'std': std, 'method': method}
        elif method == "minmax":
            min_val = self.tensor_ops.min(data)
            max_val = self.tensor_ops.max(data)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {'min': min_val, 'max': max_val, 'method': method}
        elif method == "robust":
            median = self.tensor_ops.median(data)
            q75, q25 = self.tensor_ops.quantile(data, [0.75, 0.25])
            iqr = q75 - q25
            normalized = (data - median) / (iqr + 1e-8)
            params = {'median': median, 'iqr': iqr, 'method': method}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params

    def denormalize_data(self, normalized_data: Any, params: Dict[str, Any]) -> Any:
        """
        Denormalize data using stored parameters
        
        Args:
            normalized_data: Normalized data
            params: Normalization parameters
            
        Returns:
            Denormalized data
        """
        method = params['method']
        
        if method == "standard":
            return normalized_data * params['std'] + params['mean']
        elif method == "minmax":
            return normalized_data * (params['max'] - params['min']) + params['min']
        elif method == "robust":
            return normalized_data * params['iqr'] + params['median']
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def augment_data(self, data: Any, augmentation_type: str = "noise", **kwargs) -> Any:
        """
        Augment data using specified method
        
        Args:
            data: Input data
            augmentation_type: Type of augmentation
            **kwargs: Augmentation parameters
            
        Returns:
            Augmented data
        """
        if augmentation_type == "noise":
            noise_level = kwargs.get('noise_level', 0.01)
            noise = self.tensor_ops.randn_like(data) * noise_level
            return data + noise
        elif augmentation_type == "scaling":
            scale_factor = kwargs.get('scale_factor', 1.0)
            return data * scale_factor
        elif augmentation_type == "rotation":
            # For 2D data, implement rotation
            angle = kwargs.get('angle', 0.0)
            # This is a simplified rotation - in practice, you'd want proper rotation matrices
            return data
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")

    def create_sequences(self, data: Any, sequence_length: int, stride: int = 1) -> Any:
        """
        Create sequences from time series data
        
        Args:
            data: Input time series data
            sequence_length: Length of each sequence
            stride: Stride between sequences
            
        Returns:
            Sequence data
        """
        sequences = []
        for i in range(0, len(data) - sequence_length + 1, stride):
            sequences.append(data[i:i + sequence_length])
        
        return self.tensor_ops.stack(sequences)


# Factory functions for easy dataset creation
def create_fractional_dataset(
        dataset_type: str,
        data: Any,
        targets: Optional[Any] = None,
        **kwargs) -> FractionalDataset:
    """
    Create a fractional dataset of the specified type
    
    Args:
        dataset_type: Type of dataset ('tensor', 'timeseries', 'graph')
        data: Input data
        targets: Target data (optional)
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        Configured fractional dataset
    """
    dataset_map = {
        'tensor': FractionalTensorDataset,
        'timeseries': FractionalTimeSeriesDataset,
        'graph': FractionalGraphDataset,
    }
    
    if dataset_type.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataset_class = dataset_map[dataset_type.lower()]
    
    if dataset_type.lower() == 'tensor':
        if targets is None:
            return dataset_class([data], **kwargs)
        else:
            return dataset_class([data, targets], **kwargs)
    elif dataset_type.lower() == 'timeseries':
        if targets is None:
            raise ValueError("Targets are required for time series datasets")
        return dataset_class(data, targets, **kwargs)
    elif dataset_type.lower() == 'graph':
        # For graph datasets, data should be node_features, targets should be edge_indices
        # Additional kwargs should include node_labels, etc.
        return dataset_class(data, targets, **kwargs)
    
    raise ValueError(f"Failed to create dataset of type: {dataset_type}")


def create_fractional_dataloader(
        dataset: FractionalDataset,
        **kwargs) -> FractionalDataLoader:
    """
    Create a fractional data loader
    
    Args:
        dataset: Fractional dataset
        **kwargs: DataLoader parameters
        
    Returns:
        Configured fractional data loader
    """
    return FractionalDataLoader(dataset, **kwargs)
