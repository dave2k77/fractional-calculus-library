"""
Tests for Fractional Calculus Data Loading and Processing

This module tests the data loading, dataset classes, and data processing utilities.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from hpfracc.ml.data import (
    FractionalDataset, FractionalTensorDataset, FractionalTimeSeriesDataset,
    FractionalGraphDataset, FractionalDataLoader, FractionalDataProcessor,
    create_fractional_dataset, create_fractional_dataloader
)
from hpfracc.core.definitions import FractionalOrder


class TestFractionalDataset:
    """Test base FractionalDataset class"""

    def test_fractional_dataset_creation(self):
        """Test FractionalDataset creation"""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            dataset = FractionalDataset(fractional_order=0.5)

    def test_fractional_dataset_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        # Create a concrete subclass for testing
        class ConcreteDataset(FractionalDataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, index):
                return torch.randn(3), torch.randn(1)
        
        dataset = ConcreteDataset(fractional_order=0.5)
        assert dataset.fractional_order.alpha == 0.5
        assert dataset.method == "RL"
        assert dataset.apply_fractional is True


class TestFractionalTensorDataset:
    """Test FractionalTensorDataset class"""

    def test_fractional_tensor_dataset_creation(self):
        """Test FractionalTensorDataset creation"""
        tensors = [torch.randn(10, 3), torch.randn(10)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        assert dataset.fractional_order.alpha == 0.5
        assert len(dataset) == 10
        assert len(dataset.tensors) == 2

    def test_fractional_tensor_dataset_getitem(self):
        """Test FractionalTensorDataset __getitem__"""
        tensors = [torch.randn(10, 3), torch.randn(10)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        item = dataset[0]
        assert len(item) == 2
        assert item[0].shape == (3,)  # First tensor
        assert isinstance(item[1], list)  # Second tensor as list

    def test_fractional_tensor_dataset_single_tensor(self):
        """Test FractionalTensorDataset with single tensor"""
        tensors = [torch.randn(10, 3)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        item = dataset[0]
        assert item[0].shape == (3,)  # First tensor
        assert item[1] is None  # No targets


class TestFractionalTimeSeriesDataset:
    """Test FractionalTimeSeriesDataset class"""

    def test_fractional_time_series_dataset_creation(self):
        """Test FractionalTimeSeriesDataset creation"""
        data = torch.randn(100, 5)
        targets = torch.randn(100)
        sequence_length = 10
        
        dataset = FractionalTimeSeriesDataset(
            data, targets, sequence_length, fractional_order=0.5
        )
        assert dataset.fractional_order.alpha == 0.5
        assert len(dataset) == 91  # 100 - 10 + 1
        assert dataset.sequence_length == sequence_length

    def test_fractional_time_series_dataset_getitem(self):
        """Test FractionalTimeSeriesDataset __getitem__"""
        data = torch.randn(100, 5)
        targets = torch.randn(100)
        sequence_length = 10
        
        dataset = FractionalTimeSeriesDataset(
            data, targets, sequence_length, fractional_order=0.5
        )
        
        item = dataset[0]
        assert len(item) == 2
        assert item[0].shape == (sequence_length, 5)
        assert item[1].shape == ()

    def test_fractional_time_series_dataset_edge_cases(self):
        """Test FractionalTimeSeriesDataset edge cases"""
        data = torch.randn(5, 3)
        targets = torch.randn(5)
        sequence_length = 10
        
        # Sequence length longer than data
        with pytest.raises(ValueError):
            FractionalTimeSeriesDataset(
                data, targets, sequence_length, fractional_order=0.5
            )


class TestFractionalGraphDataset:
    """Test FractionalGraphDataset class"""

    def test_fractional_graph_dataset_creation(self):
        """Test FractionalGraphDataset creation"""
        node_features = [torch.randn(10, 5)]
        edge_indices = [torch.randint(0, 10, (2, 20))]
        node_labels = [torch.randint(0, 3, (10,))]
        
        dataset = FractionalGraphDataset(
            node_features, edge_indices, node_labels, fractional_order=0.5
        )
        assert dataset.fractional_order.alpha == 0.5
        assert len(dataset) == 1  # Single graph
        assert len(dataset.node_features) == 1

    def test_fractional_graph_dataset_getitem(self):
        """Test FractionalGraphDataset __getitem__"""
        node_features = [torch.randn(10, 5)]
        edge_indices = [torch.randint(0, 10, (2, 20))]
        node_labels = [torch.randint(0, 3, (10,))]
        
        dataset = FractionalGraphDataset(
            node_features, edge_indices, node_labels, fractional_order=0.5
        )
        
        item = dataset[0]
        assert isinstance(item, dict)
        assert 'node_features' in item
        assert 'edge_index' in item
        assert 'node_labels' in item


class TestFractionalDataLoader:
    """Test FractionalDataLoader class"""

    def test_fractional_data_loader_creation(self):
        """Test FractionalDataLoader creation"""
        tensors = [torch.randn(100, 3), torch.randn(100)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        dataloader = FractionalDataLoader(
            dataset, batch_size=32, shuffle=True
        )
        assert dataloader.batch_size == 32
        assert dataloader.shuffle is True
        assert dataloader.dataset == dataset

    def test_fractional_data_loader_iteration(self):
        """Test FractionalDataLoader iteration"""
        tensors = [torch.randn(100, 3), torch.randn(100)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        dataloader = FractionalDataLoader(
            dataset, batch_size=32, shuffle=False
        )
        
        batches = list(dataloader)
        assert len(batches) == 4  # 100 / 32 = 4 batches
        
        for batch in batches:
            assert len(batch) == 2
            assert batch[0].shape[0] <= 32  # batch size
            assert batch[0].shape[1] == 3   # feature dimension

    def test_fractional_data_loader_with_workers(self):
        """Test FractionalDataLoader with multiple workers"""
        tensors = [torch.randn(100, 3), torch.randn(100)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        dataloader = FractionalDataLoader(
            dataset, batch_size=32, num_workers=2
        )
        
        # Test that it works with multiple workers
        batches = list(dataloader)
        assert len(batches) > 0


class TestFractionalDataProcessor:
    """Test FractionalDataProcessor class"""

    def test_fractional_data_processor_creation(self):
        """Test FractionalDataProcessor creation"""
        processor = FractionalDataProcessor(fractional_order=0.5)
        assert processor.fractional_order.alpha == 0.5

    def test_fractional_data_processor_normalize_data(self):
        """Test FractionalDataProcessor normalize_data method"""
        processor = FractionalDataProcessor(fractional_order=0.5)
        data = torch.randn(100, 5)
        
        normalized_data, params = processor.normalize_data(data, method="standard")
        assert normalized_data.shape == data.shape
        
        # Check that normalization was applied
        assert not torch.allclose(normalized_data, data)
        assert 'mean' in params
        assert 'std' in params

    def test_fractional_data_processor_normalize_data_minmax(self):
        """Test FractionalDataProcessor normalize_data with minmax method"""
        processor = FractionalDataProcessor(fractional_order=0.5)
        data = torch.randn(100, 5)
        
        normalized_data, params = processor.normalize_data(data, method="minmax")
        assert normalized_data.shape == data.shape
        
        # Check that normalization was applied
        assert not torch.allclose(normalized_data, data)
        assert 'min' in params
        assert 'max' in params

    def test_fractional_data_processor_denormalize_data(self):
        """Test FractionalDataProcessor denormalize_data method"""
        processor = FractionalDataProcessor(fractional_order=0.5)
        data = torch.randn(100, 5)
        
        normalized_data, params = processor.normalize_data(data, method="standard")
        denormalized_data = processor.denormalize_data(normalized_data, params)
        
        # Check that denormalization works
        assert torch.allclose(data, denormalized_data, atol=1e-6)

    def test_fractional_data_processor_augment_data(self):
        """Test FractionalDataProcessor augment_data method"""
        processor = FractionalDataProcessor(fractional_order=0.5)
        data = torch.randn(100, 5)
        
        # Test noise augmentation
        augmented_data = processor.augment_data(data, augmentation_type="noise", noise_level=0.01)
        assert augmented_data.shape == data.shape
        
        # Test scaling augmentation
        augmented_data = processor.augment_data(data, augmentation_type="scaling", scale_factor=2.0)
        assert augmented_data.shape == data.shape

    def test_fractional_data_processor_create_sequences(self):
        """Test FractionalDataProcessor create_sequences method"""
        processor = FractionalDataProcessor(fractional_order=0.5)
        data = torch.randn(100, 5)
        
        sequences = processor.create_sequences(data, sequence_length=10, stride=5)
        assert sequences.shape[1] == 10  # sequence length
        assert sequences.shape[2] == 5   # feature dimension


class TestFactoryFunctions:
    """Test factory functions for creating datasets and dataloaders"""

    def test_create_fractional_dataset_tensor(self):
        """Test create_fractional_dataset for tensor data"""
        data = torch.randn(100, 3)
        targets = torch.randn(100)
        
        dataset = create_fractional_dataset(
            'tensor', data, targets, fractional_order=0.5
        )
        assert isinstance(dataset, FractionalTensorDataset)
        assert dataset.fractional_order.alpha == 0.5

    def test_create_fractional_dataset_timeseries(self):
        """Test create_fractional_dataset for time series data"""
        data = torch.randn(100, 5)
        targets = torch.randn(100)
        
        dataset = create_fractional_dataset(
            'timeseries', data, targets, sequence_length=10, fractional_order=0.5
        )
        assert isinstance(dataset, FractionalTimeSeriesDataset)
        assert dataset.fractional_order.alpha == 0.5

    def test_create_fractional_dataset_graph(self):
        """Test create_fractional_dataset for graph data"""
        node_features = [torch.randn(10, 5)]
        edge_indices = [torch.randint(0, 10, (2, 20))]
        node_labels = [torch.randint(0, 3, (10,))]
        
        dataset = create_fractional_dataset(
            'graph', node_features, edge_indices, node_labels=node_labels, fractional_order=0.5
        )
        assert isinstance(dataset, FractionalGraphDataset)
        assert dataset.fractional_order.alpha == 0.5

    def test_create_fractional_dataset_invalid_type(self):
        """Test create_fractional_dataset with invalid type"""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_fractional_dataset('invalid', None, None, fractional_order=0.5)

    def test_create_fractional_dataloader(self):
        """Test create_fractional_dataloader"""
        tensors = [torch.randn(100, 3), torch.randn(100)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        dataloader = create_fractional_dataloader(
            dataset, batch_size=32, shuffle=True
        )
        assert isinstance(dataloader, FractionalDataLoader)
        assert dataloader.batch_size == 32
        assert dataloader.shuffle is True


class TestDataIntegration:
    """Test data integration and edge cases"""

    def test_dataset_with_different_fractional_orders(self):
        """Test datasets with different fractional orders"""
        orders = [0.1, 0.5, 0.9]
        for order in orders:
            dataset = FractionalTensorDataset(
                [torch.randn(10, 3)], fractional_order=order
            )
            assert dataset.fractional_order.alpha == order

    def test_dataloader_edge_cases(self):
        """Test dataloader edge cases"""
        # Empty dataset
        tensors = [torch.randn(0, 3), torch.randn(0)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        dataloader = FractionalDataLoader(dataset, batch_size=32)
        batches = list(dataloader)
        assert len(batches) == 0

        # Single sample
        tensors = [torch.randn(1, 3), torch.randn(1)]
        dataset = FractionalTensorDataset(tensors, fractional_order=0.5)
        
        dataloader = FractionalDataLoader(dataset, batch_size=32)
        batches = list(dataloader)
        assert len(batches) == 1

    def test_data_processor_edge_cases(self):
        """Test data processor edge cases"""
        processor = FractionalDataProcessor(fractional_order=0.5)
        
        # Empty data
        empty_data = torch.randn(0, 5)
        normalized, params = processor.normalize_data(empty_data)
        assert normalized.shape == empty_data.shape
        
        # Single sample
        single_data = torch.randn(1, 5)
        normalized, params = processor.normalize_data(single_data)
        assert normalized.shape == single_data.shape

    def test_fractional_transform_integration(self):
        """Test fractional transform integration"""
        # Create a concrete dataset to test fractional transform
        class TestDataset(FractionalDataset):
            def __len__(self):
                return 1
            
            def __getitem__(self, index):
                return torch.randn(3), torch.randn(1)
        
        dataset = TestDataset(fractional_order=0.5, apply_fractional=True)
        
        # Test that fractional transform can be called
        data = torch.randn(3)
        transformed = dataset.fractional_transform(data)
        assert transformed is not None
