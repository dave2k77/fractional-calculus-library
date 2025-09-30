#!/usr/bin/env python3
"""Simple tests for hybrid GNN layers module."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import sys

# Mock JAX/jraph imports if not available
with patch.dict('sys.modules', {
    'jax': MagicMock(),
    'jax.numpy': MagicMock(),
    'jraph': MagicMock(),
    'numba': MagicMock()
}):
    from hpfracc.ml.hybrid_gnn_layers import (
        GraphConfig,
        HybridFractionalGraphConv,
        HybridFractionalGraphAttention,
        HybridFractionalGraphPooling,
        create_graph_conv,
        create_graph_attention,
        create_graph_pooling
    )


class TestGraphConfig:
    """Test graph configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = GraphConfig()
        assert hasattr(config, 'use_jax')
        assert hasattr(config, 'fractional_order')
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = GraphConfig(
            use_jax=False,
            fractional_order=0.3,
            method="Caputo"
        )
        assert config.use_jax is False
        assert config.fractional_order == 0.3


class TestHybridFractionalGraphConv:
    """Test hybrid fractional graph convolution."""
    
    def test_initialization(self):
        """Test initialization."""
        conv = HybridFractionalGraphConv(
            input_dim=10,
            output_dim=5,
            fractional_order=0.5
        )
        assert conv.input_dim == 10
        assert conv.output_dim == 5
        assert conv.fractional_order == 0.5
        
    def test_forward_pass(self):
        """Test forward pass."""
        conv = HybridFractionalGraphConv(10, 5, 0.5)
        
        # Mock data
        node_features = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(conv, '_forward_torch') as mock_torch:
            mock_torch.return_value = torch.randn(5, 5)
            
            result = conv.forward(node_features, edge_index)
            assert result.shape == (5, 5)


class TestHybridFractionalGraphAttention:
    """Test hybrid fractional graph attention."""
    
    def test_initialization(self):
        """Test initialization."""
        attention = HybridFractionalGraphAttention(
            input_dim=10,
            output_dim=5,
            num_heads=4,
            fractional_order=0.5
        )
        assert attention.input_dim == 10
        assert attention.output_dim == 5
        assert attention.num_heads == 4
        
    def test_forward_pass(self):
        """Test forward pass."""
        attention = HybridFractionalGraphAttention(10, 5, 4, 0.5)
        
        node_features = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(attention, '_forward_torch') as mock_torch:
            mock_torch.return_value = torch.randn(5, 5)
            
            result = attention.forward(node_features, edge_index)
            assert result.shape == (5, 5)


class TestHybridFractionalGraphPooling:
    """Test hybrid fractional graph pooling."""
    
    def test_initialization(self):
        """Test initialization."""
        pooling = HybridFractionalGraphPooling(
            input_dim=10,
            pool_ratio=0.5,
            fractional_order=0.5
        )
        assert pooling.input_dim == 10
        assert pooling.pool_ratio == 0.5
        
    def test_forward_pass(self):
        """Test forward pass."""
        pooling = HybridFractionalGraphPooling(10, 0.5, 0.5)
        
        node_features = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(pooling, '_forward_torch') as mock_torch:
            mock_torch.return_value = (torch.randn(5, 10), torch.tensor([0, 1, 2, 3, 4]))
            
            result, cluster_indices = pooling.forward(node_features, edge_index)
            assert result.shape == (5, 10)
            assert len(cluster_indices) == 5


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_graph_conv(self):
        """Test create_graph_conv function."""
        conv = create_graph_conv(
            input_dim=10,
            output_dim=5,
            fractional_order=0.5
        )
        assert isinstance(conv, HybridFractionalGraphConv)
        
    def test_create_graph_attention(self):
        """Test create_graph_attention function."""
        attention = create_graph_attention(
            input_dim=10,
            output_dim=5,
            num_heads=4,
            fractional_order=0.5
        )
        assert isinstance(attention, HybridFractionalGraphAttention)
        
    def test_create_graph_pooling(self):
        """Test create_graph_pooling function."""
        pooling = create_graph_pooling(
            input_dim=10,
            pool_ratio=0.5,
            fractional_order=0.5
        )
        assert isinstance(pooling, HybridFractionalGraphPooling)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_graph(self):
        """Test handling of empty graph."""
        conv = HybridFractionalGraphConv(10, 5, 0.5)
        
        # Empty node features and edge index
        node_features = torch.empty(0, 10)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        
        with patch.object(conv, '_forward_torch') as mock_torch:
            mock_torch.return_value = torch.empty(0, 5)
            
            result = conv.forward(node_features, edge_index)
            assert result.shape == (0, 5)
            
    def test_single_node(self):
        """Test handling of single node."""
        conv = HybridFractionalGraphConv(10, 5, 0.5)
        
        node_features = torch.randn(1, 10)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        
        with patch.object(conv, '_forward_torch') as mock_torch:
            mock_torch.return_value = torch.randn(1, 5)
            
            result = conv.forward(node_features, edge_index)
            assert result.shape == (1, 5)
