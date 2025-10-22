"""
Tests for variance-aware training module.
"""

import pytest
import torch
from hpfracc.ml.variance_aware_training import VarianceMonitor
import logging

class TestVarianceMonitor:
    """Test VarianceMonitor class."""

    def test_initialization(self):
        """Test that the VarianceMonitor initializes correctly."""
        monitor = VarianceMonitor()
        assert monitor.window_size == 100
        assert monitor.variance_threshold == 0.1
        assert monitor.high_variance_threshold == 0.5

    def test_update_with_tensor(self):
        """Test the update method with a torch.Tensor."""
        monitor = VarianceMonitor()
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        monitor.update("test_tensor", tensor)

        metrics = monitor.get_metrics("test_tensor")
        assert metrics is not None
        assert metrics.mean == 3.0
        assert metrics.std == pytest.approx(1.41421356)
        assert metrics.variance == pytest.approx(2.0)
        assert metrics.sample_count == 5

    def test_update_with_high_variance(self, caplog):
        """Test that a warning is logged when variance is high."""
        monitor = VarianceMonitor()
        tensor = torch.tensor([1.0, 100.0, 200.0])
        
        with caplog.at_level(logging.WARNING):
            monitor.update("high_variance_tensor", tensor)
        
        assert "High variance detected" in caplog.text

    def test_get_history(self):
        """Test the get_history method."""
        monitor = VarianceMonitor()
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        monitor.update("test_tensor", tensor1)
        monitor.update("test_tensor", tensor2)

        history = monitor.get_history("test_tensor")
        assert len(history) == 2
        assert history[0].mean == 2.0
        assert history[1].mean == 5.0

    def test_get_summary(self):
        """Test the get_summary method."""
        monitor = VarianceMonitor()
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        monitor.update("test_tensor1", tensor1)
        monitor.update("test_tensor2", tensor2)

        summary = monitor.get_summary()
        assert "test_tensor1" in summary
        assert "test_tensor2" in summary
        assert summary["test_tensor1"]["mean"] == 2.0
        assert summary["test_tensor2"]["mean"] == 5.0
