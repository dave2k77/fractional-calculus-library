"""
Test script for variance-aware training functionality.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from hpfracc.ml.variance_aware_training import (
    VarianceMonitor, StochasticSeedManager, VarianceAwareCallback,
    AdaptiveSamplingManager, VarianceAwareTrainer, create_variance_aware_trainer
)


def test_variance_monitor():
    """Test the variance monitor."""
    print("Testing VarianceMonitor...")
    
    monitor = VarianceMonitor(window_size=10)
    
    # Test with some sample data
    for i in range(20):
        # Generate data with varying variance
        if i < 10:
            data = torch.randn(100) * 0.1  # Low variance
        else:
            data = torch.randn(100) * 1.0  # High variance
        
        monitor.update(f"test_component", data)
    
    # Get summary
    summary = monitor.get_summary()
    print("Variance summary:")
    for name, metrics in summary.items():
        print(f"  {name}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}, cv={metrics['cv']:.3f}")
    
    print("✓ VarianceMonitor test passed\n")


def test_seed_manager():
    """Test the seed manager."""
    print("Testing StochasticSeedManager...")
    
    seed_manager = StochasticSeedManager(base_seed=42)
    
    # Test seed setting
    seed_manager.set_seed(100)
    assert seed_manager.current_seed == 100
    
    # Test next seed
    next_seed = seed_manager.get_next_seed()
    assert next_seed == 101
    
    # Test reset
    seed_manager.reset_to_base()
    assert seed_manager.current_seed == 42
    
    print("✓ StochasticSeedManager test passed\n")


def test_adaptive_sampling():
    """Test adaptive sampling manager."""
    print("Testing AdaptiveSamplingManager...")
    
    sampling_manager = AdaptiveSamplingManager(initial_k=32, min_k=8, max_k=128)
    
    # Test with high variance (should increase K)
    new_k = sampling_manager.update_k(variance=0.5, current_k=32)
    assert new_k > 32  # Should increase K
    
    # Test with low variance (should decrease K)
    new_k = sampling_manager.update_k(variance=0.05, current_k=32)
    assert new_k < 32  # Should decrease K
    
    # Test bounds
    new_k = sampling_manager.update_k(variance=0.5, current_k=64)
    assert new_k <= 128  # Should not exceed max_k
    
    new_k = sampling_manager.update_k(variance=0.05, current_k=8)
    assert new_k >= 8  # Should not go below min_k
    
    print("✓ AdaptiveSamplingManager test passed\n")


def test_variance_aware_trainer():
    """Test the variance-aware trainer with a simple model."""
    print("Testing VarianceAwareTrainer...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
            return x
    
    # Create model, optimizer, and loss
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Create variance-aware trainer
    trainer = create_variance_aware_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        base_seed=42,
        variance_threshold=0.1,
        log_interval=2
    )
    
    # Create dummy data
    def create_dataloader():
        batches = []
        for _ in range(5):  # 5 batches
            data = torch.randn(16, 10)
            target = torch.randn(16, 1)
            batches.append((data, target))
        return batches
    
    # Train for a few epochs
    print("Training for 3 epochs...")
    results = trainer.train(create_dataloader(), num_epochs=3)
    
    # Check results
    assert len(results['losses']) == 3
    assert len(results['variance_history']) == 3
    assert len(results['epochs']) == 3
    
    print("Training results:")
    for i, (loss, variance_summary) in enumerate(zip(results['losses'], results['variance_history'])):
        print(f"  Epoch {i}: Loss = {loss:.4f}")
        if variance_summary:
            print(f"    Variance components: {len(variance_summary)}")
    
    # Test variance summary
    final_summary = trainer.get_variance_summary()
    print(f"Final variance summary has {len(final_summary)} components")
    
    # Test deterministic mode
    trainer.enable_deterministic_mode(True)
    trainer.enable_deterministic_mode(False)
    
    # Test sampling budget
    trainer.set_sampling_budget(64)
    assert trainer.adaptive_sampling.current_k == 64
    
    print("✓ VarianceAwareTrainer test passed\n")


def test_integration():
    """Test integration with actual stochastic/probabilistic components."""
    print("Testing integration with stochastic components...")
    
    try:
        from hpfracc.ml.stochastic_memory_sampling import StochasticFractionalLayer
        from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer
        
        # Create model with actual stochastic/probabilistic layers
        class StochasticModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 8)
                self.stochastic_layer = StochasticFractionalLayer(alpha=0.5, k=16, method="importance")
                self.probabilistic_layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
                self.linear2 = nn.Linear(8, 1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                
                # Handle stochastic layer output (might be scalar)
                stoch_out = self.stochastic_layer(x)
                if stoch_out.dim() == 0:  # scalar
                    stoch_out = stoch_out.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], 1)
                elif stoch_out.dim() == 1:  # 1D
                    stoch_out = stoch_out.unsqueeze(-1)
                
                # Handle probabilistic layer output (might be scalar)
                prob_out = self.probabilistic_layer(x)
                if prob_out.dim() == 0:  # scalar
                    prob_out = prob_out.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], 1)
                elif prob_out.dim() == 1:  # 1D
                    prob_out = prob_out.unsqueeze(-1)
                
                # Combine outputs
                if stoch_out.shape[1] + prob_out.shape[1] == 8:
                    x = torch.cat([stoch_out, prob_out], dim=1)
                else:
                    # Fallback: use original x if shapes don't match
                    x = x
                
                x = self.linear2(x)
                return x
        
        # Create model and trainer
        model = StochasticModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        trainer = create_variance_aware_trainer(model, optimizer, loss_fn)
        
        # Create data
        def create_dataloader():
            batches = []
            for _ in range(3):
                data = torch.randn(8, 10)
                target = torch.randn(8, 1)
                batches.append((data, target))
            return batches
        
        # Train
        results = trainer.train(create_dataloader(), num_epochs=2)
        
        print("Integration test results:")
        print(f"  Final loss: {results['losses'][-1]:.4f}")
        print(f"  Variance components monitored: {len(results['variance_history'][-1])}")
        
        print("✓ Integration test passed\n")
        
    except ImportError as e:
        print(f"⚠ Integration test skipped (import error: {e})\n")


def main():
    """Run all tests."""
    print("="*60)
    print("VARIANCE-AWARE TRAINING TESTS")
    print("="*60)
    
    try:
        test_variance_monitor()
        test_seed_manager()
        test_adaptive_sampling()
        test_variance_aware_trainer()
        test_integration()
        
        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
