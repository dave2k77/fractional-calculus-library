"""
Simple smoke test for the hpfracc library.
This is a minimal test that should work in CI/CD.
"""
import torch
import numpy as np

def test_basic_imports():
    """Test that basic imports work."""
    try:
        from hpfracc.ml import SpectralFractionalLayer, StochasticFractionalLayer
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic fractional calculus functionality."""
    try:
        from hpfracc.core import FractionalOrder
        from hpfracc.algorithms import optimized_grunwald_letnikov
        
        # Test basic fractional order
        alpha = FractionalOrder(0.5)
        print(f"✅ Created fractional order: {alpha}")
        
        # Test basic derivative computation
        x = np.linspace(0, 1, 10)
        result = optimized_grunwald_letnikov(x, alpha.alpha, 1.0)
        print(f"✅ Computed fractional derivative, shape: {result.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_ml_components():
    """Test ML components work."""
    try:
        from hpfracc.ml.layers import FractionalConv1D
        
        # Create a simple layer
        layer = FractionalConv1D(in_channels=1, out_channels=4, kernel_size=3)
        
        # Test forward pass
        x = torch.randn(1, 1, 10)  # batch, channels, sequence
        output = layer(x)
        print(f"✅ ML layer test passed, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ ML components test failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("🚀 Starting hpfracc smoke tests...")
    
    tests = [
        test_basic_imports,
        test_basic_functionality,
        test_ml_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training completed.")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
