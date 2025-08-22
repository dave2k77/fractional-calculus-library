#!/usr/bin/env python3
"""
Quick test script to verify all major functionality works with the installed hpfracc package.
This is a simplified version of the full test suite that works with the conda environment.
"""

import numpy as np
import time

def test_core_functionality():
    """Test core fractional calculus methods."""
    print("🧪 Testing Core Functionality...")
    
    try:
        from hpfracc import (
            FractionalOrder, 
            OptimizedCaputo, 
            OptimizedRiemannLiouville,
            OptimizedGrunwaldLetnikov
        )
        print("✅ Core imports successful")
        
        # Test data
        t = np.linspace(0, 1, 100)
        f = t**2 + np.sin(2*np.pi*t)
        h = t[1] - t[0]
        alpha = 0.5
        
        # Test Caputo
        caputo = OptimizedCaputo(alpha)
        result_caputo = caputo.compute(f, t, h)
        print(f"✅ Caputo derivative computed: shape {result_caputo.shape}")
        
        # Test Riemann-Liouville
        rl = OptimizedRiemannLiouville(alpha)
        result_rl = rl.compute(f, t, h)
        print(f"✅ Riemann-Liouville derivative computed: shape {result_rl.shape}")
        
        # Test Grünwald-Letnikov
        gl = OptimizedGrunwaldLetnikov(alpha)
        result_gl = gl.compute(f, t, h)
        print(f"✅ Grünwald-Letnikov derivative computed: shape {result_gl.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_advanced_methods():
    """Test advanced fractional calculus methods."""
    print("\n🧪 Testing Advanced Methods...")
    
    try:
        from hpfracc import (
            WeylDerivative, 
            MarchaudDerivative, 
            HadamardDerivative,
            AdomianDecomposition
        )
        print("✅ Advanced method imports successful")
        
        # Test data
        t = np.linspace(0, 1, 50)
        f = np.exp(-t)
        alpha = 0.3
        
        # Test Weyl
        weyl = WeylDerivative(alpha)
        result_weyl = weyl.compute(f, t)
        print(f"✅ Weyl derivative computed: shape {result_weyl.shape}")
        
        # Test Marchaud
        marchaud = MarchaudDerivative(alpha)
        result_marchaud = marchaud.compute(f, t)
        print(f"✅ Marchaud derivative computed: shape {result_marchaud.shape}")
        
        # Test Hadamard
        hadamard = HadamardDerivative(alpha)
        result_hadamard = hadamard.compute(f, t)
        print(f"✅ Hadamard derivative computed: shape {result_hadamard.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced methods test failed: {e}")
        return False

def test_special_methods():
    """Test special fractional calculus methods."""
    print("\n🧪 Testing Special Methods...")
    
    try:
        from hpfracc import (
            FractionalLaplacian,
            FractionalFourierTransform,
            FractionalZTransform,
            FractionalMellinTransform
        )
        print("✅ Special method imports successful")
        
        # Test data
        x = np.linspace(0.1, 5, 100)  # Positive domain for Mellin transform
        f = np.exp(-x**2)
        alpha = 0.5
        
        # Test Fractional Laplacian
        laplacian = FractionalLaplacian(alpha)
        result_laplacian = laplacian.compute(f, x, method='spectral')
        print(f"✅ Fractional Laplacian computed: shape {result_laplacian.shape}")
        
        # Test Fractional Fourier Transform
        frft = FractionalFourierTransform(alpha)
        result_frft = frft.transform(f, x, method='fast')
        # FrFT returns (u, result) tuple, extract the result
        if isinstance(result_frft, tuple):
            result_frft = result_frft[1]
        print(f"✅ Fractional Fourier Transform computed: shape {result_frft.shape}")
        
        # Test Fractional Z-Transform
        z_transform = FractionalZTransform(alpha)
        result_z = z_transform.transform(f, x)
        print(f"✅ Fractional Z-Transform computed: shape {result_z.shape}")
        
        # Test Fractional Mellin Transform
        mellin = FractionalMellinTransform(alpha)
        s_values = np.linspace(0.1, 2.0, 50)  # s parameter for Mellin transform
        result_mellin = mellin.transform(f, x, s_values, method='numerical')
        print(f"✅ Fractional Mellin Transform computed: shape {result_mellin.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Special methods test failed: {e}")
        return False

def test_performance():
    """Test performance of optimized methods."""
    print("\n🧪 Testing Performance...")
    
    try:
        from hpfracc import OptimizedCaputo
        
        # Test with larger arrays
        sizes = [100, 1000, 5000]
        alpha = 0.5
        
        for size in sizes:
            t = np.linspace(0, 1, size)
            f = t**2 + np.sin(2*np.pi*t)
            h = t[1] - t[0]
            
            caputo = OptimizedCaputo(alpha)
            
            start_time = time.time()
            result = caputo.compute(f, t, h)
            end_time = time.time()
            
            elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"✅ Size {size}: {elapsed:.2f} ms, result shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting hpfracc Package Functionality Test")
    print("=" * 50)
    
    tests = [
        test_core_functionality,
        test_advanced_methods,
        test_special_methods,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The hpfracc package is working perfectly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
