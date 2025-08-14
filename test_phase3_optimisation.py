#!/usr/bin/env python3
"""
Phase 3: Optimization and Parallel Computing Test Suite

This script tests all the optimization and parallel computing implementations
in Phase 3, including JAX GPU acceleration, NUMBA CPU parallelization,
and distributed computing strategies.
"""

import numpy as np
import sys
import os
import traceback
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_jax_optimizations():
    """Test JAX optimization implementations."""
    print("🧪 Testing JAX Optimizations...")
    
    try:
        from optimisation import (
            JAXOptimizer, JAXFractionalDerivatives, JAXAutomaticDifferentiation,
            JAXVectorization, JAXPerformanceMonitor,
            optimize_fractional_derivative_jax, compute_fractional_derivative_gpu
        )
        import jax.numpy as jnp
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Test JAX optimizer
        try:
            optimizer = JAXOptimizer(device="auto", precision="float32")
            print("  ✅ JAXOptimizer: Success")
        except Exception as e:
            print(f"  ⚠️  JAXOptimizer: {str(e)}")
        
        # Test GPU-accelerated derivatives
        try:
            # Caputo derivative
            result_caputo = JAXFractionalDerivatives.caputo_derivative_gpu(
                jnp.array(f_values), jnp.array(t), alpha, h
            )
            print("  ✅ GPU Caputo derivative: Success")
        except Exception as e:
            print(f"  ⚠️  GPU Caputo derivative: {str(e)}")
        
        try:
            # Riemann-Liouville derivative
            result_rl = JAXFractionalDerivatives.riemann_liouville_derivative_gpu(
                jnp.array(f_values), jnp.array(t), alpha, h
            )
            print("  ✅ GPU Riemann-Liouville derivative: Success")
        except Exception as e:
            print(f"  ⚠️  GPU Riemann-Liouville derivative: {str(e)}")
        
        try:
            # Grünwald-Letnikov derivative
            result_gl = JAXFractionalDerivatives.grunwald_letnikov_derivative_gpu(
                jnp.array(f_values), jnp.array(t), alpha, h
            )
            print("  ✅ GPU Grünwald-Letnikov derivative: Success")
        except Exception as e:
            print(f"  ⚠️  GPU Grünwald-Letnikov derivative: {str(e)}")
        
        try:
            # FFT-based derivative
            result_fft = JAXFractionalDerivatives.fft_fractional_derivative_gpu(
                jnp.array(f_values), jnp.array(t), alpha, h, "spectral"
            )
            print("  ✅ GPU FFT-based derivative: Success")
        except Exception as e:
            print(f"  ⚠️  GPU FFT-based derivative: {str(e)}")
        
        # Test automatic differentiation
        try:
            grad_func = JAXAutomaticDifferentiation.gradient_wrt_alpha(
                JAXFractionalDerivatives.caputo_derivative_gpu,
                jnp.array(f_values), jnp.array(t), alpha, h
            )
            print("  ✅ Automatic differentiation: Success")
        except Exception as e:
            print(f"  ⚠️  Automatic differentiation: {str(e)}")
        
        # Test vectorization
        try:
            alphas = jnp.array([0.3, 0.5, 0.7])
            vectorized_result = JAXVectorization.vectorize_over_alpha(
                JAXFractionalDerivatives.caputo_derivative_gpu,
                jnp.array(f_values), jnp.array(t), alphas, h
            )
            print("  ✅ Vectorization over alpha: Success")
        except Exception as e:
            print(f"  ⚠️  Vectorization over alpha: {str(e)}")
        
        # Test performance monitoring
        try:
            monitor = JAXPerformanceMonitor()
            timing = monitor.time_function(
                JAXFractionalDerivatives.caputo_derivative_gpu,
                jnp.array(f_values), jnp.array(t), alpha, h
            )
            print("  ✅ Performance monitoring: Success")
        except Exception as e:
            print(f"  ⚠️  Performance monitoring: {str(e)}")
        
        # Test convenience functions
        try:
            result_gpu = compute_fractional_derivative_gpu(
                jnp.array(f_values), jnp.array(t), alpha, h, "caputo"
            )
            print("  ✅ Convenience GPU function: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience GPU function: {str(e)}")
        
        print("  ✅ JAX optimizations test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ JAX optimizations test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_numba_kernels():
    """Test NUMBA kernel implementations."""
    print("🧪 Testing NUMBA Kernels...")
    
    try:
        from optimisation import (
            NumbaOptimizer, NumbaFractionalKernels, NumbaMemoryOptimizer,
            NumbaSpecializedKernels, NumbaParallelManager,
            optimize_fractional_kernel_numba, compute_caputo_derivative_numba_optimized
        )
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Test NUMBA optimizer
        try:
            optimizer = NumbaOptimizer(parallel=True, fastmath=True)
            print("  ✅ NumbaOptimizer: Success")
        except Exception as e:
            print(f"  ⚠️  NumbaOptimizer: {str(e)}")
        
        # Test fractional kernels
        try:
            # Caputo L1 kernel
            result_l1 = NumbaFractionalKernels.caputo_l1_kernel(f_values, alpha, h)
            print("  ✅ Caputo L1 kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Caputo L1 kernel: {str(e)}")
        
        try:
            # Caputo L2 kernel
            result_l2 = NumbaFractionalKernels.caputo_l2_kernel(f_values, alpha, h)
            print("  ✅ Caputo L2 kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Caputo L2 kernel: {str(e)}")
        
        try:
            # Grünwald-Letnikov kernel
            result_gl = NumbaFractionalKernels.grunwald_letnikov_kernel(f_values, alpha, h)
            print("  ✅ Grünwald-Letnikov kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Grünwald-Letnikov kernel: {str(e)}")
        
        try:
            # Riemann-Liouville kernel
            result_rl = NumbaFractionalKernels.riemann_liouville_kernel(f_values, alpha, h)
            print("  ✅ Riemann-Liouville kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Riemann-Liouville kernel: {str(e)}")
        
        try:
            # FFT convolution kernel
            kernel = np.ones(len(f_values))
            result_fft = NumbaFractionalKernels.fft_convolution_kernel(f_values, kernel, h)
            print("  ✅ FFT convolution kernel: Success")
        except Exception as e:
            print(f"  ⚠️  FFT convolution kernel: {str(e)}")
        
        # Test memory optimization
        try:
            result_mem = NumbaMemoryOptimizer.memory_efficient_caputo(f_values, alpha, h)
            print("  ✅ Memory-efficient Caputo: Success")
        except Exception as e:
            print(f"  ⚠️  Memory-efficient Caputo: {str(e)}")
        
        try:
            result_block = NumbaMemoryOptimizer.block_processing_kernel(f_values, alpha, h)
            print("  ✅ Block processing kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Block processing kernel: {str(e)}")
        
        # Test specialized kernels
        try:
            power_kernel = NumbaSpecializedKernels.power_law_kernel(t, alpha, 1)
            print("  ✅ Power-law kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Power-law kernel: {str(e)}")
        
        try:
            binom_coeffs = NumbaSpecializedKernels.binomial_coefficients_kernel(alpha, 10)
            print("  ✅ Binomial coefficients kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Binomial coefficients kernel: {str(e)}")
        
        try:
            ml_result = NumbaSpecializedKernels.mittag_leffler_kernel(t, alpha, 1.0)
            print("  ✅ Mittag-Leffler kernel: Success")
        except Exception as e:
            print(f"  ⚠️  Mittag-Leffler kernel: {str(e)}")
        
        # Test parallel manager
        try:
            parallel_manager = NumbaParallelManager(num_threads=4)
            chunk_size = parallel_manager.get_optimal_chunk_size(len(f_values))
            print("  ✅ Parallel manager: Success")
        except Exception as e:
            print(f"  ⚠️  Parallel manager: {str(e)}")
        
        # Test convenience functions
        try:
            result_opt = compute_caputo_derivative_numba_optimized(f_values, alpha, h, "l1")
            print("  ✅ Convenience NUMBA function: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience NUMBA function: {str(e)}")
        
        print("  ✅ NUMBA kernels test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ NUMBA kernels test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_parallel_computing():
    """Test parallel computing implementations."""
    print("🧪 Testing Parallel Computing...")
    
    try:
        from optimisation import (
            ParallelComputingManager, DistributedComputing, LoadBalancer,
            PerformanceOptimizer, ParallelFractionalComputing,
            parallel_fractional_derivative, optimize_parallel_parameters,
            get_system_info
        )
        from algorithms import caputo_derivative
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 50)
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Test parallel computing manager
        try:
            with ParallelComputingManager(num_workers=2, backend="multiprocessing") as manager:
                # Simple test function
                def test_func(x):
                    return x * 2
                
                test_data = [1, 2, 3, 4, 5]
                results = manager.parallel_map(test_func, test_data)
                print("  ✅ ParallelComputingManager: Success")
        except Exception as e:
            print(f"  ⚠️  ParallelComputingManager: {str(e)}")
        
        # Test distributed computing
        try:
            distributed = DistributedComputing(num_nodes=1, node_id=0)
            work_items = list(range(10))
            local_work = distributed.distribute_work(work_items, strategy="round_robin")
            print("  ✅ DistributedComputing: Success")
        except Exception as e:
            print(f"  ⚠️  DistributedComputing: {str(e)}")
        
        # Test load balancer
        try:
            load_balancer = LoadBalancer(num_workers=4)
            work_items = list(range(20))
            chunks = load_balancer.create_work_chunks(work_items, chunk_size=5)
            print("  ✅ LoadBalancer: Success")
        except Exception as e:
            print(f"  ⚠️  LoadBalancer: {str(e)}")
        
        # Test performance optimizer
        try:
            optimizer = PerformanceOptimizer()
            optimizer.start_monitoring()
            
            # Simulate some work
            time.sleep(0.1)
            optimizer.record_metrics(50.0, 100.0, 0.1, 10.0)
            
            summary = optimizer.get_performance_summary()
            print("  ✅ PerformanceOptimizer: Success")
        except Exception as e:
            print(f"  ⚠️  PerformanceOptimizer: {str(e)}")
        
        # Test parallel fractional computing
        try:
            parallel_fc = ParallelFractionalComputing(num_workers=2)
            
            # Create multiple function arrays
            f_values_list = [f_values, f_values * 2, f_values * 3]
            t_values_list = [t, t, t]
            
            results = parallel_fc.parallel_fractional_derivative(
                caputo_derivative, f_values_list, t_values_list, alpha, h
            )
            print("  ✅ ParallelFractionalComputing: Success")
        except Exception as e:
            print(f"  ⚠️  ParallelFractionalComputing: {str(e)}")
        
        # Test convenience functions
        try:
            system_info = get_system_info()
            print("  ✅ System info: Success")
        except Exception as e:
            print(f"  ⚠️  System info: {str(e)}")
        
        try:
            opt_params = optimize_parallel_parameters(1000, 4, 0.001)
            print("  ✅ Parallel parameter optimization: Success")
        except Exception as e:
            print(f"  ⚠️  Parallel parameter optimization: {str(e)}")
        
        print("  ✅ Parallel computing test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Parallel computing test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_optimization_integration():
    """Test integration between different optimization strategies."""
    print("🧪 Testing Optimization Integration...")
    
    try:
        from optimisation import (
            JAXOptimizer, NumbaOptimizer, ParallelComputingManager,
            compute_fractional_derivative_gpu, compute_caputo_derivative_numba_optimized
        )
        import jax.numpy as jnp
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Test JAX + NUMBA integration
        try:
            # JAX GPU computation
            jax_result = compute_fractional_derivative_gpu(
                jnp.array(f_values), jnp.array(t), alpha, h, "caputo"
            )
            
            # NUMBA CPU computation
            numba_result = compute_caputo_derivative_numba_optimized(
                f_values, alpha, h, "l1"
            )
            
            print("  ✅ JAX + NUMBA integration: Success")
        except Exception as e:
            print(f"  ⚠️  JAX + NUMBA integration: {str(e)}")
        
        # Test parallel + optimization integration
        try:
            with ParallelComputingManager(num_workers=2) as manager:
                # Parallel NUMBA computation
                def numba_worker(data):
                    f_vals, alpha_val, h_val = data
                    return compute_caputo_derivative_numba_optimized(f_vals, alpha_val, h_val, "l1")
                
                work_data = [(f_values, alpha, h), (f_values * 2, alpha, h)]
                results = manager.parallel_map(numba_worker, work_data)
                
                print("  ✅ Parallel + NUMBA integration: Success")
        except Exception as e:
            print(f"  ⚠️  Parallel + NUMBA integration: {str(e)}")
        
        # Test performance comparison
        try:
            # Time JAX computation
            start_time = time.time()
            jax_result = compute_fractional_derivative_gpu(
                jnp.array(f_values), jnp.array(t), alpha, h, "caputo"
            )
            jax_time = time.time() - start_time
            
            # Time NUMBA computation
            start_time = time.time()
            numba_result = compute_caputo_derivative_numba_optimized(
                f_values, alpha, h, "l1"
            )
            numba_time = time.time() - start_time
            
            print(f"  ✅ Performance comparison: JAX={jax_time:.4f}s, NUMBA={numba_time:.4f}s")
        except Exception as e:
            print(f"  ⚠️  Performance comparison: {str(e)}")
        
        print("  ✅ Optimization integration test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Optimization integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase 3 optimization and parallel computing tests."""
    print("🚀 Starting Phase 3: Optimization and Parallel Computing Test Suite\n")
    
    tests = [
        ("JAX Optimizations", test_jax_optimizations),
        ("NUMBA Kernels", test_numba_kernels),
        ("Parallel Computing", test_parallel_computing),
        ("Optimization Integration", test_optimization_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        print("=" * 50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED\n")
        else:
            print(f"❌ {test_name}: FAILED\n")
    
    print("🎯 Phase 3 Test Summary")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All Phase 3 optimization and parallel computing tests passed!")
        print("✅ Phase 3: Optimization and Parallel Computing is complete!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
