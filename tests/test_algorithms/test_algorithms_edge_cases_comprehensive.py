"""
Comprehensive edge case and error handling tests for algorithms modules.

This module provides extensive testing for edge cases, boundary conditions,
and error handling across all algorithm implementations.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from typing import Union, Callable

from hpfracc.algorithms.optimized_methods import (
    OptimizedRiemannLiouville, OptimizedCaputo, OptimizedGrunwaldLetnikov
)
from hpfracc.algorithms.optimized_methods import (
    ParallelOptimizedRiemannLiouville, ParallelOptimizedCaputo
)
from hpfracc.algorithms.special_methods import (
    FractionalLaplacian, FractionalFourierTransform
)
from hpfracc.core.fractional_implementations import MillerRossDerivative
from hpfracc.algorithms.integral_methods import (
    RiemannLiouvilleIntegral, CaputoIntegral
)
from hpfracc.core.integrals import WeylIntegral
from hpfracc.algorithms.novel_derivatives import (
    CaputoFabrizioDerivative, AtanganaBaleanuDerivative
)


class TestOptimizedMethodsEdgeCases:
    """Test edge cases for optimized methods implementations."""
    
    def test_optimized_riemann_liouville_zero_alpha(self):
        """Test optimized Riemann-Liouville with alpha = 0."""
        rl = OptimizedRiemannLiouville(0.0)
        assert rl.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_negative_alpha(self):
        """Test optimized Riemann-Liouville with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            OptimizedRiemannLiouville(-0.5)
    
    def test_optimized_riemann_liouville_large_alpha(self):
        """Test optimized Riemann-Liouville with large alpha."""
        rl = OptimizedRiemannLiouville(10.0)
        assert rl.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_empty_input(self):
        """Test optimized Riemann-Liouville with empty input."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = rl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_optimized_riemann_liouville_single_element(self):
        """Test optimized Riemann-Liouville with single element."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = rl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_constant_function(self):
        """Test optimized Riemann-Liouville with constant function."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def const_func(x):
            return np.ones_like(x) * 5.0
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(const_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_linear_function(self):
        """Test optimized Riemann-Liouville with linear function."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def linear_func(x):
            return 2.0 * x + 3.0
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(linear_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_oscillatory_function(self):
        """Test optimized Riemann-Liouville with oscillatory function."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def osc_func(x):
            return np.sin(10 * x)
        
        x_vals = np.array([0.0, 0.1, 0.2])
        result = rl.compute(osc_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_exponential_function(self):
        """Test optimized Riemann-Liouville with exponential function."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def exp_func(x):
            return np.exp(x)
        
        x_vals = np.array([0.0, 1.0, 2.0])
        result = rl.compute(exp_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_logarithmic_function(self):
        """Test optimized Riemann-Liouville with logarithmic function."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def log_func(x):
            return np.log(x + 1)
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(log_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_riemann_liouville_polynomial_function(self):
        """Test optimized Riemann-Liouville with polynomial function."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def poly_func(x):
            return x**5 + 2*x**3 + x + 1
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(poly_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_caputo_alpha_one(self):
        """Test optimized Caputo with alpha = 1 (now valid)."""
        # Caputo now supports alpha = 1.0
        caputo = OptimizedCaputo(1.0)
        assert caputo.alpha.alpha == 1.0
    
    def test_optimized_caputo_alpha_zero(self):
        """Test optimized Caputo with alpha = 0 (identity operation)."""
        # Alpha = 0 is mathematically valid (identity operation)
        caputo = OptimizedCaputo(0.0)
        assert caputo.alpha.alpha == 0.0
    
    def test_optimized_caputo_alpha_close_to_one(self):
        """Test optimized Caputo with alpha close to 1."""
        caputo = OptimizedCaputo(0.999999)
        assert caputo.alpha.alpha == 0.999999
    
    def test_optimized_caputo_alpha_close_to_zero(self):
        """Test optimized Caputo with alpha close to zero."""
        caputo = OptimizedCaputo(1e-10)
        assert caputo.alpha.alpha == 1e-10
    
    def test_optimized_grunwald_letnikov_negative_alpha(self):
        """Test optimized Grünwald-Letnikov with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            OptimizedGrunwaldLetnikov(-0.5)
    
    def test_optimized_grunwald_letnikov_zero_alpha(self):
        """Test optimized Grünwald-Letnikov with alpha = 0."""
        gl = OptimizedGrunwaldLetnikov(0.0)
        assert gl.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = gl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_optimized_grunwald_letnikov_large_alpha(self):
        """Test optimized Grünwald-Letnikov with large alpha."""
        gl = OptimizedGrunwaldLetnikov(10.0)
        assert gl.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = gl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestParallelOptimizedMethodsEdgeCases:
    """Test edge cases for parallel optimized methods implementations."""
    
    def test_parallel_optimized_riemann_liouville_zero_alpha(self):
        """Test parallel optimized Riemann-Liouville with alpha = 0."""
        porl = ParallelOptimizedRiemannLiouville(0.0)
        assert porl.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = porl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_parallel_optimized_riemann_liouville_negative_alpha(self):
        """Test parallel optimized Riemann-Liouville with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            ParallelOptimizedRiemannLiouville(-0.5)
    
    def test_parallel_optimized_riemann_liouville_large_alpha(self):
        """Test parallel optimized Riemann-Liouville with large alpha."""
        porl = ParallelOptimizedRiemannLiouville(10.0)
        assert porl.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = porl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_parallel_optimized_riemann_liouville_empty_input(self):
        """Test parallel optimized Riemann-Liouville with empty input."""
        porl = ParallelOptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = porl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_parallel_optimized_riemann_liouville_single_element(self):
        """Test parallel optimized Riemann-Liouville with single element."""
        porl = ParallelOptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = porl.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_parallel_optimized_caputo_alpha_one(self):
        """Test parallel optimized Caputo with alpha = 1 (now valid)."""
        # Caputo now supports alpha = 1.0
        caputo = ParallelOptimizedCaputo(1.0)
        assert caputo.alpha.alpha == 1.0
    
    def test_parallel_optimized_caputo_alpha_zero(self):
        """Test parallel optimized Caputo with alpha = 0 (identity operation)."""
        # Alpha = 0 is mathematically valid (identity operation)
        caputo = ParallelOptimizedCaputo(0.0)
        assert caputo.alpha.alpha == 0.0
    
    def test_parallel_optimized_caputo_alpha_close_to_one(self):
        """Test parallel optimized Caputo with alpha close to 1."""
        caputo = ParallelOptimizedCaputo(0.999999)
        assert caputo.alpha.alpha == 0.999999
    
    def test_parallel_optimized_caputo_alpha_close_to_zero(self):
        """Test parallel optimized Caputo with alpha close to zero."""
        caputo = ParallelOptimizedCaputo(1e-10)
        assert caputo.alpha.alpha == 1e-10
    
    def test_parallel_optimized_caputo_empty_input(self):
        """Test parallel optimized Caputo with empty input."""
        poc = ParallelOptimizedCaputo(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = poc.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_parallel_optimized_caputo_single_element(self):
        """Test parallel optimized Caputo with single element."""
        poc = ParallelOptimizedCaputo(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = poc.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestSpecialMethodsEdgeCases:
    """Test edge cases for special methods implementations."""
    
    def test_fractional_laplacian_zero_alpha(self):
        """Test fractional Laplacian with alpha = 0."""
        laplacian = FractionalLaplacian(0.0)
        assert laplacian.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = laplacian.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_fractional_laplacian_negative_alpha(self):
        """Test fractional Laplacian with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            FractionalLaplacian(-0.5)
    
    def test_fractional_laplacian_large_alpha(self):
        """Test fractional Laplacian with large alpha."""
        laplacian = FractionalLaplacian(10.0)
        assert laplacian.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = laplacian.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_fractional_laplacian_empty_input(self):
        """Test fractional Laplacian with empty input."""
        laplacian = FractionalLaplacian(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = laplacian.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_fractional_laplacian_single_element(self):
        """Test fractional Laplacian with single element."""
        laplacian = FractionalLaplacian(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = laplacian.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_fractional_fourier_transform_zero_alpha(self):
        """Test fractional Fourier transform with alpha = 0."""
        fft = FractionalFourierTransform(0.0)
        assert fft.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = fft.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_fractional_fourier_transform_negative_alpha(self):
        """Test fractional Fourier transform with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            FractionalFourierTransform(-0.5)
    
    def test_fractional_fourier_transform_large_alpha(self):
        """Test fractional Fourier transform with large alpha."""
        fft = FractionalFourierTransform(10.0)
        assert fft.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = fft.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_fractional_fourier_transform_empty_input(self):
        """Test fractional Fourier transform with empty input."""
        fft = FractionalFourierTransform(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = fft.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_fractional_fourier_transform_single_element(self):
        """Test fractional Fourier transform with single element."""
        fft = FractionalFourierTransform(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = fft.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_miller_ross_derivative_zero_alpha(self):
        """Test Miller-Ross derivative with alpha = 0."""
        mr = MillerRossDerivative(0.0)
        assert mr.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = mr.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_miller_ross_derivative_negative_alpha(self):
        """Test Miller-Ross derivative with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            MillerRossDerivative(-0.5)
    
    def test_miller_ross_derivative_large_alpha(self):
        """Test Miller-Ross derivative with large alpha."""
        mr = MillerRossDerivative(10.0)
        assert mr.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = mr.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_miller_ross_derivative_empty_input(self):
        """Test Miller-Ross derivative with empty input."""
        mr = MillerRossDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = mr.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_miller_ross_derivative_single_element(self):
        """Test Miller-Ross derivative with single element."""
        mr = MillerRossDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = mr.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestIntegralMethodsEdgeCases:
    """Test edge cases for integral methods implementations."""
    
    def test_riemann_liouville_integral_zero_alpha(self):
        """Test Riemann-Liouville integral with alpha = 0."""
        rl_integral = RiemannLiouvilleIntegral(0.0)
        assert rl_integral.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_riemann_liouville_integral_negative_alpha(self):
        """Test Riemann-Liouville integral with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            RiemannLiouvilleIntegral(-0.5)
    
    def test_riemann_liouville_integral_large_alpha(self):
        """Test Riemann-Liouville integral with large alpha."""
        rl_integral = RiemannLiouvilleIntegral(10.0)
        assert rl_integral.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_riemann_liouville_integral_empty_input(self):
        """Test Riemann-Liouville integral with empty input."""
        rl_integral = RiemannLiouvilleIntegral(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = rl_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_riemann_liouville_integral_single_element(self):
        """Test Riemann-Liouville integral with single element."""
        rl_integral = RiemannLiouvilleIntegral(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = rl_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_caputo_integral_zero_alpha(self):
        """Test Caputo integral with alpha = 0."""
        caputo_integral = CaputoIntegral(0.0)
        assert caputo_integral.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = caputo_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_caputo_integral_negative_alpha(self):
        """Test Caputo integral with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            CaputoIntegral(-0.5)
    
    def test_caputo_integral_large_alpha(self):
        """Test Caputo integral with large alpha."""
        caputo_integral = CaputoIntegral(10.0)
        assert caputo_integral.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = caputo_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_weyl_integral_zero_alpha(self):
        """Test Weyl integral with alpha = 0."""
        weyl_integral = WeylIntegral(0.0)
        assert weyl_integral.alpha.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = weyl_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_weyl_integral_negative_alpha(self):
        """Test Weyl integral with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            WeylIntegral(-0.5)
    
    def test_weyl_integral_large_alpha(self):
        """Test Weyl integral with large alpha."""
        weyl_integral = WeylIntegral(10.0)
        assert weyl_integral.alpha.alpha == 10.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = weyl_integral.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestNovelDerivativesEdgeCases:
    """Test edge cases for novel derivatives implementations."""
    
    def test_caputo_fabrizio_derivative_zero_alpha(self):
        """Test Caputo-Fabrizio derivative with alpha = 0."""
        cf_deriv = CaputoFabrizioDerivative(0.0)
        assert cf_deriv.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = cf_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_caputo_fabrizio_derivative_negative_alpha(self):
        """Test Caputo-Fabrizio derivative with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order α must be in \\[0, 1\\) for Caputo-Fabrizio"):
            CaputoFabrizioDerivative(-0.5)
    
    def test_caputo_fabrizio_derivative_large_alpha(self):
        """Test Caputo-Fabrizio derivative with large alpha."""
        with pytest.raises(ValueError, match="Fractional order α must be in \\[0, 1\\) for Caputo-Fabrizio"):
            CaputoFabrizioDerivative(10.0)
    
    def test_caputo_fabrizio_derivative_empty_input(self):
        """Test Caputo-Fabrizio derivative with empty input."""
        cf_deriv = CaputoFabrizioDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = cf_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_caputo_fabrizio_derivative_single_element(self):
        """Test Caputo-Fabrizio derivative with single element."""
        cf_deriv = CaputoFabrizioDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = cf_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_atangana_baleanu_derivative_zero_alpha(self):
        """Test Atangana-Baleanu derivative with alpha = 0."""
        ab_deriv = AtanganaBaleanuDerivative(0.0)
        assert ab_deriv.alpha == 0.0
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = ab_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_atangana_baleanu_derivative_negative_alpha(self):
        """Test Atangana-Baleanu derivative with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order α must be in \\[0, 1\\) for Atangana-Baleanu"):
            AtanganaBaleanuDerivative(-0.5)
    
    def test_atangana_baleanu_derivative_large_alpha(self):
        """Test Atangana-Baleanu derivative with large alpha."""
        with pytest.raises(ValueError, match="Fractional order α must be in \\[0, 1\\) for Atangana-Baleanu"):
            AtanganaBaleanuDerivative(10.0)
    
    def test_atangana_baleanu_derivative_empty_input(self):
        """Test Atangana-Baleanu derivative with empty input."""
        ab_deriv = AtanganaBaleanuDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([])
        result = ab_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_atangana_baleanu_derivative_single_element(self):
        """Test Atangana-Baleanu derivative with single element."""
        ab_deriv = AtanganaBaleanuDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0])
        result = ab_deriv.compute(test_func, x_vals)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestNumericalStabilityEdgeCases:
    """Test numerical stability and precision edge cases."""
    
    def test_high_precision_calculation(self):
        """Test calculations with high precision requirements."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return np.sin(x)
        
        x_vals = np.array([0.0, np.pi/4, np.pi/2])
        result = rl.compute(test_func, x_vals)
        
        # Check that result is finite
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_very_small_values(self):
        """Test calculations with very small values."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**0.1  # Very small power
        
        x_vals = np.array([1e-10, 1e-8, 1e-6])
        result = rl.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_very_large_values(self):
        """Test calculations with very large values."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**10  # Large power
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_near_boundary_values(self):
        """Test calculations near boundary values."""
        # Test alpha very close to 0
        rl = OptimizedRiemannLiouville(1e-10)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_near_integer_values(self):
        """Test calculations near integer alpha values."""
        # Test alpha very close to 1
        rl = OptimizedRiemannLiouville(0.999999)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl.compute(test_func, x_vals)
        
        assert np.all(np.isfinite(result))
        assert isinstance(result, np.ndarray)
    
    def test_memory_intensive_calculation(self):
        """Test calculations with memory-intensive operations."""
        rl = OptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**2
        
        # Test with large array
        x_vals = np.linspace(0, 10, 10000)
        result = rl.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_concurrent_calculations(self):
        """Test calculations with concurrent operations."""
        import threading
        import time
        
        rl = OptimizedRiemannLiouville(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        results = []
        
        def compute_derivative():
            result = rl.compute(test_func, x_vals)
            results.append(result)
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=compute_derivative)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all results are valid
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert len(result) > 0
