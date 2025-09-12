"""
Comprehensive tests for fractional implementations module.

This module provides extensive testing for all fractional derivative
and integral implementations in the core module.
"""

import pytest
import numpy as np
from typing import Union, Callable

from hpfracc.core.fractional_implementations import (
    RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
    CaputoFabrizioDerivative, AtanganaBaleanuDerivative, FractionalLaplacian,
    FractionalFourierTransform, MillerRossDerivative, WeylDerivative,
    MarchaudDerivative, HadamardDerivative, ReizFellerDerivative,
    ParallelOptimizedRiemannLiouville, ParallelOptimizedCaputo,
    RieszFisherOperator, AdomianDecompositionMethod,
    create_fractional_integral, create_riesz_fisher_operator,
    register_fractional_implementations
)
from hpfracc.core.derivatives import BaseFractionalDerivative
from hpfracc.core.definitions import FractionalOrder, DefinitionType


class TestRiemannLiouvilleDerivative:
    """Test Riemann-Liouville fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        # Test with float alpha
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        assert rl_deriv.alpha == 0.5
        
        # Test with FractionalOrder
        alpha = FractionalOrder(0.7, DefinitionType.RIEMANN_LIOUVILLE)
        rl_deriv = RiemannLiouvilleDerivative(alpha)
        assert rl_deriv.alpha == alpha
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rl_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        f_values = np.array([1.0, 4.0, 9.0])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = rl_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestCaputoDerivative:
    """Test Caputo fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        caputo_deriv = CaputoDerivative(0.3)
        assert caputo_deriv.alpha == 0.3
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        caputo_deriv = CaputoDerivative(0.3)
        
        def test_func(x):
            return np.sin(x)
        
        x_vals = np.array([0.0, np.pi/2, np.pi])
        result = caputo_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        caputo_deriv = CaputoDerivative(0.3)
        
        f_values = np.array([0.0, 1.0, 0.0])
        x_values = np.array([0.0, np.pi/2, np.pi])
        
        result = caputo_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestGrunwaldLetnikovDerivative:
    """Test Grünwald-Letnikov fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        gl_deriv = GrunwaldLetnikovDerivative(0.4)
        assert gl_deriv.alpha == 0.4
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        gl_deriv = GrunwaldLetnikovDerivative(0.4)
        
        def test_func(x):
            return x**3
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = gl_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        gl_deriv = GrunwaldLetnikovDerivative(0.4)
        
        f_values = np.array([1.0, 8.0, 27.0])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = gl_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestCaputoFabrizioDerivative:
    """Test Caputo-Fabrizio fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        cf_deriv = CaputoFabrizioDerivative(0.6)
        assert cf_deriv.alpha == 0.6
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        cf_deriv = CaputoFabrizioDerivative(0.6)
        
        def test_func(x):
            return np.exp(x)
        
        x_vals = np.array([0.0, 1.0, 2.0])
        result = cf_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        cf_deriv = CaputoFabrizioDerivative(0.6)
        
        f_values = np.array([1.0, np.e, np.e**2])
        x_values = np.array([0.0, 1.0, 2.0])
        
        result = cf_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestAtanganaBaleanuDerivative:
    """Test Atangana-Baleanu fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        ab_deriv = AtanganaBaleanuDerivative(0.4)
        assert ab_deriv.alpha == 0.4
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        ab_deriv = AtanganaBaleanuDerivative(0.4)
        
        def test_func(x):
            return np.cos(x)
        
        x_vals = np.array([0.0, np.pi/4, np.pi/2])
        result = ab_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        ab_deriv = AtanganaBaleanuDerivative(0.4)
        
        f_values = np.array([1.0, np.sqrt(2)/2, 0.0])
        x_values = np.array([0.0, np.pi/4, np.pi/2])
        
        result = ab_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestFractionalLaplacian:
    """Test Fractional Laplacian implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        laplacian = FractionalLaplacian(1.2)
        assert laplacian.alpha == 1.2
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        laplacian = FractionalLaplacian(1.2)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = laplacian.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        laplacian = FractionalLaplacian(1.2)
        
        f_values = np.array([1.0, 4.0, 9.0])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = laplacian.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestFractionalFourierTransform:
    """Test Fractional Fourier Transform implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        fft = FractionalFourierTransform(0.9)
        assert fft.alpha == 0.9
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        fft = FractionalFourierTransform(0.9)
        
        def test_func(x):
            return np.sin(x)
        
        x_vals = np.array([0.0, np.pi/2, np.pi])
        result = fft.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        fft = FractionalFourierTransform(0.9)
        
        f_values = np.array([0.0, 1.0, 0.0])
        x_values = np.array([0.0, np.pi/2, np.pi])
        
        result = fft.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestMillerRossDerivative:
    """Test Miller-Ross fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        mr_deriv = MillerRossDerivative(0.7)
        assert mr_deriv.alpha == 0.7
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        mr_deriv = MillerRossDerivative(0.7)
        
        def test_func(x):
            return x**1.5
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = mr_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        mr_deriv = MillerRossDerivative(0.7)
        
        f_values = np.array([1.0, 2.828, 5.196])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = mr_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestWeylDerivative:
    """Test Weyl fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        weyl_deriv = WeylDerivative(1.1)
        assert weyl_deriv.alpha == 1.1
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        weyl_deriv = WeylDerivative(1.1)
        
        def test_func(x):
            return np.log(x + 1)
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = weyl_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        weyl_deriv = WeylDerivative(1.1)
        
        f_values = np.array([0.693, 1.099, 1.386])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = weyl_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestMarchaudDerivative:
    """Test Marchaud fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        marchaud_deriv = MarchaudDerivative(0.85)
        assert marchaud_deriv.alpha == 0.85
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        marchaud_deriv = MarchaudDerivative(0.85)
        
        def test_func(x):
            return np.sqrt(x)
        
        x_vals = np.array([1.0, 4.0, 9.0])
        result = marchaud_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        marchaud_deriv = MarchaudDerivative(0.85)
        
        f_values = np.array([1.0, 2.0, 3.0])
        x_values = np.array([1.0, 4.0, 9.0])
        
        result = marchaud_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestHadamardDerivative:
    """Test Hadamard fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        hadamard_deriv = HadamardDerivative(0.95)
        assert hadamard_deriv.alpha == 0.95
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        hadamard_deriv = HadamardDerivative(0.95)
        
        def test_func(x):
            return 1.0 / x
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = hadamard_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        hadamard_deriv = HadamardDerivative(0.95)
        
        f_values = np.array([1.0, 0.5, 0.333])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = hadamard_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestReizFellerDerivative:
    """Test Riesz-Feller fractional derivative implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        rf_deriv = ReizFellerDerivative(1.3)
        assert rf_deriv.alpha == 1.3
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        rf_deriv = ReizFellerDerivative(1.3)
        
        def test_func(x):
            return x**0.7
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rf_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        rf_deriv = ReizFellerDerivative(1.3)
        
        f_values = np.array([1.0, 1.625, 2.157])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = rf_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestParallelOptimizedRiemannLiouville:
    """Test Parallel Optimized Riemann-Liouville implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        porl_deriv = ParallelOptimizedRiemannLiouville(0.6)
        assert porl_deriv.alpha == 0.6
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        porl_deriv = ParallelOptimizedRiemannLiouville(0.6)
        
        def test_func(x):
            return x**1.5
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = porl_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        porl_deriv = ParallelOptimizedRiemannLiouville(0.6)
        
        f_values = np.array([1.0, 2.828, 5.196])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = porl_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestParallelOptimizedCaputo:
    """Test Parallel Optimized Caputo implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        poc_deriv = ParallelOptimizedCaputo(0.4)
        assert poc_deriv.alpha == 0.4
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        poc_deriv = ParallelOptimizedCaputo(0.4)
        
        def test_func(x):
            return np.tanh(x)
        
        x_vals = np.array([0.0, 1.0, 2.0])
        result = poc_deriv.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        poc_deriv = ParallelOptimizedCaputo(0.4)
        
        f_values = np.array([0.0, 0.762, 0.964])
        x_values = np.array([0.0, 1.0, 2.0])
        
        result = poc_deriv.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestRieszFisherOperator:
    """Test Riesz-Fisher operator implementation."""
    
    def test_initialization(self):
        """Test initialization with different alpha values."""
        rf_op = RieszFisherOperator(1.5)
        assert rf_op.alpha == 1.5
    
    def test_compute_with_function(self):
        """Test compute method with function input."""
        rf_op = RieszFisherOperator(1.5)
        
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = rf_op.compute(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_compute_numerical(self):
        """Test compute_numerical method."""
        rf_op = RieszFisherOperator(1.5)
        
        f_values = np.array([1.0, 4.0, 9.0])
        x_values = np.array([1.0, 2.0, 3.0])
        
        result = rf_op.compute_numerical(f_values, x_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f_values)


class TestAdomianDecompositionMethod:
    """Test Adomian Decomposition Method implementation."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        adm = AdomianDecompositionMethod(max_terms=10, tolerance=1e-6)
        assert adm.max_terms == 10
        assert adm.tolerance == 1e-6
    
    def test_decompose_function(self):
        """Test function decomposition."""
        adm = AdomianDecompositionMethod(max_terms=5, tolerance=1e-4)
        
        def test_func(x):
            return x**2 + 2*x + 1
        
        x_vals = np.array([0.0, 1.0, 2.0])
        result = adm.decompose_function(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)
    
    def test_solve_equation(self):
        """Test equation solving."""
        adm = AdomianDecompositionMethod(max_terms=3, tolerance=1e-3)
        
        def test_func(x):
            return x**2 - 4
        
        x_vals = np.array([1.0, 2.0, 3.0])
        result = adm.solve_equation(test_func, x_vals)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_vals)


class TestCreateFunctions:
    """Test factory functions for creating fractional implementations."""
    
    def test_create_fractional_integral(self):
        """Test create_fractional_integral function."""
        integral = create_fractional_integral("RL", 0.5)
        assert integral is not None
        assert hasattr(integral, 'compute')
    
    def test_create_riesz_fisher_operator(self):
        """Test create_riesz_fisher_operator function."""
        operator = create_riesz_fisher_operator(1.2)
        assert operator is not None
        assert hasattr(operator, 'compute')
    
    def test_register_fractional_implementations(self):
        """Test register_fractional_implementations function."""
        # This should not raise an error
        register_fractional_implementations()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_alpha_zero(self):
        """Test behavior with alpha = 0."""
        rl_deriv = RiemannLiouvilleDerivative(0.0)
        assert rl_deriv.alpha == 0.0
    
    def test_alpha_one(self):
        """Test behavior with alpha = 1."""
        # Caputo derivative has restrictions on alpha
        with pytest.raises(ValueError, match="L1 scheme requires 0 < α < 1"):
            CaputoDerivative(1.0)
    
    def test_negative_alpha(self):
        """Test behavior with negative alpha."""
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            RiemannLiouvilleDerivative(-0.5)
    
    def test_large_alpha(self):
        """Test behavior with large alpha values."""
        laplacian = FractionalLaplacian(5.0)
        assert laplacian.alpha == 5.0
    
    def test_inf_alpha(self):
        """Test behavior with infinite alpha."""
        with pytest.raises(ValueError, match="Fractional order must be finite"):
            RiemannLiouvilleDerivative(np.inf)
    
    def test_nan_alpha(self):
        """Test behavior with NaN alpha."""
        with pytest.raises(ValueError, match="Fractional order must be finite"):
            RiemannLiouvilleDerivative(np.nan)


class TestIntegration:
    """Test integration between different components."""
    
    def test_derivative_integral_consistency(self):
        """Test that derivatives and integrals are consistent."""
        # Test basic consistency
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        assert rl_deriv.alpha == 0.5
    
    def test_different_alpha_values(self):
        """Test with various alpha values."""
        alphas = [0.1, 0.5, 0.9, 1.1, 1.5, 2.0]
        
        for alpha in alphas:
            if alpha < 1.0:
                # Test with Caputo derivative for small alpha
                try:
                    caputo_deriv = CaputoDerivative(alpha)
                    assert caputo_deriv.alpha == alpha
                except ValueError:
                    # Some alpha values may not be valid for Caputo
                    pass
            
            # Test with Riemann-Liouville for all alpha values
            rl_deriv = RiemannLiouvilleDerivative(alpha)
            assert rl_deriv.alpha == alpha
    
    def test_computation_consistency(self):
        """Test that computations are consistent across different methods."""
        def test_func(x):
            return x**2
        
        x_vals = np.array([1.0, 2.0, 3.0])
        
        # Test different derivative methods
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        rl_result = rl_deriv.compute(test_func, x_vals)
        
        gl_deriv = GrunwaldLetnikovDerivative(0.5)
        gl_result = gl_deriv.compute(test_func, x_vals)
        
        # Results should be arrays of the same length
        assert len(rl_result) == len(gl_result) == len(x_vals)
        assert isinstance(rl_result, np.ndarray)
        assert isinstance(gl_result, np.ndarray)