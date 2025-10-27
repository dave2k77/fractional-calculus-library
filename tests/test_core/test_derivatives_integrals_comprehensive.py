"""
Comprehensive tests for core derivatives and integrals

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
from hpfracc.core.derivatives import (
    FractionalDerivative, RiemannLiouvilleDerivative, CaputoDerivative,
    GrunwaldLetnikovDerivative, MillerRossDerivative
)
from hpfracc.core.integrals import (
    FractionalIntegral, RiemannLiouvilleIntegral, CaputoIntegral,
    GrunwaldLetnikovIntegral, MillerRossIntegral
)
from hpfracc.core.definitions import FractionalOrder


class TestFractionalDerivative:
    """Test base fractional derivative class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.derivative = FractionalDerivative(self.alpha)
    
    def test_initialization(self):
        """Test fractional derivative initialization"""
        assert self.derivative.alpha.alpha == self.alpha
        assert isinstance(self.derivative.alpha, FractionalOrder)
    
    def test_compute_method(self):
        """Test compute method"""
        def f(t):
            return t**2
        
        result = self.derivative.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_different_alpha_values(self):
        """Test different fractional order values"""
        alphas = [0.3, 0.5, 0.7, 1.0, 1.5]
        f = self.t**2
        
        for alpha in alphas:
            derivative = FractionalDerivative(alpha)
            result = derivative.compute(f, self.t, self.h)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(self.t)
            assert not np.any(np.isnan(result))


class TestRiemannLiouvilleDerivative:
    """Test Riemann-Liouville derivative implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.rl = RiemannLiouvilleDerivative(self.alpha)
    
    def test_initialization(self):
        """Test Riemann-Liouville derivative initialization"""
        assert self.rl.alpha.alpha == self.alpha
        assert isinstance(self.rl.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing derivative with function input"""
        def f(t):
            return t**2
        
        result = self.rl.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing derivative with numerical input"""
        f = self.t**2
        
        result = self.rl.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
    
    def test_mathematical_properties(self):
        """Test mathematical properties"""
        f1 = self.t**2
        f2 = np.sin(self.t)
        a, b = 2.0, 3.0
        
        # Test linearity
        combined = a * f1 + b * f2
        result_combined = self.rl.compute(combined, self.t, self.h)
        
        result_f1 = self.rl.compute(f1, self.t, self.h)
        result_f2 = self.rl.compute(f2, self.t, self.h)
        result_linear = a * result_f1 + b * result_f2
        
        np.testing.assert_allclose(result_combined, result_linear, rtol=1e-2, atol=1e-2)


class TestCaputoDerivative:
    """Test Caputo derivative implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.caputo = CaputoDerivative(self.alpha)
    
    def test_initialization(self):
        """Test Caputo derivative initialization"""
        assert self.caputo.alpha.alpha == self.alpha
        assert isinstance(self.caputo.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing derivative with function input"""
        def f(t):
            return t**2
        
        result = self.caputo.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing derivative with numerical input"""
        f = self.t**2
        
        result = self.caputo.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))


class TestGrunwaldLetnikovDerivative:
    """Test Grünwald-Letnikov derivative implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.gl = GrunwaldLetnikovDerivative(self.alpha)
    
    def test_initialization(self):
        """Test Grünwald-Letnikov derivative initialization"""
        assert self.gl.alpha.alpha == self.alpha
        assert isinstance(self.gl.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing derivative with function input"""
        def f(t):
            return t**2
        
        result = self.gl.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing derivative with numerical input"""
        f = self.t**2
        
        result = self.gl.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))


class TestMillerRossDerivative:
    """Test Miller-Ross derivative implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.mr = MillerRossDerivative(self.alpha)
    
    def test_initialization(self):
        """Test Miller-Ross derivative initialization"""
        assert self.mr.alpha.alpha == self.alpha
        assert isinstance(self.mr.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing derivative with function input"""
        def f(t):
            return t**2
        
        result = self.mr.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing derivative with numerical input"""
        f = self.t**2
        
        result = self.mr.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))


class TestFractionalIntegral:
    """Test base fractional integral class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.integral = FractionalIntegral(self.alpha)
    
    def test_initialization(self):
        """Test fractional integral initialization"""
        assert self.integral.alpha.alpha == self.alpha
        assert isinstance(self.integral.alpha, FractionalOrder)
    
    def test_compute_method(self):
        """Test compute method"""
        def f(t):
            return t**2
        
        result = self.integral.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_different_alpha_values(self):
        """Test different fractional order values"""
        alphas = [0.3, 0.5, 0.7, 1.0, 1.5]
        f = self.t**2
        
        for alpha in alphas:
            integral = FractionalIntegral(alpha)
            result = integral.compute(f, self.t, self.h)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(self.t)
            assert not np.any(np.isnan(result))


class TestRiemannLiouvilleIntegral:
    """Test Riemann-Liouville integral implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.rl_int = RiemannLiouvilleIntegral(self.alpha)
    
    def test_initialization(self):
        """Test Riemann-Liouville integral initialization"""
        assert self.rl_int.alpha.alpha == self.alpha
        assert isinstance(self.rl_int.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing integral with function input"""
        def f(t):
            return t**2
        
        result = self.rl_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing integral with numerical input"""
        f = self.t**2
        
        result = self.rl_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))


class TestCaputoIntegral:
    """Test Caputo integral implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.caputo_int = CaputoIntegral(self.alpha)
    
    def test_initialization(self):
        """Test Caputo integral initialization"""
        assert self.caputo_int.alpha.alpha == self.alpha
        assert isinstance(self.caputo_int.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing integral with function input"""
        def f(t):
            return t**2
        
        result = self.caputo_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing integral with numerical input"""
        f = self.t**2
        
        result = self.caputo_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))


class TestGrunwaldLetnikovIntegral:
    """Test Grünwald-Letnikov integral implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.gl_int = GrunwaldLetnikovIntegral(self.alpha)
    
    def test_initialization(self):
        """Test Grünwald-Letnikov integral initialization"""
        assert self.gl_int.alpha.alpha == self.alpha
        assert isinstance(self.gl_int.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing integral with function input"""
        def f(t):
            return t**2
        
        result = self.gl_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing integral with numerical input"""
        f = self.t**2
        
        result = self.gl_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))


class TestMillerRossIntegral:
    """Test Miller-Ross integral implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.mr_int = MillerRossIntegral(self.alpha)
    
    def test_initialization(self):
        """Test Miller-Ross integral initialization"""
        assert self.mr_int.alpha.alpha == self.alpha
        assert isinstance(self.mr_int.alpha, FractionalOrder)
    
    def test_compute_with_function(self):
        """Test computing integral with function input"""
        def f(t):
            return t**2
        
        result = self.mr_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_numerical(self):
        """Test computing integral with numerical input"""
        f = self.t**2
        
        result = self.mr_int.compute(f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.t)
        assert not np.any(np.isnan(result))


class TestDerivativeIntegralRelations:
    """Test relationships between derivatives and integrals"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.derivative = RiemannLiouvilleDerivative(self.alpha)
        self.integral = RiemannLiouvilleIntegral(self.alpha)
    
    def test_derivative_integral_inverse(self):
        """Test that derivative and integral are inverse operations"""
        f = self.t**2
        
        # Compute integral then derivative
        integral_result = self.integral.compute(f, self.t, self.h)
        derivative_result = self.derivative.compute(integral_result, self.t, self.h)
        
        # Should recover original function (approximately)
        np.testing.assert_allclose(derivative_result, f, rtol=1e-1, atol=1e-1)
    
    def test_integral_derivative_inverse(self):
        """Test that integral and derivative are inverse operations"""
        f = self.t**2
        
        # Compute derivative then integral
        derivative_result = self.derivative.compute(f, self.t, self.h)
        integral_result = self.integral.compute(derivative_result, self.t, self.h)
        
        # Should recover original function (approximately)
        np.testing.assert_allclose(integral_result, f, rtol=1e-1, atol=1e-1)


class TestMathematicalConsistency:
    """Test mathematical consistency across different methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.t = np.linspace(0.1, 2.0, 50)
        self.h = self.t[1] - self.t[0]
        self.f = self.t**2
    
    def test_consistency_between_derivative_methods(self):
        """Test consistency between different derivative methods"""
        rl = RiemannLiouvilleDerivative(self.alpha)
        caputo = CaputoDerivative(self.alpha)
        gl = GrunwaldLetnikovDerivative(self.alpha)
        mr = MillerRossDerivative(self.alpha)
        
        result_rl = rl.compute(self.f, self.t, self.h)
        result_caputo = caputo.compute(self.f, self.t, self.h)
        result_gl = gl.compute(self.f, self.t, self.h)
        result_mr = mr.compute(self.f, self.t, self.h)
        
        # Results should be similar (not identical due to different definitions)
        # but should have same order of magnitude
        assert np.allclose(np.abs(result_rl), np.abs(result_caputo), rtol=1.0)
        assert np.allclose(np.abs(result_rl), np.abs(result_gl), rtol=1.0)
        assert np.allclose(np.abs(result_rl), np.abs(result_mr), rtol=1.0)
    
    def test_consistency_between_integral_methods(self):
        """Test consistency between different integral methods"""
        rl_int = RiemannLiouvilleIntegral(self.alpha)
        caputo_int = CaputoIntegral(self.alpha)
        gl_int = GrunwaldLetnikovIntegral(self.alpha)
        mr_int = MillerRossIntegral(self.alpha)
        
        result_rl = rl_int.compute(self.f, self.t, self.h)
        result_caputo = caputo_int.compute(self.f, self.t, self.h)
        result_gl = gl_int.compute(self.f, self.t, self.h)
        result_mr = mr_int.compute(self.f, self.t, self.h)
        
        # Results should be similar (not identical due to different definitions)
        # but should have same order of magnitude
        assert np.allclose(np.abs(result_rl), np.abs(result_caputo), rtol=1.0)
        assert np.allclose(np.abs(result_rl), np.abs(result_gl), rtol=1.0)
        assert np.allclose(np.abs(result_rl), np.abs(result_mr), rtol=1.0)


class TestErrorHandling:
    """Test error handling in derivatives and integrals"""
    
    def test_invalid_alpha_values(self):
        """Test handling of invalid alpha values"""
        # Test negative alpha
        with pytest.raises(ValueError):
            RiemannLiouvilleDerivative(-0.5)
        
        # Test alpha >= 2
        with pytest.raises(ValueError):
            RiemannLiouvilleDerivative(2.5)
    
    def test_invalid_time_arrays(self):
        """Test handling of invalid time arrays"""
        alpha = 0.5
        derivative = RiemannLiouvilleDerivative(alpha)
        
        # Test empty time array
        with pytest.raises(ValueError):
            derivative.compute(lambda t: t, np.array([]), 0.01)
        
        # Test single point
        with pytest.raises(ValueError):
            derivative.compute(lambda t: t, np.array([1.0]), 0.01)
    
    def test_invalid_step_size(self):
        """Test handling of invalid step sizes"""
        alpha = 0.5
        derivative = RiemannLiouvilleDerivative(alpha)
        t = np.linspace(0.1, 2.0, 50)
        
        # Test negative step size
        with pytest.raises(ValueError):
            derivative.compute(lambda t: t, t, -0.01)
        
        # Test zero step size
        with pytest.raises(ValueError):
            derivative.compute(lambda t: t, t, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
