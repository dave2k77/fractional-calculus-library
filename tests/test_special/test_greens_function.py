"""
Tests for fractional Green's functions module.

This module contains comprehensive tests for all fractional Green's function
implementations including diffusion, wave, and advection equations.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.special.greens_function import (
    FractionalGreensFunction,
    FractionalDiffusionGreensFunction,
    FractionalWaveGreensFunction,
    FractionalAdvectionGreensFunction,
    create_fractional_greens_function,
    get_greens_function_properties,
    validate_greens_function_parameters
)


class TestFractionalGreensFunction:
    """Test base FractionalGreensFunction class."""
    
    def test_greens_function_creation(self):
        """Test creating FractionalGreensFunction instances."""
        greens = FractionalGreensFunction(order=0.5, equation_type="diffusion")
        assert greens.order == 0.5
        assert greens.equation_type == "diffusion"
        assert greens.dimension == 1
    
    def test_greens_function_validation(self):
        """Test FractionalGreensFunction validation."""
        # Test valid parameters
        greens = FractionalGreensFunction(order=0.5, equation_type="diffusion")
        assert greens.order == 0.5
        
        # Test invalid order
        with pytest.raises(ValueError):
            FractionalGreensFunction(order=-1.0, equation_type="diffusion")
        
        # Test invalid equation type
        with pytest.raises(ValueError):
            FractionalGreensFunction(order=0.5, equation_type="invalid")
        
        # Test invalid dimension
        with pytest.raises(ValueError):
            FractionalGreensFunction(order=0.5, equation_type="diffusion", dimension=0)
    
    def test_greens_function_repr(self):
        """Test FractionalGreensFunction string representation."""
        greens = FractionalGreensFunction(order=0.5, equation_type="diffusion", dimension=2)
        repr_str = repr(greens)
        assert "FractionalGreensFunction" in repr_str
        assert "0.5" in repr_str
        assert "diffusion" in repr_str
        assert "2" in repr_str


class TestFractionalDiffusionGreensFunction:
    """Test fractional diffusion Green's function."""
    
    def test_diffusion_greens_creation(self):
        """Test creating FractionalDiffusionGreensFunction instances."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        assert isinstance(greens, FractionalGreensFunction)
        assert greens.order == 0.5
        assert greens.equation_type == "diffusion"
        assert greens.diffusion_coefficient == 1.0
    
    def test_diffusion_greens_1d(self):
        """Test 1D diffusion Green's function."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        
        # Test spatial and temporal coordinates
        x = np.linspace(-5, 5, 100)
        t = np.linspace(0.1, 2.0, 50)
        
        # Test single point evaluation
        result = greens.evaluate(x[50], t[25])
        assert np.isfinite(result)
        assert result >= 0  # Green's function should be non-negative
        
        # Test array evaluation
        X, T = np.meshgrid(x, t)
        result_array = greens.evaluate(X, T)
        assert result_array.shape == X.shape
        assert np.all(np.isfinite(result_array))
        assert np.all(result_array >= 0)
    
    def test_diffusion_greens_2d(self):
        """Test 2D diffusion Green's function."""
        greens = FractionalDiffusionGreensFunction(
            order=0.5, diffusion_coefficient=1.0, dimension=2
        )
        
        # Test 2D spatial coordinates
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        t = np.linspace(0.1, 1.0, 10)
        
        X, Y, T = np.meshgrid(x, y, t)
        result = greens.evaluate((X, Y), T)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_diffusion_greens_3d(self):
        """Test 3D diffusion Green's function."""
        greens = FractionalDiffusionGreensFunction(
            order=0.5, diffusion_coefficient=1.0, dimension=3
        )
        
        # Test 3D spatial coordinates
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        z = np.linspace(-1, 1, 10)
        t = np.linspace(0.1, 0.5, 5)
        
        X, Y, Z, T = np.meshgrid(x, y, z, t)
        result = greens.evaluate((X, Y, Z), T)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_diffusion_greens_properties(self):
        """Test diffusion Green's function properties."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        
        # Test causality (G(x,t) = 0 for t < 0)
        x = np.array([0.0, 1.0, -1.0])
        t_negative = np.array([-0.1, -0.5, -1.0])
        
        for x_val, t_val in zip(x, t_negative):
            result = greens.evaluate(x_val, t_val)
            assert result == 0.0
        
        # Test symmetry in space (G(x,t) = G(-x,t))
        x = np.array([1.0, 2.0, 3.0])
        t = np.array([0.5, 1.0, 1.5])
        
        for x_val, t_val in zip(x, t):
            result_pos = greens.evaluate(x_val, t_val)
            result_neg = greens.evaluate(-x_val, t_val)
            np.testing.assert_allclose(result_pos, result_neg, rtol=1e-10)
    
    def test_diffusion_greens_normalization(self):
        """Test diffusion Green's function normalization."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        
        # Test that integral over space equals 1 for all t > 0
        t_values = np.array([0.1, 0.5, 1.0, 2.0])
        x = np.linspace(-20, 20, 1000)
        dx = x[1] - x[0]
        
        for t_val in t_values:
            G_values = greens.evaluate(x, t_val)
            integral = np.trapz(G_values, x)
            np.testing.assert_allclose(integral, 1.0, rtol=1e-2)


class TestFractionalWaveGreensFunction:
    """Test fractional wave Green's function."""
    
    def test_wave_greens_creation(self):
        """Test creating FractionalWaveGreensFunction instances."""
        greens = FractionalWaveGreensFunction(order=0.5, wave_speed=1.0)
        assert isinstance(greens, FractionalGreensFunction)
        assert greens.order == 0.5
        assert greens.equation_type == "wave"
        assert greens.wave_speed == 1.0
    
    def test_wave_greens_1d(self):
        """Test 1D wave Green's function."""
        greens = FractionalWaveGreensFunction(order=0.5, wave_speed=1.0)
        
        # Test spatial and temporal coordinates
        x = np.linspace(-5, 5, 100)
        t = np.linspace(0.1, 2.0, 50)
        
        # Test single point evaluation
        result = greens.evaluate(x[50], t[25])
        assert np.isfinite(result)
        
        # Test array evaluation
        X, T = np.meshgrid(x, t)
        result_array = greens.evaluate(X, T)
        assert result_array.shape == X.shape
        assert np.all(np.isfinite(result_array))
    
    def test_wave_greens_2d(self):
        """Test 2D wave Green's function."""
        greens = FractionalWaveGreensFunction(
            order=0.5, wave_speed=1.0, dimension=2
        )
        
        # Test 2D spatial coordinates
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        t = np.linspace(0.1, 1.0, 10)
        
        X, Y, T = np.meshgrid(x, y, t)
        result = greens.evaluate((X, Y), T)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))
    
    def test_wave_greens_3d(self):
        """Test 3D wave Green's function."""
        greens = FractionalWaveGreensFunction(
            order=0.5, wave_speed=1.0, dimension=3
        )
        
        # Test 3D spatial coordinates
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        z = np.linspace(-1, 1, 10)
        t = np.linspace(0.1, 0.5, 5)
        
        X, Y, Z, T = np.meshgrid(x, y, z, t)
        result = greens.evaluate((X, Y, Z), T)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))
    
    def test_wave_greens_properties(self):
        """Test wave Green's function properties."""
        greens = FractionalWaveGreensFunction(order=0.5, wave_speed=1.0)
        
        # Test causality (G(x,t) = 0 for t < 0)
        x = np.array([0.0, 1.0, -1.0])
        t_negative = np.array([-0.1, -0.5, -1.0])
        
        for x_val, t_val in zip(x, t_negative):
            result = greens.evaluate(x_val, t_val)
            assert result == 0.0
        
        # Test wave front propagation
        # For wave equation, Green's function should be zero outside the light cone
        x = np.array([2.0, 3.0, 4.0])
        t = np.array([1.0, 1.0, 1.0])  # t = 1
        
        for x_val, t_val in zip(x, t):
            if abs(x_val) > t_val:  # Outside light cone
                result = greens.evaluate(x_val, t_val)
                assert result == 0.0


class TestFractionalAdvectionGreensFunction:
    """Test fractional advection Green's function."""
    
    def test_advection_greens_creation(self):
        """Test creating FractionalAdvectionGreensFunction instances."""
        greens = FractionalAdvectionGreensFunction(order=0.5, advection_velocity=1.0)
        assert isinstance(greens, FractionalGreensFunction)
        assert greens.order == 0.5
        assert greens.equation_type == "advection"
        assert greens.advection_velocity == 1.0
    
    def test_advection_greens_1d(self):
        """Test 1D advection Green's function."""
        greens = FractionalAdvectionGreensFunction(order=0.5, advection_velocity=1.0)
        
        # Test spatial and temporal coordinates
        x = np.linspace(-5, 5, 100)
        t = np.linspace(0.1, 2.0, 50)
        
        # Test single point evaluation
        result = greens.evaluate(x[50], t[25])
        assert np.isfinite(result)
        
        # Test array evaluation
        X, T = np.meshgrid(x, t)
        result_array = greens.evaluate(X, T)
        assert result_array.shape == X.shape
        assert np.all(np.isfinite(result_array))
    
    def test_advection_greens_2d(self):
        """Test 2D advection Green's function."""
        greens = FractionalAdvectionGreensFunction(
            order=0.5, advection_velocity=1.0, dimension=2
        )
        
        # Test 2D spatial coordinates
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        t = np.linspace(0.1, 1.0, 10)
        
        X, Y, T = np.meshgrid(x, y, t)
        result = greens.evaluate((X, Y), T)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))
    
    def test_advection_greens_3d(self):
        """Test 3D advection Green's function."""
        greens = FractionalAdvectionGreensFunction(
            order=0.5, advection_velocity=1.0, dimension=3
        )
        
        # Test 3D spatial coordinates
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        z = np.linspace(-1, 1, 10)
        t = np.linspace(0.1, 0.5, 5)
        
        X, Y, Z, T = np.meshgrid(x, y, z, t)
        result = greens.evaluate((X, Y, Z), T)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))
    
    def test_advection_greens_properties(self):
        """Test advection Green's function properties."""
        greens = FractionalAdvectionGreensFunction(order=0.5, advection_velocity=1.0)
        
        # Test causality (G(x,t) = 0 for t < 0)
        x = np.array([0.0, 1.0, -1.0])
        t_negative = np.array([-0.1, -0.5, -1.0])
        
        for x_val, t_val in zip(x, t_negative):
            result = greens.evaluate(x_val, t_val)
            assert result == 0.0
        
        # Test advection behavior
        # For advection equation, Green's function should peak at x = vt
        x = np.linspace(-2, 2, 100)
        t = 1.0
        v = 1.0
        
        G_values = greens.evaluate(x, t)
        peak_index = np.argmax(G_values)
        peak_x = x[peak_index]
        
        # Peak should be close to x = vt
        np.testing.assert_allclose(peak_x, v * t, rtol=0.1)


class TestCreateFractionalGreensFunction:
    """Test factory function for creating Green's functions."""
    
    def test_create_diffusion_greens(self):
        """Test creating diffusion Green's function via factory."""
        greens = create_fractional_greens_function(
            "diffusion", order=0.5, diffusion_coefficient=1.0
        )
        assert isinstance(greens, FractionalDiffusionGreensFunction)
        assert greens.order == 0.5
        assert greens.diffusion_coefficient == 1.0
    
    def test_create_wave_greens(self):
        """Test creating wave Green's function via factory."""
        greens = create_fractional_greens_function(
            "wave", order=0.5, wave_speed=1.0
        )
        assert isinstance(greens, FractionalWaveGreensFunction)
        assert greens.order == 0.5
        assert greens.wave_speed == 1.0
    
    def test_create_advection_greens(self):
        """Test creating advection Green's function via factory."""
        greens = create_fractional_greens_function(
            "advection", order=0.5, advection_velocity=1.0
        )
        assert isinstance(greens, FractionalAdvectionGreensFunction)
        assert greens.order == 0.5
        assert greens.advection_velocity == 1.0
    
    def test_create_invalid_type(self):
        """Test creating Green's function with invalid type."""
        with pytest.raises(ValueError):
            create_fractional_greens_function("invalid", order=0.5)


class TestGreensFunctionUtilities:
    """Test utility functions for Green's functions."""
    
    def test_get_greens_function_properties(self):
        """Test getting Green's function properties."""
        # Test diffusion properties
        properties = get_greens_function_properties("diffusion")
        assert isinstance(properties, dict)
        assert "parameters" in properties
        assert "dimensions" in properties
        
        # Test wave properties
        properties = get_greens_function_properties("wave")
        assert isinstance(properties, dict)
        
        # Test advection properties
        properties = get_greens_function_properties("advection")
        assert isinstance(properties, dict)
        
        # Test invalid type
        properties = get_greens_function_properties("invalid")
        assert properties is None
    
    def test_validate_greens_function_parameters(self):
        """Test validation of Green's function parameters."""
        # Test valid diffusion parameters
        params = {"order": 0.5, "diffusion_coefficient": 1.0}
        assert validate_greens_function_parameters("diffusion", params) is None
        
        # Test valid wave parameters
        params = {"order": 0.5, "wave_speed": 1.0}
        assert validate_greens_function_parameters("wave", params) is None
        
        # Test valid advection parameters
        params = {"order": 0.5, "advection_velocity": 1.0}
        assert validate_greens_function_parameters("advection", params) is None
        
        # Test invalid parameters
        params = {"order": -0.5, "diffusion_coefficient": 1.0}
        with pytest.raises(ValueError):
            validate_greens_function_parameters("diffusion", params)
        
        # Test missing parameters
        params = {"order": 0.5}
        with pytest.raises(ValueError):
            validate_greens_function_parameters("diffusion", params)


class TestGreensFunctionPerformance:
    """Test performance characteristics of Green's functions."""
    
    def test_large_array_performance(self):
        """Test performance with large arrays."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        
        # Large arrays
        x = np.linspace(-10, 10, 1000)
        t = np.linspace(0.1, 5.0, 500)
        X, T = np.meshgrid(x, t)
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        result = greens.evaluate(X, T)
        end_time = time.time()
        
        assert end_time - start_time < 10.0  # Should complete within 10 seconds
        assert result.shape == X.shape
    
    def test_memory_usage(self):
        """Test memory usage with large arrays."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        
        # Large arrays
        x = np.linspace(-10, 10, 1000)
        t = np.linspace(0.1, 5.0, 500)
        X, T = np.meshgrid(x, t)
        
        # Should not cause memory issues
        result = greens.evaluate(X, T)
        assert result.nbytes < 1e7  # Should use less than 10MB


class TestGreensFunctionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_order(self):
        """Test behavior with zero order."""
        greens = FractionalDiffusionGreensFunction(order=0.0, diffusion_coefficient=1.0)
        x = np.linspace(-5, 5, 100)
        t = np.array([0.5])
        
        result = greens.evaluate(x, t)
        # Zero order should behave like standard diffusion
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_zero_time(self):
        """Test behavior with zero time."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        x = np.linspace(-5, 5, 100)
        t = np.array([0.0])
        
        result = greens.evaluate(x, t)
        # At t=0, Green's function should be a delta function
        # For numerical evaluation, this should be handled appropriately
        assert len(result) == len(x)
    
    def test_negative_time(self):
        """Test behavior with negative time."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        x = np.linspace(-5, 5, 100)
        t = np.array([-0.1])
        
        result = greens.evaluate(x, t)
        # Negative time should return zero (causality)
        assert np.all(result == 0.0)
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        x = np.array([])
        t = np.array([])
        
        result = greens.evaluate(x, t)
        assert len(result) == 0
    
    def test_single_points(self):
        """Test behavior with single points."""
        greens = FractionalDiffusionGreensFunction(order=0.5, diffusion_coefficient=1.0)
        x = np.array([1.0])
        t = np.array([0.5])
        
        result = greens.evaluate(x, t)
        assert len(result) == 1
        assert np.isfinite(result[0])


if __name__ == "__main__":
    pytest.main([__file__])
