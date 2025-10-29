"""
Unit tests for noise models in hpfracc.solvers.noise_models

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
from hpfracc.solvers import (
    BrownianMotion, FractionalBrownianMotion, LevyNoise, ColouredNoise,
    NoiseConfig, create_noise_model, generate_noise_trajectory
)


class TestBrownianMotion:
    """Test BrownianMotion noise model"""
    
    def test_initialization(self):
        """Test BrownianMotion initialization"""
        bm = BrownianMotion(scale=1.0)
        assert bm.scale == 1.0
        
        bm = BrownianMotion(scale=2.5)
        assert bm.scale == 2.5
    
    def test_generate_increment(self):
        """Test noise increment generation"""
        bm = BrownianMotion(scale=1.0)
        
        # Generate increment
        dW = bm.generate_increment(t=0.0, dt=0.01, size=100, seed=42)
        
        assert dW.shape == (100,)
        assert not np.any(np.isnan(dW))
        assert not np.any(np.isinf(dW))
    
    def test_statistics(self):
        """Test statistical properties"""
        bm = BrownianMotion(scale=1.0)
        dt = 0.01
        
        # Generate many samples
        samples = [bm.generate_increment(t=0.0, dt=dt, size=100, seed=i) 
                   for i in range(1000)]
        samples = np.array(samples)
        
        # Check mean ≈ 0
        mean = np.mean(samples)
        assert abs(mean) < 0.05  # Should be very close to 0
        
        # Check variance ≈ dt
        variance = np.var(samples)
        assert abs(variance - dt) < 0.01
    
    def test_variance_method(self):
        """Test variance calculation method"""
        bm = BrownianMotion(scale=2.0)
        dt = 0.05
        expected_variance = 4.0 * dt  # scale^2 * dt
        assert abs(bm.variance(dt) - expected_variance) < 1e-10
    
    def test_seed_reproducibility(self):
        """Test that seeds produce reproducible results"""
        bm = BrownianMotion(scale=1.0)
        
        # Generate two batches with same seed
        dw1 = bm.generate_increment(t=0.0, dt=0.01, size=100, seed=42)
        dw2 = bm.generate_increment(t=0.0, dt=0.01, size=100, seed=42)
        
        np.testing.assert_array_equal(dw1, dw2)


class TestFractionalBrownianMotion:
    """Test FractionalBrownianMotion noise model"""
    
    def test_initialization(self):
        """Test FractionalBrownianMotion initialization"""
        fbm = FractionalBrownianMotion(hurst=0.7, scale=1.0)
        assert fbm.hurst == 0.7
        assert fbm.scale == 1.0
    
    def test_invalid_hurst(self):
        """Test that invalid Hurst values raise errors"""
        with pytest.raises(ValueError):
            FractionalBrownianMotion(hurst=0.0, scale=1.0)
        
        with pytest.raises(ValueError):
            FractionalBrownianMotion(hurst=1.0, scale=1.0)
        
        with pytest.raises(ValueError):
            FractionalBrownianMotion(hurst=-0.5, scale=1.0)
    
    def test_generate_increment(self):
        """Test noise increment generation"""
        fbm = FractionalBrownianMotion(hurst=0.7, scale=1.0)
        
        dW = fbm.generate_increment(t=0.0, dt=0.01, size=100, seed=42)
        
        assert dW.shape == (100,)
        assert not np.any(np.isnan(dW))
    
    def test_hurst_effect(self):
        """Test that different Hurst values produce different behavior"""
        fbm_lo = FractionalBrownianMotion(hurst=0.3, scale=1.0)
        fbm_hi = FractionalBrownianMotion(hurst=0.7, scale=1.0)
        
        dw_lo = fbm_lo.generate_increment(t=0.0, dt=0.01, size=1000, seed=42)
        dw_hi = fbm_hi.generate_increment(t=0.0, dt=0.01, size=1000, seed=42)
        
        # They should be different
        assert not np.allclose(dw_lo, dw_hi)
    
    def test_standard_bm(self):
        """Test that H=0.5 gives standard Brownian motion"""
        fbm = FractionalBrownianMotion(hurst=0.5, scale=1.0)
        assert fbm.is_standard_bm


class TestLevyNoise:
    """Test LevyNoise noise model"""
    
    def test_initialization(self):
        """Test LevyNoise initialization"""
        levy = LevyNoise(alpha=1.5, beta=0.0, scale=1.0, location=0.0)
        assert levy.alpha == 1.5
        assert levy.beta == 0.0
        assert levy.scale == 1.0
        assert levy.location == 0.0
    
    def test_invalid_alpha(self):
        """Test that invalid alpha values raise errors"""
        with pytest.raises(ValueError):
            LevyNoise(alpha=0.0, beta=0.0)
        
        with pytest.raises(ValueError):
            LevyNoise(alpha=2.5, beta=0.0)
    
    def test_generate_increment(self):
        """Test noise increment generation"""
        levy = LevyNoise(alpha=1.5, beta=0.0, scale=1.0)
        
        dW = levy.generate_increment(t=0.0, dt=0.01, size=100, seed=42)
        
        assert dW.shape == (100,)
    
    def test_gaussian_case(self):
        """Test that alpha=2 gives Gaussian noise"""
        levy = LevyNoise(alpha=2.0, beta=0.0, scale=1.0)
        dW = levy.generate_increment(t=0.0, dt=0.01, size=1000, seed=42)
        
        # Should have approximately Gaussian distribution
        mean = np.mean(dW)
        assert abs(mean) < 0.05


class TestColouredNoise:
    """Test ColouredNoise (Ornstein-Uhlenbeck process)"""
    
    def test_initialization(self):
        """Test ColouredNoise initialization"""
        cn = ColouredNoise(correlation_time=1.0, amplitude=1.0, seed=42)
        assert cn.correlation_time == 1.0
        assert cn.amplitude == 1.0
    
    def test_generate_increment(self):
        """Test noise increment generation"""
        cn = ColouredNoise(correlation_time=1.0, amplitude=1.0, seed=42)
        
        dW = cn.generate_increment(t=0.0, dt=0.01, size=100, seed=42)
        
        assert dW.shape == (100,)
        assert not np.any(np.isnan(dW))
    
    def test_state_persistence(self):
        """Test that state persists across calls"""
        cn = ColouredNoise(correlation_time=1.0, amplitude=1.0, seed=42)
        
        # First call
        dW1 = cn.generate_increment(t=0.0, dt=0.01, size=10)
        
        # Second call - should use persistent state
        dW2 = cn.generate_increment(t=0.01, dt=0.01, size=10)
        
        # Should have correlation
        assert cn._state is not None
    
    def test_reset(self):
        """Test state reset functionality"""
        cn = ColouredNoise(correlation_time=1.0, amplitude=1.0, seed=42)
        
        # Generate some noise
        cn.generate_increment(t=0.0, dt=0.01, size=10)
        
        # Reset
        cn.reset()
        assert cn._state is None


class TestNoiseConfig:
    """Test NoiseConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = NoiseConfig()
        
        assert config.noise_type == "brownian"
        assert config.hurst == 0.5
        assert config.scale == 1.0
        assert config.alpha == 1.5
        assert config.beta == 0.0
        assert config.correlation_time == 1.0
        assert config.amplitude == 1.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = NoiseConfig(
            noise_type="fractional_brownian",
            hurst=0.7,
            scale=2.0
        )
        
        assert config.noise_type == "fractional_brownian"
        assert config.hurst == 0.7
        assert config.scale == 2.0


class TestCreateNoiseModel:
    """Test create_noise_model factory function"""
    
    def test_create_brownian(self):
        """Test creating Brownian motion"""
        config = NoiseConfig(noise_type="brownian")
        noise = create_noise_model(config)
        
        assert isinstance(noise, BrownianMotion)
    
    def test_create_fractional_brownian(self):
        """Test creating fractional Brownian motion"""
        config = NoiseConfig(noise_type="fractional_brownian", hurst=0.7)
        noise = create_noise_model(config)
        
        assert isinstance(noise, FractionalBrownianMotion)
        assert noise.hurst == 0.7
    
    def test_create_levy(self):
        """Test creating Lévy noise"""
        config = NoiseConfig(noise_type="levy", alpha=1.5, beta=0.5)
        noise = create_noise_model(config)
        
        assert isinstance(noise, LevyNoise)
        assert noise.alpha == 1.5
        assert noise.beta == 0.5
    
    def test_create_coloured(self):
        """Test creating coloured noise"""
        config = NoiseConfig(noise_type="coloured", correlation_time=0.5)
        noise = create_noise_model(config)
        
        assert isinstance(noise, ColouredNoise)
        assert noise.correlation_time == 0.5
    
    def test_invalid_type(self):
        """Test that invalid noise types raise errors"""
        config = NoiseConfig(noise_type="invalid_type")
        
        with pytest.raises(ValueError):
            create_noise_model(config)


class TestGenerateNoiseTrajectory:
    """Test generate_noise_trajectory function"""
    
    def test_basic_generation(self):
        """Test basic trajectory generation"""
        bm = BrownianMotion(scale=1.0)
        
        t, dW = generate_noise_trajectory(
            bm,
            t_span=(0, 1),
            num_steps=100,
            size=(1,),
            seed=42
        )
        
        assert t.shape == (101,)  # num_steps + 1
        assert dW.shape == (100, 1)  # num_steps, size
        assert t[0] == 0
        assert t[-1] == 1
    
    def test_multidimensional(self):
        """Test multidimensional noise generation"""
        bm = BrownianMotion(scale=1.0)
        
        t, dW = generate_noise_trajectory(
            bm,
            t_span=(0, 1),
            num_steps=50,
            size=(3, 2),  # 3x2 noise increments
            seed=42
        )
        
        assert dW.shape == (50, 3, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
