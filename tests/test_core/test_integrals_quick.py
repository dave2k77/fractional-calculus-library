#!/usr/bin/env python3
"""Quick tests for core/integrals.py to boost coverage."""

import pytest
import numpy as np
from hpfracc.core.integrals import RiemannLiouvilleIntegral, CaputoIntegral
from hpfracc.core.definitions import FractionalOrder


class TestIntegralsQuick:
    def setup_method(self):
        self.alpha = 0.5
        self.order = FractionalOrder(self.alpha)
        self.t = np.linspace(0, 1, 50)
        self.f = np.sin(2 * np.pi * self.t)
        self.h = self.t[1] - self.t[0]
        
    def test_rl_integral_basic(self):
        integral = RiemannLiouvilleIntegral(self.order)
        result = integral.compute(self.f, self.t, self.h)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.f.shape
        
    def test_caputo_integral_basic(self):
        integral = CaputoIntegral(self.order)
        result = integral.compute(self.f, self.t, self.h)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.f.shape
        
    def test_different_alphas(self):
        alphas = [0.1, 0.5, 0.9, 1.0]
        for alpha in alphas:
            order = FractionalOrder(alpha)
            integral = RiemannLiouvilleIntegral(order)
            result = integral.compute(self.f, self.t, self.h)
            assert isinstance(result, np.ndarray)
            
    def test_different_functions(self):
        integral = RiemannLiouvilleIntegral(self.order)
        functions = [
            np.ones_like(self.t),
            self.t,
            self.t**2,
            np.exp(-self.t)
        ]
        for func in functions:
            result = integral.compute(func, self.t, self.h)
            assert isinstance(result, np.ndarray)
            
    def test_edge_cases(self):
        integral = RiemannLiouvilleIntegral(self.order)
        
        # Zero function
        zero_f = np.zeros_like(self.t)
        result = integral.compute(zero_f, self.t, self.h)
        assert isinstance(result, np.ndarray)
        
        # Single point
        single_t = np.array([0.0])
        single_f = np.array([1.0])
        result = integral.compute(single_f, single_t, 1.0)
        assert isinstance(result, np.ndarray)
        
    def test_linearity(self):
        integral = RiemannLiouvilleIntegral(self.order)
        f1, f2 = np.sin(self.t), np.cos(self.t)
        a, b = 2.0, 3.0
        
        combined = a * f1 + b * f2
        left = integral.compute(combined, self.t, self.h)
        
        right = a * integral.compute(f1, self.t, self.h) + b * integral.compute(f2, self.t, self.h)
        
        assert np.allclose(left, right, atol=1e-10)

















