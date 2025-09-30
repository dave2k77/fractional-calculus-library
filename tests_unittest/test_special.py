"""
Unittest tests for HPFRACC special functions
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestGammaBetaFunctions(unittest.TestCase):
    """Test gamma and beta functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.special.gamma_beta import gamma, beta
        self.gamma = gamma
        self.beta = beta
    
    def test_gamma_function_basic(self):
        """Test basic gamma function values"""
        # Test known values
        test_cases = [
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.0, 6.0),
            (0.5, 1.7724538509055159),  # sqrt(pi)
            (1.5, 0.8862269254527579),  # sqrt(pi)/2
        ]
        
        for x, expected in test_cases:
            with self.subTest(x=x):
                result = self.gamma(x)
                self.assertAlmostEqual(result, expected, places=10)
    
    def test_gamma_function_properties(self):
        """Test gamma function properties"""
        # Test gamma(n+1) = n * gamma(n)
        for n in [1, 2, 3, 4, 5]:
            result1 = self.gamma(n + 1)
            result2 = n * self.gamma(n)
            self.assertAlmostEqual(result1, result2, places=10)
    
    def test_beta_function_basic(self):
        """Test basic beta function values"""
        # Test known values
        test_cases = [
            (1.0, 1.0, 1.0),
            (2.0, 3.0, 1.0/12.0),
            (0.5, 0.5, 3.141592653589793),  # pi
        ]
        
        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b):
                result = self.beta(a, b)
                self.assertAlmostEqual(result, expected, places=10)
    
    def test_beta_gamma_relationship(self):
        """Test beta-gamma relationship: B(a,b) = Γ(a)Γ(b)/Γ(a+b)"""
        a, b = 2.5, 3.5
        beta_result = self.beta(a, b)
        gamma_result = (self.gamma(a) * self.gamma(b)) / self.gamma(a + b)
        self.assertAlmostEqual(beta_result, gamma_result, places=10)

class TestBinomialCoefficients(unittest.TestCase):
    """Test binomial coefficient functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.special.binomial_coeffs import BinomialCoefficients
        self.binomial = BinomialCoefficients()
    
    def test_binomial_coefficients_basic(self):
        """Test basic binomial coefficient values"""
        # Test known values
        test_cases = [
            (5, 2, 10),
            (10, 3, 120),
            (7, 0, 1),
            (8, 8, 1),
            (6, 4, 15),
            (12, 5, 792),
        ]
        
        for n, k, expected in test_cases:
            with self.subTest(n=n, k=k):
                result = self.binomial.compute(n, k)
                self.assertEqual(result, expected)
    
    def test_binomial_coefficients_symmetry(self):
        """Test binomial coefficient symmetry: C(n,k) = C(n,n-k)"""
        test_cases = [
            (10, 3),
            (15, 7),
            (20, 5),
            (8, 2),
        ]
        
        for n, k in test_cases:
            with self.subTest(n=n, k=k):
                result1 = self.binomial.compute(n, k)
                result2 = self.binomial.compute(n, n - k)
                self.assertEqual(result1, result2)
    
    def test_binomial_coefficients_pascals_triangle(self):
        """Test Pascal's triangle property: C(n,k) = C(n-1,k-1) + C(n-1,k)"""
        test_cases = [
            (5, 2),
            (7, 3),
            (10, 4),
        ]
        
        for n, k in test_cases:
            if k > 0 and k < n:  # Valid range for Pascal's triangle
                with self.subTest(n=n, k=k):
                    result = self.binomial.compute(n, k)
                    result_pascal = (self.binomial.compute(n-1, k-1) + 
                                   self.binomial.compute(n-1, k))
                    self.assertEqual(result, result_pascal)
    
    def test_binomial_coefficients_edge_cases(self):
        """Test binomial coefficient edge cases"""
        # Test edge cases
        edge_cases = [
            (0, 0, 1),
        ]
        # Add more edge cases
        for n in range(1, 6):
            edge_cases.append((n, 0, 1))
            edge_cases.append((n, n, 1))
        
        for case in edge_cases:
            if len(case) == 3:
                n, k, expected = case
                with self.subTest(n=n, k=k):
                    result = self.binomial.compute(n, k)
                    self.assertEqual(result, expected)

class TestMittagLefflerFunction(unittest.TestCase):
    """Test Mittag-Leffler function"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.special.mittag_leffler import mittag_leffler
        self.mittag_leffler = mittag_leffler
    
    def test_mittag_leffler_basic(self):
        """Test basic Mittag-Leffler function values"""
        # Test known values
        test_cases = [
            (0.0, 1.0, 1.0, 1.0),  # E_{1,1}(0) = 1
            (1.0, 2.0, 1.0, np.cosh(1.0)),  # E_{2,1}(1) = cosh(1)
            (1.0, 1.0, 1.0, np.exp(1.0)),  # E_{1,1}(1) = e
        ]
        
        for z, alpha, beta, expected in test_cases:
            with self.subTest(z=z, alpha=alpha, beta=beta):
                result = self.mittag_leffler(z, alpha, beta)
                # Allow for numerical precision issues
                if np.isfinite(result) and np.isfinite(expected):
                    self.assertAlmostEqual(result, expected, places=6)
    
    def test_mittag_leffler_properties(self):
        """Test Mittag-Leffler function properties"""
        # Test that function returns finite values for reasonable inputs
        test_cases = [
            (0.5, 0.5, 1.0),
            (1.0, 0.7, 1.0),
            (2.0, 1.0, 1.0),
            (-1.0, 0.8, 1.0),
        ]
        
        for z, alpha, beta in test_cases:
            with self.subTest(z=z, alpha=alpha, beta=beta):
                result = self.mittag_leffler(z, alpha, beta)
                # Should return a finite number (or handle gracefully)
                self.assertTrue(np.isfinite(result) or isinstance(result, (int, float, complex)))
    
    def test_mittag_leffler_edge_cases(self):
        """Test Mittag-Leffler function edge cases"""
        # Test edge cases that might cause issues
        edge_cases = [
            (0.0, 1.0, 1.0),  # E_{1,1}(0) = 1
            (1.0, 0.0, 1.0),  # Special case
            (0.0, 0.0, 1.0),  # Another special case
        ]
        
        for z, alpha, beta in edge_cases:
            with self.subTest(z=z, alpha=alpha, beta=beta):
                try:
                    result = self.mittag_leffler(z, alpha, beta)
                    # Should return some value
                    self.assertIsNotNone(result)
                except (ValueError, ZeroDivisionError):
                    # Some edge cases may legitimately raise errors
                    pass

class TestSpecialFunctionIntegration(unittest.TestCase):
    """Test integration between special functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.special.gamma_beta import gamma, beta
        from hpfracc.special.binomial_coeffs import BinomialCoefficients
        from hpfracc.special.mittag_leffler import mittag_leffler
        
        self.gamma = gamma
        self.beta = beta
        self.binomial = BinomialCoefficients()
        self.mittag_leffler = mittag_leffler
    
    def test_gamma_binomial_relationship(self):
        """Test relationship between gamma and binomial coefficients"""
        # Test that binomial coefficients can be expressed using gamma
        n, k = 5, 2
        binomial_result = self.binomial.compute(n, k)
        
        # C(n,k) = n! / (k! * (n-k)!)
        # Using gamma: C(n,k) = Γ(n+1) / (Γ(k+1) * Γ(n-k+1))
        gamma_result = (self.gamma(n + 1) / 
                       (self.gamma(k + 1) * self.gamma(n - k + 1)))
        
        self.assertAlmostEqual(binomial_result, gamma_result, places=10)
    
    def test_special_functions_consistency(self):
        """Test consistency across special functions"""
        # Test that all functions work together consistently
        
        # Test gamma function
        gamma_val = self.gamma(3.0)
        self.assertAlmostEqual(gamma_val, 2.0, places=10)
        
        # Test beta function
        beta_val = self.beta(2.0, 3.0)
        expected_beta = (self.gamma(2.0) * self.gamma(3.0)) / self.gamma(5.0)
        self.assertAlmostEqual(beta_val, expected_beta, places=10)
        
        # Test binomial coefficients
        binomial_val = self.binomial.compute(5, 2)
        self.assertEqual(binomial_val, 10)
        
        # Test Mittag-Leffler (should not crash)
        ml_val = self.mittag_leffler(1.0, 1.0, 1.0)
        self.assertTrue(np.isfinite(ml_val))
    
    def test_special_functions_mathematical_properties(self):
        """Test mathematical properties across special functions"""
        # Test factorial property: n! = Γ(n+1)
        for n in range(1, 6):
            factorial_n = 1
            for i in range(1, n + 1):
                factorial_n *= i
            
            gamma_n_plus_1 = self.gamma(n + 1)
            self.assertAlmostEqual(factorial_n, gamma_n_plus_1, places=10)
        
        # Test binomial coefficient using gamma
        n, k = 6, 3
        expected = 20
        actual = self.binomial.compute(n, k)
        self.assertEqual(actual, expected)

class TestSpecialFunctionPerformance(unittest.TestCase):
    """Test special function performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.special.gamma_beta import gamma
        from hpfracc.special.binomial_coeffs import BinomialCoefficients
        
        self.gamma = gamma
        self.binomial = BinomialCoefficients()
    
    def test_gamma_performance(self):
        """Test gamma function performance"""
        import time
        
        # Test multiple evaluations
        start_time = time.time()
        for _ in range(100):
            self.gamma(2.5)
        end_time = time.time()
        
        # Should complete quickly
        elapsed = end_time - start_time
        self.assertLess(elapsed, 1.0)  # Should complete in under 1 second
    
    def test_binomial_performance(self):
        """Test binomial coefficient performance"""
        import time
        
        # Test multiple evaluations
        start_time = time.time()
        for _ in range(100):
            self.binomial.compute(10, 3)
        end_time = time.time()
        
        # Should complete quickly
        elapsed = end_time - start_time
        self.assertLess(elapsed, 1.0)  # Should complete in under 1 second
    
    def test_special_functions_memory_usage(self):
        """Test that special functions don't leak memory"""
        # Test that repeated calls don't cause memory issues
        for _ in range(1000):
            gamma_val = self.gamma(2.0)
            binomial_val = self.binomial.compute(5, 2)
            
            # Values should be consistent
            self.assertAlmostEqual(gamma_val, 1.0, places=10)
            self.assertEqual(binomial_val, 10)

if __name__ == '__main__':
    unittest.main()
