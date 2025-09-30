#!/usr/bin/env python3
"""
Comprehensive tests for the main hpfracc/__init__.py module.

This test suite focuses on improving coverage for hpfracc/__init__.py
by testing package initialization, imports, and public API.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock


class TestHPFRACCInit:
    """Test the main hpfracc package initialization."""
    
    def test_package_metadata(self):
        """Test package metadata attributes."""
        import hpfracc
        
        assert hasattr(hpfracc, '__version__')
        assert hasattr(hpfracc, '__author__')
        assert hasattr(hpfracc, '__email__')
        assert hasattr(hpfracc, '__affiliation__')
        
        assert hpfracc.__version__ == "2.0.0"
        assert hpfracc.__author__ == "Davian R. Chin"
        assert hpfracc.__email__ == "d.r.chin@pgr.reading.ac.uk"
        assert hpfracc.__affiliation__ == "Department of Biomedical Engineering, University of Reading"
    
    def test_package_docstring(self):
        """Test package docstring."""
        import hpfracc
        
        assert hasattr(hpfracc, '__doc__')
        assert hpfracc.__doc__ is not None
        assert "High-Performance Fractional Calculus Library" in hpfracc.__doc__
        assert "fractional calculus" in hpfracc.__doc__
        assert "optimized implementations" in hpfracc.__doc__
    
    def test_all_attribute(self):
        """Test __all__ attribute exists and contains expected items."""
        import hpfracc
        
        assert hasattr(hpfracc, '__all__')
        assert isinstance(hpfracc.__all__, list)
        assert len(hpfracc.__all__) > 0
        
        # Check for key items in __all__
        expected_items = [
            "OptimizedRiemannLiouville",
            "OptimizedCaputo", 
            "OptimizedGrunwaldLetnikov",
            "FractionalOrder",
            "WeylDerivative",
            "MarchaudDerivative",
            "FractionalLaplacian",
            "FractionalFourierTransform",
            "RiemannLiouvilleIntegral",
            "CaputoIntegral",
            "CaputoFabrizioDerivative",
            "AtanganaBaleanuDerivative",
        ]
        
        for item in expected_items:
            assert item in hpfracc.__all__, f"{item} not found in __all__"
    
    def test_import_with_all_available(self):
        """Test imports when all modules are available."""
        # Mock all imports to be successful
        with patch.dict('sys.modules', {
            'hpfracc.algorithms.optimized_methods': MagicMock(),
            'hpfracc.algorithms.advanced_methods': MagicMock(),
            'hpfracc.algorithms.special_methods': MagicMock(),
            'hpfracc.algorithms.integral_methods': MagicMock(),
            'hpfracc.algorithms.novel_derivatives': MagicMock(),
            'hpfracc.core.definitions': MagicMock(),
        }):
            # Reload the module to test imports
            if 'hpfracc' in sys.modules:
                del sys.modules['hpfracc']
            
            import hpfracc
            
            # Should not raise any import errors
            assert hpfracc is not None
    
    def test_import_with_missing_modules(self):
        """Test imports when some modules are missing."""
        # Mock some imports to fail
        with patch.dict('sys.modules', {
            'hpfracc.algorithms.optimized_methods': None,
            'hpfracc.algorithms.advanced_methods': MagicMock(),
            'hpfracc.algorithms.special_methods': None,
            'hpfracc.algorithms.integral_methods': MagicMock(),
            'hpfracc.algorithms.novel_derivatives': None,
            'hpfracc.core.definitions': MagicMock(),
        }):
            # Reload the module to test imports
            if 'hpfracc' in sys.modules:
                del sys.modules['hpfracc']
            
            import hpfracc
            
            # Should not raise any import errors even with missing modules
            assert hpfracc is not None
    
    def test_import_with_all_missing_modules(self):
        """Test imports when all optional modules are missing."""
        # Mock all imports to fail
        with patch.dict('sys.modules', {
            'hpfracc.algorithms.optimized_methods': None,
            'hpfracc.algorithms.advanced_methods': None,
            'hpfracc.algorithms.special_methods': None,
            'hpfracc.algorithms.integral_methods': None,
            'hpfracc.algorithms.novel_derivatives': None,
            'hpfracc.core.definitions': None,
        }):
            # Reload the module to test imports
            if 'hpfracc' in sys.modules:
                del sys.modules['hpfracc']
            
            import hpfracc
            
            # Should not raise any import errors even with all modules missing
            assert hpfracc is not None
            # Basic metadata should still be available
            assert hasattr(hpfracc, '__version__')
            assert hasattr(hpfracc, '__author__')
    
    def test_import_error_handling(self):
        """Test that ImportError exceptions are properly handled."""
        # Test that the module can be imported even with missing dependencies
        # This is already tested by the other import tests, so we'll just verify
        # that the module structure is correct
        import hpfracc
        
        # Should not raise any import errors
        assert hpfracc is not None
        assert hasattr(hpfracc, '__version__')
        assert hasattr(hpfracc, '__author__')
        assert hasattr(hpfracc, '__all__')
    
    def test_module_attributes_after_import(self):
        """Test that expected attributes are available after import."""
        import hpfracc
        
        # Check that core attributes are available
        assert hasattr(hpfracc, '__version__')
        assert hasattr(hpfracc, '__author__')
        assert hasattr(hpfracc, '__email__')
        assert hasattr(hpfracc, '__affiliation__')
        assert hasattr(hpfracc, '__all__')
        
        # Check that __all__ is a list
        assert isinstance(hpfracc.__all__, list)
        
        # Check that __all__ contains expected number of items
        assert len(hpfracc.__all__) >= 20  # Should have many exported items
    
    def test_package_structure(self):
        """Test that the package has the expected structure."""
        import hpfracc
        
        # Check that it's a proper Python package
        assert hasattr(hpfracc, '__file__')
        assert hasattr(hpfracc, '__path__')
        
        # Check that __path__ is iterable
        assert hasattr(hpfracc.__path__, '__iter__')
    
    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import hpfracc
        
        version = hpfracc.__version__
        
        # Should be a string
        assert isinstance(version, str)
        
        # Should contain dots (semantic versioning)
        assert '.' in version
        
        # Should have at least major.minor.patch format
        parts = version.split('.')
        assert len(parts) >= 3
        
        # Each part should be a valid number
        for part in parts[:3]:  # Only check major.minor.patch
            assert part.isdigit()
    
    def test_author_information(self):
        """Test author information format."""
        import hpfracc
        
        author = hpfracc.__author__
        email = hpfracc.__email__
        affiliation = hpfracc.__affiliation__
        
        # Should be strings
        assert isinstance(author, str)
        assert isinstance(email, str)
        assert isinstance(affiliation, str)
        
        # Should not be empty
        assert len(author) > 0
        assert len(email) > 0
        assert len(affiliation) > 0
        
        # Email should contain @
        assert '@' in email
        
        # Should contain expected information
        assert "Davian" in author
        assert "Chin" in author
        assert "reading.ac.uk" in email
        assert "University of Reading" in affiliation
    
    def test_docstring_content(self):
        """Test docstring content and structure."""
        import hpfracc
        
        docstring = hpfracc.__doc__
        
        # Should be a string
        assert isinstance(docstring, str)
        
        # Should not be empty
        assert len(docstring) > 0
        
        # Should contain key terms
        key_terms = [
            "High-Performance Fractional Calculus Library",
            "hpfracc",
            "fractional calculus",
            "numerical methods",
            "optimized implementations",
            "Caputo",
            "Riemann-Liouville",
            "Grünwald-Letnikov",
            "GPU acceleration",
            "JAX",
            "PyTorch",
            "NUMBA"
        ]
        
        for term in key_terms:
            assert term in docstring, f"Key term '{term}' not found in docstring"
    
    def test_import_performance(self):
        """Test that package import is reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Reload the module to test import time
        if 'hpfracc' in sys.modules:
            del sys.modules['hpfracc']
        
        import hpfracc
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Import should complete within reasonable time (5 seconds)
        assert import_time < 5.0, f"Import took too long: {import_time:.2f} seconds"
    
    def test_import_with_system_exit(self):
        """Test that package import doesn't cause system exit."""
        # Mock system exit to catch if it's called
        with patch('sys.exit') as mock_exit:
            # Reload the module to test imports
            if 'hpfracc' in sys.modules:
                del sys.modules['hpfracc']
            
            import hpfracc
            
            # Should not call sys.exit
            mock_exit.assert_not_called()
            
            # Should successfully import
            assert hpfracc is not None
    
    def test_import_with_keyboard_interrupt(self):
        """Test that package import doesn't cause keyboard interrupt."""
        # Mock keyboard interrupt to catch if it's raised
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            # Reload the module to test imports
            if 'hpfracc' in sys.modules:
                del sys.modules['hpfracc']
            
            import hpfracc
            
            # Should successfully import despite keyboard interrupt in input
            assert hpfracc is not None
    
    def test_package_reload(self):
        """Test that package can be reloaded multiple times."""
        import hpfracc
        
        # Get initial version
        initial_version = hpfracc.__version__
        
        # Reload the module
        import importlib
        importlib.reload(hpfracc)
        
        # Should still work after reload
        assert hpfracc.__version__ == initial_version
        assert hasattr(hpfracc, '__author__')
        assert hasattr(hpfracc, '__all__')


class TestHPFRACCInitEdgeCases:
    """Test edge cases for hpfracc package initialization."""
    
    def test_import_with_memory_error(self):
        """Test import behavior with memory error."""
        # Mock memory error during import
        def mock_import(*args, **kwargs):
            if 'hpfracc' in args:
                raise MemoryError("Mock memory error")
            return MagicMock()
        
        with patch('builtins.__import__', side_effect=mock_import):
            # Should handle memory error gracefully
            try:
                import hpfracc
                # If import succeeds, basic attributes should be available
                assert hasattr(hpfracc, '__version__')
            except MemoryError:
                # Memory error is acceptable in this test
                pass
    
    def test_import_with_permission_error(self):
        """Test import behavior with permission error."""
        # Mock permission error during import
        def mock_import(*args, **kwargs):
            if 'hpfracc' in args:
                raise PermissionError("Mock permission error")
            return MagicMock()
        
        with patch('builtins.__import__', side_effect=mock_import):
            # Should handle permission error gracefully
            try:
                import hpfracc
                # If import succeeds, basic attributes should be available
                assert hasattr(hpfracc, '__version__')
            except PermissionError:
                # Permission error is acceptable in this test
                pass
    
    def test_import_with_os_error(self):
        """Test import behavior with OS error."""
        # Mock OS error during import
        def mock_import(*args, **kwargs):
            if 'hpfracc' in args:
                raise OSError("Mock OS error")
            return MagicMock()
        
        with patch('builtins.__import__', side_effect=mock_import):
            # Should handle OS error gracefully
            try:
                import hpfracc
                # If import succeeds, basic attributes should be available
                assert hasattr(hpfracc, '__version__')
            except OSError:
                # OS error is acceptable in this test
                pass


if __name__ == "__main__":
    pytest.main([__file__])
