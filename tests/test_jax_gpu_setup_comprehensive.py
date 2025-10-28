"""
Comprehensive tests for hpfracc.jax_gpu_setup module

This module tests JAX GPU setup and configuration utilities
for fractional calculus machine learning applications.
"""

import pytest
import os
import warnings
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from hpfracc.jax_gpu_setup import (
    clear_jax_plugins,
    check_cudnn_compatibility,
    setup_jax_gpu_safe,
    setup_jax_gpu,
    get_jax_info,
    force_cpu_fallback,
    JAX_GPU_AVAILABLE
)


class TestClearJaxPlugins:
    """Test the clear_jax_plugins function"""

    def test_clear_jax_plugins_success(self):
        """Test successful clearing of JAX plugins"""
        # Set some environment variables
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        os.environ['JAX_ENABLE_XLA'] = '1'
        os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_min_max=true'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Clear plugins
        clear_jax_plugins()
        
        # Check that variables were cleared
        assert 'JAX_PLATFORM_NAME' not in os.environ
        assert 'JAX_ENABLE_XLA' not in os.environ
        assert 'XLA_FLAGS' not in os.environ
        assert 'CUDA_VISIBLE_DEVICES' not in os.environ

    def test_clear_jax_plugins_partial_failure(self):
        """Test clearing plugins with partial failure"""
        # Set some environment variables
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        
        # Mock os.environ to raise exception for one variable
        original_environ = os.environ.copy()
        
        def mock_delitem(key):
            if key == 'JAX_PLATFORM_NAME':
                raise KeyError("Simulated error")
            else:
                del original_environ[key]
        
        with patch.object(os.environ, '__delitem__', side_effect=mock_delitem):
            with patch.object(os.environ, '__contains__', side_effect=lambda x: x in original_environ):
                # Should not raise exception, just warn
                with warnings.catch_warnings(record=True) as w:
                    clear_jax_plugins()
                    assert len(w) == 1
                    assert "Failed to clear JAX environment variables" in str(w[0].message)

    def test_clear_jax_plugins_no_variables(self):
        """Test clearing plugins when no variables are set"""
        # Ensure variables are not set
        env_vars = ['JAX_PLATFORM_NAME', 'JAX_ENABLE_XLA', 'XLA_FLAGS', 'CUDA_VISIBLE_DEVICES']
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Should not raise any exceptions
        clear_jax_plugins()


class TestCheckCudnnCompatibility:
    """Test the check_cudnn_compatibility function"""

    @patch('jaxlib.__version__', '0.4.20')
    @patch('ctypes.CDLL')
    def test_check_cudnn_compatibility_success(self, mock_cdll):
        """Test successful CuDNN compatibility check"""
        # Mock successful CuDNN library loading
        mock_cdll.return_value = Mock()
        
        result = check_cudnn_compatibility()
        
        assert result['cudnn_available'] == True
        assert result['jaxlib_version'] == '0.4.20'
        assert 'warning' in result
        assert 'CuDNN version mismatch' in result['warning']

    @patch('jaxlib.__version__', '0.4.20')
    @patch('ctypes.CDLL', side_effect=OSError("Library not found"))
    def test_check_cudnn_compatibility_library_not_found(self, mock_cdll):
        """Test CuDNN compatibility check when library is not found"""
        result = check_cudnn_compatibility()
        
        assert result['cudnn_available'] == False
        assert result['jaxlib_version'] == '0.4.20'
        assert 'warning' in result

    @patch('jaxlib.__version__', '0.4.20')
    @patch('ctypes.CDLL', side_effect=Exception("Unexpected error"))
    def test_check_cudnn_compatibility_unexpected_error(self, mock_cdll):
        """Test CuDNN compatibility check with unexpected error"""
        result = check_cudnn_compatibility()
        
        assert result['cudnn_available'] == False
        assert result['jaxlib_version'] == '0.4.20'
        assert 'warning' in result

    @patch('hpfracc.jax_gpu_setup.jaxlib', side_effect=ImportError("jaxlib not available"))
    def test_check_cudnn_compatibility_jaxlib_not_available(self, mock_jaxlib):
        """Test CuDNN compatibility check when jaxlib is not available"""
        result = check_cudnn_compatibility()
        
        assert result['cudnn_available'] == False
        assert 'error' in result
        assert 'jaxlib not available' in result['error']


class TestSetupJaxGpuSafe:
    """Test the setup_jax_gpu_safe function"""

    @patch('hpfracc.jax_gpu_setup.clear_jax_plugins')
    @patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility')
    @patch('jax.devices')
    def test_setup_jax_gpu_safe_gpu_available(self, mock_devices, mock_cudnn, mock_clear):
        """Test JAX GPU setup when GPU is available"""
        # Mock GPU devices
        mock_gpu_device = Mock()
        mock_gpu_device.__str__ = Mock(return_value="gpu:0")
        mock_devices.return_value = [mock_gpu_device]
        
        # Mock CuDNN check
        mock_cudnn.return_value = {'cudnn_available': True}
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.devices = mock_devices
            
            result = setup_jax_gpu_safe()
            
            assert result == True
            mock_clear.assert_called_once()
            mock_cudnn.assert_called_once()

    @patch('hpfracc.jax_gpu_setup.clear_jax_plugins')
    @patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility')
    @patch('jax.devices')
    def test_setup_jax_gpu_safe_cpu_only(self, mock_devices, mock_cudnn, mock_clear):
        """Test JAX GPU setup when only CPU is available"""
        # Mock CPU devices
        mock_cpu_device = Mock()
        mock_cpu_device.__str__ = Mock(return_value="cpu:0")
        mock_devices.return_value = [mock_cpu_device]
        
        # Mock CuDNN check
        mock_cudnn.return_value = {'cudnn_available': False}
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.devices = mock_devices
            
            result = setup_jax_gpu_safe()
            
            assert result == False
            mock_clear.assert_called_once()
            mock_cudnn.assert_called_once()

    @patch('hpfracc.jax_gpu_setup.clear_jax_plugins')
    @patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility')
    def test_setup_jax_gpu_safe_cudnn_warning(self, mock_cudnn, mock_clear):
        """Test JAX GPU setup with CuDNN warning"""
        # Mock CuDNN check with warning
        mock_cudnn.return_value = {
            'cudnn_available': False,
            'warning': 'CuDNN version mismatch'
        }
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_gpu_device = Mock()
            mock_gpu_device.__str__ = Mock(return_value="gpu:0")
            mock_jax.devices.return_value = [mock_gpu_device]
            
            with patch('builtins.print') as mock_print:
                result = setup_jax_gpu_safe()
                
                assert result == True
                # Check that warning was printed
                mock_print.assert_called()
                assert any('CuDNN version mismatch' in str(call) for call in mock_print.call_args_list)

    @patch('hpfracc.jax_gpu_setup.clear_jax_plugins')
    @patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility')
    def test_setup_jax_gpu_safe_import_error(self, mock_cudnn, mock_clear):
        """Test JAX GPU setup with import error"""
        # Mock CuDNN check
        mock_cudnn.return_value = {'cudnn_available': True}
        
        # Mock JAX import to raise exception
        with patch('jax', side_effect=ImportError("JAX not available")):
            with patch('builtins.print') as mock_print:
                result = setup_jax_gpu_safe()
                
                assert result == False
                # Check that fallback message was printed
                mock_print.assert_called_with("ℹ️  Falling back to CPU execution")

    @patch('hpfracc.jax_gpu_setup.clear_jax_plugins')
    @patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility')
    def test_setup_jax_gpu_safe_jax_error(self, mock_cudnn, mock_clear):
        """Test JAX GPU setup with JAX runtime error"""
        # Mock CuDNN check
        mock_cudnn.return_value = {'cudnn_available': True}
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.devices.side_effect = RuntimeError("JAX runtime error")
            
            with patch('builtins.print') as mock_print:
                result = setup_jax_gpu_safe()
                
                assert result == False
                # Check that fallback message was printed
                mock_print.assert_called_with("ℹ️  Falling back to CPU execution")


class TestSetupJaxGpu:
    """Test the setup_jax_gpu function (legacy)"""

    @patch('hpfracc.jax_gpu_setup.setup_jax_gpu_safe')
    def test_setup_jax_gpu_legacy(self, mock_safe):
        """Test legacy setup_jax_gpu function"""
        mock_safe.return_value = True
        
        result = setup_jax_gpu()
        
        assert result == True
        mock_safe.assert_called_once()


class TestGetJaxInfo:
    """Test the get_jax_info function"""

    @patch('jax.__version__', '0.4.20')
    @patch('jax.devices')
    @patch('jax.default_backend')
    @patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility')
    def test_get_jax_info_success(self, mock_cudnn, mock_backend, mock_devices):
        """Test successful JAX info retrieval"""
        # Mock devices
        mock_gpu_device = Mock()
        mock_gpu_device.__str__ = Mock(return_value="gpu:0")
        mock_cpu_device = Mock()
        mock_cpu_device.__str__ = Mock(return_value="cpu:0")
        mock_devices.return_value = [mock_gpu_device, mock_cpu_device]
        
        # Mock backend
        mock_backend.return_value = 'gpu'
        
        # Mock CuDNN check
        mock_cudnn.return_value = {'cudnn_available': True}
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.__version__ = '0.4.20'
            mock_jax.devices = mock_devices
            mock_jax.default_backend = mock_backend
            
            result = get_jax_info()
            
            assert result['version'] == '0.4.20'
            assert result['devices'] == ['gpu:0', 'cpu:0']
            assert result['device_count'] == 2
            assert result['backend'] == 'gpu'
            assert result['gpu_available'] == True
            assert result['cudnn_info'] == {'cudnn_available': True}
            assert result['platform'] == 'gpu'

    @patch('jax.__version__', '0.4.20')
    @patch('jax.devices')
    @patch('jax.default_backend')
    @patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility')
    def test_get_jax_info_cpu_only(self, mock_cudnn, mock_backend, mock_devices):
        """Test JAX info retrieval with CPU only"""
        # Mock CPU devices only
        mock_cpu_device = Mock()
        mock_cpu_device.__str__ = Mock(return_value="cpu:0")
        mock_devices.return_value = [mock_cpu_device]
        
        # Mock backend
        mock_backend.return_value = 'cpu'
        
        # Mock CuDNN check
        mock_cudnn.return_value = {'cudnn_available': False}
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.__version__ = '0.4.20'
            mock_jax.devices = mock_devices
            mock_jax.default_backend = mock_backend
            
            result = get_jax_info()
            
            assert result['version'] == '0.4.20'
            assert result['devices'] == ['cpu:0']
            assert result['device_count'] == 1
            assert result['backend'] == 'cpu'
            assert result['gpu_available'] == False
            assert result['cudnn_info'] == {'cudnn_available': False}
            assert result['platform'] == 'cpu'

    def test_get_jax_info_import_error(self):
        """Test JAX info retrieval with import error"""
        with patch('jax', side_effect=ImportError("JAX not available")):
            result = get_jax_info()
            
            assert 'error' in result
            assert 'JAX not available' in result['error']

    def test_get_jax_info_runtime_error(self):
        """Test JAX info retrieval with runtime error"""
        with patch('jax', create=True) as mock_jax:
            mock_jax.devices.side_effect = RuntimeError("JAX runtime error")
            
            result = get_jax_info()
            
            assert 'error' in result
            assert 'JAX runtime error' in result['error']


class TestForceCpuFallback:
    """Test the force_cpu_fallback function"""

    @patch('jax.devices')
    @patch('jax.clear_caches')
    def test_force_cpu_fallback_success(self, mock_clear_caches, mock_devices):
        """Test successful CPU fallback"""
        # Mock CPU devices
        mock_cpu_device = Mock()
        mock_cpu_device.__str__ = Mock(return_value="cpu:0")
        mock_devices.return_value = [mock_cpu_device]
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.devices = mock_devices
            mock_jax.clear_caches = mock_clear_caches
            
            with patch('builtins.print') as mock_print:
                result = force_cpu_fallback()
                
                assert result == True
                assert os.environ['JAX_PLATFORM_NAME'] == 'cpu'
                mock_clear_caches.assert_called_once()
                mock_print.assert_called_with("✅ Forced JAX to use CPU: ['cpu:0']")

    @patch('jax.devices')
    @patch('jax.clear_caches')
    def test_force_cpu_fallback_no_cpu_devices(self, mock_clear_caches, mock_devices):
        """Test CPU fallback when no CPU devices found"""
        # Mock GPU devices only
        mock_gpu_device = Mock()
        mock_gpu_device.__str__ = Mock(return_value="gpu:0")
        mock_devices.return_value = [mock_gpu_device]
        
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.devices = mock_devices
            mock_jax.clear_caches = mock_clear_caches
            
            with patch('builtins.print') as mock_print:
                result = force_cpu_fallback()
                
                assert result == False
                assert os.environ['JAX_PLATFORM_NAME'] == 'cpu'
                mock_clear_caches.assert_called_once()
                mock_print.assert_called_with("⚠️  Failed to force CPU fallback")

    @patch('jax.clear_caches')
    def test_force_cpu_fallback_jax_error(self, mock_clear_caches):
        """Test CPU fallback with JAX error"""
        # Mock JAX import
        with patch('jax', create=True) as mock_jax:
            mock_jax.clear_caches.side_effect = RuntimeError("JAX error")
            
            with patch('builtins.print') as mock_print:
                result = force_cpu_fallback()
                
                assert result == False
                assert os.environ['JAX_PLATFORM_NAME'] == 'cpu'

    def test_force_cpu_fallback_import_error(self):
        """Test CPU fallback with import error"""
        with patch('jax', side_effect=ImportError("JAX not available")):
            result = force_cpu_fallback()
            
            assert result == False
            assert os.environ['JAX_PLATFORM_NAME'] == 'cpu'


class TestJaxGpuAvailable:
    """Test the JAX_GPU_AVAILABLE constant"""

    def test_jax_gpu_available_constant(self):
        """Test that JAX_GPU_AVAILABLE is defined"""
        assert isinstance(JAX_GPU_AVAILABLE, bool)

    def test_jax_gpu_available_import_time_setup(self):
        """Test that JAX GPU setup runs on import"""
        # The constant should be set based on the import-time setup
        # This tests that the auto-configuration works
        assert JAX_GPU_AVAILABLE is not None


# Integration tests
class TestJaxGpuSetupIntegration:
    """Integration tests for JAX GPU setup"""

    def test_full_gpu_setup_workflow(self):
        """Test complete GPU setup workflow"""
        # Clear any existing environment
        env_vars = ['JAX_PLATFORM_NAME', 'JAX_ENABLE_XLA', 'XLA_FLAGS', 'CUDA_VISIBLE_DEVICES']
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Mock successful GPU setup
        with patch('hpfracc.jax_gpu_setup.clear_jax_plugins') as mock_clear:
            with patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility') as mock_cudnn:
                with patch('jax', create=True) as mock_jax:
                    # Mock GPU devices
                    mock_gpu_device = Mock()
                    mock_gpu_device.__str__ = Mock(return_value="gpu:0")
                    mock_jax.devices.return_value = [mock_gpu_device]
                    mock_jax.__version__ = '0.4.20'
                    mock_jax.default_backend.return_value = 'gpu'
                    
                    # Mock CuDNN check
                    mock_cudnn.return_value = {'cudnn_available': True}
                    
                    # Test setup
                    result = setup_jax_gpu_safe()
                    
                    assert result == True
                    mock_clear.assert_called_once()
                    mock_cudnn.assert_called_once()

    def test_cpu_fallback_workflow(self):
        """Test CPU fallback workflow"""
        # Clear any existing environment
        env_vars = ['JAX_PLATFORM_NAME', 'JAX_ENABLE_XLA', 'XLA_FLAGS', 'CUDA_VISIBLE_DEVICES']
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Mock CPU-only setup
        with patch('hpfracc.jax_gpu_setup.clear_jax_plugins') as mock_clear:
            with patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility') as mock_cudnn:
                with patch('jax', create=True) as mock_jax:
                    # Mock CPU devices
                    mock_cpu_device = Mock()
                    mock_cpu_device.__str__ = Mock(return_value="cpu:0")
                    mock_jax.devices.return_value = [mock_cpu_device]
                    mock_jax.__version__ = '0.4.20'
                    mock_jax.default_backend.return_value = 'cpu'
                    
                    # Mock CuDNN check
                    mock_cudnn.return_value = {'cudnn_available': False}
                    
                    # Test setup
                    result = setup_jax_gpu_safe()
                    
                    assert result == False
                    mock_clear.assert_called_once()
                    mock_cudnn.assert_called_once()

    def test_error_handling_workflow(self):
        """Test error handling workflow"""
        # Clear any existing environment
        env_vars = ['JAX_PLATFORM_NAME', 'JAX_ENABLE_XLA', 'XLA_FLAGS', 'CUDA_VISIBLE_DEVICES']
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Mock error scenario
        with patch('hpfracc.jax_gpu_setup.clear_jax_plugins') as mock_clear:
            with patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility') as mock_cudnn:
                with patch('jax', side_effect=ImportError("JAX not available")):
                    # Test setup
                    result = setup_jax_gpu_safe()
                    
                    assert result == False
                    mock_clear.assert_called_once()
                    mock_cudnn.assert_called_once()

    def test_info_retrieval_workflow(self):
        """Test information retrieval workflow"""
        with patch('jax', create=True) as mock_jax:
            # Mock devices and info
            mock_gpu_device = Mock()
            mock_gpu_device.__str__ = Mock(return_value="gpu:0")
            mock_cpu_device = Mock()
            mock_cpu_device.__str__ = Mock(return_value="cpu:0")
            mock_jax.devices.return_value = [mock_gpu_device, mock_cpu_device]
            mock_jax.__version__ = '0.4.20'
            mock_jax.default_backend.return_value = 'gpu'
            
            with patch('hpfracc.jax_gpu_setup.check_cudnn_compatibility') as mock_cudnn:
                mock_cudnn.return_value = {'cudnn_available': True}
                
                # Test info retrieval
                info = get_jax_info()
                
                assert info['version'] == '0.4.20'
                assert info['gpu_available'] == True
                assert info['device_count'] == 2
                assert 'cudnn_info' in info
