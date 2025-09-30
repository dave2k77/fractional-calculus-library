"""
Comprehensive tests for hpfracc/ml/backends.py to improve coverage
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import warnings

from hpfracc.ml.backends import (
    BackendType, BackendManager, get_backend_manager, 
    set_backend_manager, get_active_backend, switch_backend,
    TORCH_AVAILABLE, JAX_AVAILABLE, NUMBA_AVAILABLE
)


class TestBackendManagerInitialization(unittest.TestCase):
    """Test BackendManager initialization and configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Reset global backend manager
        set_backend_manager(None)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_initialization_default(self, mock_numba, mock_jax, mock_torch):
        """Test default initialization"""
        mock_torch.cuda.is_available.return_value = True
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = True
        
        with patch('builtins.print') as mock_print:
            manager = BackendManager()
            
            self.assertEqual(manager.preferred_backend, BackendType.AUTO)
            self.assertFalse(manager.force_cpu)
            self.assertTrue(manager.enable_jit)
            self.assertTrue(manager.enable_gpu)
            self.assertIsInstance(manager.available_backends, list)
            self.assertIsInstance(manager.backend_configs, dict)
            
            # Check print statements
            self.assertTrue(mock_print.called)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_initialization_custom_config(self, mock_numba, mock_jax, mock_torch):
        """Test initialization with custom configuration"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager(
            preferred_backend=BackendType.JAX,
            force_cpu=True,
            enable_jit=False,
            enable_gpu=False
        )
        
        self.assertEqual(manager.preferred_backend, BackendType.JAX)
        self.assertTrue(manager.force_cpu)
        self.assertFalse(manager.enable_jit)
        self.assertFalse(manager.enable_gpu)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_force_cpu_prevents_gpu_detection(self, mock_numba, mock_jax, mock_torch):
        """Test that force_cpu prevents GPU detection"""
        mock_torch.cuda.is_available.return_value = True
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = True
        
        with patch('builtins.print') as mock_print:
            manager = BackendManager(force_cpu=True)
            
            # Should not print GPU detection messages
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            self.assertFalse(any('GPU support detected' in call for call in print_calls))


class TestBackendDetection(unittest.TestCase):
    """Test backend detection and availability checking"""
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_detect_available_backends_all_available(self, mock_numba, mock_jax, mock_torch):
        """Test detection when all backends are available"""
        mock_torch.cuda.is_available.return_value = True
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = True
        
        manager = BackendManager()
        available = manager._detect_available_backends()
        
        self.assertIn(BackendType.TORCH, available)
        self.assertIn(BackendType.JAX, available)
        self.assertIn(BackendType.NUMBA, available)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_detect_available_backends_partial_availability(self, mock_numba, mock_jax, mock_torch):
        """Test detection when only some backends are available"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        # Mock imports to simulate partial availability
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            manager = BackendManager()
            available = manager._detect_available_backends()
            
            self.assertNotIn(BackendType.TORCH, available)
    
    @patch('hpfracc.ml.backends.TORCH_AVAILABLE', False)
    @patch('hpfracc.ml.backends.JAX_AVAILABLE', False)
    @patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False)
    def test_detect_available_backends_none_available(self):
        """Test detection when no backends are available"""
        with self.assertRaises(RuntimeError) as context:
            BackendManager()
        
        self.assertIn("No computation backends available", str(context.exception))


class TestBackendSelection(unittest.TestCase):
    """Test backend selection logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_backend_manager(None)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_select_optimal_backend_auto_prefers_torch(self, mock_numba, mock_jax, mock_torch):
        """Test that AUTO backend selection prefers PyTorch"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager(preferred_backend=BackendType.AUTO)
        self.assertEqual(manager.active_backend, BackendType.TORCH)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_select_optimal_backend_auto_fallback_to_jax(self, mock_numba, mock_jax, mock_torch):
        """Test AUTO backend selection falls back to JAX when PyTorch unavailable"""
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            manager = BackendManager(preferred_backend=BackendType.AUTO, enable_gpu=True)
            self.assertEqual(manager.active_backend, BackendType.JAX)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_select_optimal_backend_auto_fallback_to_numba(self, mock_numba, mock_jax, mock_torch):
        """Test AUTO backend selection falls back to NUMBA"""
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            with patch('hpfracc.ml.backends.JAX_AVAILABLE', False):
                manager = BackendManager(preferred_backend=BackendType.AUTO)
                self.assertEqual(manager.active_backend, BackendType.NUMBA)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_select_optimal_backend_preferred_not_available(self, mock_numba, mock_jax, mock_torch):
        """Test selection when preferred backend is not available"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                manager = BackendManager(preferred_backend=BackendType.TORCH)
                
                self.assertEqual(len(w), 1)
                self.assertIn("Preferred backend torch not available", str(w[0].message))
                self.assertEqual(manager.active_backend, BackendType.JAX)


class TestBackendConfigurations(unittest.TestCase):
    """Test backend configuration initialization"""
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_initialize_backend_configs_torch(self, mock_numba, mock_jax, mock_torch):
        """Test PyTorch backend configuration"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float32 = np.float32
        mock_torch.compile = Mock()
        
        manager = BackendManager()
        configs = manager._initialize_backend_configs()
        
        if BackendType.TORCH in configs:
            torch_config = configs[BackendType.TORCH]
            self.assertEqual(torch_config['device'], 'cuda')
            self.assertEqual(torch_config['dtype'], np.float32)
            self.assertTrue(torch_config['enable_amp'])
            self.assertTrue(torch_config['enable_compile'])
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_initialize_backend_configs_jax(self, mock_numba, mock_jax, mock_torch):
        """Test JAX backend configuration"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_jax.numpy = Mock()
        mock_jax.numpy.float32 = np.float32
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager(enable_gpu=True)
        configs = manager._initialize_backend_configs()
        
        if BackendType.JAX in configs:
            jax_config = configs[BackendType.JAX]
            self.assertEqual(jax_config['device'], 'gpu')
            self.assertEqual(jax_config['dtype'], np.float32)
            self.assertTrue(jax_config['enable_jit'])
            self.assertFalse(jax_config['enable_x64'])
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_initialize_backend_configs_numba(self, mock_numba, mock_jax, mock_torch):
        """Test NUMBA backend configuration"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = True
        mock_numba.float32 = np.float32
        
        manager = BackendManager()
        configs = manager._initialize_backend_configs()
        
        if BackendType.NUMBA in configs:
            numba_config = configs[BackendType.NUMBA]
            self.assertEqual(numba_config['device'], 'gpu')
            self.assertEqual(numba_config['dtype'], np.float32)
            self.assertTrue(numba_config['enable_jit'])
            self.assertTrue(numba_config['enable_parallel'])


class TestBackendManagerOperations(unittest.TestCase):
    """Test BackendManager operations and methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_backend_manager(None)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_get_backend_config(self, mock_numba, mock_jax, mock_torch):
        """Test getting backend configuration"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        config = manager.get_backend_config()
        
        self.assertIsInstance(config, dict)
        
        # Test with specific backend
        if BackendType.TORCH in manager.available_backends:
            torch_config = manager.get_backend_config(BackendType.TORCH)
            self.assertIsInstance(torch_config, dict)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_switch_backend_success(self, mock_numba, mock_jax, mock_torch):
        """Test successful backend switching"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        if BackendType.JAX in manager.available_backends:
            with patch('builtins.print') as mock_print:
                result = manager.switch_backend(BackendType.JAX)
                
                self.assertTrue(result)
                self.assertEqual(manager.active_backend, BackendType.JAX)
                mock_print.assert_called_with("🔄 Switched to jax backend")
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_switch_backend_failure(self, mock_numba, mock_jax, mock_torch):
        """Test backend switching failure"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            # Create a new manager without TORCH to test the failure case
            with patch.object(manager, 'available_backends', [BackendType.JAX, BackendType.NUMBA]):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = manager.switch_backend(BackendType.TORCH)
                    
                    self.assertFalse(result)
                    self.assertEqual(len(w), 1)
                    self.assertIn("Backend torch not available", str(w[0].message))
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_get_tensor_lib(self, mock_numba, mock_jax, mock_torch):
        """Test getting tensor library"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        # Test with current active backend
        tensor_lib = manager.get_tensor_lib()
        self.assertIsNotNone(tensor_lib)
        
        # Test switching and getting different libraries
        if BackendType.JAX in manager.available_backends:
            manager.switch_backend(BackendType.JAX)
            jax_lib = manager.get_tensor_lib()
            # The actual jax.numpy module is returned, not the mock
            self.assertIsNotNone(jax_lib)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_get_tensor_lib_unknown_backend(self, mock_numba, mock_jax, mock_torch):
        """Test getting tensor library for unknown backend"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        # Mock an unknown backend
        manager.active_backend = "unknown"
        
        with self.assertRaises(RuntimeError) as context:
            manager.get_tensor_lib()
        
        self.assertIn("Unknown backend", str(context.exception))


class TestTensorCreation(unittest.TestCase):
    """Test tensor creation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_backend_manager(None)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_create_tensor_torch(self, mock_numba, mock_jax, mock_torch):
        """Test tensor creation with PyTorch backend"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        if BackendType.TORCH in manager.available_backends:
            mock_torch.tensor.return_value = Mock()
            
            data = [1, 2, 3]
            result = manager.create_tensor(data)
            
            mock_torch.tensor.assert_called()
            self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_create_tensor_jax(self, mock_numba, mock_jax, mock_torch):
        """Test tensor creation with JAX backend"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            manager = BackendManager()
            
            if BackendType.JAX in manager.available_backends:
                # The actual jax.numpy.array is called, not the mock
                data = [1, 2, 3]
                result = manager.create_tensor(data)
                
                self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_create_tensor_numba(self, mock_numba, mock_jax, mock_torch):
        """Test tensor creation with NUMBA backend"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            with patch('hpfracc.ml.backends.JAX_AVAILABLE', False):
                manager = BackendManager()
                
                if BackendType.NUMBA in manager.available_backends:
                    with patch('numpy.array') as mock_np_array:
                        mock_np_array.return_value = Mock()
                        
                        data = [1, 2, 3]
                        result = manager.create_tensor(data)
                        
                        mock_np_array.assert_called()
                        self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_create_tensor_with_dtype_preservation(self, mock_numba, mock_jax, mock_torch):
        """Test tensor creation with dtype preservation for integer types"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        if BackendType.TORCH in manager.available_backends:
            mock_torch.tensor.return_value = Mock()
            mock_torch.long = np.int64
            
            # Test with integer data
            data = np.array([1, 2, 3], dtype=np.int32)
            result = manager.create_tensor(data)
            
            # Should preserve integer dtype
            call_args = mock_torch.tensor.call_args
            self.assertEqual(call_args[1]['dtype'], np.int64)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_create_tensor_unknown_backend(self, mock_numba, mock_jax, mock_torch):
        """Test tensor creation with unknown backend"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        # Mock an unknown backend
        manager.active_backend = "unknown"
        
        with self.assertRaises(RuntimeError) as context:
            manager.create_tensor([1, 2, 3])
        
        self.assertIn("Unknown backend", str(context.exception))


class TestDeviceOperations(unittest.TestCase):
    """Test device operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_backend_manager(None)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_to_device_torch(self, mock_numba, mock_jax, mock_torch):
        """Test moving tensor to device with PyTorch"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        if BackendType.TORCH in manager.available_backends:
            mock_tensor = Mock()
            mock_tensor.to.return_value = mock_tensor
            
            result = manager.to_device(mock_tensor, "cuda")
            
            mock_tensor.to.assert_called_with("cuda")
            self.assertEqual(result, mock_tensor)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_to_device_jax(self, mock_numba, mock_jax, mock_torch):
        """Test moving tensor to device with JAX"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            manager = BackendManager()
            
            if BackendType.JAX in manager.available_backends:
                mock_tensor = Mock()
                
                result = manager.to_device(mock_tensor, "gpu")
                
                # JAX handles device placement differently, should return tensor as-is
                self.assertEqual(result, mock_tensor)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_to_device_numba(self, mock_numba, mock_jax, mock_torch):
        """Test moving tensor to device with NUMBA"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            with patch('hpfracc.ml.backends.JAX_AVAILABLE', False):
                manager = BackendManager()
                
                if BackendType.NUMBA in manager.available_backends:
                    mock_tensor = Mock()
                    
                    result = manager.to_device(mock_tensor, "cpu")
                    
                    # NUMBA handles device placement differently, should return tensor as-is
                    self.assertEqual(result, mock_tensor)


class TestFunctionCompilation(unittest.TestCase):
    """Test function compilation"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_backend_manager(None)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_compile_function_torch_with_compile(self, mock_numba, mock_jax, mock_torch):
        """Test function compilation with PyTorch when compile is available"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        if BackendType.TORCH in manager.available_backends:
            mock_torch.compile = Mock()
            mock_torch.compile.return_value = Mock()
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            
            mock_torch.compile.assert_called_with(test_func)
            self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_compile_function_torch_without_compile(self, mock_numba, mock_jax, mock_torch):
        """Test function compilation with PyTorch when compile is not available"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        manager = BackendManager()
        
        if BackendType.TORCH in manager.available_backends:
            # Remove compile attribute
            del mock_torch.compile
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            
            # Should return function as-is
            self.assertEqual(result, test_func)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_compile_function_jax_with_jit(self, mock_numba, mock_jax, mock_torch):
        """Test function compilation with JAX when JIT is enabled"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            manager = BackendManager(enable_jit=True)
            
            if BackendType.JAX in manager.available_backends:
                mock_jax.jit = Mock()
                mock_jax.jit.return_value = Mock()
                
                def test_func(x):
                    return x * 2
                
                result = manager.compile_function(test_func)
                
                mock_jax.jit.assert_called_with(test_func)
                self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_compile_function_numba_with_jit(self, mock_numba, mock_jax, mock_torch):
        """Test function compilation with NUMBA when JIT is enabled"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False):
            with patch('hpfracc.ml.backends.JAX_AVAILABLE', False):
                manager = BackendManager(enable_jit=True)
                
                if BackendType.NUMBA in manager.available_backends:
                    mock_numba.jit = Mock()
                    mock_numba.jit.return_value = Mock()
                    
                    def test_func(x):
                        return x * 2
                    
                    result = manager.compile_function(test_func)
                    
                    mock_numba.jit.assert_called_with(test_func)
                    self.assertIsNotNone(result)


class TestGlobalFunctions(unittest.TestCase):
    """Test global backend management functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_backend_manager(None)
    
    def test_get_backend_manager_singleton(self):
        """Test that get_backend_manager returns singleton"""
        manager1 = get_backend_manager()
        manager2 = get_backend_manager()
        
        self.assertIs(manager1, manager2)
    
    def test_set_backend_manager(self):
        """Test setting custom backend manager"""
        custom_manager = Mock()
        set_backend_manager(custom_manager)
        
        retrieved_manager = get_backend_manager()
        self.assertIs(retrieved_manager, custom_manager)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_get_active_backend(self, mock_numba, mock_jax, mock_torch):
        """Test getting active backend"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        backend = get_active_backend()
        self.assertIsInstance(backend, BackendType)
    
    @patch('hpfracc.ml.backends.torch')
    @patch('hpfracc.ml.backends.jax')
    @patch('hpfracc.ml.backends.numba')
    def test_switch_backend_global(self, mock_numba, mock_jax, mock_torch):
        """Test global backend switching"""
        mock_torch.cuda.is_available.return_value = False
        mock_jax.devices.return_value = [Mock()]
        mock_numba.cuda.is_available.return_value = False
        
        if BackendType.JAX in get_backend_manager().available_backends:
            with patch('builtins.print'):
                result = switch_backend(BackendType.JAX)
                self.assertTrue(result)
                
                active_backend = get_active_backend()
                self.assertEqual(active_backend, BackendType.JAX)


if __name__ == '__main__':
    unittest.main()
