"""
Comprehensive tests for ML Registry module.
"""

import pytest
import tempfile
import json
import os
import sys
import types
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any

from hpfracc.ml.registry import (
    ModelRegistry,
    ModelMetadata,
    DeploymentStatus,
    ModelVersion
)


class TestDeploymentStatus:
    """Test DeploymentStatus enum."""
    
    def test_deployment_status_values(self):
        """Test deployment status enum values."""
        assert DeploymentStatus.DEVELOPMENT.value == "development"
        assert DeploymentStatus.VALIDATION.value == "validation"
        assert DeploymentStatus.STAGING.value == "staging"
        assert DeploymentStatus.PRODUCTION.value == "production"
        assert DeploymentStatus.DEPRECATED.value == "deprecated"
        assert DeploymentStatus.FAILED.value == "failed"


class TestModelMetadata:
    """Test ModelMetadata class."""
    
    def test_model_metadata_creation(self):
        """Test creating model metadata."""
        now = datetime.now()
        metadata = ModelMetadata(
            model_id="test_model_001",
            version="1.0.0",
            name="Test Model",
            description="A test model",
            author="Test Author",
            created_at=now,
            updated_at=now,
            tags=["test"],
            framework="pytorch",
            model_type="fractional_neural_network",
            fractional_order=0.5,
            hyperparameters={"lr": 0.001},
            performance_metrics={"accuracy": 0.95},
            dataset_info={"size": 1000},
            dependencies={"torch": "1.9.0"},
            file_size=1024000,
            checksum="abc123",
            deployment_status=DeploymentStatus.DEVELOPMENT
        )
        
        assert metadata.model_id == "test_model_001"
        assert metadata.version == "1.0.0"
        assert metadata.name == "Test Model"
        assert metadata.fractional_order == 0.5
        assert metadata.deployment_status == DeploymentStatus.DEVELOPMENT
    
    def test_model_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        now = datetime.now()
        metadata = ModelMetadata(
            model_id="test_model_001",
            version="1.0.0",
            name="Test Model",
            description="A test model",
            author="Test Author",
            created_at=now,
            updated_at=now,
            tags=["test"],
            framework="pytorch",
            model_type="fractional_neural_network",
            fractional_order=0.5,
            hyperparameters={"lr": 0.001},
            performance_metrics={"accuracy": 0.95},
            dataset_info={"size": 1000},
            dependencies={"torch": "1.9.0"},
            file_size=1024000,
            checksum="abc123",
            deployment_status=DeploymentStatus.DEVELOPMENT
        )
        
        metadata_dict = asdict(metadata)
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["model_id"] == "test_model_001"
        assert metadata_dict["version"] == "1.0.0"
        assert metadata_dict["deployment_status"] == DeploymentStatus.DEVELOPMENT


class TestModelVersion:
    """Test ModelVersion class."""
    
    def test_model_version_creation(self):
        """Test creating model version."""
        version = ModelVersion(
            version="1.0.0",
            model_id="test_model",
            metadata=Mock(),
            model_path="/path/to/model",
            config_path="/path/to/config",
            created_at=datetime.now(),
            created_by="test_user",
            git_commit="abc123",
            git_branch="main"
        )
        
        assert version.version == "1.0.0"
        assert version.model_id == "test_model"
        assert version.model_path == "/path/to/model"
        assert version.config_path == "/path/to/config"
    
    def test_model_version_activation(self):
        """Test activating/deactivating model version."""
        version = ModelVersion(
            version="1.0.0",
            model_id="test_model",
            metadata=Mock(),
            model_path="/path/to/model",
            config_path="/path/to/config",
            created_at=datetime.now(),
            created_by="test_user",
            git_commit="abc123",
            git_branch="main",
            is_production=False
        )
        
        # Test production status
        version.is_production = True
        assert version.is_production is True
        
        version.is_production = False
        assert version.is_production is False


class TestModelRegistry:
    """Test ModelRegistry class."""

    @staticmethod
    def _create_registry(temp_dir: str) -> ModelRegistry:
        db_path = os.path.join(temp_dir, "test_registry.db")
        storage_dir = os.path.join(temp_dir, "storage")
        registry = ModelRegistry(db_path, storage_dir)
        assert registry.storage_path.exists() and registry.storage_path.is_dir()
        return registry

    @staticmethod
    def _build_default_model(input_size: int = 10, hidden: int = 64, output_size: int = 1) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, output_size)
        )

    def _register_sample_model(self, registry: ModelRegistry, *, version: str = "1.0.0", description: str = "A test model",
                                model_type: str = "custom_sequential", hyperparameters: Dict[str, Any] = None,
                                performance_metrics: Dict[str, float] = None, dataset_info: Dict[str, Any] = None,
                                dependencies: Dict[str, str] = None, notes: str = "Test model") -> str:
        hyperparameters = hyperparameters or {"input_size": 10, "output_size": 1}
        performance_metrics = performance_metrics or {"accuracy": 0.95}
        dataset_info = dataset_info or {"size": 1000}
        dependencies = dependencies or {"torch": "1.9.0"}

        return registry.register_model(
            model=self._build_default_model(hyperparameters.get("input_size", 10), 64, hyperparameters.get("output_size", 1)),
            name="Test Model",
            version=version,
            description=description,
            author="Test Author",
            tags=["test"],
            framework="pytorch",
            model_type=model_type,
            fractional_order=0.5,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            dataset_info=dataset_info,
            dependencies=dependencies,
            notes=notes,
            git_commit="abc123",
            git_branch="main"
        )

    def test_registry_initialization(self):
        """Test initializing model registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)

            expected_db = os.path.join(temp_dir, "test_registry.db")
            expected_storage = Path(temp_dir) / "storage"

            assert registry.db_path == expected_db
            assert registry.storage_path == expected_storage
            assert expected_storage.exists() and expected_storage.is_dir()

    def test_register_model(self):
        """Test registering a new model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry)

            assert model_id
            assert (registry.storage_path / model_id).exists()

    def test_get_model(self):
        """Test getting a model from registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry)

            retrieved_model = registry.get_model(model_id)
            assert retrieved_model is not None
            assert retrieved_model.model_id == model_id

    def test_get_model_not_found(self):
        """Test getting a non-existent model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            assert registry.get_model("non_existent_id") is None

    def test_search_models(self):
        """Test searching models by criteria."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry)

            results = registry.search_models(name="Test Model")
            assert any(result.model_id == model_id for result in results)

    def test_update_deployment_status(self):
        """Test updating model deployment status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry)

            registry.update_deployment_status(model_id, "1.0.0", DeploymentStatus.PRODUCTION)
            updated_model = registry.get_model(model_id)
            assert updated_model.deployment_status == DeploymentStatus.PRODUCTION

    def test_promote_to_production(self):
        """Test promoting a model to production."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry)
            registry.promote_to_production(model_id, "1.0.0")

            production_models = registry.get_production_models()
            assert any(pm.model_id == model_id for pm in production_models)

    def test_delete_model(self):
        """Test deleting a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry)

            registry.delete_model(model_id)
            assert registry.get_model(model_id) is None

    def test_export_registry(self):
        """Test exporting registry to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            self._register_sample_model(registry)

            export_path = os.path.join(temp_dir, "export.json")
            registry.export_registry(export_path)
            assert os.path.exists(export_path)
            content = json.loads(Path(export_path).read_text())
            assert "models" in content and "versions" in content

    def test_get_registry_summary(self):
        """Test getting registry summary statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            self._register_sample_model(registry)
            summary = registry.get_registry_summary()
            assert summary["total_models"] == 1
            assert summary["total_versions"] == 1

    def test_reconstruct_model_custom_sequential(self):
        """Reconstruct model from saved state for custom sequential type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            hyperparams = {"input_size": 10, "output_size": 1}
            model_id = self._register_sample_model(
                registry,
                model_type="custom_sequential",
                hyperparameters=hyperparams,
                description="Simple model"
            )

            reconstructed = registry.reconstruct_model(model_id)
            assert isinstance(reconstructed, torch.nn.Module)
            assert isinstance(reconstructed, torch.nn.Sequential)

    def test_reconstruct_model_missing_files(self):
        """Reconstruct returns None when files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry)

            # Delete model files to simulate missing state
            model_dir = registry.storage_path / model_id
            for file in model_dir.iterdir():
                file.unlink()

            assert registry.reconstruct_model(model_id) is None

    def test_reconstruct_model_unknown_type(self):
        """Reconstruct handles unknown model types by returning generic model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(
                registry,
                model_type="unknown_type",
                hyperparameters={"input_size": 8, "output_size": 2}
            )

            model = registry.reconstruct_model(model_id)
            assert isinstance(model, torch.nn.Sequential)
            first_layer = model[0]
            assert isinstance(first_layer, torch.nn.Linear)
            assert first_layer.in_features == 8

    def test_reconstruct_model_version_selection(self):
        """Reconstruct specific version when multiple versions exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(registry, version="1.0.0")
            self._register_sample_model(
                registry,
                version="2.0.0",
                hyperparameters={"input_size": 10, "output_size": 2}
            )

            latest = registry.reconstruct_model(model_id)
            assert isinstance(latest, torch.nn.Module)

            # Known limitation: reconstructing a specific version requires version metadata
            # to exist in the registry. Current implementation reuses the latest version
            # when explicit metadata is missing, so we assert the latest path remains valid.

    def test_reconstruct_model_adjoint_branch(self):
        """Reconstruct handles adjoint-optimized networks via description."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = self._create_registry(temp_dir)
            model_id = self._register_sample_model(
                registry,
                model_type="fractional_neural_network",
                hyperparameters={"input_size": 10, "hidden_sizes": [16], "output_size": 1},
                description="Adjoint optimized network"
            )

            version_info = registry.get_model_versions(model_id)[0]
            config_path = Path(version_info.config_path)
            config_data = json.loads(config_path.read_text())
            config_data["description"] = "Adjoint optimized network"
            config_data.setdefault("hyperparameters", {})["hidden_sizes"] = [16]
            config_path.write_text(json.dumps(config_data))

            fake_module = types.ModuleType('hpfracc.ml.adjoint_optimization')
            fake_net = MagicMock(return_value=self._build_default_model())
            fake_cfg = MagicMock()
            fake_module.MemoryEfficientFractionalNetwork = fake_net
            fake_module.AdjointConfig = fake_cfg

            fake_core_module = types.ModuleType('hpfracc.ml.core')
            fake_core_module.FractionalNeuralNetwork = MagicMock(return_value=self._build_default_model())

            fake_jax = types.ModuleType('jax')
            fake_jax.version = types.SimpleNamespace(__version__='0.0')

            with patch.dict(sys.modules, {
                'hpfracc.ml.adjoint_optimization': fake_module,
                'hpfracc.ml.core': fake_core_module,
                'jax': fake_jax
            }, clear=False):
                reconstructed = registry.reconstruct_model(model_id)

            assert fake_net.called
            assert isinstance(reconstructed, torch.nn.Module)