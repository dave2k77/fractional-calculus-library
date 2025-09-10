"""
Comprehensive tests for ML Registry module.
"""

import pytest
import tempfile
import json
import os
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

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
    
    def test_registry_initialization(self):
        """Test initializing model registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            assert registry.db_path == db_path
            assert registry.storage_path == Path(temp_dir)
    
    def test_register_model(self):
        """Test registering a new model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Create a simple PyTorch model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"},
                notes="Test model"
            )
            
            assert model_id is not None
            assert len(model_id) > 0
    
    def test_get_model(self):
        """Test getting a model from registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Register a model first
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"}
            )
            
            # Get the model
            retrieved_model = registry.get_model(model_id)
            assert retrieved_model is not None
            assert retrieved_model.model_id == model_id
    
    def test_get_model_not_found(self):
        """Test getting a non-existent model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Try to get a non-existent model
            retrieved_model = registry.get_model("non_existent_id")
            assert retrieved_model is None
    
    def test_search_models(self):
        """Test searching models by criteria."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Register a model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"}
            )
            
            # Search for models
            results = registry.search_models(name="Test Model")
            assert len(results) > 0
            assert any(result.model_id == model_id for result in results)
    
    def test_update_deployment_status(self):
        """Test updating model deployment status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Register a model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"}
            )
            
            # Update deployment status
            registry.update_deployment_status(model_id, "1.0.0", DeploymentStatus.PRODUCTION)
            
            # Verify the update
            updated_model = registry.get_model(model_id)
            assert updated_model.deployment_status == DeploymentStatus.PRODUCTION
    
    def test_promote_to_production(self):
        """Test promoting a model to production."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Register a model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"}
            )
            
            # Promote to production
            registry.promote_to_production(model_id, "1.0.0")
            
            # Verify promotion
            production_models = registry.get_production_models()
            assert len(production_models) > 0
            assert any(pm.model_id == model_id for pm in production_models)
    
    def test_delete_model(self):
        """Test deleting a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Register a model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"}
            )
            
            # Delete the model
            registry.delete_model(model_id)
            
            # Verify deletion
            deleted_model = registry.get_model(model_id)
            assert deleted_model is None
    
    def test_export_registry(self):
        """Test exporting registry to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Register a model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"}
            )
            
            # Export registry
            export_path = os.path.join(temp_dir, "export.json")
            registry.export_registry(export_path)
            
            # Verify export file exists
            assert os.path.exists(export_path)
            
            # Verify export content
            with open(export_path, 'r') as f:
                export_data = json.load(f)
            assert "models" in export_data
            assert "versions" in export_data
    
    def test_get_registry_summary(self):
        """Test getting registry summary statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_registry.db")
            registry = ModelRegistry(db_path, temp_dir)
            
            # Register a model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            model_id = registry.register_model(
                model=model,
                name="Test Model",
                version="1.0.0",
                description="A test model",
                author="Test Author",
                tags=["test"],
                framework="pytorch",
                model_type="fractional_neural_network",
                fractional_order=0.5,
                hyperparameters={"lr": 0.001},
                performance_metrics={"accuracy": 0.95},
                dataset_info={"size": 1000},
                dependencies={"torch": "1.9.0"}
            )
            
            # Get summary
            summary = registry.get_registry_summary()
            assert isinstance(summary, dict)
            assert "total_models" in summary
            assert "total_versions" in summary
            assert summary["total_models"] > 0