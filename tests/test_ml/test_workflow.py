"""
Tests for workflow management module.

This module tests the workflow management functionality including quality gates,
model validation, and development/production workflows.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from hpfracc.ml.workflow import (
    QualityMetric,
    QualityThreshold,
    QualityGate,
    ModelValidator,
    DevelopmentWorkflow,
    ProductionWorkflow
)
from hpfracc.ml.registry import ModelRegistry


class TestQualityMetric:
    """Test QualityMetric enum."""

    def test_quality_metric_values(self):
        """Test that quality metrics have correct values."""
        assert QualityMetric.ACCURACY.value == "accuracy"
        assert QualityMetric.PRECISION.value == "precision"
        assert QualityMetric.RECALL.value == "recall"
        assert QualityMetric.F1_SCORE.value == "f1_score"
        assert QualityMetric.LOSS.value == "loss"
        assert QualityMetric.INFERENCE_TIME.value == "inference_time"
        assert QualityMetric.MEMORY_USAGE.value == "memory_usage"
        assert QualityMetric.MODEL_SIZE.value == "model_size"
        assert QualityMetric.CUSTOM.value == "custom"


class TestQualityThreshold:
    """Test QualityThreshold class."""

    def test_quality_threshold_creation(self):
        """Test creating a quality threshold."""
        threshold = QualityThreshold(
            metric=QualityMetric.ACCURACY,
            min_value=0.8,
            max_value=0.95,
            target_value=0.9,
            tolerance=0.05
        )

        assert threshold.metric == QualityMetric.ACCURACY
        assert threshold.min_value == 0.8
        assert threshold.max_value == 0.95
        assert threshold.target_value == 0.9
        assert threshold.tolerance == 0.05

    def test_quality_threshold_defaults(self):
        """Test quality threshold with default values."""
        threshold = QualityThreshold(metric=QualityMetric.ACCURACY)

        assert threshold.metric == QualityMetric.ACCURACY
        assert threshold.min_value is None
        assert threshold.max_value is None
        assert threshold.target_value is None
        assert threshold.tolerance == 0.05

    def test_check_threshold_min_value(self):
        """Test threshold checking with min_value."""
        threshold = QualityThreshold(
            metric=QualityMetric.ACCURACY,
            min_value=0.8
        )

        assert threshold.check_threshold(0.9) == True
        assert threshold.check_threshold(0.8) == True
        assert threshold.check_threshold(0.7) == False

    def test_check_threshold_max_value(self):
        """Test threshold checking with max_value."""
        threshold = QualityThreshold(
            metric=QualityMetric.ACCURACY,
            max_value=0.95
        )

        assert threshold.check_threshold(0.9) == True
        assert threshold.check_threshold(0.95) == True
        assert threshold.check_threshold(0.96) == False

    def test_check_threshold_target_value(self):
        """Test threshold checking with target_value."""
        threshold = QualityThreshold(
            metric=QualityMetric.ACCURACY,
            target_value=0.9,
            tolerance=0.05
        )

        assert threshold.check_threshold(0.9) == True
        assert threshold.check_threshold(0.88) == True  # Within tolerance
        assert threshold.check_threshold(0.95) == True  # Within tolerance
        assert threshold.check_threshold(0.8) == False  # Outside tolerance
        assert threshold.check_threshold(1.0) == False  # Outside tolerance

    def test_check_threshold_combined(self):
        """Test threshold checking with multiple constraints."""
        threshold = QualityThreshold(
            metric=QualityMetric.ACCURACY,
            min_value=0.8,
            max_value=0.95,
            target_value=0.9,
            tolerance=0.05
        )

        assert threshold.check_threshold(0.9) == True
        assert threshold.check_threshold(0.88) == True  # Within tolerance
        assert threshold.check_threshold(0.95) == True
        assert threshold.check_threshold(0.8) == False  # Below target tolerance
        assert threshold.check_threshold(0.75) == False  # Below min
        assert threshold.check_threshold(0.96) == False  # Above max
        assert threshold.check_threshold(0.7) == False  # Below min


class TestQualityGate:
    """Test QualityGate class."""

    def test_quality_gate_creation(self):
        """Test creating a quality gate."""
        thresholds = [
            QualityThreshold(metric=QualityMetric.ACCURACY, min_value=0.8),
            QualityThreshold(metric=QualityMetric.LOSS, max_value=0.3)
        ]
        
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            thresholds=thresholds,
            required=True,
            weight=0.8
        )

        assert gate.name == "test_gate"
        assert gate.description == "Test gate"
        assert gate.thresholds == thresholds
        assert gate.required == True
        assert gate.weight == 0.8

    def test_quality_gate_defaults(self):
        """Test quality gate with default values."""
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            thresholds=[]
        )

        assert gate.name == "test_gate"
        assert gate.description == "Test gate"
        assert gate.thresholds == []
        assert gate.required == True
        assert gate.weight == 1.0

    def test_evaluate_all_passed(self):
        """Test quality gate evaluation when all thresholds pass."""
        thresholds = [
            QualityThreshold(metric=QualityMetric.ACCURACY, min_value=0.8),
            QualityThreshold(metric=QualityMetric.LOSS, max_value=0.3)
        ]
        
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            thresholds=thresholds
        )

        metrics = {"accuracy": 0.9, "loss": 0.2}
        result = gate.evaluate(metrics)

        assert result["gate_name"] == "test_gate"
        assert result["passed"] == True
        assert result["required"] == True
        assert "accuracy" in result["results"]
        assert "loss" in result["results"]

    def test_evaluate_some_failed(self):
        """Test quality gate evaluation when some thresholds fail."""
        thresholds = [
            QualityThreshold(metric=QualityMetric.ACCURACY, min_value=0.8),
            QualityThreshold(metric=QualityMetric.LOSS, max_value=0.3)
        ]
        
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            thresholds=thresholds
        )

        metrics = {"accuracy": 0.9, "loss": 0.4}  # Loss fails
        result = gate.evaluate(metrics)

        assert result["gate_name"] == "test_gate"
        assert result["passed"] == False
        assert result["required"] == True

    def test_evaluate_missing_metric(self):
        """Test quality gate evaluation with missing metric."""
        thresholds = [
            QualityThreshold(metric=QualityMetric.ACCURACY, min_value=0.8)
        ]
        
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            thresholds=thresholds
        )

        metrics = {"loss": 0.2}  # Missing accuracy
        result = gate.evaluate(metrics)

        assert result["gate_name"] == "test_gate"
        assert result["passed"] == False
        assert result["required"] == True


class TestModelValidator:
    """Test ModelValidator class."""

    def test_init_default(self):
        """Test ModelValidator initialization with default parameters."""
        validator = ModelValidator()

        assert len(validator.quality_gates) == 3  # Default gates
        assert validator.config == {}

    def test_init_custom(self):
        """Test ModelValidator initialization with custom parameters."""
        validator = ModelValidator(config={"test": "value"})

        assert validator.config == {"test": "value"}
        assert len(validator.quality_gates) == 3  # Still has default gates

    def test_add_quality_gate(self):
        """Test adding a quality gate."""
        validator = ModelValidator()
        initial_count = len(validator.quality_gates)
        
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            thresholds=[QualityThreshold(metric=QualityMetric.ACCURACY, min_value=0.8)]
        )
        
        validator.add_quality_gate(gate)
        
        assert len(validator.quality_gates) == initial_count + 1
        assert validator.quality_gates[-1] == gate

    def test_validate_model_basic(self):
        """Test basic model validation."""
        validator = ModelValidator()

        # Mock model
        model = nn.Linear(10, 5)

        # Mock test data
        test_data = torch.randn(100, 10)
        test_labels = torch.randn(100, 5)

        # Mock custom metrics
        custom_metrics = {"accuracy": 0.9}

        result = validator.validate_model(model, test_data, test_labels, custom_metrics)

        assert "validation_passed" in result
        assert "gate_results" in result
        assert "metrics" in result
        assert "final_score" in result

    def test_validate_model_failed_gate(self):
        """Test model validation when quality gate fails."""
        validator = ModelValidator()

        # Add a quality gate with high threshold
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            thresholds=[QualityThreshold(metric=QualityMetric.ACCURACY, min_value=0.95)]
        )
        validator.add_quality_gate(gate)

        # Mock model
        model = nn.Linear(10, 5)

        # Mock test data
        test_data = torch.randn(100, 10)
        test_labels = torch.randn(100, 5)

        # Mock custom metrics with low accuracy
        custom_metrics = {"accuracy": 0.8}

        result = validator.validate_model(model, test_data, test_labels, custom_metrics)

        assert "validation_passed" in result
        assert "gate_results" in result
        assert "metrics" in result
        assert "final_score" in result


class TestDevelopmentWorkflow:
    """Test DevelopmentWorkflow class."""

    def test_init(self):
        """Test DevelopmentWorkflow initialization."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        
        workflow = DevelopmentWorkflow(registry, validator)

        assert workflow.registry == registry
        assert workflow.validator == validator

    def test_register_development_model(self):
        """Test registering a development model."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        workflow = DevelopmentWorkflow(registry, validator)

        # Mock registry.register_model to return a model ID
        registry.register_model.return_value = "test_model_id"

        model = nn.Linear(10, 5)
        model_id = workflow.register_development_model(
            model=model,
            name="test_model",
            version="1.0.0",
            description="Test model",
            author="test_author",
            tags=["test"],
            fractional_order=0.5,
            hyperparameters={},
            performance_metrics={},
            dataset_info={},
            dependencies={}
        )

        assert model_id == "test_model_id"
        registry.register_model.assert_called_once()

    def test_validate_development_model(self):
        """Test validating a development model."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        workflow = DevelopmentWorkflow(registry, validator)

        # Mock registry methods
        mock_metadata = {"name": "test_model"}
        mock_version = Mock()
        mock_version.version = "1.0.0"
        registry.get_model.return_value = mock_metadata
        registry.get_model_versions.return_value = [mock_version]
        registry.reconstruct_model.return_value = nn.Linear(10, 5)

        # Mock validator
        validation_result = {"validation_passed": True}
        validator.validate_model.return_value = validation_result

        result = workflow.validate_development_model(
            model_id="test_model_id",
            test_data=torch.randn(100, 10),
            test_labels=torch.randn(100, 5)
        )

        assert result == validation_result
        validator.validate_model.assert_called_once()

    def test_validate_development_model_not_found(self):
        """Test validating a non-existent development model."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        workflow = DevelopmentWorkflow(registry, validator)

        # Mock registry to return None (model not found)
        registry.get_model.return_value = None

        with pytest.raises(ValueError, match="Model not found"):
            workflow.validate_development_model(
                model_id="nonexistent_model",
                test_data=torch.randn(100, 10),
                test_labels=torch.randn(100, 5)
            )


class TestProductionWorkflow:
    """Test ProductionWorkflow class."""

    def test_init(self):
        """Test ProductionWorkflow initialization."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        
        workflow = ProductionWorkflow(registry, validator)

        assert workflow.registry == registry
        assert workflow.validator == validator

    def test_promote_to_production(self):
        """Test promoting a model to production."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        workflow = ProductionWorkflow(registry, validator)

        # Mock registry methods
        mock_metadata = {"name": "test_model"}
        mock_version = Mock()
        mock_version.version = "1.0.0"
        registry.get_model.return_value = mock_metadata
        registry.get_model_versions.return_value = [mock_version]
        registry.reconstruct_model.return_value = nn.Linear(10, 5)

        # Mock validator
        validation_result = {"validation_passed": True}
        validator.validate_model.return_value = validation_result

        result = workflow.promote_to_production(
            model_id="test_model_id",
            version="1.0.0",
            test_data=torch.randn(100, 10),
            test_labels=torch.randn(100, 5)
        )

        assert "promoted" in result
        assert result["promoted"] == True
        assert "validation_results" in result

    def test_promote_to_production_validation_failed(self):
        """Test promoting a model that fails validation."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        workflow = ProductionWorkflow(registry, validator)

        # Mock registry methods
        mock_metadata = {"name": "test_model"}
        mock_version = Mock()
        mock_version.version = "1.0.0"
        registry.get_model.return_value = mock_metadata
        registry.get_model_versions.return_value = [mock_version]
        registry.reconstruct_model.return_value = nn.Linear(10, 5)

        # Mock validator to return failed validation
        validation_result = {"validation_passed": False}
        validator.validate_model.return_value = validation_result

        result = workflow.promote_to_production(
            model_id="test_model_id",
            version="1.0.0",
            test_data=torch.randn(100, 10),
            test_labels=torch.randn(100, 5)
        )

        assert "promoted" in result
        assert result["promoted"] == False
        assert "validation_results" in result


class TestWorkflowIntegration:
    """Test integration between development and production workflows."""

    def test_workflow_integration(self):
        """Test that workflows can be instantiated and work together."""
        registry = Mock(spec=ModelRegistry)
        validator = Mock(spec=ModelValidator)
        
        # Test that both workflows can be created
        dev_workflow = DevelopmentWorkflow(registry, validator)
        prod_workflow = ProductionWorkflow(registry, validator)

        assert dev_workflow is not None
        assert prod_workflow is not None
        assert dev_workflow != prod_workflow

        # Test that they have the expected attributes
        assert hasattr(dev_workflow, 'registry')
        assert hasattr(dev_workflow, 'validator')
        assert hasattr(prod_workflow, 'registry')
        assert hasattr(prod_workflow, 'validator')