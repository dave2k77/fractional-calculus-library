"""
Development vs. Production Workflow Management

This module provides the workflow system that manages the transition of models
from development to production, including validation, quality gates, and deployment.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
from enum import Enum

from .registry import ModelRegistry, DeploymentStatus


class QualityMetric(Enum):
    """Quality metrics for model validation"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LOSS = "loss"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    MODEL_SIZE = "model_size"
    CUSTOM = "custom"


@dataclass
class QualityThreshold:
    """Quality threshold for a specific metric"""
    metric: QualityMetric
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = 0.05

    def check_threshold(self, value: float) -> bool:
        """Check if value meets threshold requirements"""
        # Normalize to plain floats to avoid dtype quirks
        v = float(value)
        t = float(self.target_value) if self.target_value is not None else None
        tol = float(self.tolerance)

        # If target specified and within tolerance, consider it a pass regardless of min/max bounds
        # Allow small numerical error margin
        if t is not None and abs(v - t) <= (tol + 1e-12):
            return True
        if self.min_value is not None and v < float(self.min_value):
            return False
        if self.max_value is not None and v > float(self.max_value):
            return False
        if t is not None:
            return abs(v - t) <= (tol + 1e-12)
        return True


@dataclass
class QualityGate:
    """Quality gate configuration"""
    name: str
    description: str
    thresholds: List[QualityThreshold]
    required: bool = True
    weight: float = 1.0

    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate quality gate against metrics"""
        results = {}
        passed = True

        for threshold in self.thresholds:
            metric_name = threshold.metric.value
            if metric_name in metrics:
                value = metrics[metric_name]
                threshold_passed = threshold.check_threshold(value)
                results[metric_name] = {
                    'value': value,
                    'passed': threshold_passed,
                    'threshold': threshold
                }
                if not threshold_passed:
                    passed = False
            else:
                results[metric_name] = {
                    'value': None,
                    'passed': False,
                    'threshold': threshold,
                    'error': 'Metric not found'
                }
                passed = False

        return {
            'gate_name': self.name,
            'passed': passed,
            'required': self.required,
            'weight': self.weight,
            'results': results
        }


class ModelValidator:
    """
    Model validation system

    This class provides comprehensive validation of models before they can be
    promoted to production, including quality gates and performance testing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.quality_gates = self._setup_default_quality_gates()
        self.logger = logging.getLogger(__name__)

    def _setup_default_quality_gates(self) -> List[QualityGate]:
        """Setup default quality gates"""
        gates = [
            QualityGate(
                name="Basic Performance",
                description="Basic performance requirements",
                thresholds=[
                    QualityThreshold(QualityMetric.ACCURACY, min_value=0.8),
                    QualityThreshold(QualityMetric.LOSS, max_value=0.3),
                ],
                required=True,
                weight=1.0
            ),
            QualityGate(
                name="Efficiency",
                description="Model efficiency requirements",
                thresholds=[
                    QualityThreshold(
                        QualityMetric.INFERENCE_TIME, max_value=100.0),  # ms
                    QualityThreshold(QualityMetric.MEMORY_USAGE,
                                     max_value=512.0),   # MB
                ],
                required=False,
                weight=0.7
            ),
            QualityGate(
                name="Model Size",
                description="Model size constraints",
                thresholds=[
                    QualityThreshold(QualityMetric.MODEL_SIZE,
                                     max_value=100.0),    # MB
                ],
                required=False,
                weight=0.5
            )
        ]
        return gates

    def add_quality_gate(self, gate: QualityGate):
        """Add a custom quality gate"""
        self.quality_gates.append(gate)

    def validate_model(
        self,
        model: torch.nn.Module,
        test_data: Any = None,
        test_labels: Any = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Validate a model against quality gates

        Args:
            model: Model to validate
            test_data: Test data for evaluation
            test_labels: Test labels for evaluation
            custom_metrics: Additional custom metrics

        Returns:
            Validation results
        """
        self.logger.info(
            f"Starting validation for model: {type(model).__name__}")

        # Calculate standard metrics or accept precomputed metrics dict
        if isinstance(test_data, dict) and test_labels is None:
            metrics = dict(test_data)
        else:
            metrics = self._calculate_standard_metrics(
                model, test_data, test_labels)

        # Add custom metrics if provided
        if custom_metrics:
            metrics.update(custom_metrics)

        # Evaluate quality gates
        gate_results = []
        overall_score = 0.0
        total_weight = 0.0

        for gate in self.quality_gates:
            gate_result = gate.evaluate(metrics)
            gate_results.append(gate_result)

            if gate_result['passed']:
                overall_score += gate.weight
            total_weight += gate.weight

        # Calculate final score
        final_score = self._compute_validation_score(gate_results)

        # Determine if validation passed
        required_gates_passed = all(
            gate['passed'] for gate in gate_results
            if gate['required']
        )

        validation_passed = required_gates_passed and final_score >= 0.7

        results = {
            'validation_passed': validation_passed,
            'final_score': final_score,
            'overall_score': overall_score,
            'total_weight': total_weight,
            'required_gates_passed': required_gates_passed,
            'metrics': metrics,
            'gate_results': gate_results,
            'timestamp': datetime.now().isoformat()
        }

        self.logger.info(
            f"Validation completed. Passed: {validation_passed}, Score: {final_score:.3f}")

        return self._generate_validation_report(gate_results, final_score, metrics)

    def _compute_validation_score(self, gate_results: List[Dict[str, Any]]) -> float:
        overall_score = 0.0
        total_weight = 0.0
        for gr in gate_results:
            if gr.get('passed'):
                overall_score += float(gr.get('weight', 1.0))
            total_weight += float(gr.get('weight', 1.0))
        return overall_score / total_weight if total_weight > 0 else 0.0

    def _generate_validation_report(self, gate_results: List[Dict[str, Any]], final_score: float, metrics: Dict[str, float]) -> Dict[str, Any]:
        required_gates_passed = all(gr.get('passed') for gr in gate_results if gr.get('required'))
        validation_passed = required_gates_passed and final_score >= 0.7
        return {
            'validation_passed': validation_passed,
            'final_score': final_score,
            'required_gates_passed': required_gates_passed,
            'metrics': metrics,
            'gate_results': gate_results,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_standard_metrics(
        self,
        model: torch.nn.Module,
        test_data: Any,
        test_labels: Any
    ) -> Dict[str, float]:
        """Calculate standard performance metrics"""
        model.eval()

        with torch.no_grad():
            # Measure inference time
            start_time = datetime.now()
            predictions = model(test_data)
            end_time = datetime.now()
            # Convert to ms
            inference_time = (end_time - start_time).total_seconds() * 1000

            # Calculate accuracy
            if (predictions.dim() > 1 and predictions.size(1) > 1 and
                    test_labels.dim() > 1 and test_labels.size(1) > 1):
                # Multi-dimensional regression - calculate R² score instead of
                # accuracy
                ss_res = torch.sum(
                    (test_labels - predictions) ** 2, dim=1).sum()
                ss_tot = torch.sum(
                    (test_labels - test_labels.mean(dim=0)) ** 2, dim=1).sum()
                r2 = 1 - (ss_res / ss_tot)
                accuracy = r2.item()  # Use R² as accuracy for regression
            elif hasattr(predictions, 'argmax') and predictions.dim() > 1:
                # Classification problem
                predicted_labels = predictions.argmax(dim=1)
                accuracy = (predicted_labels ==
                            test_labels).float().mean().item()
            else:
                # Single-dimensional regression - calculate R² score instead of
                # accuracy
                ss_res = torch.sum((test_labels - predictions) ** 2)
                ss_tot = torch.sum((test_labels - test_labels.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                accuracy = r2.item()  # Use R² as accuracy for regression

            # Calculate loss
            if hasattr(torch.nn.functional, 'cross_entropy'):
                loss = torch.nn.functional.cross_entropy(
                    predictions, test_labels).item()
            else:
                loss = torch.nn.functional.mse_loss(
                    predictions, test_labels).item()

            # Measure memory usage
            model_size = sum(p.numel() * p.element_size()
                             for p in model.parameters()) / (1024 * 1024)  # MB

            # Estimate memory usage during inference
            memory_usage = model_size * 2  # Rough estimate

        return {
            'accuracy': accuracy,
            'loss': loss,
            'inference_time': inference_time,
            'model_size': model_size,
            'memory_usage': memory_usage
        }


class DevelopmentWorkflow:
    """
    Development workflow management

    This class manages the development phase of models, including:
    - Model training and experimentation
    - Development validation
    - Model registration in development
    """

    def __init__(self, registry: Optional[ModelRegistry] = None, validator: Optional[ModelValidator] = None, model_registry: Optional[ModelRegistry] = None):
        # Accept both registry and model_registry for API compatibility
        self.registry = model_registry or registry or ModelRegistry()
        self.validator = validator or ModelValidator()
        self.logger = logging.getLogger(__name__)

    def register_development_model(
        self,
        model: torch.nn.Module,
        name: str,
        version: str,
        description: str,
        author: str,
        tags: List[str],
        fractional_order: float,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        dataset_info: Dict[str, Any],
        dependencies: Dict[str, str],
        notes: str = "",
        git_commit: str = "",
        git_branch: str = "dev"
    ) -> str:
        """Register a model in development"""
        self.logger.info(f"Registering development model: {name} v{version}")

        model_id = self.registry.register_model(
            model=model,
            name=name,
            version=version,
            description=description,
            author=author,
            tags=tags,
            framework="pytorch",
            model_type="fractional_neural_network",
            fractional_order=fractional_order,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            dataset_info=dataset_info,
            dependencies=dependencies,
            notes=notes,
            git_commit=git_commit,
            git_branch=git_branch
        )

        self.logger.info(f"Development model registered with ID: {model_id}")
        return model_id

    def validate_development_model(
        self,
        model_id: str,
        test_data: Any,
        test_labels: Any,
        custom_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Validate a development model"""
        self.logger.info(f"Validating development model: {model_id}")

        # Get model from registry
        model_metadata = self.registry.get_model(model_id)
        if not model_metadata:
            raise ValueError(f"Model not found: {model_id}")

        # Load model
        model_versions = self.registry.get_model_versions(model_id)
        if not model_versions:
            raise ValueError(f"No versions found for model: {model_id}")

        latest_version = model_versions[0]
        model = self.registry.reconstruct_model(
            model_id, latest_version.version)

        if model is None:
            raise ValueError(f"Failed to reconstruct model: {model_id}")

        # Validate model
        validation_results = self.validator.validate_model(
            model, test_data, test_labels, custom_metrics
        )

        # Update model status based on validation
        if validation_results['validation_passed']:
            self.registry.update_deployment_status(
                model_id, latest_version.version, DeploymentStatus.VALIDATION
            )
            self.logger.info(
                f"Model {model_id} passed validation and moved to VALIDATION status")
        else:
            self.registry.update_deployment_status(
                model_id, latest_version.version, DeploymentStatus.FAILED
            )
            self.logger.warning(f"Model {model_id} failed validation")

        return validation_results


class ProductionWorkflow:
    """
    Production workflow management

    This class manages the production deployment of models, including:
    - Production validation
    - Quality gate evaluation
    - Production deployment
    - Monitoring and rollback
    """

    def __init__(self, registry: Optional[ModelRegistry] = None, validator: Optional[ModelValidator] = None, model_registry: Optional[ModelRegistry] = None):
        self.registry = model_registry or registry or ModelRegistry()
        self.validator = validator or ModelValidator()
        self.logger = logging.getLogger(__name__)

    def promote_to_production(
        self,
        model_id: str,
        version: str,
        test_data: Any,
        test_labels: Any,
        custom_metrics: Optional[Dict[str, float]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Promote a model to production

        Args:
            model_id: Model ID to promote
            version: Version to promote
            test_data: Test data for final validation
            test_labels: Test labels for final validation
            custom_metrics: Additional metrics
            force: Force promotion even if validation fails

        Returns:
            Promotion results
        """
        self.logger.info(
            f"Promoting model {model_id} v{version} to production")

        # Get model metadata
        model_metadata = self.registry.get_model(model_id)
        if not model_metadata:
            raise ValueError(f"Model not found: {model_id}")

        # Get model version
        model_versions = self.registry.get_model_versions(model_id)
        target_version = None
        for mv in model_versions:
            if mv.version == version:
                target_version = mv
                break

        if not target_version:
            raise ValueError(
                f"Version {version} not found for model {model_id}")

        # Load model
        model = self.registry.reconstruct_model(model_id, version)

        if model is None:
            raise ValueError(f"Failed to reconstruct model: {model_id}")

        # Final validation
        validation_results = self.validator.validate_model(
            model, test_data, test_labels, custom_metrics
        )

        if not validation_results['validation_passed'] and not force:
            self.logger.error(f"Model {model_id} failed production validation")
            return {
                'promoted': False,
                'reason': 'Validation failed',
                'validation_results': validation_results
            }

        # Promote to production
        self.registry.promote_to_production(model_id, version)

        self.logger.info(
            f"Model {model_id} v{version} successfully promoted to production")

        return {
            'promoted': True,
            'model_id': model_id,
            'version': version,
            'validation_results': validation_results,
            'promoted_at': datetime.now().isoformat()
        }

    def rollback_production(
        self,
        model_id: str,
        target_version: str
    ) -> Dict[str, Any]:
        """
        Rollback production model to a previous version

        Args:
            model_id: Model ID to rollback
            target_version: Version to rollback to

        Returns:
            Rollback results
        """
        self.logger.info(
            f"Rolling back model {model_id} to version {target_version}")

        # Verify target version exists
        model_versions = self.registry.get_model_versions(model_id)
        target_version_exists = any(
            mv.version == target_version for mv in model_versions)

        if not target_version_exists:
            raise ValueError(
                f"Target version {target_version} not found for model {model_id}")

        # Promote target version to production
        self.registry.promote_to_production(model_id, target_version)

        self.logger.info(
            f"Model {model_id} successfully rolled back to version {target_version}")

        return {
            'rolled_back': True,
            'model_id': model_id,
            'target_version': target_version,
            'rolled_back_at': datetime.now().isoformat()
        }

    def get_production_status(self) -> Dict[str, Any]:
        """Get current production status"""
        production_models = self.registry.get_production_models()

        status = {
            'total_production_models': len(production_models),
            'models': []
        }

        for model_version in production_models:
            model_info = {
                'model_id': model_version.model_id,
                'name': model_version.metadata.name,
                'version': model_version.version,
                'deployment_status': model_version.metadata.deployment_status.value,
                'created_at': model_version.created_at.isoformat(),
                'created_by': model_version.created_by,
                'git_commit': model_version.git_commit,
                'git_branch': model_version.git_branch}
            status['models'].append(model_info)

        return status

    def monitor_production_models(
        self,
        monitoring_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monitor production models for performance degradation

        Args:
            monitoring_data: Current monitoring data

        Returns:
            Monitoring results and alerts
        """
        self.logger.info("Monitoring production models")

        alerts = []
        production_models = self.registry.get_production_models()

        for model_version in production_models:
            model_id = model_version.model_id

            if model_id in monitoring_data:
                current_metrics = monitoring_data[model_id]

                # Check for performance degradation
                if 'accuracy' in current_metrics:
                    if current_metrics['accuracy'] < 0.7:  # Threshold for alert
                        alerts.append({
                            'model_id': model_id,
                            'model_name': model_version.metadata.name,
                            'version': model_version.version,
                            'alert_type': 'performance_degradation',
                            'metric': 'accuracy',
                            'current_value': current_metrics['accuracy'],
                            'threshold': 0.7,
                            'timestamp': datetime.now().isoformat()
                        })

                if 'inference_time' in current_metrics:
                    if current_metrics['inference_time'] > 200:  # ms
                        alerts.append({
                            'model_id': model_id,
                            'model_name': model_version.metadata.name,
                            'version': model_version.version,
                            'alert_type': 'performance_degradation',
                            'metric': 'inference_time',
                            'current_value': current_metrics['inference_time'],
                            'threshold': 200,
                            'timestamp': datetime.now().isoformat()
                        })

        return {
            'monitoring_timestamp': datetime.now().isoformat(),
            'total_models_monitored': len(production_models),
            'alerts': alerts,
            'alert_count': len(alerts)
        }
