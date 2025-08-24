"""
Model Registry and Versioning System

This module provides a comprehensive model registry that tracks model versions,
metadata, and deployment status to support the development vs. production workflow.
"""

import json
import pickle
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np
from enum import Enum


class DeploymentStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for a model"""
    model_id: str
    version: str
    name: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    framework: str  # e.g., "pytorch", "tensorflow"
    model_type: str  # e.g., "fractional_neural_network", "fractional_attention"
    fractional_order: float
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    dependencies: Dict[str, str]
    file_size: int
    checksum: str
    deployment_status: DeploymentStatus
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['deployment_status'] = self.deployment_status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['deployment_status'] = DeploymentStatus(data['deployment_status'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Version information for a model"""
    version: str
    model_id: str
    metadata: ModelMetadata
    model_path: str
    config_path: str
    created_at: datetime
    created_by: str
    git_commit: str
    git_branch: str
    is_production: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['metadata'] = self.metadata.to_dict()
        data['created_at'] = self.created_at.isoformat()
        return data


class ModelRegistry:
    """
    Central model registry for tracking and managing models
    
    This class provides a comprehensive system for:
    - Storing model metadata and versions
    - Tracking deployment status
    - Managing development vs. production models
    - Version control and rollback capabilities
    """
    
    def __init__(self, db_path: str = "models/registry.db", storage_path: str = "models/"):
        self.db_path = db_path
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                author TEXT,
                created_at TEXT,
                updated_at TEXT,
                tags TEXT,
                framework TEXT,
                model_type TEXT,
                fractional_order REAL,
                hyperparameters TEXT,
                performance_metrics TEXT,
                dataset_info TEXT,
                dependencies TEXT,
                file_size INTEGER,
                checksum TEXT,
                deployment_status TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS versions (
                version TEXT,
                model_id TEXT,
                model_path TEXT,
                config_path TEXT,
                created_at TEXT,
                created_by TEXT,
                git_commit TEXT,
                git_branch TEXT,
                is_production BOOLEAN,
                PRIMARY KEY (version, model_id),
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().isoformat()
        unique_string = f"{name}_{version}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def register_model(
        self,
        model: torch.nn.Module,
        name: str,
        version: str,
        description: str,
        author: str,
        tags: List[str],
        framework: str,
        model_type: str,
        fractional_order: float,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        dataset_info: Dict[str, Any],
        dependencies: Dict[str, str],
        notes: str = "",
        git_commit: str = "",
        git_branch: str = "main"
    ) -> str:
        """
        Register a new model in the registry
        
        Args:
            model: PyTorch model to register
            name: Model name
            version: Model version
            description: Model description
            author: Model author
            tags: List of tags
            framework: Framework used
            model_type: Type of model
            fractional_order: Fractional order used
            hyperparameters: Model hyperparameters
            performance_metrics: Performance metrics
            dataset_info: Dataset information
            dependencies: Package dependencies
            notes: Additional notes
            git_commit: Git commit hash
            git_branch: Git branch name
            
        Returns:
            Model ID
        """
        model_id = self._generate_model_id(name, version)
        
        # Create storage directories
        model_dir = self.storage_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model files
        model_path = model_dir / f"{name}_v{version}.pth"
        config_path = model_dir / f"{name}_v{version}_config.json"
        
        # Save model state
        torch.save(model.state_dict(), model_path)
        
        # Save configuration
        config_data = {
            'name': name,
            'version': version,
            'model_type': model_type,
            'fractional_order': fractional_order,
            'hyperparameters': hyperparameters,
            'framework': framework
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Calculate file size and checksum
        file_size = model_path.stat().st_size
        checksum = self._calculate_checksum(str(model_path))
        
        # Create metadata
        now = datetime.now()
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            author=author,
            created_at=now,
            updated_at=now,
            tags=tags,
            framework=framework,
            model_type=model_type,
            fractional_order=fractional_order,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            dataset_info=dataset_info,
            dependencies=dependencies,
            file_size=file_size,
            checksum=checksum,
            deployment_status=DeploymentStatus.DEVELOPMENT,
            notes=notes
        )
        
        # Create version
        model_version = ModelVersion(
            version=version,
            model_id=model_id,
            metadata=metadata,
            model_path=str(model_path),
            config_path=str(config_path),
            created_at=now,
            created_by=author,
            git_commit=git_commit,
            git_branch=git_branch,
            is_production=False
        )
        
        # Store in database
        self._store_model(metadata)
        self._store_version(model_version)
        
        return model_id
    
    def _store_model(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.model_id,
            metadata.version,
            metadata.name,
            metadata.description,
            metadata.author,
            metadata.created_at.isoformat(),
            metadata.updated_at.isoformat(),
            json.dumps(metadata.tags),
            metadata.framework,
            metadata.model_type,
            metadata.fractional_order,
            json.dumps(metadata.hyperparameters),
            json.dumps(metadata.performance_metrics),
            json.dumps(metadata.dataset_info),
            json.dumps(metadata.dependencies),
            metadata.file_size,
            metadata.checksum,
            metadata.deployment_status.value,
            metadata.notes
        ))
        
        conn.commit()
        conn.close()
    
    def _store_version(self, model_version: ModelVersion):
        """Store model version in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_version.version,
            model_version.model_id,
            model_version.model_path,
            model_version.config_path,
            model_version.created_at.isoformat(),
            model_version.created_by,
            model_version.git_commit,
            model_version.git_branch,
            model_version.is_production
        ))
        
        conn.commit()
        conn.close()
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM models WHERE model_id = ?', (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        # Reconstruct metadata
        metadata = ModelMetadata(
            model_id=row[0],
            version=row[1],
            name=row[2],
            description=row[3],
            author=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            tags=json.loads(row[7]),
            framework=row[8],
            model_type=row[9],
            fractional_order=row[10],
            hyperparameters=json.loads(row[11]),
            performance_metrics=json.loads(row[12]),
            dataset_info=json.loads(row[13]),
            dependencies=json.loads(row[14]),
            file_size=row[15],
            checksum=row[16],
            deployment_status=DeploymentStatus(row[17]),
            notes=row[18]
        )
        
        return metadata
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM versions WHERE model_id = ? ORDER BY created_at DESC', (model_id,))
        rows = cursor.fetchall()
        conn.close()
        
        versions = []
        for row in rows:
            metadata = self.get_model(model_id)
            if metadata:
                model_version = ModelVersion(
                    version=row[0],
                    model_id=row[1],
                    metadata=metadata,
                    model_path=row[2],
                    config_path=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    created_by=row[5],
                    git_commit=row[6],
                    git_branch=row[7],
                    is_production=bool(row[8])
                )
                versions.append(model_version)
        
        return versions
    
    def update_deployment_status(self, model_id: str, version: str, status: DeploymentStatus):
        """Update deployment status of a model version"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update deployment status
        cursor.execute('''
            UPDATE versions SET is_production = ? WHERE model_id = ? AND version = ?
        ''', (status == DeploymentStatus.PRODUCTION, model_id, version))
        
        # Update model metadata
        cursor.execute('''
            UPDATE models SET deployment_status = ?, updated_at = ? WHERE model_id = ?
        ''', (status.value, datetime.now().isoformat(), model_id))
        
        conn.commit()
        conn.close()
    
    def promote_to_production(self, model_id: str, version: str):
        """Promote a model version to production"""
        # First, demote all other versions of this model
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE versions SET is_production = 0 WHERE model_id = ?
        ''', (model_id,))
        
        # Then promote the specified version
        cursor.execute('''
            UPDATE versions SET is_production = 1 WHERE model_id = ? AND version = ?
        ''', (model_id, version))
        
        # Update model status
        cursor.execute('''
            UPDATE models SET deployment_status = ?, updated_at = ? WHERE model_id = ?
        ''', (DeploymentStatus.PRODUCTION.value, datetime.now().isoformat(), model_id))
        
        conn.commit()
        conn.close()
    
    def get_production_models(self) -> List[ModelVersion]:
        """Get all production models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.*, m.name, m.description FROM versions v
            JOIN models m ON v.model_id = m.model_id
            WHERE v.is_production = 1
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        production_models = []
        for row in rows:
            metadata = self.get_model(row[1])
            if metadata:
                model_version = ModelVersion(
                    version=row[0],
                    model_id=row[1],
                    metadata=metadata,
                    model_path=row[2],
                    config_path=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    created_by=row[5],
                    git_commit=row[6],
                    git_branch=row[7],
                    is_production=bool(row[8])
                )
                production_models.append(model_version)
        
        return production_models
    
    def search_models(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        deployment_status: Optional[DeploymentStatus] = None,
        author: Optional[str] = None
    ) -> List[ModelMetadata]:
        """Search for models based on criteria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM models WHERE 1=1'
        params = []
        
        if name:
            query += ' AND name LIKE ?'
            params.append(f'%{name}%')
        
        if tags:
            for tag in tags:
                query += ' AND tags LIKE ?'
                params.append(f'%{tag}%')
        
        if model_type:
            query += ' AND model_type = ?'
            params.append(model_type)
        
        if deployment_status:
            query += ' AND deployment_status = ?'
            params.append(deployment_status.value)
        
        if author:
            query += ' AND author LIKE ?'
            params.append(f'%{author}%')
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        models = []
        for row in rows:
            metadata = ModelMetadata(
                model_id=row[0],
                version=row[1],
                name=row[2],
                description=row[3],
                author=row[4],
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                tags=json.loads(row[7]),
                framework=row[8],
                model_type=row[9],
                fractional_order=row[10],
                hyperparameters=json.loads(row[11]),
                performance_metrics=json.loads(row[12]),
                dataset_info=json.loads(row[13]),
                dependencies=json.loads(row[14]),
                file_size=row[15],
                checksum=row[16],
                deployment_status=DeploymentStatus(row[17]),
                notes=row[18]
            )
            models.append(metadata)
        
        return models
    
    def delete_model(self, model_id: str):
        """Delete a model and all its versions"""
        # Get model files
        versions = self.get_model_versions(model_id)
        
        # Delete files
        for version in versions:
            if Path(version.model_path).exists():
                Path(version.model_path).unlink()
            if Path(version.config_path).exists():
                Path(version.config_path).unlink()
        
        # Delete from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM versions WHERE model_id = ?', (model_id,))
        cursor.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
        
        conn.commit()
        conn.close()
    
    def export_registry(self, file_path: str):
        """Export registry to JSON file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all models
        cursor.execute('SELECT * FROM models')
        models = cursor.fetchall()
        
        # Get all versions
        cursor.execute('SELECT * FROM versions')
        versions = cursor.fetchall()
        
        conn.close()
        
        # Export data
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'models': models,
            'versions': versions
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total models
        cursor.execute('SELECT COUNT(*) FROM models')
        total_models = cursor.fetchone()[0]
        
        # Total versions
        cursor.execute('SELECT COUNT(*) FROM versions')
        total_versions = cursor.fetchone()[0]
        
        # Models by status
        cursor.execute('SELECT deployment_status, COUNT(*) FROM models GROUP BY deployment_status')
        status_counts = dict(cursor.fetchall())
        
        # Models by type
        cursor.execute('SELECT model_type, COUNT(*) FROM models GROUP BY model_type')
        type_counts = dict(cursor.fetchall())
        
        # Production models
        cursor.execute('SELECT COUNT(*) FROM versions WHERE is_production = 1')
        production_models = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'production_models': production_models,
            'status_distribution': status_counts,
            'type_distribution': type_counts
        }

    def reconstruct_model(self, model_id: str, version: str = None) -> Optional[torch.nn.Module]:
        """
        Reconstruct a model from its saved state dict and configuration
        
        Args:
            model_id: Model ID
            version: Model version (if None, uses latest)
            
        Returns:
            Reconstructed PyTorch model
        """
        # Get model metadata
        model_metadata = self.get_model(model_id)
        if not model_metadata:
            return None
        
        # Get model version
        model_versions = self.get_model_versions(model_id)
        if not model_versions:
            return None
        
        target_version = None
        if version is None:
            target_version = model_versions[0]  # Latest version
        else:
            for mv in model_versions:
                if mv.version == version:
                    target_version = mv
                    break
        
        if not target_version:
            return None
        
        # Load configuration
        config_path = Path(target_version.config_path)
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Load state dict
        model_path = Path(target_version.model_path)
        if not model_path.exists():
            return None
        
        state_dict = torch.load(model_path)
        
        # Reconstruct model based on type
        model_type = config_data.get('model_type', 'FractionalNeuralNetwork')
        hyperparameters = config_data.get('hyperparameters', {})
        
        if model_type == "fractional_neural_network":
            # Check if it's an adjoint-optimized network
            if "adjoint" in config_data.get('description', '').lower():
                from .adjoint_optimization import MemoryEfficientFractionalNetwork, AdjointConfig
                adjoint_config = AdjointConfig(
                    use_adjoint=True,
                    memory_efficient=False,  # Default to False for compatibility
                    checkpoint_frequency=5
                )
                model = MemoryEfficientFractionalNetwork(
                    input_size=hyperparameters.get('input_size', 10),
                    hidden_sizes=hyperparameters.get('hidden_sizes', [64, 32]),
                    output_size=hyperparameters.get('output_size', 1),  # Use saved output_size
                    fractional_order=hyperparameters.get('fractional_order', 0.5),
                    adjoint_config=adjoint_config
                )
            else:
                from .core import FractionalNeuralNetwork
                model = FractionalNeuralNetwork(
                    input_size=hyperparameters.get('input_size', 10),
                    hidden_sizes=hyperparameters.get('hidden_sizes', [64, 32]),
                    output_size=hyperparameters.get('output_size', 1),  # Use saved output_size
                    fractional_order=hyperparameters.get('fractional_order', 0.5)
                )
        else:
            # For other model types, create a simple placeholder
            # In practice, you'd want to handle each model type specifically
            model = torch.nn.Sequential(
                torch.nn.Linear(hyperparameters.get('input_size', 10), 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, hyperparameters.get('output_size', 1))
            )
        
        # Load the state dict
        model.load_state_dict(state_dict)
        return model
