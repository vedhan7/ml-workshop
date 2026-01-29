"""
Model registry for versioning, selection, and management.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from src.config import get_settings
from src.database.connection import get_db_context
from src.database.models import ModelMetadata
from src.ml_pipeline.models import BaseForecaster

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """
    Manages model versioning, storage, and selection.
    
    Provides:
    - Model registration with metadata
    - Version management
    - Production model selection
    - Model retrieval by criteria
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model_dir = self.settings.model_dir
    
    def register(
        self,
        model: BaseForecaster,
        version: str = None,
        horizon_hours: int = 24,
        is_production: bool = False,
    ) -> ModelMetadata:
        """
        Register a trained model in the registry.
        
        Args:
            model: Trained forecaster model
            version: Version string (auto-generated if not provided)
            horizon_hours: Prediction horizon in hours
            is_production: Whether to set as production model
            
        Returns:
            ModelMetadata record
        """
        if not model.is_fitted:
            raise ValueError("Cannot register unfitted model")
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        
        # Save model to disk
        model_filename = f"{model.model_type}_{version}_{horizon_hours}h.pkl"
        model_path = self.model_dir / model_filename
        model.save(model_path)
        
        # Get metrics from model metadata
        metrics = model.metadata.get("metrics", {})
        
        # Create database record
        with get_db_context() as db:
            # If setting as production, unset current production model
            if is_production:
                db.query(ModelMetadata).filter(
                    ModelMetadata.model_type == model.model_type,
                    ModelMetadata.horizon_hours == horizon_hours,
                    ModelMetadata.is_production == True,
                ).update({"is_production": False})
            
            metadata = ModelMetadata(
                name=model.name,
                version=version,
                model_type=model.model_type,
                model_path=str(model_path),
                hyperparameters=model.hyperparameters,
                feature_columns=model.feature_columns,
                target_column=model.target_col,
                horizon_hours=horizon_hours,
                metrics=metrics,
                training_samples=model.metadata.get("training_samples"),
                is_active=True,
                is_production=is_production,
            )
            
            # Parse training dates if available
            if "training_start" in model.metadata:
                metadata.training_start = datetime.fromisoformat(model.metadata["training_start"])
            if "training_end" in model.metadata:
                metadata.training_end = datetime.fromisoformat(model.metadata["training_end"])
            
            db.add(metadata)
            db.commit()
            db.refresh(metadata)
            model_id = metadata.id
        
        logger.info(
            "model_registered",
            model_id=model_id,
            model_type=model.model_type,
            version=version,
            is_production=is_production,
        )
        
        return metadata
    
    def get_production_model(
        self,
        model_type: str = None,
        horizon_hours: int = 24,
    ) -> Optional[BaseForecaster]:
        """
        Get the current production model.
        
        Args:
            model_type: Optional model type filter
            horizon_hours: Prediction horizon
            
        Returns:
            Production model or None
        """
        with get_db_context() as db:
            query = db.query(ModelMetadata).filter(
                ModelMetadata.is_production == True,
                ModelMetadata.is_active == True,
                ModelMetadata.horizon_hours == horizon_hours,
            )
            
            if model_type:
                query = query.filter(ModelMetadata.model_type == model_type)
            
            metadata = query.first()
            
            if metadata is None:
                # Fallback: get best model by MAE
                return self.get_best_model(model_type, horizon_hours)
            
            return BaseForecaster.load(metadata.model_path)
    
    def get_best_model(
        self,
        model_type: str = None,
        horizon_hours: int = 24,
        metric: str = "mae",
    ) -> Optional[BaseForecaster]:
        """
        Get the best model by a specific metric.
        
        Args:
            model_type: Optional model type filter
            horizon_hours: Prediction horizon
            metric: Metric to use for selection (lower is better)
            
        Returns:
            Best model or None
        """
        with get_db_context() as db:
            query = db.query(ModelMetadata).filter(
                ModelMetadata.is_active == True,
                ModelMetadata.horizon_hours == horizon_hours,
            )
            
            if model_type:
                query = query.filter(ModelMetadata.model_type == model_type)
            
            models = query.all()
            
            if not models:
                return None
            
            # Find best by metric
            best_model = None
            best_score = float("inf")
            
            for m in models:
                if m.metrics and metric in m.metrics:
                    score = m.metrics[metric]
                    if score < best_score:
                        best_score = score
                        best_model = m
            
            if best_model:
                return BaseForecaster.load(best_model.model_path)
            
            # Fallback: return most recent
            latest = max(models, key=lambda m: m.created_at)
            return BaseForecaster.load(latest.model_path)
    
    def get_model_by_id(self, model_id: int) -> Optional[BaseForecaster]:
        """Get a specific model by its ID."""
        with get_db_context() as db:
            metadata = db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).first()
            
            if metadata is None:
                return None
            
            return BaseForecaster.load(metadata.model_path)
    
    def list_models(
        self,
        model_type: str = None,
        horizon_hours: int = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        """
        List available models.
        
        Returns:
            List of model metadata dictionaries
        """
        with get_db_context() as db:
            query = db.query(ModelMetadata)
            
            if active_only:
                query = query.filter(ModelMetadata.is_active == True)
            if model_type:
                query = query.filter(ModelMetadata.model_type == model_type)
            if horizon_hours:
                query = query.filter(ModelMetadata.horizon_hours == horizon_hours)
            
            query = query.order_by(ModelMetadata.created_at.desc()).limit(limit)
            models = query.all()
            
            return [
                {
                    "id": m.id,
                    "name": m.name,
                    "version": m.version,
                    "model_type": m.model_type,
                    "horizon_hours": m.horizon_hours,
                    "metrics": m.metrics,
                    "is_production": m.is_production,
                    "created_at": m.created_at.isoformat(),
                }
                for m in models
            ]
    
    def set_production(self, model_id: int) -> bool:
        """
        Set a model as the production model.
        
        Args:
            model_id: ID of the model to promote
            
        Returns:
            True if successful
        """
        with get_db_context() as db:
            model = db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).first()
            
            if model is None:
                return False
            
            # Unset current production for this type/horizon
            db.query(ModelMetadata).filter(
                ModelMetadata.model_type == model.model_type,
                ModelMetadata.horizon_hours == model.horizon_hours,
                ModelMetadata.is_production == True,
            ).update({"is_production": False})
            
            # Set new production
            model.is_production = True
            db.commit()
            
            logger.info(
                "production_model_set",
                model_id=model_id,
                model_type=model.model_type,
            )
            
            return True
    
    def deactivate(self, model_id: int) -> bool:
        """Deactivate a model (soft delete)."""
        with get_db_context() as db:
            result = db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).update({"is_active": False, "is_production": False})
            db.commit()
            
            return result > 0
    
    def cleanup_old_models(
        self,
        keep_recent: int = 5,
        keep_production: bool = True,
    ) -> int:
        """
        Clean up old model files and records.
        
        Args:
            keep_recent: Number of recent models to keep per type/horizon
            keep_production: Whether to always keep production models
            
        Returns:
            Number of models cleaned up
        """
        cleaned = 0
        
        with get_db_context() as db:
            # Group by model_type and horizon
            from sqlalchemy import func
            
            groups = db.query(
                ModelMetadata.model_type,
                ModelMetadata.horizon_hours,
            ).distinct().all()
            
            for model_type, horizon in groups:
                query = db.query(ModelMetadata).filter(
                    ModelMetadata.model_type == model_type,
                    ModelMetadata.horizon_hours == horizon,
                    ModelMetadata.is_active == True,
                )
                
                if keep_production:
                    query = query.filter(ModelMetadata.is_production == False)
                
                models = query.order_by(ModelMetadata.created_at.desc()).all()
                
                # Keep recent, delete rest
                for model in models[keep_recent:]:
                    # Delete file
                    model_path = Path(model.model_path)
                    if model_path.exists():
                        model_path.unlink()
                    
                    # Deactivate record
                    model.is_active = False
                    cleaned += 1
            
            db.commit()
        
        logger.info("cleanup_complete", cleaned=cleaned)
        return cleaned
