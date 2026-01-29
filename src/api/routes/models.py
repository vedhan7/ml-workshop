"""
Models endpoint for listing and managing trained models.
"""

from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_model_registry
from src.api.schemas import (
    ModelInfo,
    ModelListResponse,
    ModelDetailResponse,
    ErrorResponse,
)
from src.ml_pipeline.registry import ModelRegistry

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get(
    "",
    response_model=ModelListResponse,
)
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    horizon_hours: Optional[int] = Query(None, description="Filter by prediction horizon"),
    active_only: bool = Query(True, description="Only show active models"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of models to return"),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    List all available trained models.
    
    Returns a list of models with their basic information and metrics.
    """
    models = registry.list_models(
        model_type=model_type,
        horizon_hours=horizon_hours,
        active_only=active_only,
        limit=limit,
    )
    
    model_infos = [
        ModelInfo(
            id=m["id"],
            name=m["name"],
            version=m["version"],
            model_type=m["model_type"],
            horizon_hours=m["horizon_hours"],
            is_production=m["is_production"],
            metrics=m.get("metrics"),
            created_at=datetime.fromisoformat(m["created_at"]),
        )
        for m in models
    ]
    
    return ModelListResponse(
        models=model_infos,
        total=len(model_infos),
    )


@router.get(
    "/{model_id}",
    response_model=ModelDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_model_detail(
    model_id: int,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Get detailed information about a specific model.
    
    Includes hyperparameters, feature columns, and feature importance.
    """
    from src.database.connection import get_db_context
    from src.database.models import ModelMetadata
    
    with get_db_context() as db:
        metadata = db.query(ModelMetadata).filter(
            ModelMetadata.id == model_id
        ).first()
        
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Try to load model for feature importance
        feature_importance = None
        try:
            model = registry.get_model_by_id(model_id)
            if model and hasattr(model, "get_feature_importance"):
                feature_importance = model.get_feature_importance()
            elif model and "feature_importance" in model.metadata:
                feature_importance = dict(
                    sorted(
                        model.metadata["feature_importance"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:20]
                )
        except Exception as e:
            logger.warning("failed_to_load_model", error=str(e))
        
        return ModelDetailResponse(
            id=metadata.id,
            name=metadata.name,
            version=metadata.version,
            model_type=metadata.model_type,
            horizon_hours=metadata.horizon_hours,
            is_production=metadata.is_production,
            is_active=metadata.is_active,
            metrics=metadata.metrics,
            hyperparameters=metadata.hyperparameters,
            feature_columns=metadata.feature_columns,
            training_samples=metadata.training_samples,
            training_start=metadata.training_start,
            training_end=metadata.training_end,
            created_at=metadata.created_at,
            feature_importance=feature_importance,
        )


@router.post(
    "/{model_id}/set-production",
    responses={404: {"model": ErrorResponse}},
)
async def set_production_model(
    model_id: int,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Set a model as the production model.
    
    This will be used for predictions when no specific model is requested.
    """
    success = registry.set_production(model_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "message": f"Model {model_id} set as production",
        "model_id": model_id,
    }


@router.delete(
    "/{model_id}",
    responses={404: {"model": ErrorResponse}},
)
async def deactivate_model(
    model_id: int,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Deactivate a model (soft delete).
    
    The model will no longer appear in listings or be available for predictions.
    """
    success = registry.deactivate(model_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "message": f"Model {model_id} deactivated",
        "model_id": model_id,
    }


@router.get("/production/{horizon_hours}")
async def get_production_model(
    horizon_hours: int,
    model_type: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Get the current production model for a specific horizon.
    """
    from src.database.connection import get_db_context
    from src.database.models import ModelMetadata
    
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
            return {
                "message": "No production model set for this horizon",
                "horizon_hours": horizon_hours,
                "has_production_model": False,
            }
        
        return {
            "model_id": metadata.id,
            "name": metadata.name,
            "model_type": metadata.model_type,
            "horizon_hours": horizon_hours,
            "has_production_model": True,
            "metrics": metadata.metrics,
        }
