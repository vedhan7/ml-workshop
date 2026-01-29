"""
Metrics endpoint for system and model performance monitoring.
"""

from datetime import datetime, timedelta
from typing import Optional

import structlog
from fastapi import APIRouter, Depends

from src.api.dependencies import get_model_registry
from src.api.schemas import (
    MetricsResponse,
    SystemMetrics,
    DataMetrics,
    ModelPerformanceMetrics,
)
from src.database.connection import get_db_context
from src.database.models import ElectricityConsumption, ModelMetadata, Prediction
from src.ml_pipeline.registry import ModelRegistry

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("", response_model=MetricsResponse)
async def get_metrics(
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Get comprehensive system and model metrics.
    """
    with get_db_context() as db:
        # System metrics
        total_predictions = db.query(Prediction).count()
        
        yesterday = datetime.utcnow() - timedelta(days=1)
        predictions_last_24h = db.query(Prediction).filter(
            Prediction.prediction_made_at >= yesterday
        ).count()
        
        active_models = db.query(ModelMetadata).filter(
            ModelMetadata.is_active == True
        ).count()
        
        production_models = db.query(ModelMetadata).filter(
            ModelMetadata.is_production == True
        ).count()
        
        system = SystemMetrics(
            total_predictions=total_predictions,
            predictions_last_24h=predictions_last_24h,
            avg_prediction_time_ms=50.0,  # Placeholder
            active_models=active_models,
            production_models=production_models,
        )
        
        # Data metrics
        total_records = db.query(ElectricityConsumption).count()
        
        latest_record = db.query(ElectricityConsumption).order_by(
            ElectricityConsumption.timestamp.desc()
        ).first()
        
        earliest_record = db.query(ElectricityConsumption).order_by(
            ElectricityConsumption.timestamp.asc()
        ).first()
        
        coverage_days = 0
        if latest_record and earliest_record:
            coverage_days = (latest_record.timestamp - earliest_record.timestamp).days
        
        data = DataMetrics(
            total_records=total_records,
            latest_timestamp=latest_record.timestamp if latest_record else None,
            data_coverage_days=coverage_days,
            missing_data_percent=0.0,  # Would need to calculate
        )
        
        # Model performance metrics
        model_metrics = []
        models = db.query(ModelMetadata).filter(
            ModelMetadata.is_active == True
        ).all()
        
        for model in models:
            if model.metrics:
                model_metrics.append(ModelPerformanceMetrics(
                    model_id=model.id,
                    model_type=model.model_type,
                    horizon_hours=model.horizon_hours,
                    mae=model.metrics.get("mae", 0.0),
                    rmse=model.metrics.get("rmse", 0.0),
                    mape=model.metrics.get("mape", 0.0),
                    last_evaluated=model.created_at,
                ))
        
        return MetricsResponse(
            system=system,
            data=data,
            models=model_metrics,
            generated_at=datetime.utcnow(),
        )


@router.get("/predictions")
async def get_prediction_accuracy():
    """
    Get prediction accuracy metrics by comparing predictions to actuals.
    """
    with get_db_context() as db:
        # Get predictions that have actual values
        predictions = db.query(Prediction).filter(
            Prediction.actual_value.isnot(None)
        ).order_by(Prediction.prediction_made_at.desc()).limit(1000).all()
        
        if not predictions:
            return {
                "message": "No predictions with actual values available",
                "total_evaluated": 0,
            }
        
        # Calculate metrics
        errors = []
        for p in predictions:
            if p.actual_value is not None:
                error = abs(p.actual_value - p.predicted_value)
                errors.append({
                    "absolute_error": error,
                    "percentage_error": (error / p.actual_value * 100) if p.actual_value != 0 else 0,
                })
        
        import numpy as np
        abs_errors = [e["absolute_error"] for e in errors]
        pct_errors = [e["percentage_error"] for e in errors]
        
        return {
            "total_evaluated": len(errors),
            "mae": float(np.mean(abs_errors)) if abs_errors else 0,
            "rmse": float(np.sqrt(np.mean(np.array(abs_errors) ** 2))) if abs_errors else 0,
            "mape": float(np.mean(pct_errors)) if pct_errors else 0,
            "median_error": float(np.median(abs_errors)) if abs_errors else 0,
        }


@router.get("/data-quality")
async def get_data_quality_metrics():
    """
    Get data quality metrics for the consumption data.
    """
    from src.data_pipeline.ingestion import DataIngestionService
    from src.data_pipeline.validation import DataValidator
    
    service = DataIngestionService()
    df = service.load_from_database()
    
    if df.empty:
        return {
            "message": "No data available",
            "total_records": 0,
        }
    
    validator = DataValidator()
    result = validator.validate(df)
    
    return {
        "total_records": len(df),
        "is_valid": result.is_valid,
        "issues": [
            {
                "type": i.issue_type,
                "severity": i.severity.value,
                "message": i.message,
                "affected_rows": i.affected_rows,
            }
            for i in result.issues
        ],
        "summary": result.summary,
    }


@router.get("/model/{model_id}")
async def get_model_metrics(
    model_id: int,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Get detailed metrics for a specific model.
    """
    with get_db_context() as db:
        model = db.query(ModelMetadata).filter(
            ModelMetadata.id == model_id
        ).first()
        
        if not model:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Get predictions made by this model
        predictions = db.query(Prediction).filter(
            Prediction.model_id == model_id
        ).all()
        
        predictions_with_actuals = [
            p for p in predictions if p.actual_value is not None
        ]
        
        return {
            "model_id": model_id,
            "model_type": model.model_type,
            "training_metrics": model.metrics,
            "total_predictions": len(predictions),
            "predictions_evaluated": len(predictions_with_actuals),
            "hyperparameters": model.hyperparameters,
            "feature_count": len(model.feature_columns) if model.feature_columns else 0,
        }
