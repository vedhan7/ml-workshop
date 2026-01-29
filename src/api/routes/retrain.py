"""
Retrain endpoint for triggering model retraining.
"""

import uuid
from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from src.api.dependencies import get_model_registry
from src.api.schemas import RetrainRequest, RetrainResponse, ErrorResponse
from src.data_pipeline.ingestion import DataIngestionService
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.ml_pipeline.training import ModelTrainer
from src.ml_pipeline.hyperparameter import HyperparameterOptimizer
from src.ml_pipeline.registry import ModelRegistry

logger = structlog.get_logger(__name__)

router = APIRouter()

# Simple in-memory job tracking (use Redis/database in production)
_training_jobs: dict[str, dict] = {}


def _run_training(
    job_id: str,
    model_type: str,
    horizon_hours: int,
    optimize: bool,
    hyperparameters: dict,
    set_as_production: bool,
):
    """Background task for model training."""
    try:
        _training_jobs[job_id]["status"] = "running"
        _training_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
        
        logger.info("training_started", job_id=job_id, model_type=model_type)
        
        # Load data
        ingestion = DataIngestionService()
        df = ingestion.load_from_database()
        
        if df.empty:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["error"] = "No data available for training"
            return
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        featured_df = feature_engineer.fit_transform(df)
        
        if len(featured_df) < 100:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["error"] = "Insufficient data after feature engineering"
            return
        
        # Get feature columns
        feature_cols = feature_engineer.feature_columns
        
        # Hyperparameter optimization if requested
        final_params = hyperparameters or {}
        if optimize:
            _training_jobs[job_id]["status"] = "optimizing"
            
            optimizer = HyperparameterOptimizer(
                model_type=model_type,
                n_trials=30,  # Reduce for faster results
            )
            
            X = featured_df[feature_cols]
            y = featured_df["consumption_kwh"]
            
            final_params = optimizer.optimize(X, y)
            _training_jobs[job_id]["best_params"] = final_params
        
        # Train model
        _training_jobs[job_id]["status"] = "training"
        
        trainer = ModelTrainer(
            model_type=model_type,
            feature_columns=feature_cols,
        )
        
        model, metrics = trainer.train(
            featured_df,
            model_params=final_params,
            save_model=True,
        )
        
        # Register model
        registry = ModelRegistry()
        metadata = registry.register(
            model=model,
            horizon_hours=horizon_hours,
            is_production=set_as_production,
        )
        
        _training_jobs[job_id]["status"] = "completed"
        _training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _training_jobs[job_id]["model_id"] = metadata.id
        _training_jobs[job_id]["metrics"] = metrics
        
        logger.info(
            "training_completed",
            job_id=job_id,
            model_id=metadata.id,
            metrics=metrics,
        )
        
    except Exception as e:
        logger.error("training_failed", job_id=job_id, error=str(e))
        _training_jobs[job_id]["status"] = "failed"
        _training_jobs[job_id]["error"] = str(e)


@router.post(
    "",
    response_model=RetrainResponse,
    responses={400: {"model": ErrorResponse}},
)
async def trigger_retrain(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Trigger model retraining.
    
    Returns a job ID that can be used to track progress.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Initialize job tracking
    _training_jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "model_type": request.model_type.value,
        "horizon_hours": request.horizon_hours,
    }
    
    # Schedule background task
    background_tasks.add_task(
        _run_training,
        job_id=job_id,
        model_type=request.model_type.value,
        horizon_hours=request.horizon_hours,
        optimize=request.optimize_hyperparameters,
        hyperparameters=request.hyperparameters,
        set_as_production=request.set_as_production,
    )
    
    return RetrainResponse(
        job_id=job_id,
        status="queued",
        message=f"Training job {job_id} queued for {request.model_type.value} model",
    )


@router.get("/{job_id}")
async def get_training_status(job_id: str):
    """
    Get the status of a training job.
    """
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _training_jobs[job_id]
    
    return {
        "job_id": job_id,
        **job,
    }


@router.get("")
async def list_training_jobs():
    """
    List all training jobs.
    """
    return {
        "jobs": [
            {"job_id": job_id, **job}
            for job_id, job in sorted(
                _training_jobs.items(),
                key=lambda x: x[1].get("created_at", ""),
                reverse=True,
            )
        ]
    }
