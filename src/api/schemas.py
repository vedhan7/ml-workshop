"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# Enums
class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class PredictionHorizon(str, Enum):
    HOUR_1 = "1h"
    HOUR_24 = "24h"
    WEEK = "168h"


# Prediction Schemas
class PredictionRequest(BaseModel):
    """Request for generating predictions."""
    
    horizon: PredictionHorizon = Field(
        default=PredictionHorizon.HOUR_24,
        description="Prediction horizon"
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="Start time for predictions (defaults to now)"
    )
    location_id: Optional[str] = Field(
        default=None,
        description="Location identifier"
    )
    model_id: Optional[int] = Field(
        default=None,
        description="Specific model ID to use (uses production if not specified)"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence intervals"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "horizon": "24h",
                "start_time": "2026-01-29T00:00:00",
                "include_confidence": True
            }
        }


class PredictionPoint(BaseModel):
    """Single prediction point."""
    
    timestamp: datetime
    predicted_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response containing predictions."""
    
    predictions: list[PredictionPoint]
    model_used: str
    model_version: str
    horizon: str
    generated_at: datetime
    confidence_level: float = 0.95
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "timestamp": "2026-01-29T00:00:00",
                        "predicted_value": 1250.5,
                        "lower_bound": 1200.0,
                        "upper_bound": 1300.0
                    }
                ],
                "model_used": "xgboost",
                "model_version": "v20260128_120000",
                "horizon": "24h",
                "generated_at": "2026-01-28T20:00:00",
                "confidence_level": 0.95
            }
        }


# Model Schemas
class ModelInfo(BaseModel):
    """Information about a trained model."""
    
    id: int
    name: str
    version: str
    model_type: str
    horizon_hours: int
    is_production: bool
    metrics: Optional[dict] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    """Response containing list of models."""
    
    models: list[ModelInfo]
    total: int


class ModelDetailResponse(BaseModel):
    """Detailed information about a model."""
    
    id: int
    name: str
    version: str
    model_type: str
    horizon_hours: int
    is_production: bool
    is_active: bool
    metrics: Optional[dict] = None
    hyperparameters: Optional[dict] = None
    feature_columns: Optional[list[str]] = None
    training_samples: Optional[int] = None
    training_start: Optional[datetime] = None
    training_end: Optional[datetime] = None
    created_at: datetime
    feature_importance: Optional[dict] = None


# Retrain Schemas
class RetrainRequest(BaseModel):
    """Request to trigger model retraining."""
    
    model_type: ModelType = Field(
        default=ModelType.XGBOOST,
        description="Type of model to train"
    )
    horizon_hours: int = Field(
        default=24,
        description="Prediction horizon in hours"
    )
    use_latest_data: bool = Field(
        default=True,
        description="Whether to use the latest available data"
    )
    optimize_hyperparameters: bool = Field(
        default=False,
        description="Whether to run hyperparameter optimization"
    )
    set_as_production: bool = Field(
        default=False,
        description="Whether to set the new model as production"
    )
    hyperparameters: Optional[dict] = Field(
        default=None,
        description="Custom hyperparameters to use"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "xgboost",
                "horizon_hours": 24,
                "use_latest_data": True,
                "optimize_hyperparameters": False,
                "set_as_production": False
            }
        }


class RetrainResponse(BaseModel):
    """Response from retraining request."""
    
    job_id: str
    status: str
    message: str
    model_id: Optional[int] = None
    metrics: Optional[dict] = None


# Metrics Schemas
class SystemMetrics(BaseModel):
    """System-level metrics."""
    
    total_predictions: int
    predictions_last_24h: int
    avg_prediction_time_ms: float
    active_models: int
    production_models: int


class DataMetrics(BaseModel):
    """Data-related metrics."""
    
    total_records: int
    latest_timestamp: Optional[datetime]
    data_coverage_days: int
    missing_data_percent: float


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""
    
    model_id: int
    model_type: str
    horizon_hours: int
    mae: float
    rmse: float
    mape: float
    last_evaluated: datetime


class MetricsResponse(BaseModel):
    """Complete metrics response."""
    
    system: SystemMetrics
    data: DataMetrics
    models: list[ModelPerformanceMetrics]
    generated_at: datetime


# Data Ingestion Schemas
class DataIngestionRequest(BaseModel):
    """Request for data ingestion."""
    
    file_path: str
    timestamp_col: str = "timestamp"
    consumption_col: str = "consumption_kwh"
    location_id: Optional[str] = None


class DataIngestionResponse(BaseModel):
    """Response from data ingestion."""
    
    records_ingested: int
    validation_issues: list[dict]
    status: str


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
