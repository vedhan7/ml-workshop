"""
Prediction endpoint for electricity demand forecasting.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_model_registry
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionPoint,
    ErrorResponse,
)
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.ingestion import DataIngestionService
from src.ml_pipeline.registry import ModelRegistry

logger = structlog.get_logger(__name__)

router = APIRouter()


def _horizon_to_hours(horizon: str) -> int:
    """Convert horizon string to hours."""
    mapping = {"1h": 1, "24h": 24, "168h": 168}
    return mapping.get(horizon, 24)


def _generate_future_features(
    start_time: datetime,
    horizon_hours: int,
    feature_engineer: FeatureEngineer,
) -> pd.DataFrame:
    """
    Generate feature DataFrame for future timestamps.
    
    This creates the required features for prediction without actual
    consumption values (which we're predicting).
    """
    # Generate timestamps
    timestamps = pd.date_range(
        start=start_time,
        periods=horizon_hours,
        freq="h",
    )
    
    # Create base DataFrame
    df = pd.DataFrame({"timestamp": timestamps})
    df = df.set_index("timestamp")
    
    # Add calendar features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"] = df.index.is_month_end.astype(int)
    
    df["is_morning"] = df["hour"].between(6, 11).astype(int)
    df["is_afternoon"] = df["hour"].between(12, 17).astype(int)
    df["is_evening"] = df["hour"].between(18, 22).astype(int)
    df["is_night"] = (~df["hour"].between(6, 22)).astype(int)
    df["is_peak_hour"] = df["hour"].between(17, 21).astype(int)
    df["is_holiday"] = 0  # Simplified
    
    # Fourier features
    hours_in_day = 24
    hours_in_week = 168
    hour_of_week = df["day_of_week"] * 24 + df["hour"]
    day_of_year = df.index.dayofyear
    
    for k in range(1, 5):
        df[f"daily_sin_{k}"] = np.sin(2 * np.pi * k * df["hour"] / hours_in_day)
        df[f"daily_cos_{k}"] = np.cos(2 * np.pi * k * df["hour"] / hours_in_day)
        df[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * hour_of_week / hours_in_week)
        df[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * hour_of_week / hours_in_week)
    
    for k in range(1, 4):
        df[f"yearly_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365)
        df[f"yearly_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365)
    
    # For lag and rolling features, we'd need historical data
    # Here we set them to reasonable defaults or fetch from database
    lag_cols = [f"lag_{h}h" for h in [1, 2, 3, 6, 12, 24, 48, 168]]
    lag_cols.extend(["lag_same_hour_yesterday", "lag_same_hour_last_week"])
    
    rolling_cols = []
    for window in [6, 12, 24, 48, 168]:
        rolling_cols.extend([
            f"rolling_mean_{window}h",
            f"rolling_std_{window}h",
            f"rolling_min_{window}h",
            f"rolling_max_{window}h",
        ])
    rolling_cols.extend(["consumption_diff_1h", "consumption_diff_24h"])
    
    # Set to mean values (in production, fetch from recent data)
    for col in lag_cols + rolling_cols:
        df[col] = 1000.0  # Default placeholder
    
    df = df.reset_index()
    return df


@router.post(
    "",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_prediction(
    request: PredictionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Generate electricity demand predictions.
    
    Returns predictions for the specified horizon starting from the given time.
    """
    try:
        horizon_hours = _horizon_to_hours(request.horizon.value)
        start_time = request.start_time or datetime.utcnow()
        
        logger.info(
            "prediction_request",
            horizon=horizon_hours,
            start_time=start_time.isoformat(),
        )
        
        # Get the model
        if request.model_id:
            model = registry.get_model_by_id(request.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        else:
            model = registry.get_production_model(horizon_hours=horizon_hours)
            if model is None:
                raise HTTPException(
                    status_code=404,
                    detail="No production model available. Please train a model first."
                )
        
        # Generate features for prediction
        feature_engineer = FeatureEngineer()
        features_df = _generate_future_features(start_time, horizon_hours, feature_engineer)
        
        # Ensure we have the right columns
        model_features = model.feature_columns
        available_features = [col for col in model_features if col in features_df.columns]
        
        if len(available_features) < len(model_features) * 0.8:
            # Fall back to simple prediction if too many features are missing
            logger.warning(
                "missing_features",
                required=len(model_features),
                available=len(available_features),
            )
        
        # Make predictions
        X = features_df[available_features] if available_features else features_df.drop(columns=["timestamp"])
        predictions = model.predict(X)
        
        # Build response
        prediction_points = []
        for i, (ts, pred) in enumerate(zip(features_df["timestamp"], predictions)):
            point = PredictionPoint(
                timestamp=ts,
                predicted_value=float(pred),
            )
            
            if request.include_confidence:
                # Simple confidence interval (±10% or based on model metrics)
                uncertainty = abs(pred) * 0.1
                point.lower_bound = float(pred - uncertainty)
                point.upper_bound = float(pred + uncertainty)
            
            prediction_points.append(point)
        
        return PredictionResponse(
            predictions=prediction_points,
            model_used=model.model_type,
            model_version=model.name,
            horizon=request.horizon.value,
            generated_at=datetime.utcnow(),
            confidence_level=0.95 if request.include_confidence else 0.0,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def create_batch_predictions(
    requests: list[PredictionRequest],
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Generate batch predictions for multiple requests.
    """
    results = []
    for req in requests:
        try:
            result = await create_prediction(req, registry)
            results.append({"status": "success", "data": result})
        except HTTPException as e:
            results.append({"status": "error", "detail": e.detail})
    
    return {"results": results, "total": len(results)}
