"""
Scheduled task definitions for background processing.
"""

from datetime import datetime, timedelta

import structlog

from src.config import get_settings
from src.database.connection import get_db_context
from src.database.models import ElectricityConsumption, Prediction
from src.data_pipeline.ingestion import DataIngestionService
from src.data_pipeline.validation import DataValidator
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.ml_pipeline.training import ModelTrainer
from src.ml_pipeline.registry import ModelRegistry

logger = structlog.get_logger(__name__)


def daily_data_ingestion():
    """
    Daily task to ingest new data from configured sources.
    
    In production, this would fetch from:
    - Utility company APIs
    - Weather services
    - Other data sources
    """
    logger.info("starting_daily_ingestion")
    
    try:
        settings = get_settings()
        ingestion = DataIngestionService()
        
        # Check for new data files in raw directory
        raw_dir = settings.data_dir / "raw"
        
        if raw_dir.exists():
            for csv_file in raw_dir.glob("*.csv"):
                try:
                    count = ingestion.ingest_csv(csv_file)
                    logger.info("ingested_file", file=csv_file.name, records=count)
                except Exception as e:
                    logger.error("ingestion_failed", file=csv_file.name, error=str(e))
        
        # Fetch weather data if API key is configured
        if settings.openweather_api_key:
            # Default coordinates (can be configured)
            ingestion.fetch_weather_data(
                latitude=13.0827,  # Chennai
                longitude=80.2707,
                start_date=datetime.utcnow() - timedelta(days=1),
                end_date=datetime.utcnow(),
            )
        
        logger.info("daily_ingestion_complete")
        
    except Exception as e:
        logger.error("daily_ingestion_failed", error=str(e))


def weekly_model_retrain():
    """
    Weekly task to retrain models with latest data.
    
    Trains models for all configured horizons and updates
    production if the new model performs better.
    """
    logger.info("starting_weekly_retrain")
    
    try:
        settings = get_settings()
        registry = ModelRegistry()
        
        # Load and prepare data
        ingestion = DataIngestionService()
        df = ingestion.load_from_database()
        
        if df.empty or len(df) < 1000:
            logger.warning("insufficient_data_for_training", rows=len(df))
            return
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        featured_df = feature_engineer.fit_transform(df)
        
        # Train for each horizon
        for horizon in settings.prediction_horizons:
            try:
                logger.info("training_horizon", horizon=horizon)
                
                trainer = ModelTrainer(
                    model_type="xgboost",
                    feature_columns=feature_engineer.feature_columns,
                )
                
                # Shift target for horizon
                df_horizon = featured_df.copy()
                df_horizon["consumption_kwh"] = df_horizon["consumption_kwh"].shift(-horizon)
                df_horizon = df_horizon.dropna()
                
                model, metrics = trainer.train(df_horizon, save_model=True)
                
                # Get current production model metrics
                current_prod = registry.list_models(
                    model_type="xgboost",
                    horizon_hours=horizon,
                )
                current_prod = [m for m in current_prod if m["is_production"]]
                
                # Set as production if better than current
                should_promote = True
                if current_prod and "metrics" in current_prod[0]:
                    old_mae = current_prod[0]["metrics"].get("mae", float("inf"))
                    new_mae = metrics.get("mae", float("inf"))
                    should_promote = new_mae < old_mae * 0.95  # 5% improvement threshold
                
                registry.register(
                    model=model,
                    horizon_hours=horizon,
                    is_production=should_promote,
                )
                
                logger.info(
                    "model_trained",
                    horizon=horizon,
                    mae=metrics.get("mae"),
                    promoted=should_promote,
                )
                
            except Exception as e:
                logger.error("horizon_training_failed", horizon=horizon, error=str(e))
        
        # Cleanup old models
        registry.cleanup_old_models(keep_recent=3, keep_production=True)
        
        logger.info("weekly_retrain_complete")
        
    except Exception as e:
        logger.error("weekly_retrain_failed", error=str(e))


def hourly_prediction_batch():
    """
    Hourly task to generate and store batch predictions.
    
    Creates predictions for the next 24 hours using the
    production model.
    """
    logger.info("starting_hourly_predictions")
    
    try:
        registry = ModelRegistry()
        
        # Get production model for 24h horizon
        model = registry.get_production_model(horizon_hours=24)
        
        if model is None:
            logger.warning("no_production_model_available")
            return
        
        # Generate predictions for next 24 hours
        import pandas as pd
        import numpy as np
        
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        timestamps = pd.date_range(start=now, periods=24, freq="h")
        
        # Generate features (simplified - use actual feature engineering in production)
        features = []
        for ts in timestamps:
            feature = {
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "month": ts.month,
                "is_weekend": int(ts.dayofweek >= 5),
            }
            # Add more features as needed
            features.append(feature)
        
        X = pd.DataFrame(features)
        
        # Get available features
        available = [c for c in model.feature_columns if c in X.columns]
        
        # Fill missing features with defaults
        for col in model.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        predictions = model.predict(X[model.feature_columns])
        
        # Store predictions
        with get_db_context() as db:
            from src.database.models import ModelMetadata
            
            # Get model metadata ID
            metadata = db.query(ModelMetadata).filter(
                ModelMetadata.model_path.like(f"%{model.name}%")
            ).first()
            
            model_id = metadata.id if metadata else 1
            
            for ts, pred in zip(timestamps, predictions):
                prediction = Prediction(
                    model_id=model_id,
                    target_timestamp=ts,
                    predicted_value=float(pred),
                    horizon_hours=24,
                )
                db.add(prediction)
            
            db.commit()
        
        logger.info("hourly_predictions_complete", count=len(predictions))
        
    except Exception as e:
        logger.error("hourly_predictions_failed", error=str(e))


def data_quality_check():
    """
    Daily task to check data quality and alert on issues.
    """
    logger.info("starting_data_quality_check")
    
    try:
        ingestion = DataIngestionService()
        df = ingestion.load_from_database()
        
        if df.empty:
            logger.warning("no_data_for_quality_check")
            return
        
        validator = DataValidator()
        result = validator.validate(df)
        
        # Log issues
        for issue in result.issues:
            log_method = logger.warning if issue.severity.value in ["warning", "error"] else logger.info
            log_method(
                "data_quality_issue",
                type=issue.issue_type,
                severity=issue.severity.value,
                message=issue.message,
                affected_rows=issue.affected_rows,
            )
        
        logger.info(
            "data_quality_check_complete",
            is_valid=result.is_valid,
            issue_count=len(result.issues),
        )
        
    except Exception as e:
        logger.error("data_quality_check_failed", error=str(e))
