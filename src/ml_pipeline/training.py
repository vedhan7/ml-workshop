"""
Model training orchestration for electricity demand forecasting.
Handles cross-validation, multi-horizon training, and MLflow logging.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from src.config import get_settings
from src.ml_pipeline.models import BaseForecaster, get_model
from src.ml_pipeline.evaluation import ModelEvaluator

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """
    Orchestrates model training with time-series cross-validation.
    
    Supports:
    - Expanding window cross-validation
    - Walk-forward validation
    - Multi-horizon training
    - MLflow experiment tracking
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        feature_columns: list[str] = None,
        target_col: str = "consumption_kwh",
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_mlflow: bool = True,
    ):
        self.model_type = model_type
        self.feature_columns = feature_columns
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.use_mlflow = use_mlflow
        
        self.settings = get_settings()
        self.evaluator = ModelEvaluator()
        
        self._mlflow_initialized = False
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        if self._mlflow_initialized or not self.use_mlflow:
            return
        
        try:
            import mlflow
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)
            self._mlflow_initialized = True
            logger.info("mlflow_initialized", uri=self.settings.mlflow_tracking_uri)
        except Exception as e:
            logger.warning("mlflow_init_failed", error=str(e))
            self.use_mlflow = False
    
    def train(
        self,
        df: pd.DataFrame,
        model_params: dict = None,
        save_model: bool = True,
    ) -> tuple[BaseForecaster, dict]:
        """
        Train a model on the full dataset.
        
        Args:
            df: DataFrame with features and target
            model_params: Optional model hyperparameters
            save_model: Whether to save the trained model
            
        Returns:
            Tuple of (trained model, metrics dict)
        """
        logger.info("starting_training", model_type=self.model_type, rows=len(df))
        
        # Prepare data
        X, y = self._prepare_data(df)
        
        # Split into train/test
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info("data_split", train_size=len(X_train), test_size=len(X_test))
        
        # Create and train model
        model = get_model(self.model_type, **(model_params or {}))
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        metrics = self.evaluator.evaluate(y_test.values, predictions)
        
        # Add metadata
        model.metadata.update({
            "training_start": df[self.timestamp_col].min().isoformat(),
            "training_end": df[self.timestamp_col].max().isoformat(),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": metrics,
        })
        
        # Log to MLflow
        if self.use_mlflow:
            self._log_to_mlflow(model, metrics, model_params)
        
        # Save model
        if save_model:
            model_path = self._save_model(model)
            model.metadata["model_path"] = str(model_path)
        
        logger.info("training_complete", metrics=metrics)
        return model, metrics
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        model_params: dict = None,
    ) -> dict:
        """
        Perform time-series cross-validation.
        
        Uses expanding window approach suitable for time series.
        
        Args:
            df: DataFrame with features and target
            model_params: Optional model hyperparameters
            
        Returns:
            Dictionary with CV results and statistics
        """
        logger.info("starting_cross_validation", folds=self.cv_folds)
        
        X, y = self._prepare_data(df)
        
        # Calculate fold sizes
        n_samples = len(X)
        min_train_size = int(n_samples * 0.5)  # Minimum 50% for first fold
        fold_size = (n_samples - min_train_size) // self.cv_folds
        
        fold_metrics = []
        
        for fold in range(self.cv_folds):
            # Expanding window: train on all data up to this point
            train_end = min_train_size + fold * fold_size
            test_end = min(train_end + fold_size, n_samples)
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            
            if len(X_test) == 0:
                continue
            
            # Train and evaluate
            model = get_model(self.model_type, **(model_params or {}))
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            metrics = self.evaluator.evaluate(y_test.values, predictions)
            metrics["fold"] = fold
            metrics["train_size"] = len(X_train)
            metrics["test_size"] = len(X_test)
            
            fold_metrics.append(metrics)
            logger.info("fold_complete", fold=fold, mae=metrics["mae"])
        
        # Aggregate results
        results = {
            "folds": fold_metrics,
            "mean_mae": np.mean([m["mae"] for m in fold_metrics]),
            "mean_rmse": np.mean([m["rmse"] for m in fold_metrics]),
            "mean_mape": np.mean([m["mape"] for m in fold_metrics]),
            "std_mae": np.std([m["mae"] for m in fold_metrics]),
            "std_rmse": np.std([m["rmse"] for m in fold_metrics]),
        }
        
        logger.info("cv_complete", mean_mae=results["mean_mae"], std_mae=results["std_mae"])
        return results
    
    def train_multi_horizon(
        self,
        df: pd.DataFrame,
        horizons: list[int] = None,
        model_params: dict = None,
    ) -> dict[int, tuple[BaseForecaster, dict]]:
        """
        Train separate models for different prediction horizons.
        
        Args:
            df: DataFrame with features and target
            horizons: List of horizons in hours (default: [1, 24, 168])
            model_params: Optional model hyperparameters
            
        Returns:
            Dictionary mapping horizon to (model, metrics) tuples
        """
        horizons = horizons or self.settings.prediction_horizons
        results = {}
        
        for horizon in horizons:
            logger.info("training_horizon", horizon=horizon)
            
            # Shift target for the horizon
            df_horizon = df.copy()
            df_horizon[self.target_col] = df_horizon[self.target_col].shift(-horizon)
            df_horizon = df_horizon.dropna()
            
            model, metrics = self.train(
                df_horizon,
                model_params=model_params,
                save_model=True,
            )
            
            model.metadata["horizon_hours"] = horizon
            results[horizon] = (model, metrics)
        
        return results
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from dataframe."""
        df = df.sort_values(self.timestamp_col)
        
        if self.feature_columns:
            X = df[self.feature_columns]
        else:
            # Use all columns except timestamp and target
            exclude = [self.timestamp_col, self.target_col]
            X = df.drop(columns=[c for c in exclude if c in df.columns])
        
        y = df[self.target_col]
        
        return X, y
    
    def _save_model(self, model: BaseForecaster) -> Path:
        """Save model to disk."""
        model_dir = self.settings.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model.model_type}_{timestamp}.pkl"
        model_path = model_dir / filename
        
        model.save(model_path)
        return model_path
    
    def _log_to_mlflow(
        self,
        model: BaseForecaster,
        metrics: dict,
        params: dict = None,
    ):
        """Log training run to MLflow."""
        if not self.use_mlflow:
            return
        
        self._init_mlflow()
        
        try:
            import mlflow
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    "model_type": model.model_type,
                    **(params or {}),
                })
                
                # Log metrics
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(name, value)
                
                # Log model
                mlflow.sklearn.log_model(model.model, "model")
                
        except Exception as e:
            logger.warning("mlflow_logging_failed", error=str(e))


def train_model(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    **kwargs,
) -> tuple[BaseForecaster, dict]:
    """Convenience function for training a model."""
    trainer = ModelTrainer(model_type=model_type, **kwargs)
    return trainer.train(df)
