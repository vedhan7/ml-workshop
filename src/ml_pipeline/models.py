"""
Model definitions for electricity demand forecasting.
Includes XGBoost, LightGBM, Prophet, and ensemble models.
"""

import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from sklearn.ensemble import RandomForestRegressor

logger = structlog.get_logger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    model_type: str = "base"
    
    def __init__(self, name: str = None, **kwargs):
        self.name = name or f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.feature_columns: list[str] = []
        self.target_col: str = "consumption_kwh"
        self.is_fitted: bool = False
        self.hyperparameters: dict = kwargs
        self.metadata: dict = {}
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save(self, path: str | Path) -> Path:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "name": self.name,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "target_col": self.target_col,
            "hyperparameters": self.hyperparameters,
            "metadata": self.metadata,
            "is_fitted": self.is_fitted,
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info("model_saved", path=str(path))
        return path
    
    @classmethod
    def load(cls, path: str | Path) -> "BaseForecaster":
        """Load model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        # Get the correct class
        model_type = model_data["model_type"]
        model_class = MODEL_REGISTRY.get(model_type, cls)
        
        instance = model_class.__new__(model_class)
        instance.model = model_data["model"]
        instance.name = model_data["name"]
        instance.model_type = model_data["model_type"]
        instance.feature_columns = model_data["feature_columns"]
        instance.target_col = model_data["target_col"]
        instance.hyperparameters = model_data["hyperparameters"]
        instance.metadata = model_data.get("metadata", {})
        instance.is_fitted = model_data["is_fitted"]
        
        logger.info("model_loaded", path=str(path), model_type=model_type)
        return instance
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return self.hyperparameters.copy()
    
    def set_params(self, **params) -> "BaseForecaster":
        """Set model parameters."""
        self.hyperparameters.update(params)
        return self


class XGBoostForecaster(BaseForecaster):
    """XGBoost-based forecaster for electricity demand."""
    
    model_type = "xgboost"
    
    def __init__(
        self,
        name: str = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(name=name)
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            **kwargs,
        }
    
    def _create_model(self):
        """Create XGBoost model instance."""
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(**self.hyperparameters)
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        """Fit XGBoost model."""
        self.feature_columns = list(X.columns)
        self.model = self._create_model()
        
        logger.info("fitting_xgboost", features=len(self.feature_columns), samples=len(X))
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature importance
        self.metadata["feature_importance"] = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X[self.feature_columns]
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n: int = 20) -> dict:
        """Get top N feature importances."""
        if "feature_importance" not in self.metadata:
            return {}
        
        importance = self.metadata["feature_importance"]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])


class LightGBMForecaster(BaseForecaster):
    """LightGBM-based forecaster for electricity demand."""
    
    model_type = "lightgbm"
    
    def __init__(
        self,
        name: str = None,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        random_state: int = 42,
        verbose: int = -1,
        **kwargs,
    ):
        super().__init__(name=name)
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "verbose": verbose,
            **kwargs,
        }
    
    def _create_model(self):
        """Create LightGBM model instance."""
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(**self.hyperparameters)
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMForecaster":
        """Fit LightGBM model."""
        self.feature_columns = list(X.columns)
        self.model = self._create_model()
        
        logger.info("fitting_lightgbm", features=len(self.feature_columns), samples=len(X))
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature importance
        self.metadata["feature_importance"] = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X[self.feature_columns]
        return self.model.predict(X)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet-based forecaster for electricity demand."""
    
    model_type = "prophet"
    
    def __init__(
        self,
        name: str = None,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        **kwargs,
    ):
        super().__init__(name=name)
        self.hyperparameters = {
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "seasonality_mode": seasonality_mode,
            "changepoint_prior_scale": changepoint_prior_scale,
            **kwargs,
        }
        self.timestamp_col = "timestamp"
    
    def _create_model(self):
        """Create Prophet model instance."""
        try:
            from prophet import Prophet
            return Prophet(**self.hyperparameters)
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProphetForecaster":
        """
        Fit Prophet model.
        
        Note: Prophet expects a DataFrame with 'ds' (timestamp) and 'y' (target) columns.
        """
        self.model = self._create_model()
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            "ds": X[self.timestamp_col] if self.timestamp_col in X.columns else X.index,
            "y": y.values,
        })
        
        # Add regressors if additional features exist
        extra_features = [col for col in X.columns if col != self.timestamp_col]
        for col in extra_features[:10]:  # Limit regressors
            prophet_df[col] = X[col].values
            self.model.add_regressor(col)
        
        self.feature_columns = extra_features[:10]
        
        logger.info("fitting_prophet", samples=len(prophet_df))
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Prophet."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare future dataframe
        future = pd.DataFrame({
            "ds": X[self.timestamp_col] if self.timestamp_col in X.columns else X.index,
        })
        
        # Add regressors
        for col in self.feature_columns:
            if col in X.columns:
                future[col] = X[col].values
        
        forecast = self.model.predict(future)
        return forecast["yhat"].values



class RandomForestForecaster(BaseForecaster):
    """Random Forest forecasting model."""
    
    model_type = "random_forest"
    
    def __init__(
        self,
        name: str = None,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(name=name)
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **kwargs,
        }
    
    def _create_model(self):
        """Create Random Forest model instance."""
        return RandomForestRegressor(**self.hyperparameters)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        """Fit Random Forest model."""
        self.feature_columns = list(X.columns)
        self.model = self._create_model()
        
        logger.info("fitting_random_forest", features=len(self.feature_columns), samples=len(X))
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature importance
        if hasattr(self.model, "feature_importances_"):
            self.metadata["feature_importance"] = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X[self.feature_columns]
        return self.model.predict(X)


class EnsembleForecaster(BaseForecaster):
    """Ensemble of multiple forecasters with weighted averaging."""
    
    model_type = "ensemble"
    
    def __init__(
        self,
        name: str = None,
        models: list[BaseForecaster] = None,
        weights: list[float] = None,
    ):
        super().__init__(name=name)
        self.models = models or []
        self.weights = weights
    
    def add_model(self, model: BaseForecaster, weight: float = 1.0) -> "EnsembleForecaster":
        """Add a model to the ensemble."""
        self.models.append(model)
        if self.weights is None:
            self.weights = []
        self.weights.append(weight)
        return self
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleForecaster":
        """Fit all models in the ensemble."""
        for i, model in enumerate(self.models):
            logger.info("fitting_ensemble_model", model_idx=i, model_type=model.model_type)
            model.fit(X, y)
        
        self.is_fitted = True
        self.feature_columns = self.models[0].feature_columns if self.models else []
        
        # Normalize weights
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted average predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.weights:
            weights = np.array(self.weights).reshape(-1, 1)
            return np.sum(predictions * weights, axis=0)
        else:
            return np.mean(predictions, axis=0)


# Model registry for easy loading
MODEL_REGISTRY = {
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
    "prophet": ProphetForecaster,
    "random_forest": RandomForestForecaster,
    "ensemble": EnsembleForecaster,
}


def get_model(model_type: str, **kwargs) -> BaseForecaster:
    """Factory function to get a model by type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](**kwargs)
