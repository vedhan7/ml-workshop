"""
Model evaluation metrics for electricity demand forecasting.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    max_error: float
    median_absolute_error: float
    directional_accuracy: Optional[float] = None
    peak_accuracy: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "smape": self.smape,
            "r2": self.r2,
            "max_error": self.max_error,
            "median_absolute_error": self.median_absolute_error,
            "directional_accuracy": self.directional_accuracy,
            "peak_accuracy": self.peak_accuracy,
        }


class ModelEvaluator:
    """
    Evaluates forecasting models with various metrics.
    
    Metrics computed:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Square Error
    - MAPE: Mean Absolute Percentage Error
    - SMAPE: Symmetric Mean Absolute Percentage Error
    - R²: Coefficient of Determination
    - Directional Accuracy: % of correct direction predictions
    - Peak Accuracy: Accuracy during peak demand hours
    """
    
    def __init__(self, peak_hours: list[int] = None):
        """
        Initialize evaluator.
        
        Args:
            peak_hours: Hours considered as peak demand (default: 17-21)
        """
        self.peak_hours = peak_hours or list(range(17, 22))
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Optional timestamps for time-based metrics
            
        Returns:
            Dictionary of metric names to values
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        metrics = {
            "mae": self._mae(y_true, y_pred),
            "rmse": self._rmse(y_true, y_pred),
            "mape": self._mape(y_true, y_pred),
            "smape": self._smape(y_true, y_pred),
            "r2": self._r2(y_true, y_pred),
            "max_error": self._max_error(y_true, y_pred),
            "median_absolute_error": self._median_absolute_error(y_true, y_pred),
        }
        
        # Directional accuracy (requires at least 2 points)
        if len(y_true) > 1:
            metrics["directional_accuracy"] = self._directional_accuracy(y_true, y_pred)
        
        # Peak accuracy (requires timestamps)
        if timestamps is not None:
            metrics["peak_mae"] = self._peak_accuracy(y_true, y_pred, timestamps)
        
        return metrics
    
    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return float("inf")
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def _smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = np.abs(y_true) + np.abs(y_pred)
        # Avoid division by zero
        mask = denominator != 0
        if not mask.any():
            return 0.0
        return float(
            np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        )
    
    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (Coefficient of Determination)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - (ss_res / ss_tot))
    
    def _max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Maximum Absolute Error."""
        return float(np.max(np.abs(y_true - y_pred)))
    
    def _median_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median Absolute Error."""
        return float(np.median(np.abs(y_true - y_pred)))
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional Accuracy.
        
        Measures how often the model correctly predicts whether the value
        will increase or decrease from the previous time step.
        """
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        
        if total == 0:
            return 0.0
        
        return float(correct / total * 100)
    
    def _peak_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> float:
        """
        MAE during peak hours.
        
        Peak hours are typically when demand is highest (e.g., 17:00-21:00).
        """
        hours = pd.Series(timestamps).dt.hour
        peak_mask = hours.isin(self.peak_hours).values
        
        if not peak_mask.any():
            return float("nan")
        
        return self._mae(y_true[peak_mask], y_pred[peak_mask])
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            y_true: True values
            predictions: Dictionary mapping model name to predictions
            timestamps: Optional timestamps for time-based metrics
            
        Returns:
            DataFrame comparing all models
        """
        results = []
        for name, y_pred in predictions.items():
            metrics = self.evaluate(y_true, y_pred, timestamps)
            metrics["model"] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index("model")
        
        # Rank models by MAE (lower is better)
        df["mae_rank"] = df["mae"].rank()
        
        return df.sort_values("mae")
    
    def residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> dict:
        """
        Analyze prediction residuals.
        
        Returns statistics about the residuals and potential patterns.
        """
        residuals = y_true - y_pred
        
        analysis = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "skewness": float(self._skewness(residuals)),
            "kurtosis": float(self._kurtosis(residuals)),
            "percentile_5": float(np.percentile(residuals, 5)),
            "percentile_95": float(np.percentile(residuals, 95)),
        }
        
        # Check for systematic bias
        if analysis["mean_residual"] > 0.1 * np.std(y_true):
            analysis["bias_warning"] = "Model tends to underpredict"
        elif analysis["mean_residual"] < -0.1 * np.std(y_true):
            analysis["bias_warning"] = "Model tends to overpredict"
        else:
            analysis["bias_warning"] = None
        
        # Hourly residuals if timestamps provided
        if timestamps is not None:
            hours = pd.Series(timestamps).dt.hour
            hourly_residuals = pd.DataFrame({"hour": hours, "residual": residuals})
            analysis["hourly_mean_residual"] = (
                hourly_residuals.groupby("hour")["residual"].mean().to_dict()
            )
        
        return analysis
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.sum(((x - mean) / std) ** 3) / n)
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.sum(((x - mean) / std) ** 4) / n - 3)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> dict:
    """Convenience function for model evaluation."""
    evaluator = ModelEvaluator()
    return evaluator.evaluate(y_true, y_pred, timestamps)
