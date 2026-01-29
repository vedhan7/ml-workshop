"""
Custom sklearn-compatible transformers for feature engineering.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for creating lag features.
    
    Creates shifted versions of the target column to capture
    autocorrelation patterns.
    """
    
    def __init__(
        self,
        target_col: str = "consumption_kwh",
        lag_periods: list[int] = None,
    ):
        self.target_col = target_col
        self.lag_periods = lag_periods or [1, 24, 168]
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag features."""
        X = X.copy()
        for lag in self.lag_periods:
            X[f"lag_{lag}"] = X[self.target_col].shift(lag)
        return X
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        return [f"lag_{lag}" for lag in self.lag_periods]


class RollingStatisticsTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for rolling window statistics.
    
    Computes mean, std, min, max over rolling windows.
    """
    
    def __init__(
        self,
        target_col: str = "consumption_kwh",
        windows: list[int] = None,
        statistics: list[str] = None,
    ):
        self.target_col = target_col
        self.windows = windows or [24, 168]
        self.statistics = statistics or ["mean", "std"]
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics features."""
        X = X.copy()
        for window in self.windows:
            # Shift by 1 to avoid leakage
            rolled = X[self.target_col].shift(1).rolling(window=window)
            
            for stat in self.statistics:
                if stat == "mean":
                    X[f"rolling_{stat}_{window}"] = rolled.mean()
                elif stat == "std":
                    X[f"rolling_{stat}_{window}"] = rolled.std()
                elif stat == "min":
                    X[f"rolling_{stat}_{window}"] = rolled.min()
                elif stat == "max":
                    X[f"rolling_{stat}_{window}"] = rolled.max()
        
        return X
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        return [
            f"rolling_{stat}_{window}"
            for window in self.windows
            for stat in self.statistics
        ]


class CalendarFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for calendar features.
    
    Extracts hour, day of week, month, etc. from timestamp.
    """
    
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        include_cyclic: bool = True,
    ):
        self.timestamp_col = timestamp_col
        self.include_cyclic = include_cyclic
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create calendar features."""
        X = X.copy()
        
        # Ensure datetime
        ts = pd.to_datetime(X[self.timestamp_col])
        
        # Basic features
        X["hour"] = ts.dt.hour
        X["day_of_week"] = ts.dt.dayofweek
        X["day_of_month"] = ts.dt.day
        X["month"] = ts.dt.month
        X["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        
        # Cyclic encoding for periodic features
        if self.include_cyclic:
            X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
            X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
            X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
            X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
            X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        
        return X
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        features = ["hour", "day_of_week", "day_of_month", "month", "is_weekend"]
        if self.include_cyclic:
            features.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"])
        return features


class FourierFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for Fourier features.
    
    Creates sine and cosine terms to capture seasonality patterns.
    """
    
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        periods: dict = None,
        order: int = 3,
    ):
        self.timestamp_col = timestamp_col
        # Default periods: daily (24h), weekly (168h), yearly (8760h)
        self.periods = periods or {"daily": 24, "weekly": 168}
        self.order = order
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create Fourier features."""
        X = X.copy()
        
        ts = pd.to_datetime(X[self.timestamp_col])
        
        # Calculate hour position in each period
        hour_of_year = (ts.dt.dayofyear - 1) * 24 + ts.dt.hour
        
        for name, period in self.periods.items():
            for k in range(1, self.order + 1):
                if name == "daily":
                    t = ts.dt.hour
                elif name == "weekly":
                    t = ts.dt.dayofweek * 24 + ts.dt.hour
                else:
                    t = hour_of_year
                
                X[f"{name}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
                X[f"{name}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
        
        return X
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        features = []
        for name in self.periods:
            for k in range(1, self.order + 1):
                features.extend([f"{name}_sin_{k}", f"{name}_cos_{k}"])
        return features


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder for categorical features.
    
    Encodes categories by their mean target value.
    """
    
    def __init__(
        self,
        categorical_cols: list[str] = None,
        smoothing: float = 10.0,
    ):
        self.categorical_cols = categorical_cols or []
        self.smoothing = smoothing
        self.encodings_ = {}
        self.global_mean_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Compute target encodings."""
        self.global_mean_ = y.mean()
        
        for col in self.categorical_cols:
            if col not in X.columns:
                continue
            
            # Compute category statistics
            stats = X.groupby(col)[[]].agg(["count"])
            stats.columns = ["count"]
            stats["mean"] = X.groupby(col).apply(lambda x: y.loc[x.index].mean())
            
            # Smoothed encoding
            stats["encoding"] = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            
            self.encodings_[col] = stats["encoding"].to_dict()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply target encodings."""
        X = X.copy()
        
        for col in self.categorical_cols:
            if col not in X.columns:
                continue
            
            X[f"{col}_encoded"] = X[col].map(self.encodings_.get(col, {}))
            X[f"{col}_encoded"] = X[f"{col}_encoded"].fillna(self.global_mean_)
        
        return X
