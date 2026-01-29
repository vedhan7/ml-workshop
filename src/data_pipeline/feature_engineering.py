"""
Feature engineering pipeline for electricity demand forecasting.
Creates time-series features from raw consumption data.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from src.config import get_settings
from src.data_pipeline.transformers import (
    CalendarFeatureTransformer,
    FourierFeatureTransformer,
    LagFeatureTransformer,
    RollingStatisticsTransformer,
)

logger = structlog.get_logger(__name__)


class FeatureEngineer:
    """
    Creates features for electricity demand forecasting.
    
    Features created:
    - Lag features (past consumption values)
    - Rolling statistics (mean, std, min, max)
    - Calendar features (hour, day, month, is_weekend, is_holiday)
    - Fourier terms (for seasonality)
    - Weather features (if available)
    """
    
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        target_col: str = "consumption_kwh",
        lag_hours: list[int] = None,
        rolling_windows: list[int] = None,
        fourier_order: int = 4,
        include_holidays: bool = True,
    ):
        """
        Initialize feature engineer.
        
        Args:
            timestamp_col: Name of timestamp column
            target_col: Name of target column
            lag_hours: List of lag hours (default: [1, 2, 3, 6, 12, 24, 48, 168])
            rolling_windows: List of rolling window sizes in hours
            fourier_order: Order for Fourier features
            include_holidays: Whether to include holiday features
        """
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.lag_hours = lag_hours or [1, 2, 3, 6, 12, 24, 48, 168]
        self.rolling_windows = rolling_windows or [6, 12, 24, 48, 168]
        self.fourier_order = fourier_order
        self.include_holidays = include_holidays
        
        self.settings = get_settings()
        self._feature_columns: list[str] = []
    
    @property
    def feature_columns(self) -> list[str]:
        """Get list of generated feature column names."""
        return self._feature_columns
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create all features from raw data.
        
        Args:
            df: DataFrame with timestamp and consumption columns
            weather_df: Optional weather data DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("starting_feature_engineering", rows=len(df))
        
        df = df.copy()
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Ensure timestamp is datetime
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Set timestamp as index for easier time operations
        df = df.set_index(self.timestamp_col)
        
        # Create lag features
        df = self._create_lag_features(df)
        
        # Create rolling statistics
        df = self._create_rolling_features(df)
        
        # Create calendar features
        df = self._create_calendar_features(df)
        
        # Create Fourier features for seasonality
        df = self._create_fourier_features(df)
        
        # Merge weather data if available
        if weather_df is not None:
            df = self._merge_weather_features(df, weather_df)
        
        # Reset index
        df = df.reset_index()
        
        # Drop rows with NaN from lag/rolling features
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        # Track feature columns (exclude timestamp and target)
        self._feature_columns = [
            col for col in df.columns
            if col not in [self.timestamp_col, self.target_col]
            and not col.startswith("_")
        ]
        
        logger.info(
            "feature_engineering_complete",
            features=len(self._feature_columns),
            rows=len(df),
            dropped_rows=dropped,
        )
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for past consumption values."""
        for lag in self.lag_hours:
            col_name = f"lag_{lag}h"
            df[col_name] = df[self.target_col].shift(lag)
        
        # Same hour previous day and week
        df["lag_same_hour_yesterday"] = df[self.target_col].shift(24)
        df["lag_same_hour_last_week"] = df[self.target_col].shift(168)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        for window in self.rolling_windows:
            # Shift by 1 to avoid data leakage
            rolled = df[self.target_col].shift(1).rolling(window=window)
            
            df[f"rolling_mean_{window}h"] = rolled.mean()
            df[f"rolling_std_{window}h"] = rolled.std()
            df[f"rolling_min_{window}h"] = rolled.min()
            df[f"rolling_max_{window}h"] = rolled.max()
        
        # Rate of change
        df["consumption_diff_1h"] = df[self.target_col].diff(1)
        df["consumption_diff_24h"] = df[self.target_col].diff(24)
        
        return df
    
    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar-based features."""
        # Basic time features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
        
        # Boolean features
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)
        
        # Time of day categories
        df["is_morning"] = df["hour"].between(6, 11).astype(int)
        df["is_afternoon"] = df["hour"].between(12, 17).astype(int)
        df["is_evening"] = df["hour"].between(18, 22).astype(int)
        df["is_night"] = (~df["hour"].between(6, 22)).astype(int)
        
        # Peak hours (typically 17:00-21:00)
        df["is_peak_hour"] = df["hour"].between(17, 21).astype(int)
        
        if self.include_holidays:
            df = self._add_holiday_features(df)
        
        return df
    
    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday features (basic implementation)."""
        # Simple holiday detection (can be extended with a proper holiday library)
        # For now, mark common Indian holidays and weekends
        
        # Initialize as 0
        df["is_holiday"] = 0
        
        # Mark some fixed holidays (can be configured)
        fixed_holidays = [
            (1, 1),   # New Year
            (1, 26),  # Republic Day (India)
            (8, 15),  # Independence Day (India)
            (10, 2),  # Gandhi Jayanti
            (12, 25), # Christmas
        ]
        
        for month, day in fixed_holidays:
            mask = (df.index.month == month) & (df.index.day == day)
            df.loc[mask, "is_holiday"] = 1
        
        return df
    
    def _create_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Fourier features to capture seasonality."""
        # Daily seasonality (24 hours)
        hours_in_day = 24
        for k in range(1, self.fourier_order + 1):
            df[f"daily_sin_{k}"] = np.sin(2 * np.pi * k * df["hour"] / hours_in_day)
            df[f"daily_cos_{k}"] = np.cos(2 * np.pi * k * df["hour"] / hours_in_day)
        
        # Weekly seasonality (168 hours)
        hours_in_week = 168
        hour_of_week = df["day_of_week"] * 24 + df["hour"]
        for k in range(1, self.fourier_order + 1):
            df[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * hour_of_week / hours_in_week)
            df[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * hour_of_week / hours_in_week)
        
        # Yearly seasonality (8760 hours)
        day_of_year = df.index.dayofyear
        for k in range(1, min(3, self.fourier_order) + 1):
            df[f"yearly_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365)
            df[f"yearly_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365)
        
        return df
    
    def _merge_weather_features(
        self,
        df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge weather data with consumption data."""
        weather_df = weather_df.copy()
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        weather_df = weather_df.set_index("timestamp")
        
        # Select weather columns
        weather_cols = [
            col for col in weather_df.columns
            if col in ["temperature_celsius", "humidity_percent", "wind_speed_mps",
                       "cloud_cover_percent", "feels_like_celsius"]
        ]
        
        if not weather_cols:
            return df
        
        # Merge on nearest timestamp
        df = df.join(weather_df[weather_cols], how="left")
        
        # Forward fill missing weather data
        for col in weather_cols:
            df[col] = df[col].ffill().bfill()
        
        return df
    
    def save_features(
        self,
        df: pd.DataFrame,
        name: str,
        version: str = "v1",
    ) -> Path:
        """
        Save feature set to parquet file.
        
        Args:
            df: Featured DataFrame
            name: Name for the feature set
            version: Version string
            
        Returns:
            Path to saved file
        """
        processed_dir = self.settings.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = processed_dir / f"{name}_{version}.parquet"
        df.to_parquet(file_path, index=False)
        
        logger.info("saved_features", path=str(file_path), rows=len(df))
        return file_path
    
    def load_features(self, name: str, version: str = "v1") -> pd.DataFrame:
        """
        Load feature set from parquet file.
        
        Args:
            name: Name of the feature set
            version: Version string
            
        Returns:
            Featured DataFrame
        """
        file_path = self.settings.data_dir / "processed" / f"{name}_{version}.parquet"
        df = pd.read_parquet(file_path)
        
        # Restore feature columns list
        self._feature_columns = [
            col for col in df.columns
            if col not in [self.timestamp_col, self.target_col]
        ]
        
        return df


def create_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    target_col: str = "consumption_kwh",
    **kwargs,
) -> pd.DataFrame:
    """Convenience function for feature engineering."""
    engineer = FeatureEngineer(
        timestamp_col=timestamp_col,
        target_col=target_col,
        **kwargs,
    )
    return engineer.fit_transform(df)
