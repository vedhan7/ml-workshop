"""
Tests for data pipeline module.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.validation import DataValidator, ValidationSeverity
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.transformers import (
    LagFeatureTransformer,
    RollingStatisticsTransformer,
    CalendarFeatureTransformer,
)


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        validator = DataValidator()
        result = validator.validate(pd.DataFrame())
        
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "empty_dataframe"
    
    def test_validate_missing_columns(self):
        """Test validation when required columns are missing."""
        validator = DataValidator()
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        result = validator.validate(df)
        
        assert not result.is_valid
        assert any(i.issue_type == "missing_column" for i in result.issues)
    
    def test_validate_valid_data(self):
        """Test validation of valid data."""
        validator = DataValidator()
        
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "consumption_kwh": np.random.uniform(800, 1200, 100),
        })
        
        result = validator.validate(df)
        assert result.is_valid
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        validator = DataValidator(outlier_threshold=1.5)
        
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="h")
        consumption = np.random.uniform(900, 1100, 100)
        consumption[50] = 5000  # Outlier
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "consumption_kwh": consumption,
        })
        
        result = validator.validate(df)
        assert any(i.issue_type == "outliers" for i in result.issues)
    
    def test_detect_negative_values(self):
        """Test detection of negative values."""
        validator = DataValidator()
        
        timestamps = pd.date_range(start="2024-01-01", periods=10, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "consumption_kwh": [100, 200, -50, 300, 400, 500, 600, 700, 800, 900],
        })
        
        result = validator.validate(df)
        assert any(i.issue_type == "negative_values" for i in result.issues)
    
    def test_detect_timestamp_gaps(self):
        """Test detection of timestamp gaps."""
        validator = DataValidator()
        
        # Create timestamps with a gap
        timestamps = list(pd.date_range(start="2024-01-01", periods=5, freq="h"))
        timestamps.append(pd.Timestamp("2024-01-01 10:00:00"))  # 5 hour gap
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "consumption_kwh": [100, 200, 300, 400, 500, 600],
        })
        
        result = validator.validate(df)
        assert any(i.issue_type == "timestamp_gaps" for i in result.issues)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = pd.date_range(start="2024-01-01", periods=500, freq="h")
        return pd.DataFrame({
            "timestamp": timestamps,
            "consumption_kwh": np.random.uniform(800, 1200, 500),
        })
    
    def test_fit_transform_creates_features(self, sample_data):
        """Test that fit_transform creates features."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        assert len(result) > 0
        assert len(engineer.feature_columns) > 0
    
    def test_lag_features_created(self, sample_data):
        """Test that lag features are created."""
        engineer = FeatureEngineer(lag_hours=[1, 24])
        result = engineer.fit_transform(sample_data)
        
        assert "lag_1h" in result.columns
        assert "lag_24h" in result.columns
    
    def test_calendar_features_created(self, sample_data):
        """Test that calendar features are created."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
    
    def test_rolling_features_created(self, sample_data):
        """Test that rolling features are created."""
        engineer = FeatureEngineer(rolling_windows=[24])
        result = engineer.fit_transform(sample_data)
        
        assert any("rolling" in col for col in result.columns)
    
    def test_fourier_features_created(self, sample_data):
        """Test that Fourier features are created."""
        engineer = FeatureEngineer(fourier_order=2)
        result = engineer.fit_transform(sample_data)
        
        assert any("daily_sin" in col for col in result.columns)
        assert any("daily_cos" in col for col in result.columns)


class TestTransformers:
    """Tests for sklearn-compatible transformers."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="h")
        return pd.DataFrame({
            "timestamp": timestamps,
            "consumption_kwh": np.random.uniform(800, 1200, 100),
        })
    
    def test_lag_transformer(self, sample_data):
        """Test LagFeatureTransformer."""
        transformer = LagFeatureTransformer(lag_periods=[1, 24])
        result = transformer.fit_transform(sample_data)
        
        assert "lag_1" in result.columns
        assert "lag_24" in result.columns
    
    def test_rolling_transformer(self, sample_data):
        """Test RollingStatisticsTransformer."""
        transformer = RollingStatisticsTransformer(
            windows=[24],
            statistics=["mean", "std"],
        )
        result = transformer.fit_transform(sample_data)
        
        assert "rolling_mean_24" in result.columns
        assert "rolling_std_24" in result.columns
    
    def test_calendar_transformer(self, sample_data):
        """Test CalendarFeatureTransformer."""
        transformer = CalendarFeatureTransformer(include_cyclic=True)
        result = transformer.fit_transform(sample_data)
        
        assert "hour" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
