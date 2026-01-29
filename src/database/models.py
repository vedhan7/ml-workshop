"""
SQLAlchemy models for electricity demand forecasting.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class ElectricityConsumption(Base):
    """
    Raw electricity consumption data.
    Stores hourly/sub-hourly electricity usage readings.
    """
    __tablename__ = "electricity_consumption"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    consumption_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Optional location/meter identifiers
    location_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    meter_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Data quality flags
    is_interpolated: Mapped[bool] = mapped_column(Boolean, default=False)
    is_outlier: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("timestamp", "location_id", name="uq_consumption_timestamp_location"),
    )
    
    def __repr__(self) -> str:
        return f"<ElectricityConsumption(timestamp={self.timestamp}, kwh={self.consumption_kwh})>"


class WeatherData(Base):
    """
    Weather data for feature engineering.
    Temperature, humidity, and other weather conditions.
    """
    __tablename__ = "weather_data"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Core weather features
    temperature_celsius: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed_mps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cloud_cover_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Derived features
    feels_like_celsius: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Location
    location_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    
    # Source metadata
    source: Mapped[str] = mapped_column(String(50), default="openweathermap")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("timestamp", "location_id", name="uq_weather_timestamp_location"),
    )
    
    def __repr__(self) -> str:
        return f"<WeatherData(timestamp={self.timestamp}, temp={self.temperature_celsius}°C)>"


class ModelMetadata(Base):
    """
    Metadata for trained ML models.
    Tracks model versions, parameters, and performance metrics.
    """
    __tablename__ = "model_metadata"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Model identification
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # xgboost, lightgbm, prophet, etc.
    
    # Model file location
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Training configuration
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_columns: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    target_column: Mapped[str] = mapped_column(String(100), default="consumption_kwh")
    
    # Prediction horizon
    horizon_hours: Mapped[int] = mapped_column(Integer, default=24)
    
    # Performance metrics
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Example: {"mae": 10.5, "rmse": 15.2, "mape": 5.3}
    
    # Training data info
    training_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    training_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="model")
    
    def __repr__(self) -> str:
        return f"<ModelMetadata(name={self.name}, version={self.version}, type={self.model_type})>"


class Prediction(Base):
    """
    Stored predictions/forecasts.
    Allows tracking prediction accuracy over time.
    """
    __tablename__ = "predictions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Model reference
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_metadata.id"), nullable=False)
    model: Mapped["ModelMetadata"] = relationship("ModelMetadata", back_populates="predictions")
    
    # Prediction timing
    prediction_made_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    target_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Prediction values
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Confidence interval
    prediction_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Actual value (filled in later for accuracy tracking)
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Location
    location_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    
    # Horizon
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    
    def __repr__(self) -> str:
        return f"<Prediction(target={self.target_timestamp}, predicted={self.predicted_value})>"
    
    @property
    def error(self) -> Optional[float]:
        """Calculate prediction error if actual value is available."""
        if self.actual_value is not None:
            return self.actual_value - self.predicted_value
        return None
    
    @property
    def absolute_percentage_error(self) -> Optional[float]:
        """Calculate absolute percentage error if actual value is available."""
        if self.actual_value is not None and self.actual_value != 0:
            return abs((self.actual_value - self.predicted_value) / self.actual_value) * 100
        return None


class FeatureSet(Base):
    """
    Engineered feature sets for model training.
    Stores precomputed features for efficiency.
    """
    __tablename__ = "feature_sets"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Feature set identification
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Data range
    start_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Feature information
    feature_columns: Mapped[list] = mapped_column(JSON, nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Storage location (parquet file path)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Feature engineering parameters
    parameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_featureset_name_version"),
    )
    
    def __repr__(self) -> str:
        return f"<FeatureSet(name={self.name}, version={self.version}, features={self.feature_count})>"
