"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Database
    database_url: str = Field(
        default="sqlite:///./data/electricity.db",
        description="Database connection URL"
    )
    
    # Paths
    model_dir: Path = Field(
        default=Path("./data/models"),
        description="Directory for saved model artifacts"
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base data directory"
    )
    
    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_debug: bool = Field(default=False, description="Enable debug mode")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Weather API (optional)
    openweather_api_key: str | None = Field(
        default=None,
        description="OpenWeatherMap API key for weather data"
    )
    
    # MLflow (optional)
    mlflow_tracking_uri: str = Field(
        default="./mlruns",
        description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="electricity-demand-forecast",
        description="MLflow experiment name"
    )
    
    # Model defaults
    default_model_type: str = Field(
        default="xgboost",
        description="Default model type for predictions"
    )
    prediction_horizons: list[int] = Field(
        default=[1, 24, 168],
        description="Prediction horizons in hours (1h, 24h, 7d)"
    )
    
    # Training
    train_test_split_ratio: float = Field(
        default=0.8,
        description="Train/test split ratio"
    )
    cv_folds: int = Field(
        default=5,
        description="Number of cross-validation folds"
    )
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
