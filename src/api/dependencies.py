"""
FastAPI dependencies for injection.
"""

from typing import Generator

from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.database.connection import get_db
from src.ml_pipeline.registry import ModelRegistry


def get_settings_dependency() -> Settings:
    """Dependency for getting settings."""
    return get_settings()


def get_db_dependency() -> Generator[Session, None, None]:
    """Dependency for database sessions."""
    yield from get_db()


def get_model_registry() -> ModelRegistry:
    """Dependency for model registry."""
    return ModelRegistry()
