"""
Database connection and session management.
"""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.database.models import Base

# Global engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        
        # SQLite specific settings
        connect_args = {}
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        
        _engine = create_engine(
            settings.database_url,
            connect_args=connect_args,
            echo=settings.log_level == "DEBUG",
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


def init_db() -> None:
    """Initialize database, creating all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    Yields a session and ensures cleanup.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Use this in scripts and background tasks.
    
    Usage:
        with get_db_context() as db:
            db.query(...)
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_engine() -> None:
    """Reset engine and session factory. Useful for testing."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
