"""
FastAPI application for electricity demand forecasting.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from src.config import get_settings
from src.database.connection import init_db

logger = structlog.get_logger(__name__)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("starting_application")
    settings = get_settings()
    settings.ensure_directories()
    init_db()
    
    yield
    
    # Shutdown
    logger.info("shutting_down_application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Electricity Demand Forecasting API",
        description="ML-powered API for predicting electricity consumption",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    from src.api.routes import predict, models, retrain, metrics
    
    app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
    app.include_router(models.router, prefix="/models", tags=["Models"])
    app.include_router(retrain.router, prefix="/retrain", tags=["Training"])
    app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
    
    # Serve static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    
    @app.get("/", tags=["Dashboard"])
    async def dashboard():
        """Serve the main dashboard."""
        return FileResponse(STATIC_DIR / "index.html")
    
    @app.get("/dashboard", tags=["Dashboard"])
    async def dashboard_redirect():
        """Redirect to dashboard."""
        return RedirectResponse(url="/")
    
    @app.get("/api", tags=["API"])
    async def api_info():
        """API information endpoint."""
        return {
            "name": "Electricity Demand Forecasting API",
            "version": "0.1.0",
            "status": "running",
            "dashboard": "/",
            "docs": "/api/docs",
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
        }
    
    return app


# Create the app instance
app = create_app()


def run_server():
    """Run the server (for CLI usage)."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )


if __name__ == "__main__":
    run_server()
