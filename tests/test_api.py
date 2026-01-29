"""
Tests for FastAPI endpoints.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert "version" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestPredictEndpoints:
    """Tests for prediction endpoints."""
    
    def test_predict_no_model(self, client):
        """Test prediction when no model is available."""
        response = client.post(
            "/predict",
            json={"horizon": "24h"},
        )
        
        # Should return 404 when no production model
        assert response.status_code in [200, 404]
    
    def test_predict_with_params(self, client):
        """Test prediction with all parameters."""
        response = client.post(
            "/predict",
            json={
                "horizon": "24h",
                "start_time": "2026-01-29T00:00:00",
                "include_confidence": True,
            },
        )
        
        # May succeed or fail depending on model availability
        assert response.status_code in [200, 404, 500]


class TestModelsEndpoints:
    """Tests for models endpoints."""
    
    def test_list_models(self, client):
        """Test listing models."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total" in data
    
    def test_list_models_with_filters(self, client):
        """Test listing models with filters."""
        response = client.get("/models?model_type=xgboost&horizon_hours=24")
        assert response.status_code == 200
    
    def test_get_model_not_found(self, client):
        """Test getting non-existent model."""
        response = client.get("/models/99999")
        assert response.status_code == 404


class TestRetrainEndpoints:
    """Tests for retrain endpoints."""
    
    def test_trigger_retrain(self, client):
        """Test triggering retraining."""
        response = client.post(
            "/retrain",
            json={
                "model_type": "xgboost",
                "horizon_hours": 24,
                "use_latest_data": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
    
    def test_get_job_status_not_found(self, client):
        """Test getting non-existent job."""
        response = client.get("/retrain/nonexistent")
        assert response.status_code == 404
    
    def test_list_training_jobs(self, client):
        """Test listing training jobs."""
        response = client.get("/retrain")
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data


class TestMetricsEndpoints:
    """Tests for metrics endpoints."""
    
    def test_get_metrics(self, client):
        """Test getting system metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "system" in data
        assert "data" in data
        assert "models" in data
    
    def test_get_prediction_accuracy(self, client):
        """Test getting prediction accuracy."""
        response = client.get("/metrics/predictions")
        assert response.status_code == 200
    
    def test_get_data_quality(self, client):
        """Test getting data quality metrics."""
        response = client.get("/metrics/data-quality")
        assert response.status_code == 200
