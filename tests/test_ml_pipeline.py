"""
Tests for ML pipeline module.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml_pipeline.models import XGBoostForecaster, LightGBMForecaster, get_model
from src.ml_pipeline.evaluation import ModelEvaluator


class TestModels:
    """Tests for forecasting models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        n = 500
        X = pd.DataFrame({
            "hour": np.random.randint(0, 24, n),
            "day_of_week": np.random.randint(0, 7, n),
            "lag_1": np.random.uniform(800, 1200, n),
            "lag_24": np.random.uniform(800, 1200, n),
            "rolling_mean_24": np.random.uniform(900, 1100, n),
        })
        y = pd.Series(np.random.uniform(800, 1200, n))
        return X, y
    
    def test_xgboost_fit_predict(self, sample_data):
        """Test XGBoost model fit and predict."""
        X, y = sample_data
        
        model = XGBoostForecaster(n_estimators=10)
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_lightgbm_fit_predict(self, sample_data):
        """Test LightGBM model fit and predict."""
        X, y = sample_data
        
        model = LightGBMForecaster(n_estimators=10)
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_get_model_factory(self):
        """Test model factory function."""
        model = get_model("xgboost", n_estimators=10)
        assert isinstance(model, XGBoostForecaster)
        
        model = get_model("lightgbm", n_estimators=10)
        assert isinstance(model, LightGBMForecaster)
    
    def test_get_model_invalid_type(self):
        """Test model factory with invalid type."""
        with pytest.raises(ValueError):
            get_model("invalid_model")
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test model serialization."""
        X, y = sample_data
        
        model = XGBoostForecaster(n_estimators=10)
        model.fit(X, y)
        
        # Save
        model_path = tmp_path / "model.pkl"
        model.save(model_path)
        
        # Load
        loaded_model = XGBoostForecaster.load(model_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.model_type == "xgboost"
        
        # Predictions should match
        orig_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)


class TestModelEvaluator:
    """Tests for model evaluation."""
    
    def test_mae(self):
        """Test MAE calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "mae" in metrics
        assert abs(metrics["mae"] - 10.0) < 0.01
    
    def test_rmse(self):
        """Test RMSE calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "rmse" in metrics
        assert metrics["rmse"] == 0.0
    
    def test_mape(self):
        """Test MAPE calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 220, 330])
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "mape" in metrics
        assert metrics["mape"] == pytest.approx(10.0, rel=0.01)
    
    def test_r2(self):
        """Test R² calculation."""
        evaluator = ModelEvaluator()
        
        # Perfect predictions
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = y_true.copy()
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "r2" in metrics
        assert metrics["r2"] == pytest.approx(1.0)
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        evaluator = ModelEvaluator()
        
        # Perfect direction matching
        y_true = np.array([100, 150, 200, 150, 100])
        y_pred = np.array([100, 160, 210, 160, 110])
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "directional_accuracy" in metrics
        assert metrics["directional_accuracy"] == 100.0
    
    def test_compare_models(self):
        """Test model comparison."""
        evaluator = ModelEvaluator()
        
        y_true = np.random.uniform(800, 1200, 100)
        
        predictions = {
            "model_a": y_true + np.random.normal(0, 50, 100),
            "model_b": y_true + np.random.normal(0, 100, 100),
        }
        
        comparison = evaluator.compare_models(y_true, predictions)
        
        assert len(comparison) == 2
        assert "mae_rank" in comparison.columns
    
    def test_residual_analysis(self):
        """Test residual analysis."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 210, 310, 410, 510])
        
        analysis = evaluator.residual_analysis(y_true, y_pred)
        
        assert "mean_residual" in analysis
        assert "std_residual" in analysis
        assert analysis["mean_residual"] == pytest.approx(-10.0)
