"""
Hyperparameter optimization using Optuna.
"""

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import structlog

from src.ml_pipeline.models import get_model
from src.ml_pipeline.evaluation import ModelEvaluator

logger = structlog.get_logger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna with Bayesian optimization.
    
    Supports:
    - XGBoost parameter tuning
    - LightGBM parameter tuning
    - Custom parameter spaces
    - Time-series cross-validation
    """
    
    DEFAULT_SEARCH_SPACES = {
        "xgboost": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 12),
            "learning_rate": ("float_log", 0.01, 0.3),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("float_log", 1e-8, 10.0),
            "reg_lambda": ("float_log", 1e-8, 10.0),
        },
        "lightgbm": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 12),
            "learning_rate": ("float_log", 0.01, 0.3),
            "num_leaves": ("int", 20, 100),
            "min_child_samples": ("int", 5, 50),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("float_log", 1e-8, 10.0),
            "reg_lambda": ("float_log", 1e-8, 10.0),
            "reg_lambda": ("float_log", 1e-8, 10.0),
        },
        "random_forest": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 5, 20),
            "min_samples_split": ("int", 2, 10),
            "min_samples_leaf": ("int", 1, 5),
        },
    }
    
    def __init__(
        self,
        model_type: str = "xgboost",
        search_space: dict = None,
        n_trials: int = 50,
        cv_folds: int = 3,
        metric: str = "mae",
        direction: str = "minimize",
        timeout: int = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            model_type: Type of model to optimize
            search_space: Custom search space (overrides default)
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            metric: Metric to optimize
            direction: 'minimize' or 'maximize'
            timeout: Optional timeout in seconds
        """
        self.model_type = model_type
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACES.get(model_type, {})
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.direction = direction
        self.timeout = timeout
        
        self.evaluator = ModelEvaluator()
        self.best_params: dict = {}
        self.best_score: float = float("inf") if direction == "minimize" else float("-inf")
        self.study = None
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: list[str] = None,
    ) -> dict:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_columns: Optional list of feature columns to use
            
        Returns:
            Dictionary with best parameters
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        # Prepare data
        if feature_columns:
            X = X[feature_columns]
        
        logger.info(
            "starting_optimization",
            model_type=self.model_type,
            n_trials=self.n_trials,
            metric=self.metric,
        )
        
        # Create objective function
        def objective(trial):
            params = self._suggest_params(trial)
            score = self._evaluate_params(X, y, params)
            return score
        
        # Create and run study
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
        )
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(
            "optimization_complete",
            best_score=self.best_score,
            best_params=self.best_params,
        )
        
        return self.best_params
    
    def _suggest_params(self, trial) -> dict:
        """Suggest parameters for a trial based on search space."""
        params = {}
        
        for name, config in self.search_space.items():
            param_type = config[0]
            
            if param_type == "int":
                params[name] = trial.suggest_int(name, config[1], config[2])
            elif param_type == "float":
                params[name] = trial.suggest_float(name, config[1], config[2])
            elif param_type == "float_log":
                params[name] = trial.suggest_float(name, config[1], config[2], log=True)
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, config[1])
        
        return params
    
    def _evaluate_params(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict,
    ) -> float:
        """Evaluate parameters using cross-validation."""
        n_samples = len(X)
        min_train_size = int(n_samples * 0.5)
        fold_size = (n_samples - min_train_size) // self.cv_folds
        
        scores = []
        
        for fold in range(self.cv_folds):
            train_end = min_train_size + fold * fold_size
            test_end = min(train_end + fold_size, n_samples)
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            
            if len(X_test) == 0:
                continue
            
            try:
                model = get_model(self.model_type, **params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                metrics = self.evaluator.evaluate(y_test.values, predictions)
                scores.append(metrics[self.metric])
            except Exception as e:
                logger.warning("trial_failed", error=str(e))
                # Return a bad score for failed trials
                return float("inf") if self.direction == "minimize" else float("-inf")
        
        return np.mean(scores) if scores else float("inf")
    
    def get_importance(self) -> Optional[dict]:
        """Get hyperparameter importance from the study."""
        if self.study is None:
            return None
        
        try:
            import optuna
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception:
            return None
    
    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return None
        
        trials = []
        for trial in self.study.trials:
            row = {
                "trial": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                **trial.params,
            }
            trials.append(row)
        
        return pd.DataFrame(trials)


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_trials: int = 50,
    **kwargs,
) -> dict:
    """Convenience function for hyperparameter optimization."""
    optimizer = HyperparameterOptimizer(
        model_type=model_type,
        n_trials=n_trials,
        **kwargs,
    )
    return optimizer.optimize(X, y)
