#!/usr/bin/env python
"""
CLI script for training electricity demand forecasting models.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.database.connection import init_db
from src.data_pipeline.ingestion import DataIngestionService
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.validation import DataValidator
from src.ml_pipeline.training import ModelTrainer
from src.ml_pipeline.hyperparameter import HyperparameterOptimizer
from src.ml_pipeline.registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="Train electricity demand forecasting model")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm", "prophet", "random_forest"],
        default="xgboost",
        help="Model type to train",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Prediction horizon in hours",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--set-production",
        action="store_true",
        help="Set as production model after training",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to CSV data file (optional, uses database if not provided)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"Electricity Demand Forecasting - Model Training")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Horizon: {args.horizon} hours")
    print(f"Optimize: {args.optimize}")
    print(f"Set Production: {args.set_production}")
    print()
    
    # Initialize
    settings = get_settings()
    settings.ensure_directories()
    init_db()
    
    # Load data
    print("Loading data...")
    if args.data_file:
        import pandas as pd
        df = pd.read_csv(args.data_file, parse_dates=["timestamp"])
        print(f"Loaded {len(df)} records from file")
    else:
        ingestion = DataIngestionService()
        df = ingestion.load_from_database()
        if df.empty:
            print("No data in database. Please run generate_sample_data.py first.")
            print("Then ingest data using the API or by adding to the database.")
            return 1
        print(f"Loaded {len(df)} records from database")
    
    # Validate data
    print("\nValidating data...")
    validator = DataValidator()
    validation_result = validator.validate(df)
    
    if not validation_result.is_valid:
        print("Data validation issues found:")
        for issue in validation_result.issues:
            print(f"  [{issue.severity.value}] {issue.issue_type}: {issue.message}")
        
        if validation_result.has_critical_issues():
            print("\nCritical issues found. Please fix data before training.")
            return 1
    
    print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Feature engineering
    print("\nEngineering features...")
    feature_engineer = FeatureEngineer()
    featured_df = feature_engineer.fit_transform(df)
    print(f"Created {len(feature_engineer.feature_columns)} features")
    print(f"Samples after feature engineering: {len(featured_df)}")
    
    if len(featured_df) < 100:
        print("Insufficient data after feature engineering. Need at least 100 samples.")
        return 1
    
    # Prepare for horizon
    featured_df["consumption_kwh"] = featured_df["consumption_kwh"].shift(-args.horizon)
    featured_df = featured_df.dropna()
    
    # Hyperparameter optimization
    model_params = {}
    if args.optimize:
        print(f"\nOptimizing hyperparameters ({args.n_trials} trials)...")
        optimizer = HyperparameterOptimizer(
            model_type=args.model,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
        )
        
        X = featured_df[feature_engineer.feature_columns]
        y = featured_df["consumption_kwh"]
        
        model_params = optimizer.optimize(X, y)
        print(f"Best parameters: {model_params}")
        print(f"Best score (MAE): {optimizer.best_score:.4f}")
    
    # Train model
    print(f"\nTraining {args.model} model...")
    trainer = ModelTrainer(
        model_type=args.model,
        feature_columns=feature_engineer.feature_columns,
        cv_folds=args.cv_folds,
    )
    
    model, metrics = trainer.train(
        featured_df,
        model_params=model_params,
        save_model=True,
    )
    
    print("\nTraining Results:")
    print(f"  MAE:  {metrics.get('mae', 'N/A'):.4f}")
    print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
    print(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
    print(f"  R²:   {metrics.get('r2', 'N/A'):.4f}")
    
    # Register model
    print("\nRegistering model...")
    registry = ModelRegistry()
    metadata = registry.register(
        model=model,
        horizon_hours=args.horizon,
        is_production=args.set_production,
    )
    
    print(f"Model registered with ID: {metadata.id}")
    print(f"Version: {metadata.version}")
    print(f"Production: {metadata.is_production}")
    
    # Cross-validation
    print(f"\nCross-validation ({args.cv_folds} folds)...")
    cv_results = trainer.cross_validate(
        featured_df,
        model_params=model_params,
    )
    
    print(f"CV Mean MAE:  {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")
    print(f"CV Mean RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
    
    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
