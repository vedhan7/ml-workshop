"""
Model drift detection for monitoring prediction quality.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""
    has_drift: bool
    drift_score: float
    metric: str
    threshold: float
    details: dict
    
    def to_dict(self) -> dict:
        return {
            "has_drift": self.has_drift,
            "drift_score": self.drift_score,
            "metric": self.metric,
            "threshold": self.threshold,
            "details": self.details,
        }


class DriftDetector:
    """
    Detects drift in model predictions and data distributions.
    
    Methods:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Mean/variance monitoring
    """
    
    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        mean_threshold: float = 0.1,
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.mean_threshold = mean_threshold
    
    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> DriftResult:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change (drift detected)
        """
        # Create bins from baseline
        bins = np.histogram_bin_edges(baseline, bins=n_bins)
        
        # Get distributions
        baseline_counts, _ = np.histogram(baseline, bins=bins)
        current_counts, _ = np.histogram(current, bins=bins)
        
        # Convert to proportions (avoid division by zero)
        baseline_props = (baseline_counts + 1) / (len(baseline) + n_bins)
        current_props = (current_counts + 1) / (len(current) + n_bins)
        
        # Calculate PSI
        psi = np.sum(
            (current_props - baseline_props) * np.log(current_props / baseline_props)
        )
        
        return DriftResult(
            has_drift=psi >= self.psi_threshold,
            drift_score=float(psi),
            metric="psi",
            threshold=self.psi_threshold,
            details={
                "baseline_mean": float(np.mean(baseline)),
                "current_mean": float(np.mean(current)),
                "baseline_std": float(np.std(baseline)),
                "current_std": float(np.std(current)),
            },
        )
    
    def kolmogorov_smirnov_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """
        Perform Kolmogorov-Smirnov test for distribution shift.
        """
        from scipy import stats
        
        statistic, p_value = stats.ks_2samp(baseline, current)
        
        return DriftResult(
            has_drift=p_value < 0.05,  # 5% significance level
            drift_score=float(statistic),
            metric="ks_statistic",
            threshold=self.ks_threshold,
            details={
                "p_value": float(p_value),
                "statistic": float(statistic),
            },
        )
    
    def check_mean_drift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """
        Check for drift in mean values.
        """
        baseline_mean = np.mean(baseline)
        current_mean = np.mean(current)
        
        if baseline_mean == 0:
            relative_change = float("inf") if current_mean != 0 else 0
        else:
            relative_change = abs(current_mean - baseline_mean) / abs(baseline_mean)
        
        return DriftResult(
            has_drift=relative_change > self.mean_threshold,
            drift_score=float(relative_change),
            metric="mean_drift",
            threshold=self.mean_threshold,
            details={
                "baseline_mean": float(baseline_mean),
                "current_mean": float(current_mean),
                "absolute_change": float(current_mean - baseline_mean),
            },
        )
    
    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, DriftResult]:
        """
        Detect drift in multiple features.
        """
        results = {}
        
        for col in feature_columns:
            if col not in baseline_df.columns or col not in current_df.columns:
                continue
            
            baseline = baseline_df[col].dropna().values
            current = current_df[col].dropna().values
            
            if len(baseline) < 10 or len(current) < 10:
                continue
            
            results[col] = self.calculate_psi(baseline, current)
        
        return results
    
    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> dict[str, DriftResult]:
        """
        Detect drift in prediction distribution.
        """
        return {
            "psi": self.calculate_psi(baseline_predictions, current_predictions),
            "mean_drift": self.check_mean_drift(baseline_predictions, current_predictions),
        }
    
    def monitor_accuracy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        baseline_mae: float,
    ) -> DriftResult:
        """
        Monitor prediction accuracy compared to baseline.
        """
        current_mae = float(np.mean(np.abs(predictions - actuals)))
        
        if baseline_mae == 0:
            relative_change = float("inf") if current_mae != 0 else 0
        else:
            relative_change = (current_mae - baseline_mae) / baseline_mae
        
        return DriftResult(
            has_drift=relative_change > self.mean_threshold,
            drift_score=relative_change,
            metric="accuracy_drift",
            threshold=self.mean_threshold,
            details={
                "baseline_mae": baseline_mae,
                "current_mae": current_mae,
                "degradation_percent": relative_change * 100,
            },
        )


def detect_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    method: str = "psi",
) -> DriftResult:
    """Convenience function for drift detection."""
    detector = DriftDetector()
    
    if method == "psi":
        return detector.calculate_psi(baseline, current)
    elif method == "ks":
        return detector.kolmogorov_smirnov_test(baseline, current)
    elif method == "mean":
        return detector.check_mean_drift(baseline, current)
    else:
        raise ValueError(f"Unknown method: {method}")
