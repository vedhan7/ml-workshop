"""
Data validation module for quality checks on electricity consumption data.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    column: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    affected_rows: int = 0
    affected_indices: list = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "column": i.column,
                    "type": i.issue_type,
                    "severity": i.severity.value,
                    "message": i.message,
                    "affected_rows": i.affected_rows,
                }
                for i in self.issues
            ],
            "summary": self.summary,
        }


class DataValidator:
    """
    Validates electricity consumption data for quality issues.
    
    Checks performed:
    - Missing values
    - Outliers (using IQR method)
    - Timestamp gaps/continuity
    - Data type validation
    - Range validation
    """
    
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        consumption_col: str = "consumption_kwh",
        expected_frequency: str = "h",  # hourly
        outlier_threshold: float = 3.0,  # IQR multiplier
    ):
        self.timestamp_col = timestamp_col
        self.consumption_col = consumption_col
        self.expected_frequency = expected_frequency
        self.outlier_threshold = outlier_threshold
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform all validation checks on the dataframe.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(is_valid=True)
        
        if df.empty:
            result.add_issue(ValidationIssue(
                column="",
                issue_type="empty_dataframe",
                severity=ValidationSeverity.CRITICAL,
                message="DataFrame is empty",
            ))
            return result
        
        # Required columns check
        self._check_required_columns(df, result)
        if result.has_critical_issues():
            return result
        
        # Individual checks
        self._check_missing_values(df, result)
        self._check_data_types(df, result)
        self._check_timestamp_continuity(df, result)
        self._check_outliers(df, result)
        self._check_value_ranges(df, result)
        self._check_duplicates(df, result)
        
        # Summary statistics
        result.summary = self._generate_summary(df)
        
        logger.info(
            "validation_complete",
            is_valid=result.is_valid,
            issue_count=len(result.issues),
        )
        
        return result
    
    def _check_required_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check if required columns exist."""
        for col in [self.timestamp_col, self.consumption_col]:
            if col not in df.columns:
                result.add_issue(ValidationIssue(
                    column=col,
                    issue_type="missing_column",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Required column '{col}' is missing",
                ))
    
    def _check_missing_values(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for missing values in key columns."""
        for col in [self.timestamp_col, self.consumption_col]:
            if col not in df.columns:
                continue
            
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                severity = (
                    ValidationSeverity.ERROR if missing_pct > 5
                    else ValidationSeverity.WARNING
                )
                result.add_issue(ValidationIssue(
                    column=col,
                    issue_type="missing_values",
                    severity=severity,
                    message=f"{missing_count} missing values ({missing_pct:.1f}%)",
                    affected_rows=missing_count,
                    affected_indices=df[df[col].isna()].index.tolist()[:100],
                ))
    
    def _check_data_types(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check if columns have correct data types."""
        # Timestamp check
        if self.timestamp_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
                result.add_issue(ValidationIssue(
                    column=self.timestamp_col,
                    issue_type="invalid_dtype",
                    severity=ValidationSeverity.ERROR,
                    message=f"Expected datetime, got {df[self.timestamp_col].dtype}",
                ))
        
        # Consumption should be numeric
        if self.consumption_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[self.consumption_col]):
                result.add_issue(ValidationIssue(
                    column=self.consumption_col,
                    issue_type="invalid_dtype",
                    severity=ValidationSeverity.ERROR,
                    message=f"Expected numeric, got {df[self.consumption_col].dtype}",
                ))
    
    def _check_timestamp_continuity(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for gaps in timestamp sequence."""
        if self.timestamp_col not in df.columns:
            return
        
        df_sorted = df.sort_values(self.timestamp_col)
        timestamps = pd.to_datetime(df_sorted[self.timestamp_col])
        
        # Expected gap based on frequency
        try:
            expected_delta = pd.Timedelta(self.expected_frequency)
        except ValueError:
            # Handle pandas value error for units without numbers (e.g. 'h' -> '1h')
            if self.expected_frequency.isalpha():
                expected_delta = pd.Timedelta(f"1{self.expected_frequency}")
            else:
                raise
        
        # Calculate actual gaps
        gaps = timestamps.diff()
        unexpected_gaps = gaps[gaps > expected_delta * 1.5]
        
        if len(unexpected_gaps) > 0:
            # Find the largest gaps
            largest_gaps = unexpected_gaps.nlargest(5)
            gap_descriptions = [
                f"{gap} at index {idx}" for idx, gap in largest_gaps.items()
            ]
            
            result.add_issue(ValidationIssue(
                column=self.timestamp_col,
                issue_type="timestamp_gaps",
                severity=ValidationSeverity.WARNING,
                message=f"{len(unexpected_gaps)} timestamp gaps detected. Largest: {', '.join(gap_descriptions[:3])}",
                affected_rows=len(unexpected_gaps),
            ))
    
    def _check_outliers(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for outliers using IQR method."""
        if self.consumption_col not in df.columns:
            return
        
        values = df[self.consumption_col].dropna()
        if len(values) < 10:
            return
        
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr
        
        outliers = df[
            (df[self.consumption_col] < lower_bound) |
            (df[self.consumption_col] > upper_bound)
        ]
        
        if len(outliers) > 0:
            outlier_pct = (len(outliers) / len(df)) * 100
            severity = (
                ValidationSeverity.ERROR if outlier_pct > 5
                else ValidationSeverity.WARNING
            )
            result.add_issue(ValidationIssue(
                column=self.consumption_col,
                issue_type="outliers",
                severity=severity,
                message=f"{len(outliers)} outliers detected ({outlier_pct:.1f}%). Range: [{lower_bound:.2f}, {upper_bound:.2f}]",
                affected_rows=len(outliers),
                affected_indices=outliers.index.tolist()[:100],
            ))
    
    def _check_value_ranges(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for physically impossible values."""
        if self.consumption_col not in df.columns:
            return
        
        # Negative consumption is usually invalid
        negative_count = (df[self.consumption_col] < 0).sum()
        if negative_count > 0:
            result.add_issue(ValidationIssue(
                column=self.consumption_col,
                issue_type="negative_values",
                severity=ValidationSeverity.ERROR,
                message=f"{negative_count} negative consumption values detected",
                affected_rows=negative_count,
            ))
        
        # Zero consumption might be suspicious
        zero_count = (df[self.consumption_col] == 0).sum()
        if zero_count > 0:
            zero_pct = (zero_count / len(df)) * 100
            if zero_pct > 10:
                result.add_issue(ValidationIssue(
                    column=self.consumption_col,
                    issue_type="zero_values",
                    severity=ValidationSeverity.WARNING,
                    message=f"{zero_count} zero values ({zero_pct:.1f}%)",
                    affected_rows=zero_count,
                ))
    
    def _check_duplicates(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for duplicate timestamps."""
        if self.timestamp_col not in df.columns:
            return
        
        duplicates = df[df.duplicated(subset=[self.timestamp_col], keep=False)]
        if len(duplicates) > 0:
            result.add_issue(ValidationIssue(
                column=self.timestamp_col,
                issue_type="duplicates",
                severity=ValidationSeverity.ERROR,
                message=f"{len(duplicates)} duplicate timestamp entries",
                affected_rows=len(duplicates),
            ))
    
    def _generate_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics."""
        summary = {
            "total_rows": len(df),
            "date_range": {},
            "consumption_stats": {},
        }
        
        if self.timestamp_col in df.columns and not df[self.timestamp_col].isna().all():
            timestamps = pd.to_datetime(df[self.timestamp_col])
            summary["date_range"] = {
                "start": timestamps.min().isoformat(),
                "end": timestamps.max().isoformat(),
                "span_days": (timestamps.max() - timestamps.min()).days,
            }
        
        if self.consumption_col in df.columns:
            values = df[self.consumption_col].dropna()
            if len(values) > 0:
                summary["consumption_stats"] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                }
        
        return summary


def impute_missing_values(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    consumption_col: str = "consumption_kwh",
    method: str = "interpolate",
) -> pd.DataFrame:
    """
    Impute missing values in the consumption column.
    
    Args:
        df: DataFrame to impute
        timestamp_col: Timestamp column name
        consumption_col: Consumption column name
        method: Imputation method ('interpolate', 'mean', 'median', 'forward_fill')
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    df = df.sort_values(timestamp_col)
    
    if method == "interpolate":
        df[consumption_col] = df[consumption_col].interpolate(method="time")
    elif method == "mean":
        df[consumption_col] = df[consumption_col].fillna(df[consumption_col].mean())
    elif method == "median":
        df[consumption_col] = df[consumption_col].fillna(df[consumption_col].median())
    elif method == "forward_fill":
        df[consumption_col] = df[consumption_col].ffill()
    
    return df
