"""
Anomaly Detection Engine
Detects 8 core failure scenarios in real-time.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from .schema import TableMetadata, ColumnType


class SeverityLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies detected."""
    LATE_ARRIVAL = "late_arrival"
    ROW_COUNT_SPIKE = "row_count_spike"
    NULL_RATE_EXPLOSION = "null_rate_explosion"
    CARDINALITY_COLLAPSE = "cardinality_collapse"
    DISTRIBUTION_SHIFT = "distribution_shift"
    SCHEMA_DRIFT = "schema_drift"
    PII_EXPOSURE = "pii_exposure"
    COST_ANOMALY = "cost_anomaly"


@dataclass
class AnomalyAlert:
    """Alert generated when an anomaly is detected."""
    anomaly_type: AnomalyType
    table_name: str
    column_name: Optional[str]
    severity: SeverityLevel
    message: str
    detected_at: str
    baseline_value: Optional[Any] = None
    observed_value: Optional[Any] = None
    deviation_percent: Optional[float] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            'anomaly_type': self.anomaly_type.value,
            'table_name': self.table_name,
            'column_name': self.column_name,
            'severity': self.severity.value,
            'message': self.message,
            'detected_at': self.detected_at,
            'baseline_value': self.baseline_value,
            'observed_value': self.observed_value,
            'deviation_percent': self.deviation_percent,
            'details': self.details,
        }


class AnomalyDetector:
    """
    Detects anomalies in data using statistical methods.
    Handles 8 core incident scenarios.
    """
    
    def __init__(self, baseline_metadata: TableMetadata):
        """
        Initialize detector with baseline schema.
        
        Args:
            baseline_metadata: Known-good schema metadata
        """
        self.baseline = baseline_metadata
        self.alerts: List[AnomalyAlert] = []
    
    def detect(self, df: pd.DataFrame) -> List[AnomalyAlert]:
        """
        Run full anomaly detection on a DataFrame.
        
        Args:
            df: Data to check
            
        Returns:
            List of AnomalyAlert objects
        """
        self.alerts = []
        
        # Scenario 2: Row count spike
        self._check_row_count(df)
        
        # Scenario 3: Null rate explosion
        self._check_null_rates(df)
        
        # Scenario 4: Cardinality collapse
        self._check_cardinality(df)
        
        # Scenario 5: Distribution shift
        self._check_distribution_shift(df)
        
        # Scenario 6: Schema drift (missing/new columns + column type changes)
        self._check_schema_drift(df)
        self._check_type_drift(df)

        # Scenario 7: PII exposure
        self._check_pii_exposure(df)

        return self.alerts
    
    def _check_row_count(self, df: pd.DataFrame):
        """Scenario 2: Detect row count spike."""
        observed_count = len(df)
        baseline_count = self.baseline.row_count
        
        # Allow ±20% variation
        tolerance = baseline_count * 0.2
        min_expected = baseline_count - tolerance
        max_expected = baseline_count + tolerance
        
        if observed_count < min_expected or observed_count > max_expected:
            deviation = ((observed_count - baseline_count) / baseline_count) * 100
            severity = SeverityLevel.CRITICAL if abs(deviation) > 50 else SeverityLevel.WARNING
            
            alert = AnomalyAlert(
                anomaly_type=AnomalyType.ROW_COUNT_SPIKE,
                table_name=self.baseline.table_name,
                column_name=None,
                severity=severity,
                message=f"Row count is {observed_count}, expected {baseline_count}",
                detected_at=datetime.utcnow().isoformat(),
                baseline_value=baseline_count,
                observed_value=observed_count,
                deviation_percent=deviation,
            )
            self.alerts.append(alert)
    
    def _check_null_rates(self, df: pd.DataFrame):
        """Scenario 3: Detect null rate explosion."""
        for col in df.columns:
            if col not in self.baseline.columns:
                continue
            
            observed_null_rate = df[col].isna().sum() / len(df) if len(df) > 0 else 0
            baseline_null_rate = self.baseline.columns[col].null_rate
            
            # Alert if null rate more than doubles
            max_allowed = baseline_null_rate * 2
            
            if observed_null_rate > max_allowed:
                deviation = ((observed_null_rate - baseline_null_rate) / (baseline_null_rate + 0.001)) * 100
                
                alert = AnomalyAlert(
                    anomaly_type=AnomalyType.NULL_RATE_EXPLOSION,
                    table_name=self.baseline.table_name,
                    column_name=col,
                    severity=SeverityLevel.CRITICAL,
                    message=f"Column '{col}' null rate is {observed_null_rate:.2%}, baseline was {baseline_null_rate:.2%}",
                    detected_at=datetime.utcnow().isoformat(),
                    baseline_value=baseline_null_rate,
                    observed_value=observed_null_rate,
                    deviation_percent=deviation,
                )
                self.alerts.append(alert)
    
    def _check_cardinality(self, df: pd.DataFrame):
        """Scenario 4: Detect cardinality collapse."""
        for col in df.columns:
            if col not in self.baseline.columns:
                continue
            
            observed_cardinality = df[col].nunique()
            baseline_cardinality = self.baseline.columns[col].cardinality
            
            # Allow ±50% variation
            min_expected = baseline_cardinality * 0.5
            max_expected = baseline_cardinality * 1.5
            
            if observed_cardinality < min_expected:
                deviation = ((observed_cardinality - baseline_cardinality) / baseline_cardinality) * 100
                alert = AnomalyAlert(
                    anomaly_type=AnomalyType.CARDINALITY_COLLAPSE,
                    table_name=self.baseline.table_name,
                    column_name=col,
                    severity=SeverityLevel.CRITICAL,
                    message=f"Column '{col}' cardinality dropped to {observed_cardinality}, baseline was {baseline_cardinality}",
                    detected_at=datetime.utcnow().isoformat(),
                    baseline_value=baseline_cardinality,
                    observed_value=observed_cardinality,
                    deviation_percent=deviation,
                )
                self.alerts.append(alert)
    def _check_distribution_shift(self, df: pd.DataFrame):
        """Scenario 5: Detect distribution shift in numeric columns."""
        for col in df.columns:
            if col not in self.baseline.columns:
                continue
            
            baseline_col_meta = self.baseline.columns[col]
            
            # Only check numeric columns
            if baseline_col_meta.column_type not in [ColumnType.INTEGER, ColumnType.FLOAT]:
                continue
            
            # Get numeric data
            observed_values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(observed_values) < 10:  # Need enough samples
                continue
            
            baseline_mean = baseline_col_meta.mean_value
            baseline_std = baseline_col_meta.std_value
            
            if baseline_mean is None or baseline_std is None:
                continue
            
            observed_mean = float(observed_values.mean())
            
            # Check if mean shifted significantly (>3 sigma)
            if baseline_std > 0:
                z_score = abs(observed_mean - baseline_mean) / baseline_std
                if z_score > 3.0:
                    deviation = ((observed_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                    alert = AnomalyAlert(
                        anomaly_type=AnomalyType.DISTRIBUTION_SHIFT,
                        table_name=self.baseline.table_name,
                        column_name=col,
                        severity=SeverityLevel.WARNING,
                        message=f"Column '{col}' mean shifted from {baseline_mean:.2f} to {observed_mean:.2f}",
                        detected_at=datetime.utcnow().isoformat(),
                        baseline_value=baseline_mean,
                        observed_value=observed_mean,
                        deviation_percent=deviation,
                        details={'z_score': z_score},
                    )
                    self.alerts.append(alert)
    
    def _check_schema_drift(self, df: pd.DataFrame):
        """Scenario 6: Detect schema drift (missing or new columns)."""
        expected_columns = set(self.baseline.columns.keys())
        observed_columns = set(df.columns)
        
        # Missing columns
        missing = expected_columns - observed_columns
        if missing:
            alert = AnomalyAlert(
                anomaly_type=AnomalyType.SCHEMA_DRIFT,
                table_name=self.baseline.table_name,
                column_name=None,
                severity=SeverityLevel.CRITICAL,
                message=f"Missing columns: {', '.join(sorted(missing))}",
                detected_at=datetime.utcnow().isoformat(),
                details={'missing_columns': list(missing)},
            )
            self.alerts.append(alert)
        
        # New columns
        new = observed_columns - expected_columns
        if new:
            alert = AnomalyAlert(
                anomaly_type=AnomalyType.SCHEMA_DRIFT,
                table_name=self.baseline.table_name,
                column_name=None,
                severity=SeverityLevel.WARNING,
                message=f"Unexpected new columns: {', '.join(sorted(new))}",
                detected_at=datetime.utcnow().isoformat(),
                details={'new_columns': list(new)},
            )
            self.alerts.append(alert)
    
    def _check_type_drift(self, df: pd.DataFrame):
        """
        Scenario 6b: Detect a column whose *type* changed vs. the baseline.

        A column can stay present (so plain schema drift misses it) yet arrive
        with a different type — e.g. an upstream break sends Quantity as the
        text "12 units" instead of an int. That silently breaks every numeric
        check downstream, so we flag it explicitly.
        """
        from .schema import SchemaDiscovery

        for col in df.columns:
            if col not in self.baseline.columns:
                continue
            baseline_type = self.baseline.columns[col].column_type
            if baseline_type == ColumnType.UNKNOWN:
                continue
            observed_type = SchemaDiscovery._infer_type(df[col])
            if observed_type == ColumnType.UNKNOWN:
                continue

            # Numeric-family types are interchangeable (int vs float is fine).
            numeric = {ColumnType.INTEGER, ColumnType.FLOAT}
            same_family = baseline_type in numeric and observed_type in numeric
            if observed_type != baseline_type and not same_family:
                # A numeric column arriving as string is the dangerous case.
                severity = (
                    SeverityLevel.CRITICAL
                    if baseline_type in numeric
                    else SeverityLevel.WARNING
                )
                alert = AnomalyAlert(
                    anomaly_type=AnomalyType.SCHEMA_DRIFT,
                    table_name=self.baseline.table_name,
                    column_name=col,
                    severity=severity,
                    message=(
                        f"Column '{col}' type changed from "
                        f"{baseline_type.value} to {observed_type.value}"
                    ),
                    detected_at=datetime.utcnow().isoformat(),
                    baseline_value=baseline_type.value,
                    observed_value=observed_type.value,
                    details={"type_drift": True},
                )
                self.alerts.append(alert)

    def _check_pii_exposure(self, df: pd.DataFrame):
        """Scenario 7: Detect PII in columns that shouldn't have it."""
        # Heuristic PII patterns
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'
        
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                sample_values = df[col].dropna().head(100).astype(str)
                
                # Email detection
                email_count = sample_values.str.contains(email_pattern, regex=True, na=False).sum()
                if email_count > len(sample_values) * 0.3:  # If >30% look like emails
                    alert = AnomalyAlert(
                        anomaly_type=AnomalyType.PII_EXPOSURE,
                        table_name=self.baseline.table_name,
                        column_name=col,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Column '{col}' appears to contain email addresses (PII exposure detected)",
                        detected_at=datetime.utcnow().isoformat(),
                        details={'pattern_matched': 'email', 'sample_count': int(email_count)},
                    )
                    self.alerts.append(alert)
                
                # SSN detection
                ssn_count = sample_values.str.contains(ssn_pattern, regex=True, na=False).sum()
                if ssn_count > 0:
                    alert = AnomalyAlert(
                        anomaly_type=AnomalyType.PII_EXPOSURE,
                        table_name=self.baseline.table_name,
                        column_name=col,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Column '{col}' appears to contain SSN data (CRITICAL PII exposure)",
                        detected_at=datetime.utcnow().isoformat(),
                        details={'pattern_matched': 'ssn', 'sample_count': int(ssn_count)},
                    )
                    self.alerts.append(alert)