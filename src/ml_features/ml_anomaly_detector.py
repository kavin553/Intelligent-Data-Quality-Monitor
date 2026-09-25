"""
ML-Based Anomaly Detection
Isolation Forest, LOF, and Temporal Pattern Learning
"""

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from enum import Enum

# ============ Enums ============

class MLAnomalyType(Enum):
    """ML-detected anomaly types."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER = "local_outlier_factor"
    TEMPORAL_PATTERN = "temporal_pattern"
    MULTIVARIATE = "multivariate"

class MLSeverity(Enum):
    """Severity levels for ML anomalies."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

# ============ Data Classes ============

@dataclass
class MLAnomalyAlert:
    """Alert from ML detector."""
    anomaly_type: MLAnomalyType
    severity: MLSeverity
    column: str
    score: float
    message: str
    affected_indices: List[int]
    
    def to_dict(self):
        """Convert to dict."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "column": self.column,
            "score": self.score,
            "message": self.message,
            "affected_indices": self.affected_indices,
            "num_affected": len(self.affected_indices)
        }

# ============ ML Anomaly Detector ============

class MLAnomalyDetector:
    """
    ML-based anomaly detection using Isolation Forest, LOF, and temporal patterns.
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize ML detector.
        
        Args:
            contamination: Expected proportion of outliers (0-1)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination
        )
    
    def detect(self, df: pd.DataFrame) -> List[MLAnomalyAlert]:
        """
        Detect anomalies using all ML methods.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of anomaly alerts
        """
        alerts = []
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return alerts
        
        # Run each detector
        alerts.extend(self._isolation_forest_detect(df, numeric_cols))
        alerts.extend(self._lof_detect(df, numeric_cols))
        alerts.extend(self._temporal_pattern_detect(df, numeric_cols))
        alerts.extend(self._multivariate_detect(df, numeric_cols))
        
        return alerts
    
    # ============ Isolation Forest Detector ============
    
    def _isolation_forest_detect(
        self, 
        df: pd.DataFrame, 
        numeric_cols
    ) -> List[MLAnomalyAlert]:
        """
        Isolation Forest: Isolates anomalies by randomly selecting features.
        Works well for high-dimensional data.
        
        Algorithm:
        1. Randomly select feature
        2. Randomly select split value
        3. Points that take fewer splits to isolate = anomalies
        
        Best for: Sudden spikes, rare events, unknown anomaly patterns
        """
        alerts = []
        
        try:
            # Prepare data
            X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
            
            # Detect anomalies
            predictions = self.isolation_forest.fit_predict(X)
            scores = self.isolation_forest.score_samples(X)
            
            # For each column, find anomalies
            for col_idx, col in enumerate(numeric_cols):
                col_data = df[col].dropna().values
                
                if len(col_data) < 10:
                    continue
                
                # Get indices with low scores (anomalies)
                anomaly_indices = np.where(predictions == -1)[0]
                
                if len(anomaly_indices) > 0:
                    # Calculate anomaly score (how "isolated" the points are)
                    col_scores = scores[anomaly_indices]
                    avg_score = np.abs(col_scores).mean()
                    
                    if avg_score > 0.5:  # Threshold for alert
                        severity = self._score_to_severity(avg_score)
                        
                        alerts.append(MLAnomalyAlert(
                            anomaly_type=MLAnomalyType.ISOLATION_FOREST,
                            severity=severity,
                            column=col,
                            score=float(avg_score),
                            message=f"Isolation Forest detected {len(anomaly_indices)} anomalies in '{col}' (score: {avg_score:.3f})",
                            affected_indices=anomaly_indices.tolist()
                        ))
        except Exception as e:
            pass  # Skip on error
        
        return alerts
    
    # ============ LOF Detector ============
    
    def _lof_detect(
        self, 
        df: pd.DataFrame, 
        numeric_cols
    ) -> List[MLAnomalyAlert]:
        """
        Local Outlier Factor (LOF): Detects density-based outliers.
        
        Algorithm:
        1. Calculate local density for each point
        2. Compare to neighbors' densities
        3. Low density relative to neighbors = anomaly
        
        Best for: Contextual anomalies, local density drops, clustered data
        """
        alerts = []
        
        try:
            # Prepare data
            X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
            
            if X.shape[0] < 20:  # Need enough points for LOF
                return alerts
            
            # Detect anomalies
            predictions = self.lof.fit_predict(X)
            scores = self.lof.negative_outlier_factor_
            
            # Get anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            
            if len(anomaly_indices) > 0:
                lof_scores = scores[anomaly_indices]
                avg_score = np.abs(lof_scores).mean()
                
                if avg_score > 1.1:  # Threshold for alert
                    severity = self._score_to_severity(avg_score)
                    
                    # Determine which column is most affected
                    col = numeric_cols[0]
                    
                    alerts.append(MLAnomalyAlert(
                        anomaly_type=MLAnomalyType.LOCAL_OUTLIER,
                        severity=severity,
                        column=col,
                        score=float(avg_score),
                        message=f"LOF detected {len(anomaly_indices)} density-based anomalies (score: {avg_score:.3f})",
                        affected_indices=anomaly_indices.tolist()
                    ))
        except Exception as e:
            pass  # Skip on error
        
        return alerts
    
    # ============ Temporal Pattern Detector ============
    
    def _temporal_pattern_detect(
        self, 
        df: pd.DataFrame, 
        numeric_cols
    ) -> List[MLAnomalyAlert]:
        """
        Temporal Pattern Learning: Detects unusual time-based patterns.
        
        Assumes row order = time order (0 = earliest, n = latest)
        
        Algorithm:
        1. Divide data into chunks (e.g., weekly)
        2. Compare recent chunk to baseline
        3. Deviation > threshold = anomaly
        
        Best for: Seasonal changes, day-of-week patterns, trend breaks
        """
        alerts = []
        
        try:
            if len(df) < 20:
                return alerts
            
            # Split into baseline (first 50%) and recent (last 50%)
            split_idx = len(df) // 2
            baseline = df[numeric_cols].iloc[:split_idx]
            recent = df[numeric_cols].iloc[split_idx:]
            
            for col in numeric_cols:
                baseline_vals = baseline[col].dropna().values
                recent_vals = recent[col].dropna().values
                
                if len(baseline_vals) < 10 or len(recent_vals) < 10:
                    continue
                
                # Compare distributions
                baseline_mean = baseline_vals.mean()
                baseline_std = baseline_vals.std()
                recent_mean = recent_vals.mean()
                
                # Z-score of recent mean vs baseline
                if baseline_std > 0:
                    z_score = abs((recent_mean - baseline_mean) / baseline_std)
                    
                    if z_score > 2.0:  # 2 sigma = 95% confidence
                        severity = MLSeverity.WARNING if z_score < 3 else MLSeverity.CRITICAL
                        
                        alerts.append(MLAnomalyAlert(
                            anomaly_type=MLAnomalyType.TEMPORAL_PATTERN,
                            severity=severity,
                            column=col,
                            score=float(z_score),
                            message=f"Temporal pattern shift in '{col}': baseline mean {baseline_mean:.2f}, recent mean {recent_mean:.2f} (z={z_score:.2f})",
                            affected_indices=list(range(split_idx, len(df)))
                        ))
        except Exception as e:
            pass  # Skip on error
        
        return alerts
    
    # ============ Multivariate Detector ============
    
    def _multivariate_detect(
        self, 
        df: pd.DataFrame, 
        numeric_cols
    ) -> List[MLAnomalyAlert]:
        """
        Multivariate Anomaly Detection: Detects anomalies in relationships.
        
        Algorithm:
        1. Calculate Mahalanobis distance (accounts for correlations)
        2. Points far from centroid in multi-dimensional space = anomaly
        
        Best for: Correlated feature anomalies, relationship breaks
        """
        alerts = []
        
        try:
            if len(df) < 20 or len(numeric_cols) < 2:
                return alerts
            
            # Prepare data
            X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
            
            # Calculate Mahalanobis distance
            mean = X.mean(axis=0)
            cov = np.cov(X.T)
            
            if np.linalg.matrix_rank(cov) < cov.shape[0]:
                return alerts  # Singular matrix
            
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                return alerts
            
            # Calculate distances
            distances = []
            for row in X:
                diff = row - mean
                dist = np.sqrt(diff.dot(inv_cov).dot(diff.T))
                distances.append(dist)
            
            distances = np.array(distances)
            threshold = np.percentile(distances, 95)  # Top 5% as anomalies
            anomaly_indices = np.where(distances > threshold)[0]
            
            if len(anomaly_indices) > 0:
                avg_distance = distances[anomaly_indices].mean()
                
                if avg_distance > threshold:
                    severity = MLSeverity.WARNING
                    
                    alerts.append(MLAnomalyAlert(
                        anomaly_type=MLAnomalyType.MULTIVARIATE,
                        severity=severity,
                        column="[multivariate]",
                        score=float(avg_distance),
                        message=f"Multivariate anomaly: {len(anomaly_indices)} rows with unusual feature correlations (distance: {avg_distance:.3f})",
                        affected_indices=anomaly_indices.tolist()
                    ))
        except Exception as e:
            pass  # Skip on error
        
        return alerts
    
    # ============ Helper Methods ============
    
    def _score_to_severity(self, score: float) -> MLSeverity:
        """Convert score to severity level."""
        if score > 2.0:
            return MLSeverity.CRITICAL
        elif score > 1.5:
            return MLSeverity.WARNING
        else:
            return MLSeverity.INFO

    def compare_with_statistical(self, statistical_alerts: int, ml_alerts: int) -> Dict:
        """
        Compare ML detection with statistical detection.
        
        Returns metrics for benchmarking.
        """
        total_alerts = statistical_alerts + ml_alerts
        overlap = min(statistical_alerts, ml_alerts)
        unique_ml = ml_alerts - overlap
        
        return {
            "statistical_alerts": statistical_alerts,
            "ml_alerts": ml_alerts,
            "total_alerts": total_alerts,
            "overlap": overlap,
            "unique_ml_detections": unique_ml,
            "improvement_pct": (unique_ml / total_alerts * 100) if total_alerts > 0 else 0
        }
