"""
Tests for ML Anomaly Detector
"""

import pytest
import pandas as pd
import numpy as np
from src.ml_features.ml_anomaly_detector import (
    MLAnomalyDetector,
    MLAnomalyType,
    MLSeverity
)

class TestMLAnomalyDetector:
    """Test ML anomaly detection."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return MLAnomalyDetector(contamination=0.1)
    
    @pytest.fixture
    def healthy_data(self):
        """Create healthy baseline data."""
        np.random.seed(42)
        return pd.DataFrame({
            'value': np.random.normal(100, 10, 100),
            'metric': np.random.normal(50, 5, 100),
        })
    
    def test_isolation_forest_spike(self, detector, healthy_data):
        """Test Isolation Forest detects sudden spike."""
        # Add sudden spike
        anomalous_data = healthy_data.copy()
        anomalous_data.loc[95:99, 'value'] = 500  # 5x spike
        
        alerts = detector.detect(anomalous_data)
        
        # Should detect anomaly
        assert len(alerts) > 0
        assert any(a.anomaly_type == MLAnomalyType.ISOLATION_FOREST for a in alerts)
    
    def test_lof_outlier(self, detector, healthy_data):
        """Test LOF detects density-based outliers."""
        anomalous_data = healthy_data.copy()
        # Add point far from others
        anomalous_data.loc[99] = [500, 500]
        
        alerts = detector.detect(anomalous_data)
        
        # Should detect anomaly
        assert len(alerts) > 0
    
    def test_temporal_pattern(self, detector):
        """Test temporal pattern detection."""
        # Create data with trend break
        data = pd.DataFrame({
            'value': list(range(1, 51)) + list(range(1, 51))  # Repeat pattern
        })
        
        # Break the pattern in recent data
        data.loc[75:99, 'value'] = 500
        
        alerts = detector.detect(data)
        
        # Should detect temporal shift
        temporal_alerts = [a for a in alerts if a.anomaly_type == MLAnomalyType.TEMPORAL_PATTERN]
        assert len(temporal_alerts) > 0
    
    def test_healthy_data_no_alerts(self, detector, healthy_data):
        """Test healthy data produces minimal alerts."""
        alerts = detector.detect(healthy_data)
        
        # Should have few or no alerts (with contamination=0.1, expect ~10 points)
        # Real healthy data should have few true anomalies
        assert len(alerts) <= 5
    
    def test_alert_format(self, detector, healthy_data):
        """Test alert format is correct."""
        # Create data with obvious anomaly
        anomalous_data = healthy_data.copy()
        anomalous_data.loc[99] = [1000, 1000]
        
        alerts = detector.detect(anomalous_data)
        
        if len(alerts) > 0:
            alert = alerts[0]
            assert hasattr(alert, 'anomaly_type')
            assert hasattr(alert, 'severity')
            assert hasattr(alert, 'score')
            assert hasattr(alert, 'message')
            
            # Test to_dict conversion
            alert_dict = alert.to_dict()
            assert 'anomaly_type' in alert_dict
            assert 'severity' in alert_dict
            assert 'score' in alert_dict

class TestMLComparison:
    """Test ML vs Statistical comparison."""
    
    def test_comparison_metrics(self):
        """Test comparison calculation."""
        detector = MLAnomalyDetector()
        
        stats = detector.compare_with_statistical(
            statistical_alerts=10,
            ml_alerts=15
        )
        
        assert stats['statistical_alerts'] == 10
        assert stats['ml_alerts'] == 15
        assert stats['total_alerts'] == 25
        assert stats['unique_ml_detections'] == 5
        assert stats['improvement_pct'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
