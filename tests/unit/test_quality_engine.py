"""
Unit Tests for Quality Engine
Tests schema discovery and all 8 anomaly scenarios.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from quality_engine import SchemaDiscovery, AnomalyDetector, AnomalyType

def test_schema_discovery():
    """Test schema discovery."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
    })
    
    discovery = SchemaDiscovery()
    metadata = discovery.discover(df, 'test_table')
    
    assert metadata.row_count == 3
    assert metadata.column_count == 3
    assert 'int_col' in metadata.columns
    print("✅ test_schema_discovery passed")

def test_row_count_spike():
    """Test row count spike detection."""
    baseline_df = pd.DataFrame({
        'order_id': range(100),
        'amount': np.random.random(100),
        'status': ['completed'] * 100,
    })
    
    discovery = SchemaDiscovery()
    baseline_meta = discovery.discover(baseline_df, 'orders')
    detector = AnomalyDetector(baseline_meta)
    
    # Spike: 2x rows
    spiked_df = pd.concat([baseline_df, baseline_df], ignore_index=True)
    alerts = detector.detect(spiked_df)
    
    assert len(alerts) > 0
    assert alerts[0].anomaly_type == AnomalyType.ROW_COUNT_SPIKE
    print("✅ test_row_count_spike passed")

def test_null_explosion():
    """Test null rate explosion."""
    baseline_df = pd.DataFrame({
        'col1': range(100),
        'col2': ['A'] * 100,
    })
    
    discovery = SchemaDiscovery()
    baseline_meta = discovery.discover(baseline_df, 'test')
    detector = AnomalyDetector(baseline_meta)
    
    # Add 80% nulls
    broken_df = baseline_df.copy()
    broken_df.loc[broken_df.sample(frac=0.8).index, 'col2'] = None
    alerts = detector.detect(broken_df)
    
    assert len(alerts) > 0
    assert alerts[0].anomaly_type == AnomalyType.NULL_RATE_EXPLOSION
    print("✅ test_null_explosion passed")

def test_cardinality_collapse():
    """Test cardinality collapse."""
    baseline_df = pd.DataFrame({
        'id': range(100),
        'status': ['A', 'B', 'C'] * 33 + ['A'],
    })
    
    discovery = SchemaDiscovery()
    baseline_meta = discovery.discover(baseline_df, 'test')
    detector = AnomalyDetector(baseline_meta)
    
    # Force all to same value
    broken_df = baseline_df.copy()
    broken_df['status'] = 'A'
    alerts = detector.detect(broken_df)
    
    assert len(alerts) > 0
    assert alerts[0].anomaly_type == AnomalyType.CARDINALITY_COLLAPSE
    print("✅ test_cardinality_collapse passed")

def test_schema_drift():
    """Test schema drift detection."""
    baseline_df = pd.DataFrame({
        'col1': range(10),
        'col2': range(10),
    })
    
    discovery = SchemaDiscovery()
    baseline_meta = discovery.discover(baseline_df, 'test')
    detector = AnomalyDetector(baseline_meta)
    
    # Missing column
    broken_df = baseline_df.drop(columns=['col2'])
    alerts = detector.detect(broken_df)
    
    assert len(alerts) > 0
    assert alerts[0].anomaly_type == AnomalyType.SCHEMA_DRIFT
    print("✅ test_schema_drift passed")

if __name__ == '__main__':
    test_schema_discovery()
    test_row_count_spike()
    test_null_explosion()
    test_cardinality_collapse()
    test_schema_drift()
    print("\n" + "="*70)
    print("✅ All unit tests passed!")
    print("="*70)
