"""
Tests for Graph Optimizer
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from lineage.database import LineageDB
from lineage.graph_optimizer import GraphOptimizer

class TestGraphOptimizer:
    """Test graph optimization features."""
    
    @pytest.fixture
    def db_with_data(self):
        """Create DB with test data."""
        db = LineageDB()
        
        # Create 10 tables
        table_ids = []
        for i in range(10):
            tid = db.add_table(
                f"table_{i}",
                "transformation",
                f"owner_{i}",
                f"owner_{i}@co.com",
                "high" if i < 5 else "medium",
                "daily"
            )
            table_ids.append(tid)
        
        # Create chain: 0->1->2->3->...->9
        for i in range(9):
            db.add_dependency(table_ids[i], table_ids[i+1], latency_minutes=60)
        
        # Add fan-out from table 2
        db.add_dependency(table_ids[2], table_ids[7], latency_minutes=30)
        db.add_dependency(table_ids[2], table_ids[8], latency_minutes=45)
        
        return db
    
    @pytest.fixture
    def optimizer(self, db_with_data):
        """Create optimizer instance."""
        return GraphOptimizer(db_with_data)
    
    def test_incremental_update_tracking(self, optimizer):
        """Test incremental update tracking."""
        result = optimizer.track_incremental_update(1, 2)
        
        assert result["status"] == "tracked"
        assert result["update_number"] == 1
        assert result["computation_time_ms"] < 100
    
    def test_graph_metrics(self, optimizer):
        """Test graph metrics computation."""
        metrics = optimizer.compute_graph_metrics()
        
        assert metrics.total_tables == 10
        assert metrics.total_dependencies > 0
        assert metrics.avg_degree > 0
        assert metrics.max_degree > 0
        assert 0 <= metrics.graph_density <= 1
        assert metrics.computation_time_ms < 1000
    
    def test_probabilistic_propagation_basic(self, optimizer):
        """Test probabilistic failure propagation runs without error."""
        result = optimizer.propagate_failure_probabilistic(
            source_table_id=0,
            failure_probability=1.0,
            latency_threshold_minutes=120
        )
        
        assert result.source_table_id == 0
        assert result.expected_failures >= 0
        assert result.worst_case_failures >= result.best_case_failures
        assert result.computation_time_ms > 0
    
    def test_probabilistic_probabilities_valid(self, optimizer):
        """Test probability values are valid."""
        result = optimizer.propagate_failure_probabilistic(
            source_table_id=0,
            failure_probability=1.0,
            latency_threshold_minutes=60
        )
        
        # All probabilities should be between 0 and 1
        for prob in result.failure_probabilities.values():
            assert 0 <= prob <= 1.0
    
    def test_cache_stats(self, optimizer):
        """Test cache functionality."""
        stats = optimizer.get_cache_stats()
        assert stats["cached_queries"] == 0
        
        result = optimizer.clear_cache()
        assert result["status"] == "cache_cleared"
    
    def test_incremental_updates_accumulate(self, optimizer):
        """Test multiple incremental updates are tracked."""
        result1 = optimizer.track_incremental_update(1, 2)
        result2 = optimizer.track_incremental_update(2, 3)
        result3 = optimizer.track_incremental_update(3, 4)
        
        assert result1["update_number"] == 1
        assert result2["update_number"] == 2
        assert result3["update_number"] == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
