"""
Integration Tests for Lineage Graph & Blast Radius Calculator
"""

import sys
sys.path.insert(0, 'src')

from lineage.database import LineageDB
from lineage.blast_radius import BlastRadiusCalculator


def test_simple_chain():
    """Test: A -> B -> C (simple chain)"""
    db = LineageDB()
    
    a_id = db.add_table('A', 'source', 'team1', 'team1@co.com', 'critical', 'hourly')
    b_id = db.add_table('B', 'transformation', 'team2', 'team2@co.com', 'high', 'daily')
    c_id = db.add_table('C', 'dashboard', 'team3', 'team3@co.com', 'critical', 'daily')
    
    db.add_dependency(a_id, b_id, latency_minutes=60)
    db.add_dependency(b_id, c_id, latency_minutes=60)
    
    calculator = BlastRadiusCalculator(db)
    report = calculator.calculate(a_id)
    
    assert report.total_affected == 2
    assert report.direct_affected == 1
    assert report.indirect_affected == 1
    assert report.max_depth == 2
    assert report.critical_affected == 1
    assert report.high_affected == 1
    
    print("✅ test_simple_chain passed")


def test_fan_out():
    """Test: A -> B, A -> C, A -> D (fan-out)"""
    db = LineageDB()
    
    a_id = db.add_table('A', 'source', 'team1', 'team1@co.com', 'critical', 'hourly')
    b_id = db.add_table('B', 'dashboard', 'team2', 'team2@co.com', 'critical', 'daily')
    c_id = db.add_table('C', 'dashboard', 'team3', 'team3@co.com', 'high', 'daily')
    d_id = db.add_table('D', 'ml_model', 'team4', 'team4@co.com', 'high', 'real-time')
    
    db.add_dependency(a_id, b_id, latency_minutes=30)
    db.add_dependency(a_id, c_id, latency_minutes=45)
    db.add_dependency(a_id, d_id, latency_minutes=5)
    
    calculator = BlastRadiusCalculator(db)
    report = calculator.calculate(a_id)
    
    assert report.total_affected == 3
    assert report.direct_affected == 3
    assert report.critical_affected == 1
    assert report.high_affected == 2
    assert report.near_term_impact_count == 3
    
    print("✅ test_fan_out passed")


def test_complex_graph():
    """Test: Complex graph with multiple paths"""
    db = LineageDB()
    
    orders_id = db.add_table('orders', 'source', 'data-eng', 'data-eng@co.com', 'critical', 'hourly')
    summary_id = db.add_table('order_summary', 'transformation', 'analytics', 'analytics@co.com', 'high', 'daily')
    report_id = db.add_table('revenue_report', 'dashboard', 'finance', 'finance@co.com', 'critical', 'daily')
    dashboard_id = db.add_table('executive_dashboard', 'dashboard', 'finance', 'finance@co.com', 'critical', 'daily')
    metrics_id = db.add_table('user_metrics', 'transformation', 'data-eng', 'data-eng@co.com', 'high', 'hourly')
    ml_id = db.add_table('personalization_ml', 'ml_model', 'ml-team', 'ml@co.com', 'high', 'daily')
    recommender_id = db.add_table('recommendation_engine', 'ml_model', 'ml-team', 'ml@co.com', 'critical', 'real-time')
    
    db.add_dependency(orders_id, summary_id, latency_minutes=60)
    db.add_dependency(summary_id, report_id, latency_minutes=60)
    db.add_dependency(report_id, dashboard_id, latency_minutes=120)
    db.add_dependency(orders_id, metrics_id, latency_minutes=30)
    db.add_dependency(metrics_id, ml_id, latency_minutes=60)
    db.add_dependency(metrics_id, recommender_id, latency_minutes=5)
    
    calculator = BlastRadiusCalculator(db)
    report = calculator.calculate(orders_id)
    
    assert report.total_affected == 6
    assert report.direct_affected == 2
    assert report.indirect_affected == 4
    assert report.critical_affected == 3
    assert report.high_affected == 3
    assert 'pagerduty' in report.escalation_channels.values()
    
    print("✅ test_complex_graph passed")


def test_orphan_table():
    """Test: Table with no dependencies"""
    db = LineageDB()
    
    orphan_id = db.add_table('orphan', 'transformation', 'team1', 'team1@co.com', 'low', 'daily')
    
    calculator = BlastRadiusCalculator(db)
    report = calculator.calculate(orphan_id)
    
    assert report.total_affected == 0
    assert report.direct_affected == 0
    assert report.computation_time_ms < 1.0
    
    print("✅ test_orphan_table passed")


def test_computation_speed():
    """Test: BFS computation is sub-millisecond fast"""
    db = LineageDB()
    
    # Create 10 tables in a chain (realistic depth)
    table_ids = []
    for i in range(10):
        tid = db.add_table(f'table_{i:02d}', 'transformation', f'team_{i%3}', f'team{i%3}@co.com', 'high', 'hourly')
        table_ids.append(tid)
    
    # Create chain: 0->1->2->...->9
    for i in range(9):
        db.add_dependency(table_ids[i], table_ids[i+1], latency_minutes=30)
    
    calculator = BlastRadiusCalculator(db)
    report = calculator.calculate(table_ids[0], max_depth=50)
    
    # Should find all 9 downstream tables
    assert report.total_affected == 9, f"Expected 9, got {report.total_affected}"
    
    # Computation should be very fast (BFS is O(V+E))
    assert report.computation_time_ms < 5.0, f"Should be fast, took {report.computation_time_ms}ms"
    
    print(f"✅ test_computation_speed passed ({report.computation_time_ms:.2f}ms for 10-table chain)")


if __name__ == '__main__':
    test_simple_chain()
    test_fan_out()
    test_complex_graph()
    test_orphan_table()
    test_computation_speed()
    
    print("\n" + "="*70)
    print("✅ All 5 integration tests passed!")
    print("="*70)
