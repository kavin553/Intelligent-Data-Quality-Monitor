"""
Load Testing: Simulate 100K table graph
"""

import time
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from lineage.database import LineageDB
from lineage.blast_radius import BlastRadiusCalculator
from lineage.graph_optimizer import GraphOptimizer

def generate_large_graph(num_tables: int = 10000) -> LineageDB:
    """Generate synthetic table graph."""
    print(f"Generating {num_tables} table graph...")
    
    db = LineageDB()
    table_ids = []
    
    # Create tables
    start = time.time()
    for i in range(num_tables):
        tid = db.add_table(
            f"table_{i}",
            random.choice(["source", "transformation", "dashboard"]),
            f"owner_{i % 100}",
            f"owner_{i % 100}@company.com",
            random.choice(["critical", "high", "medium"]),
            random.choice(["hourly", "daily"])
        )
        table_ids.append(tid)
        
        if (i + 1) % 1000 == 0:
            print(f"  Created {i + 1} tables ({time.time() - start:.1f}s)")
    
    # Create dependencies
    print("Creating dependencies...")
    start = time.time()
    dep_count = 0
    
    for i in range(num_tables):
        num_deps = random.randint(0, 8)
        for _ in range(num_deps):
            if i > 0:
                upstream_idx = random.randint(0, i - 1)
                upstream_id = table_ids[upstream_idx]
                downstream_id = table_ids[i]
                latency = random.randint(5, 240)
                db.add_dependency(upstream_id, downstream_id, latency)
                dep_count += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Created {dep_count} dependencies ({time.time() - start:.1f}s)")
    
    print(f"✅ Graph: {num_tables} tables, {dep_count} dependencies")
    return db

def benchmark_blast_radius(db: LineageDB, num_queries: int = 50):
    """Benchmark blast radius."""
    print(f"\n📊 Benchmarking blast radius ({num_queries} queries)...")
    
    calculator = BlastRadiusCalculator(db)
    times = []
    
    for i in range(num_queries):
        source_id = random.randint(1, len(db.get_all_tables()))
        
        start = time.time()
        report = calculator.calculate(source_id)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            avg = sum(times) / len(times)
            print(f"  Query {i + 1}: {elapsed:.2f}ms (avg: {avg:.2f}ms)")
    
    print(f"\nBlast Radius Stats:")
    print(f"  Mean: {sum(times)/len(times):.2f}ms")
    print(f"  Min: {min(times):.2f}ms")
    print(f"  Max: {max(times):.2f}ms")

def benchmark_graph_metrics(db: LineageDB):
    """Benchmark graph metrics."""
    print(f"\n📊 Computing graph metrics...")
    
    optimizer = GraphOptimizer(db)
    start = time.time()
    metrics = optimizer.compute_graph_metrics()
    elapsed = (time.time() - start) * 1000
    
    print(f"  Time: {elapsed:.2f}ms")
    print(f"  Tables: {metrics.total_tables}")
    print(f"  Dependencies: {metrics.total_dependencies}")
    print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")

def benchmark_propagation(db: LineageDB, num_queries: int = 30):
    """Benchmark probabilistic propagation."""
    print(f"\n📊 Benchmarking propagation ({num_queries} queries)...")
    
    optimizer = GraphOptimizer(db)
    times = []
    
    for i in range(num_queries):
        source_id = random.randint(1, len(db.get_all_tables()))
        
        start = time.time()
        result = optimizer.propagate_failure_probabilistic(source_id)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            avg = sum(times) / len(times)
            print(f"  Query {i + 1}: {elapsed:.2f}ms (avg: {avg:.2f}ms)")
    
    print(f"\nPropagation Stats:")
    print(f"  Mean: {sum(times)/len(times):.2f}ms")
    print(f"  Min: {min(times):.2f}ms")
    print(f"  Max: {max(times):.2f}ms")

if __name__ == "__main__":
    print("=" * 60)
    print("DataShield Load Testing")
    print("=" * 60)
    
    db = generate_large_graph(num_tables=10000)
    benchmark_blast_radius(db, num_queries=50)
    benchmark_graph_metrics(db)
    benchmark_propagation(db, num_queries=30)
    
    print("\n" + "=" * 60)
    print("✅ Load testing complete!")
    print("=" * 60)
