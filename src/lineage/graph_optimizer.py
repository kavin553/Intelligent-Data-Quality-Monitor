"""
Graph Optimization & Scaling
Incremental updates, probabilistic failure propagation, and performance optimization
for 100K+ table lineage graphs.
"""

from typing import List, Dict, Set, Tuple
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum

# ============ Enums & Data Classes ============

class PropagationModel(Enum):
    """Failure propagation models."""
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    WEIGHTED = "weighted"

@dataclass
class GraphMetrics:
    """Performance metrics for the graph."""
    total_tables: int
    total_dependencies: int
    avg_degree: float
    max_degree: int
    graph_density: float
    computation_time_ms: float
    memory_usage_mb: float

@dataclass
class PropagationResult:
    """Result of probabilistic failure propagation."""
    source_table_id: int
    affected_tables: List[int]
    failure_probabilities: Dict[int, float]
    expected_failures: float
    worst_case_failures: int
    best_case_failures: int
    computation_time_ms: float

# ============ Graph Optimizer ============

class GraphOptimizer:
    """
    Optimized lineage graph for 100K+ tables.
    """
    
    def __init__(self, lineage_db):
        """Initialize optimizer."""
        self.lineage_db = lineage_db
        self.cache = {}
        self.last_update_time = time.time()
        self.update_count = 0
    
    # ============ Incremental Update Tracking ============
    
    def track_incremental_update(self, upstream_id: int, downstream_id: int) -> Dict:
        """Track incremental dependency update."""
        self.update_count += 1
        start = time.time()
        
        affected_nodes = self._get_affected_cache_keys(downstream_id)
        for key in affected_nodes:
            if key in self.cache:
                del self.cache[key]
        
        elapsed = (time.time() - start) * 1000
        
        return {
            "status": "tracked",
            "update_number": self.update_count,
            "affected_cache_entries": len(affected_nodes),
            "computation_time_ms": elapsed,
            "optimization": "Only cached entries invalidated"
        }
    
    def _get_affected_cache_keys(self, table_id: int) -> Set[str]:
        """Get cache keys affected by a table update."""
        affected = set()
        for key in self.cache.keys():
            if str(table_id) in str(key):
                affected.add(key)
        return affected
    
    # ============ Probabilistic Failure Propagation ============
    
    def propagate_failure_probabilistic(
        self, 
        source_table_id: int, 
        failure_probability: float = 1.0,
        latency_threshold_minutes: int = 120
    ) -> PropagationResult:
        """
        Probabilistic failure propagation model.
        
        Algorithm:
        1. Start with P(source fails) = failure_probability
        2. For each downstream table, calculate propagation probability
        3. Account for latency: high latency = lower cascade probability
           P(propagation) *= exp(-latency_minutes / time_constant)
        """
        start = time.time()
        
        affected_tables = []
        failure_probs = {}
        
        # BFS with probability tracking
        queue = [(source_table_id, failure_probability, 0)]
        visited = set()
        
        while queue:
            current_id, current_prob, depth = queue.pop(0)
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Get direct dependents (tables that depend on current_id)
            dependents = self.lineage_db.get_direct_dependents(current_id)
            
            for dependent in dependents:
                downstream_id = dependent.downstream_table_id
                latency = dependent.latency_minutes
                
                # Probability decays with latency
                latency_factor = np.exp(-latency / latency_threshold_minutes)
                propagation_prob = current_prob * latency_factor
                
                if propagation_prob > 0.05:  # Only track significant probabilities
                    if downstream_id not in affected_tables:
                        affected_tables.append(downstream_id)
                    
                    # Store max probability for this table
                    if downstream_id not in failure_probs:
                        failure_probs[downstream_id] = propagation_prob
                    else:
                        failure_probs[downstream_id] = max(failure_probs[downstream_id], propagation_prob)
                    
                    queue.append((downstream_id, propagation_prob, depth + 1))
        
        elapsed = (time.time() - start) * 1000
        
        # Calculate statistics
        expected_failures = sum(failure_probs.values()) if failure_probs else 0
        worst_case = len(failure_probs)
        best_case = len([p for p in failure_probs.values() if p > 0.9]) if failure_probs else 0
        
        return PropagationResult(
            source_table_id=source_table_id,
            affected_tables=affected_tables,
            failure_probabilities=failure_probs,
            expected_failures=expected_failures,
            worst_case_failures=worst_case,
            best_case_failures=best_case,
            computation_time_ms=elapsed
        )
    
    # ============ Graph Metrics & Analysis ============
    
    def compute_graph_metrics(self) -> GraphMetrics:
        """Compute graph metrics for performance analysis."""
        start = time.time()
        
        tables = self.lineage_db.get_all_tables()
        total_tables = len(tables)
        
        # Count dependencies
        total_deps = 0
        degrees = []
        
        for table in tables:
            table_id = table.table_id
            dependents = self.lineage_db.get_direct_dependents(table_id)
            dependencies = self.lineage_db.get_direct_dependencies(table_id)
            
            degree = len(dependents) + len(dependencies)
            degrees.append(degree)
            total_deps += len(dependents)
        
        # Calculate metrics
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        # Density
        max_possible_edges = total_tables * (total_tables - 1)
        density = total_deps / max_possible_edges if max_possible_edges > 0 else 0
        
        elapsed = (time.time() - start) * 1000
        memory_mb = (total_tables * 200 + total_deps * 100) / (1024 * 1024)
        
        return GraphMetrics(
            total_tables=total_tables,
            total_dependencies=total_deps,
            avg_degree=avg_degree,
            max_degree=max_degree,
            graph_density=density,
            computation_time_ms=elapsed,
            memory_usage_mb=memory_mb
        )
    
    # ============ Performance Benchmarking ============
    
    def benchmark_blast_radius(self, source_table_id: int, iterations: int = 100) -> Dict:
        """Benchmark blast radius calculation."""
        from lineage.blast_radius import BlastRadiusCalculator
        
        times = []
        calculator = BlastRadiusCalculator(self.lineage_db)
        
        for _ in range(iterations):
            start = time.time()
            report = calculator.calculate(source_table_id)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        times = np.array(times)
        
        return {
            "source_table_id": source_table_id,
            "iterations": iterations,
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p99_ms": float(np.percentile(times, 99)),
            "p95_ms": float(np.percentile(times, 95)),
            "std_dev_ms": float(np.std(times))
        }
    
    # ============ Caching Utilities ============
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_queries": len(self.cache),
            "memory_kb": sum(len(str(v)) for v in self.cache.values()) / 1024,
            "total_updates": self.update_count
        }
    
    def clear_cache(self):
        """Clear all cached queries."""
        self.cache.clear()
        return {"status": "cache_cleared", "message": "All cached queries invalidated"}
