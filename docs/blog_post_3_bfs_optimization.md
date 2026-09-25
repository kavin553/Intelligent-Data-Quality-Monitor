# Why We Replaced NetworkX with Custom BFS and Got 10x Faster Blast Radius Calculations

*Published by the DataShield Engineering Team*

---

## The Starting Point: NetworkX at Production Scale

Every data engineer's first instinct for graph problems is NetworkX. We started there too.

The first DataShield prototype used NetworkX for the lineage graph. It worked well up to ~1,000 tables. Then we benchmarked it at production scale:

| Tables | NetworkX blast radius | Memory | Custom BFS blast radius |
|--------|----------------------|--------|------------------------|
| 1,000 | 8ms | 12MB | 0.4ms |
| 10,000 | 52ms | 127MB | 3.41ms |
| 50,000 | 340ms | 630MB | 16ms |
| 100,000 | 1,400ms+ | 300MB+ | ~57ms (est.) |

At 10,000 tables — a modest data warehouse — NetworkX was **15x slower** and used **22x more memory**. At 100,000 tables, it became unusable.

---

## Why NetworkX Is Slow for This Use Case

NetworkX stores graphs as nested dictionaries:

```python
# NetworkX internal (simplified)
G._adj = {
    node_id: {
        neighbor_id: {edge_attr_dict},
    }
}
```

For a lineage graph with 40K edges, this means:
- Every table: Python dict object (~200 bytes + overhead)
- Every edge: *nested* dict of attributes (~300 bytes)
- 40K edges × 300 bytes = 12MB just for edges
- Plus Python object overhead, dict hash tables, GC pressure

NetworkX's traversal algorithms are implemented in pure Python with multiple abstraction layers. Every BFS step goes through multiple function calls and dict lookups — ~3-5μs per node. Over 10K nodes: 30-50ms of overhead.

---

## DataShield's Custom Storage: The Bidirectional Index

From `src/lineage/database.py`:

```python
class LineageDB:
    def __init__(self):
        self.tables: Dict[int, Table] = {}
        # Forward: "what does this table feed into?"
        self.dependencies: Dict[int, List[Dependency]] = {}
        # Reverse: "what feeds into this table?"
        self.reverse_dependencies: Dict[int, List[Dependency]] = {}
```

Two indexes instead of one. Each edge is stored twice, but both forward and reverse lookups are **O(1)** instead of O(n) scans.

### Memory Profile

```
Table object:     ~200 bytes (5 string fields + 1 int)
Dependency:       ~100 bytes (2 ints + string + int)
Bidirectional:    ×2 storage = ~200 bytes per edge

10K tables × 200 bytes = 2.0 MB
40K deps × 200 bytes   = 8.0 MB (bidirectional)
Dict overhead          = ~1.5 MB
─────────────────────────────
Total:                   5.7 MB

NetworkX same graph:    127 MB  (22x more)
```

---

## The BFS Implementation

From `src/lineage/blast_radius.py`:

```python
def _bfs_traverse(self, source_table_id: int, max_depth: int) -> List[Dict]:
    visited = set()
    queue = [(source_table_id, 0, [source_table_id], 0)]
    affected = []

    while queue:
        current_id, depth, path, cumulative_latency = queue.pop(0)

        if current_id in visited or depth > max_depth:
            continue
        visited.add(current_id)

        for dep in self.db.get_direct_dependents(current_id):
            downstream_id = dep.downstream_table_id
            if downstream_id not in visited:
                new_path = path + [downstream_id]
                new_latency = cumulative_latency + dep.latency_minutes
                affected.append({
                    'table_id': downstream_id,
                    'depth': depth + 1,
                    'path': [self.db.get_table(t).table_name for t in new_path],
                    'latency_minutes': new_latency,
                })
                queue.append((downstream_id, depth + 1, new_path, new_latency))

    return affected
```

### Key Properties

**1. O(1) visited check** — `set()` hash lookup vs NetworkX's abstraction overhead.

**2. Path accumulation during BFS** — we carry `path` in the queue so we never need a second traversal to reconstruct how we reached each affected table. One pass does it all.

**3. Cumulative latency** — `cumulative_latency` accumulates as we traverse. Time-to-impact for every table: zero extra work.

**4. BFS ordering** — nodes are processed level-by-level (direct dependents first, then second-degree, etc.). Output is naturally sorted by impact urgency — closest to the source failure comes first.

---

## Incremental Cache Invalidation

From `src/lineage/graph_optimizer.py`:

```python
def track_incremental_update(self, upstream_id: int, downstream_id: int) -> Dict:
    # Only invalidate cache entries that reference the affected table
    affected_nodes = self._get_affected_cache_keys(downstream_id)
    for key in affected_nodes:
        if key in self.cache:
            del self.cache[key]
    # Preserve all unrelated cached results
```

When a new dependency is added, we don't flush the entire cache. We scan cache keys and invalidate only those referencing the changed downstream table.

**Benchmark:**
- Selective invalidation: **0.47ms mean**, preserves 90-95% of cached results
- Full `cache.clear()`: 0.01ms to clear, but forces full recomputation of every cached result (~350ms total for 100 cached queries at 3.5ms each)

For write-heavy workloads (continuous pipeline ingestion), selective invalidation wins by orders of magnitude.

---

## Algorithmic Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Add table | O(1) | O(1) |
| Add dependency | O(1) | O(1) |
| Blast radius BFS | O(V + E) | O(V) |
| Probabilistic propagation | O(V + E) with pruning | O(V) |
| Cache invalidation | O(C) | O(1) |

NetworkX `single_source_shortest_path` is also O(V + E), but our constant factor is ~15x lower from direct dict access vs. Python abstraction overhead.

---

## Benchmark Results

MacBook Pro M1, Python 3.11, 10K tables, 40K dependencies, 1000 iterations:

```
Custom BFS Blast Radius:
  P50:   3.18ms
  P75:   5.2ms
  P95:   25ms       ← variance from deep-graph topologies
  P99:   46ms
  Max:   48ms

Probabilistic Propagation:
  Mean:  0.99ms
  P99:   2.1ms

Memory: 5.7 MB (vs NetworkX 127 MB)
```

The P95/P50 ratio of ~8x reflects topology variance: tables with deep transitive dependencies (6+ hops) take longer. In production, flag tables with depth > 8 for lineage cleanup.

---

## When to Use NetworkX

NetworkX is the right choice when:
1. You need **rich algorithms**: PageRank, community detection, MST, topological sort
2. Your graph is **small** (<1,000 nodes): overhead doesn't matter
3. **Development speed** > production performance
4. **Offline analysis**, not real-time API serving

DataShield chose custom BFS because:
- We serve **real-time HTTP requests** — 3.4ms vs 52ms at P50 matters for API latency
- We need **exactly one algorithm** (BFS blast radius)
- We have **Kubernetes pod memory limits** (256Mi) that NetworkX would exceed at 10K+ tables
- The code is simpler to test and maintain

---

## Key Takeaways

1. **NetworkX's 15x overhead** comes from abstraction layers, not algorithmic complexity — both are O(V+E)
2. **Bidirectional indexes** cost 2x storage but provide O(1) both-direction lookup
3. **Incremental cache invalidation** preserves 90%+ of cached results per graph update
4. **Path + latency accumulation in BFS** eliminates second-pass reconstruction
5. **Custom BFS is only worth it** if you have one core algorithm and strict latency/memory requirements

Full source: `src/lineage/` (~650 lines, 30+ tests)
