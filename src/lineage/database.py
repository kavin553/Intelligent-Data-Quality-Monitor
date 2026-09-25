"""
Lineage Database - In-Memory Graph Storage
Stores table metadata and dependencies for blast radius calculation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime


@dataclass
class Table:
    """Represents a table in the lineage graph."""
    table_id: int
    table_name: str
    table_type: str  # 'source', 'transformation', 'dashboard', 'ml_model', 'report'
    owner: str
    owner_email: str
    criticality: str  # 'critical', 'high', 'medium', 'low'
    refresh_frequency: str  # 'hourly', 'daily', '6-hourly', 'real-time'
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            'table_id': self.table_id,
            'table_name': self.table_name,
            'table_type': self.table_type,
            'owner': self.owner,
            'owner_email': self.owner_email,
            'criticality': self.criticality,
            'refresh_frequency': self.refresh_frequency,
        }


@dataclass
class Dependency:
    """Represents a dependency between two tables."""
    upstream_table_id: int
    downstream_table_id: int
    dependency_type: str  # 'direct', 'indirect', 'macro', 'sql'
    latency_minutes: int  # How long until downstream breaks?
    strength: str = 'strong'  # 'strong' (required), 'weak' (optional)
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            'upstream_table_id': self.upstream_table_id,
            'downstream_table_id': self.downstream_table_id,
            'dependency_type': self.dependency_type,
            'latency_minutes': self.latency_minutes,
            'strength': self.strength,
        }


class LineageDB:
    """
    In-memory lineage graph database.
    Stores tables and their dependencies.
    """
    
    def __init__(self):
        """Initialize the database."""
        self.tables: Dict[int, Table] = {}  # table_id -> Table
        self.dependencies: Dict[int, List[Dependency]] = {}  # upstream_id -> [Dependencies]
        self.reverse_dependencies: Dict[int, List[Dependency]] = {}  # downstream_id -> [Dependencies]
        self.next_table_id = 1
    
    def add_table(self, table_name: str, table_type: str, owner: str, 
                  owner_email: str, criticality: str, refresh_frequency: str) -> int:
        """
        Add a table to the lineage graph.
        
        Returns: table_id
        """
        table_id = self.next_table_id
        self.next_table_id += 1
        
        table = Table(
            table_id=table_id,
            table_name=table_name,
            table_type=table_type,
            owner=owner,
            owner_email=owner_email,
            criticality=criticality,
            refresh_frequency=refresh_frequency,
        )
        
        self.tables[table_id] = table
        self.dependencies[table_id] = []
        self.reverse_dependencies[table_id] = []
        
        return table_id
    
    def add_dependency(self, upstream_id: int, downstream_id: int, 
                       latency_minutes: int, dependency_type: str = 'direct'):
        """
        Add a dependency: downstream depends on upstream.
        
        Args:
            upstream_id: Table that produces data
            downstream_id: Table that consumes data
            latency_minutes: How long until downstream breaks if upstream fails?
            dependency_type: Type of dependency
        """
        if upstream_id not in self.tables or downstream_id not in self.tables:
            raise ValueError(f"Table ID not found")
        
        dep = Dependency(
            upstream_table_id=upstream_id,
            downstream_table_id=downstream_id,
            dependency_type=dependency_type,
            latency_minutes=latency_minutes,
        )
        
        self.dependencies[upstream_id].append(dep)
        self.reverse_dependencies[downstream_id].append(dep)
    
    def get_table(self, table_id: int) -> Optional[Table]:
        """Get table by ID."""
        return self.tables.get(table_id)
    
    def get_table_by_name(self, table_name: str) -> Optional[Table]:
        """Get table by name."""
        for table in self.tables.values():
            if table.table_name == table_name:
                return table
        return None
    
    def get_direct_dependents(self, table_id: int) -> List[Dependency]:
        """Get tables that depend on this table (downstream)."""
        return self.dependencies.get(table_id, [])
    
    def get_direct_dependencies(self, table_id: int) -> List[Dependency]:
        """Get tables this table depends on (upstream)."""
        return self.reverse_dependencies.get(table_id, [])
    
    def get_all_tables(self) -> List[Table]:
        """Get all tables."""
        return list(self.tables.values())
    
    def print_graph(self):
        """Pretty print the graph structure."""
        print("\n" + "="*70)
        print("LINEAGE GRAPH")
        print("="*70 + "\n")
        
        for table_id, table in self.tables.items():
            dependents = self.get_direct_dependents(table_id)
            
            if dependents:
                print(f"{table.table_name} ({table.criticality})")
                for dep in dependents:
                    downstream = self.get_table(dep.downstream_table_id)
                    print(f"  └─[{dep.latency_minutes}min]──> {downstream.table_name}")
            else:
                print(f"{table.table_name} (no dependents)")
        
        print("\n" + "="*70 + "\n")
