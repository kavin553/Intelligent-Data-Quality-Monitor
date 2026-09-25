"""
Blast Radius Calculator
Calculates impact of a single table failure across the entire dependency graph.

Uses Breadth-First Search (BFS) to find all affected tables.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
import time


@dataclass
class AffectedTable:
    """Represents a table affected by a failure."""
    table_id: int
    table_name: str
    table_type: str
    criticality: str
    owner: str
    owner_email: str
    depth: int  # Number of hops from source (1 = direct, 2 = indirect)
    path: List[str]  # [source, dep1, dep2, this]
    latency_minutes: int  # How long until this table breaks?
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            'table_id': self.table_id,
            'table_name': self.table_name,
            'table_type': self.table_type,
            'criticality': self.criticality,
            'owner': self.owner,
            'owner_email': self.owner_email,
            'depth': self.depth,
            'path': ' → '.join(self.path),
            'latency_minutes': self.latency_minutes,
        }


@dataclass
class BlastRadiusReport:
    """Complete blast radius analysis for a failure."""
    source_table_id: int
    source_table_name: str
    source_owner: str
    failure_time: str
    
    # Affected tables
    affected_tables: List[AffectedTable] = field(default_factory=list)
    total_affected: int = 0
    direct_affected: int = 0  # Depth = 1
    indirect_affected: int = 0  # Depth > 1
    
    # Severity breakdown
    critical_affected: int = 0
    high_affected: int = 0
    medium_affected: int = 0
    
    # Impact timing
    immediate_impact_count: int = 0  # Breaks in <5 minutes
    near_term_impact_count: int = 0  # Breaks in 5-60 minutes
    delayed_impact_count: int = 0    # Breaks in >1 hour
    
    # Escalation info
    owners_to_notify: Dict[str, List[str]] = field(default_factory=dict)  # owner -> [tables]
    teams_to_notify: Set[str] = field(default_factory=set)
    escalation_channels: Dict[str, str] = field(default_factory=dict)  # owner -> channel
    
    # Computation
    computation_time_ms: float = 0.0
    max_depth: int = 0


class BlastRadiusCalculator:
    """
    Calculates the blast radius (impact) of a table failure.
    
    Uses Breadth-First Search to traverse the dependency graph.
    Answers:
    1. What tables break if this one fails?
    2. How many hops until critical dashboard breaks?
    3. Who do we need to notify?
    4. What's the business impact?
    """
    
    def __init__(self, lineage_db):
        """
        Initialize calculator with lineage database.
        
        Args:
            lineage_db: LineageDB instance
        """
        self.db = lineage_db
    
    def calculate(self, source_table_id: int, max_depth: int = 10) -> BlastRadiusReport:
        """
        Calculate blast radius for a failing table using BFS.
        
        Args:
            source_table_id: ID of the table that broke
            max_depth: Maximum traversal depth (default: 10 hops)
            
        Returns:
            BlastRadiusReport with complete impact analysis
        """
        start_time = time.time()
        
        # Get source table info
        source_table = self.db.get_table(source_table_id)
        if not source_table:
            raise ValueError(f"Table {source_table_id} not found")
        
        # Initialize report
        report = BlastRadiusReport(
            source_table_id=source_table_id,
            source_table_name=source_table.table_name,
            source_owner=source_table.owner,
            failure_time=self._get_current_time(),
        )
        
        # BFS to find all affected tables
        affected = self._bfs_traverse(source_table_id, max_depth)
        
        # Process each affected table
        for affected_info in affected:
            table = self.db.get_table(affected_info['table_id'])
            if not table:
                continue
            
            affected_table = AffectedTable(
                table_id=affected_info['table_id'],
                table_name=table.table_name,
                table_type=table.table_type,
                criticality=table.criticality,
                owner=table.owner,
                owner_email=table.owner_email,
                depth=affected_info['depth'],
                path=affected_info['path'],
                latency_minutes=affected_info['latency_minutes'],
            )
            
            report.affected_tables.append(affected_table)
            report.total_affected += 1
            
            # Count by depth
            if affected_info['depth'] == 1:
                report.direct_affected += 1
            else:
                report.indirect_affected += 1
            
            # Count by criticality
            if table.criticality == 'critical':
                report.critical_affected += 1
            elif table.criticality == 'high':
                report.high_affected += 1
            elif table.criticality == 'medium':
                report.medium_affected += 1
            
            # Count by impact timing
            if affected_info['latency_minutes'] < 5:
                report.immediate_impact_count += 1
            elif affected_info['latency_minutes'] < 60:
                report.near_term_impact_count += 1
            else:
                report.delayed_impact_count += 1
            
            # Track owners to notify
            owner = table.owner
            if owner not in report.owners_to_notify:
                report.owners_to_notify[owner] = []
            report.owners_to_notify[owner].append(table.table_name)
            report.teams_to_notify.add(owner)
        
        # Calculate escalation priority
        report.escalation_channels = self._determine_escalation(report)
        
        # Set max depth
        if report.affected_tables:
            report.max_depth = max(t.depth for t in report.affected_tables)
        
        # Compute time
        report.computation_time_ms = (time.time() - start_time) * 1000
        
        return report
    
    def _bfs_traverse(self, source_table_id: int, max_depth: int) -> List[Dict]:
        """
        Breadth-First Search to find all affected tables.
        
        BFS is key here because it processes tables level-by-level:
        - Depth 1: Direct dependents (closest impact)
        - Depth 2: Their dependents (indirect impact)
        - Depth N: Transitive impacts
        
        Returns:
            List of affected tables with depth, path, and latency info
        """
        visited = set()
        queue = [(source_table_id, 0, [source_table_id], 0)]  # (id, depth, path, cumulative_latency)
        affected = []
        
        while queue:
            # Pop from FRONT of queue (BFS property)
            current_id, depth, path, cumulative_latency = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get tables that depend on current_id
            dependents = self.db.get_direct_dependents(current_id)
            
            for dep in dependents:
                downstream_id = dep.downstream_table_id
                
                if downstream_id not in visited:
                    new_depth = depth + 1
                    new_path = path + [downstream_id]
                    new_latency = cumulative_latency + dep.latency_minutes

                    if new_depth <= max_depth:
                        affected.append({
                            'table_id': downstream_id,
                            'depth': new_depth,
                            'path': [self.db.get_table(t).table_name for t in new_path],
                            'latency_minutes': new_latency,
                        })
                        queue.append((downstream_id, new_depth, new_path, new_latency))
        
        return affected
    
    def _determine_escalation(self, report: BlastRadiusReport) -> Dict[str, str]:
        """
        Determine escalation channels based on impact severity.
        
        Rules:
        - If ANY critical table affected → CRITICAL (page on-call via PagerDuty)
        - If >5 high tables affected → HIGH (Slack + email)
        - If >10 tables affected → MEDIUM (Slack only)
        - Otherwise → LOW (log only)
        """
        escalations = {}
        
        if report.critical_affected > 0:
            # CRITICAL: page on-call
            for owner in report.owners_to_notify:
                escalations[owner] = 'pagerduty'
        elif report.high_affected > 5:
            # HIGH: Slack + email
            for owner in report.owners_to_notify:
                escalations[owner] = 'slack_and_email'
        elif report.total_affected > 10:
            # MEDIUM: Slack
            for owner in report.owners_to_notify:
                escalations[owner] = 'slack'
        else:
            # LOW: log
            for owner in report.owners_to_notify:
                escalations[owner] = 'log'
        
        return escalations
    
    @staticmethod
    def _get_current_time() -> str:
        """Get current time in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
