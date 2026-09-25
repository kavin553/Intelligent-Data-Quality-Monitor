"""
DataShield Quality Engine
Detects 8 core data quality incidents in real-time.
"""

from .schema import (
    SchemaDiscovery,
    ColumnMetadata,
    TableMetadata,
    ColumnType,
)
from .anomaly_detector import (
    AnomalyDetector,
    AnomalyAlert,
    AnomalyType,
    SeverityLevel,
)

__version__ = "0.1.0"
__all__ = [
    "SchemaDiscovery",
    "ColumnMetadata",
    "TableMetadata",
    "ColumnType",
    "AnomalyDetector",
    "AnomalyAlert",
    "AnomalyType",
    "SeverityLevel",
]
