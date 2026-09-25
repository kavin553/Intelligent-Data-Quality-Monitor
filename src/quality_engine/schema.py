"""
Schema Discovery & Management
Automatically detects column types, statistics, and data contracts.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


class ColumnType(Enum):
    """Supported column types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""
    name: str
    column_type: ColumnType
    nullable: bool
    cardinality: int
    null_count: int
    null_rate: float
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    sample_values: List[Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict, handling enum."""
        return {
            'name': self.name,
            'column_type': self.column_type.value,
            'nullable': self.nullable,
            'cardinality': self.cardinality,
            'null_count': self.null_count,
            'null_rate': self.null_rate,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            'sample_values': self.sample_values,
        }


@dataclass
class TableMetadata:
    """Metadata for an entire table."""
    table_name: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnMetadata]
    discovered_at: str
    total_size_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            'table_name': self.table_name,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'columns': {k: v.to_dict() for k, v in self.columns.items()},
            'discovered_at': self.discovered_at,
            'total_size_mb': self.total_size_mb,
        }


class SchemaDiscovery:
    """Discovers and tracks schema metadata."""
    
    def discover(self, df: pd.DataFrame, table_name: str) -> TableMetadata:
        """
        Automatically discover schema from DataFrame.
        
        Args:
            df: Pandas DataFrame
            table_name: Name of the table
            
        Returns:
            TableMetadata with all discovered properties
        """
        columns = {}
        
        for col in df.columns:
            col_type = self._infer_type(df[col])
            null_count = df[col].isna().sum()
            null_rate = null_count / len(df) if len(df) > 0 else 0.0
            cardinality = df[col].nunique()
            
            metadata = ColumnMetadata(
                name=col,
                column_type=col_type,
                nullable=null_count > 0,
                cardinality=cardinality,
                null_count=null_count,
                null_rate=null_rate,
                min_value=self._safe_minmax(df[col], 'min'),
                max_value=self._safe_minmax(df[col], 'max'),
                mean_value=self._safe_mean(df[col]),
                std_value=self._safe_std(df[col]),
                sample_values=df[col].dropna().head(5).tolist(),
            )
            columns[col] = metadata
        
        # Estimate table size in MB
        total_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        table_metadata = TableMetadata(
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            discovered_at=datetime.utcnow().isoformat(),
            total_size_mb=round(total_size_mb, 2),
        )
        
        return table_metadata
    
    @staticmethod
    def _infer_type(series: pd.Series) -> ColumnType:
        """Infer column type from pandas Series."""
        if len(series) == 0 or series.isna().all():
            return ColumnType.UNKNOWN
        
        dtype = series.dtype
        
        if pd.api.types.is_bool_dtype(dtype):
            return ColumnType.BOOLEAN
        elif pd.api.types.is_integer_dtype(dtype):
            return ColumnType.INTEGER
        elif pd.api.types.is_float_dtype(dtype):
            return ColumnType.FLOAT
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            return ColumnType.STRING
        else:
            return ColumnType.UNKNOWN
    
    @staticmethod
    def _safe_minmax(series: pd.Series, method: str):
        """Safely get min or max, handling different types."""
        try:
            non_null = series.dropna()
            if len(non_null) == 0:
                return None
            if method == 'min':
                return non_null.min()
            else:
                return non_null.max()
        except (TypeError, ValueError):
            return None
    
    @staticmethod
    def _safe_mean(series: pd.Series) -> Optional[float]:
        """Safely calculate mean."""
        try:
            non_null = series.dropna()
            if len(non_null) == 0:
                return None
            if pd.api.types.is_numeric_dtype(series.dtype):
                return float(non_null.mean())
        except (TypeError, ValueError):
            pass
        return None
    
    @staticmethod
    def _safe_std(series: pd.Series) -> Optional[float]:
        """Safely calculate std deviation."""
        try:
            non_null = series.dropna()
            if len(non_null) == 0:
                return None
            if pd.api.types.is_numeric_dtype(series.dtype):
                return float(non_null.std())
        except (TypeError, ValueError):
            pass
        return None
