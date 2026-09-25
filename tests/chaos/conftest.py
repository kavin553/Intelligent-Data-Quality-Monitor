"""
Shared fixtures for chaos engineering tests.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Make src importable from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from quality_engine.schema import SchemaDiscovery
from lineage.database import LineageDB


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_baseline_df():
    """
    Standard 1 000-row DataFrame used as a clean baseline.

    Columns:
        user_id  – unique integer identifier
        age      – integer in [20, 70)
        salary   – float salary value
        city     – one of five categorical cities
        active   – boolean flag
    """
    rng = np.random.default_rng(seed=42)
    n = 1_000

    cities = ["New York", "London", "Tokyo", "Berlin", "Sydney"]

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1, dtype=np.int64),
            "age": rng.integers(20, 70, size=n),
            "salary": rng.uniform(30_000, 150_000, size=n).round(2),
            "city": rng.choice(cities, size=n),
            "active": rng.choice([True, False], size=n),
        }
    )
    return df


@pytest.fixture
def sample_baseline_metadata(sample_baseline_df):
    """
    Pre-computed SchemaDiscovery result for *sample_baseline_df*.

    Returns a TableMetadata instance that the AnomalyDetector can use
    directly as its baseline.
    """
    discovery = SchemaDiscovery()
    return discovery.discover(sample_baseline_df, "users")


# ---------------------------------------------------------------------------
# Lineage fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_lineage_db():
    """
    A LineageDB containing five tables arranged in a simple linear chain:

        raw_events (id=1)
            └─[10min]──> cleaned_events (id=2)
                            └─[5min]──> aggregated_stats (id=3)
                                            └─[15min]──> daily_report (id=4)
                                                            └─[30min]──> executive_dashboard (id=5)

    All tables use realistic metadata values so blast-radius calculations
    exercise the full reporting path.
    """
    db = LineageDB()

    t1 = db.add_table(
        table_name="raw_events",
        table_type="source",
        owner="data_engineering",
        owner_email="de@example.com",
        criticality="high",
        refresh_frequency="real-time",
    )
    t2 = db.add_table(
        table_name="cleaned_events",
        table_type="transformation",
        owner="data_engineering",
        owner_email="de@example.com",
        criticality="high",
        refresh_frequency="hourly",
    )
    t3 = db.add_table(
        table_name="aggregated_stats",
        table_type="transformation",
        owner="analytics",
        owner_email="analytics@example.com",
        criticality="medium",
        refresh_frequency="daily",
    )
    t4 = db.add_table(
        table_name="daily_report",
        table_type="report",
        owner="analytics",
        owner_email="analytics@example.com",
        criticality="medium",
        refresh_frequency="daily",
    )
    t5 = db.add_table(
        table_name="executive_dashboard",
        table_type="dashboard",
        owner="bi_team",
        owner_email="bi@example.com",
        criticality="critical",
        refresh_frequency="daily",
    )

    db.add_dependency(upstream_id=t1, downstream_id=t2, latency_minutes=10)
    db.add_dependency(upstream_id=t2, downstream_id=t3, latency_minutes=5)
    db.add_dependency(upstream_id=t3, downstream_id=t4, latency_minutes=15)
    db.add_dependency(upstream_id=t4, downstream_id=t5, latency_minutes=30)

    return db
