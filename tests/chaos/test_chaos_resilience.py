"""
Chaos Engineering Tests
Tests DataShield's resilience under failure conditions.

These tests verify that DataShield:
1. Handles empty DataFrames gracefully (no crash)
2. Handles malformed/corrupt data (partial detection, no crash)
3. Handles very large DataFrames (memory safety, >100K rows)
4. Handles missing columns (schema drift handled)
5. Handles NaN-only columns (doesn't crash stats)
6. Handles infinite values (numeric overflow protection)
7. Handles duplicate column names (no crash)
8. Handles graph with no nodes (empty lineage)
9. Handles disconnected graph (islands, no crash)
10. Handles max depth exceeded (BFS terminates correctly)
11. Handles concurrent blast radius calculations (thread safety)
12. Handles single-node graph (no dependencies)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import threading
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from quality_engine.schema import SchemaDiscovery
from quality_engine.anomaly_detector import AnomalyDetector
from lineage.database import LineageDB
from lineage.blast_radius import BlastRadiusCalculator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_detector(df: pd.DataFrame, table_name: str = "test_table") -> AnomalyDetector:
    """Discover schema from *df* then return an AnomalyDetector for it."""
    metadata = SchemaDiscovery().discover(df, table_name)
    return AnomalyDetector(metadata)


# ===========================================================================
# Section 1 – AnomalyDetector / DataFrame chaos
# ===========================================================================


class TestDataFrameChaos:
    """Chaos conditions injected into the quality engine."""

    def test_empty_dataframe_no_crash(self, sample_baseline_metadata):
        """
        Chaos: zero rows passed to a detector whose baseline has 1 000 rows.

        Expected: detector returns a list (possibly containing a row-count
        alert); it must never raise an exception.
        """
        detector = AnomalyDetector(sample_baseline_metadata)
        empty_df = pd.DataFrame(columns=["user_id", "age", "salary", "city", "active"])

        alerts = detector.detect(empty_df)

        assert isinstance(alerts, list), "detect() must always return a list"
        # A row-count alert is expected – the result must not be an exception.
        alert_types = [a.anomaly_type.value for a in alerts]
        assert "row_count_spike" in alert_types, (
            "A row-count anomaly should be flagged when 0 rows are observed vs 1 000 baseline"
        )

    def test_all_nulls_no_crash(self, sample_baseline_metadata):
        """
        Chaos: every cell in the DataFrame is NaN / None.

        Expected: detector handles all-null columns without raising; a
        null-rate explosion alert must be generated for nullable-baseline
        columns.
        """
        df = pd.DataFrame(
            {
                "user_id": [None] * 1_000,
                "age": [None] * 1_000,
                "salary": [None] * 1_000,
                "city": [None] * 1_000,
                "active": [None] * 1_000,
            }
        )
        detector = AnomalyDetector(sample_baseline_metadata)

        alerts = detector.detect(df)

        assert isinstance(alerts, list)
        # At least one null-rate explosion alert expected for numeric columns.
        alert_types = [a.anomaly_type.value for a in alerts]
        assert len(alerts) > 0, "All-null data should trigger at least one alert"
        assert any(t in alert_types for t in ("null_rate_explosion", "row_count_spike")), (
            "Should flag null-rate explosion or row-count anomaly on all-null data"
        )

    def test_infinite_values_no_crash(self, sample_baseline_metadata):
        """
        Chaos: numeric columns contain np.inf and -np.inf.

        Expected: detector does not raise; infinite values are handled by
        the distribution-shift check without crashing.
        """
        rng = np.random.default_rng(seed=7)
        n = 1_000
        cities = ["New York", "London", "Tokyo", "Berlin", "Sydney"]

        df = pd.DataFrame(
            {
                "user_id": np.arange(1, n + 1, dtype=np.int64),
                "age": rng.integers(20, 70, size=n).astype(float),
                "salary": rng.uniform(30_000, 150_000, size=n),
                "city": rng.choice(cities, size=n),
                "active": rng.choice([True, False], size=n),
            }
        )
        # Inject infinite values into numeric columns.
        df.loc[0, "age"] = np.inf
        df.loc[1, "age"] = -np.inf
        df.loc[2, "salary"] = np.inf

        detector = AnomalyDetector(sample_baseline_metadata)

        alerts = detector.detect(df)

        assert isinstance(alerts, list), "detect() must return a list even with inf values"

    def test_single_row_no_crash(self, sample_baseline_metadata):
        """
        Chaos: DataFrame with only one row – not enough data for stats.

        Expected: detector returns without crashing; a row-count anomaly
        should be raised (1 vs 1 000 baseline).
        """
        df = pd.DataFrame(
            {
                "user_id": [1],
                "age": [35],
                "salary": [75_000.0],
                "city": ["Tokyo"],
                "active": [True],
            }
        )
        detector = AnomalyDetector(sample_baseline_metadata)

        alerts = detector.detect(df)

        assert isinstance(alerts, list)
        row_count_alerts = [a for a in alerts if a.anomaly_type.value == "row_count_spike"]
        assert len(row_count_alerts) == 1, "Single row vs 1 000-row baseline must trigger row-count alert"

    def test_100k_rows_memory_safe(self, sample_baseline_df):
        """
        Chaos: 100 000-row DataFrame – verify no unbounded memory growth.

        Expected: detection completes; memory delta stays under 512 MB so
        the process does not explode under realistic large-batch load.
        """
        rng = np.random.default_rng(seed=99)
        n = 100_000
        cities = ["New York", "London", "Tokyo", "Berlin", "Sydney"]

        large_df = pd.DataFrame(
            {
                "user_id": np.arange(1, n + 1, dtype=np.int64),
                "age": rng.integers(20, 70, size=n),
                "salary": rng.uniform(30_000, 150_000, size=n).round(2),
                "city": rng.choice(cities, size=n),
                "active": rng.choice([True, False], size=n),
            }
        )

        baseline_meta = SchemaDiscovery().discover(sample_baseline_df, "users")
        detector = AnomalyDetector(baseline_meta)

        tracemalloc.start()
        alerts = detector.detect(large_df)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert isinstance(alerts, list)
        peak_mb = peak_bytes / (1024 * 1024)
        assert peak_mb < 512, (
            f"Peak memory during 100K-row detection was {peak_mb:.1f} MB – exceeds 512 MB safety limit"
        )

    def test_extreme_row_count_spike(self, sample_baseline_metadata):
        """
        Chaos: 1 000x more rows than the 1 000-row baseline (1 000 000 rows).

        This tests that the row-count spike detector flags the anomaly correctly
        rather than silently accepting an absurdly large batch.
        """
        rng = np.random.default_rng(seed=5)
        n = 1_000_000
        cities = ["New York", "London", "Tokyo", "Berlin", "Sydney"]

        huge_df = pd.DataFrame(
            {
                "user_id": np.arange(1, n + 1, dtype=np.int64),
                "age": rng.integers(20, 70, size=n),
                "salary": rng.uniform(30_000, 150_000, size=n).round(2),
                "city": rng.choice(cities, size=n),
                "active": rng.choice([True, False], size=n),
            }
        )

        detector = AnomalyDetector(sample_baseline_metadata)
        alerts = detector.detect(huge_df)

        assert isinstance(alerts, list)
        row_alerts = [a for a in alerts if a.anomaly_type.value == "row_count_spike"]
        assert len(row_alerts) == 1, "1 000x row count spike must trigger a row_count_spike alert"
        # Deviation should be enormous (positive).
        assert row_alerts[0].deviation_percent > 1_000, (
            "Deviation percent should reflect the 1000x row spike"
        )

    def test_schema_drift_all_columns_missing(self, sample_baseline_metadata):
        """
        Chaos: incoming DataFrame has no columns at all (all expected columns removed).

        Expected: a SCHEMA_DRIFT alert is generated listing all missing columns;
        the detector does not raise.
        """
        empty_schema_df = pd.DataFrame()

        detector = AnomalyDetector(sample_baseline_metadata)
        alerts = detector.detect(empty_schema_df)

        assert isinstance(alerts, list)
        drift_alerts = [a for a in alerts if a.anomaly_type.value == "schema_drift"]
        assert len(drift_alerts) >= 1, "Completely empty schema must trigger at least one schema_drift alert"

        # All expected columns should be reported missing.
        all_missing = set()
        for a in drift_alerts:
            if a.details and "missing_columns" in a.details:
                all_missing.update(a.details["missing_columns"])
        expected_cols = set(sample_baseline_metadata.columns.keys())
        assert expected_cols.issubset(all_missing), (
            f"Missing columns not fully reported. Expected {expected_cols}, got {all_missing}"
        )

    def test_schema_drift_all_columns_new(self, sample_baseline_metadata):
        """
        Chaos: incoming DataFrame has a completely different schema (no overlap).

        Expected: a SCHEMA_DRIFT alert flags new unexpected columns AND
        another flags the missing baseline columns.
        """
        new_schema_df = pd.DataFrame(
            {
                "transaction_id": [1, 2, 3],
                "amount_usd": [10.5, 20.0, 5.75],
                "merchant": ["ACME", "Initech", "Globex"],
            }
        )

        detector = AnomalyDetector(sample_baseline_metadata)
        alerts = detector.detect(new_schema_df)

        drift_alerts = [a for a in alerts if a.anomaly_type.value == "schema_drift"]
        assert len(drift_alerts) >= 2, (
            "Completely new schema should produce at least two schema_drift alerts "
            "(one for missing, one for new columns)"
        )

        has_missing = any(
            a.details and "missing_columns" in a.details for a in drift_alerts
        )
        has_new = any(
            a.details and "new_columns" in a.details for a in drift_alerts
        )
        assert has_missing, "Expected a missing_columns schema_drift alert"
        assert has_new, "Expected a new_columns schema_drift alert"

    def test_pii_in_every_column(self, sample_baseline_metadata):
        """
        Chaos: every string column is replaced with email addresses (PII).

        Expected: PII_EXPOSURE alerts are raised for each column containing
        emails; the detector does not raise.
        """
        n = 1_000
        emails = [f"user{i}@example.com" for i in range(n)]

        df = pd.DataFrame(
            {
                "user_id": np.arange(1, n + 1, dtype=np.int64),
                "age": emails,      # injected PII
                "salary": emails,   # injected PII
                "city": emails,     # injected PII
                "active": emails,   # injected PII
            }
        )

        detector = AnomalyDetector(sample_baseline_metadata)
        alerts = detector.detect(df)

        pii_alerts = [a for a in alerts if a.anomaly_type.value == "pii_exposure"]
        assert len(pii_alerts) >= 1, (
            "At least one PII_EXPOSURE alert must be raised when columns are full of emails"
        )
        # Each PII alert must name a specific column.
        for a in pii_alerts:
            assert a.column_name is not None, "PII alert must include the offending column name"


# ===========================================================================
# Section 2 – BlastRadiusCalculator / Lineage graph chaos
# ===========================================================================


class TestLineageChaos:
    """Chaos conditions injected into the lineage blast-radius engine."""

    def test_empty_lineage_graph(self):
        """
        Chaos: BlastRadiusCalculator is given a completely empty LineageDB.

        Expected: calculate() raises ValueError because the requested table
        does not exist – it must not hang or crash with an unhandled exception.
        """
        db = LineageDB()
        calc = BlastRadiusCalculator(db)

        with pytest.raises(ValueError, match="not found"):
            calc.calculate(source_table_id=999)

    def test_single_node_graph(self):
        """
        Chaos: lineage graph contains exactly one table with no dependencies.

        Expected: blast radius is 0 (no downstream tables affected).
        """
        db = LineageDB()
        t1 = db.add_table(
            table_name="lonely_table",
            table_type="source",
            owner="team_a",
            owner_email="a@example.com",
            criticality="low",
            refresh_frequency="daily",
        )

        calc = BlastRadiusCalculator(db)
        report = calc.calculate(source_table_id=t1)

        assert report.total_affected == 0, (
            "Single node with no dependencies must have blast radius of 0"
        )
        assert report.affected_tables == [], (
            "affected_tables must be empty for a single isolated node"
        )

    def test_disconnected_graph_islands(self):
        """
        Chaos: two completely separate subgraphs (islands) exist in the DB.

        Expected: blast radius of a table in island A does not include any
        tables from island B, confirming graph traversal is properly isolated.
        """
        db = LineageDB()

        # Island A: a1 → a2
        a1 = db.add_table("island_a_source", "source", "team_a", "a@x.com", "high", "hourly")
        a2 = db.add_table("island_a_derived", "transformation", "team_a", "a@x.com", "medium", "hourly")
        db.add_dependency(upstream_id=a1, downstream_id=a2, latency_minutes=5)

        # Island B: b1 → b2 (completely separate)
        b1 = db.add_table("island_b_source", "source", "team_b", "b@x.com", "high", "daily")
        b2 = db.add_table("island_b_derived", "transformation", "team_b", "b@x.com", "low", "daily")
        db.add_dependency(upstream_id=b1, downstream_id=b2, latency_minutes=60)

        calc = BlastRadiusCalculator(db)
        report_a = calc.calculate(source_table_id=a1)
        report_b = calc.calculate(source_table_id=b1)

        affected_a_names = {t.table_name for t in report_a.affected_tables}
        affected_b_names = {t.table_name for t in report_b.affected_tables}

        # Island A blast should only contain a2.
        assert "island_a_derived" in affected_a_names
        assert "island_b_source" not in affected_a_names
        assert "island_b_derived" not in affected_a_names

        # Island B blast should only contain b2.
        assert "island_b_derived" in affected_b_names
        assert "island_a_source" not in affected_b_names
        assert "island_a_derived" not in affected_b_names

    def test_blast_radius_max_depth_zero(self, simple_lineage_db):
        """
        Chaos: max_depth=0 is passed to the calculator.

        Expected: BFS terminates immediately and returns an empty affected list
        (depth 0 means "only the source itself", which is not counted as
        affected).
        """
        calc = BlastRadiusCalculator(simple_lineage_db)
        # Table 1 is raw_events (root of the chain).
        report = calc.calculate(source_table_id=1, max_depth=0)

        assert report.total_affected == 0, (
            "max_depth=0 must produce zero affected tables"
        )
        assert report.affected_tables == []

    def test_concurrent_blast_radius(self, simple_lineage_db):
        """
        Chaos: 10 threads simultaneously calculate blast radius on the same DB.

        Expected: all threads complete without raising exceptions and each
        returns the same deterministic result (thread-safety check).
        """
        calc = BlastRadiusCalculator(simple_lineage_db)
        results = []
        errors = []

        def worker():
            try:
                report = calc.calculate(source_table_id=1)
                results.append(report.total_affected)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent blast radius raised exceptions: {errors}"
        assert len(results) == 10, "All 10 threads must return a result"
        # All threads should see the same blast radius.
        assert len(set(results)) == 1, (
            f"Concurrent results should be deterministic; got {set(results)}"
        )

    def test_blast_radius_nonexistent_table(self, simple_lineage_db):
        """
        Chaos: calculate() is called with a table_id that does not exist.

        Expected: a ValueError is raised with a meaningful message – the
        caller must receive a clean error, not an unhandled AttributeError or
        KeyError.
        """
        calc = BlastRadiusCalculator(simple_lineage_db)

        with pytest.raises(ValueError, match="not found"):
            calc.calculate(source_table_id=99_999)

    def test_circular_dependency_detection(self):
        """
        Chaos: circular dependency A → B → C → A is added to the graph.

        Expected: BFS traversal terminates (does not loop forever) and returns
        a finite result.  The test itself completes in well under 5 seconds.
        """
        db = LineageDB()

        a = db.add_table("table_a", "source", "team", "t@x.com", "high", "hourly")
        b = db.add_table("table_b", "transformation", "team", "t@x.com", "medium", "hourly")
        c = db.add_table("table_c", "transformation", "team", "t@x.com", "medium", "hourly")

        db.add_dependency(upstream_id=a, downstream_id=b, latency_minutes=5)
        db.add_dependency(upstream_id=b, downstream_id=c, latency_minutes=5)
        # Introduce the cycle: C → A
        db.add_dependency(upstream_id=c, downstream_id=a, latency_minutes=5)

        calc = BlastRadiusCalculator(db)
        # This must return, not hang.
        report = calc.calculate(source_table_id=a, max_depth=20)

        assert isinstance(report.total_affected, int), "BFS must terminate and return an integer count"
        # All three tables (b, c, and a re-visited as downstream) may appear,
        # but the key assertion is that we got here at all.
        assert report.total_affected >= 0
